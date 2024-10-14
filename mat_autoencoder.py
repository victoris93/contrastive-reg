import wandb
import math
import xarray as xr
import asyncio
import submitit
import pickle
import sys
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import gc
from collections import defaultdict
from nilearn.connectome import sym_matrix_to_vec
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import (
    train_test_split, KFold
)
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset
from tqdm.auto import tqdm
# from augmentations import augs, aug_args
import glob, os, shutil
from nilearn.datasets import fetch_atlas_schaefer_2018
import random
from geoopt.optim import RiemannianAdam
import sys
from ContModeling.modeling import test_mat_autoencoder, train_mat_autoencoder
from ContModeling.utils import get_best_fold, mape_between_subjects, mean_correlations_between_subjects
from ContModeling.losses import LogEuclideanLoss, NormLoss
from ContModeling.models import MatAutoEncoder
from ContModeling.helper_classes import MatData, FoldTrain

print("Fetching device...")
torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Input to the train autoencoder function is train_dataset.dataset.matrices

# +
@hydra.main(config_path=".", config_name="mat_autoencoder_config")

def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
 
    results_dir = os.path.join(cfg.output_dir, cfg.experiment_name)
    os.makedirs(results_dir, exist_ok=True)
    recon_mat_dir = os.path.join(results_dir, cfg.reconstructed_dir)
    os.makedirs(recon_mat_dir, exist_ok=True)
    model_params_dir = os.path.join(results_dir, cfg.model_weight_dir)
    os.makedirs(model_params_dir, exist_ok=True)
    embedding_dir = os.path.join(results_dir, cfg.embedding_dir)
    os.makedirs(embedding_dir, exist_ok=True)
    
    random_state = np.random.RandomState(seed=cfg.seed)
    
    dataset_path = cfg.dataset_path
    targets = list(cfg.targets)
    dataset = MatData(dataset_path, targets, threshold=0)
    indices = np.arange(len(dataset))

    if cfg.external_test_mode: # deserves a separate function in the long run
        test_scanners = list(cfg.test_scanners)
        xr_dataset = xr.open_dataset(cfg.dataset_path)
        scanner_mask = np.sum([xr_dataset.isin(scanner).scanner.values for scanner in test_scanners],
                              axis = 0).astype(bool)
        test_idx = indices[scanner_mask]
        print("Size of test set: ", len(test_idx))
        train_val_idx = indices[~scanner_mask]
        print("Size of train set: ", len(train_val_idx))
        del xr_dataset
    else:
        train_val_idx, test_idx = train_test_split(indices, test_size=cfg.test_size, random_state=random_state)

    train_val_dataset = Subset(dataset, train_val_idx)
    test_dataset = Subset(dataset, test_idx)
    np.save(f"{results_dir}/test_idx.npy", test_idx)
    
    kf = KFold(n_splits=cfg.kfolds, shuffle=True, random_state=random_state)
    multi_gpu = cfg.multi_gpu
    
    if multi_gpu:
        print("Using multi-gpu")
        log_folder = Path("logs")
        executor = submitit.AutoExecutor(folder=str(log_folder / "%j"))
        executor.update_parameters(
            timeout_min=240,
            slurm_partition="gpu_short",
            gpus_per_node=1,
            tasks_per_node=1,
            nodes=1
        )
        fold_jobs = []
        with executor.batch():
            for fold, (train_idx, val_idx) in enumerate(kf.split(train_val_idx)):
                run_train_fold = FoldTrain()
                job = executor.submit(run_train_fold, fold, train_mat_autoencoder, train_idx, val_idx, train_val_dataset, model_params_dir = model_params_dir, cfg =cfg, random_state = random_state)
                fold_jobs.append(job)

        async def get_result(fold_jobs):
            fold_results = []
            for aws in tqdm(asyncio.as_completed([j.awaitable().result() for j in fold_jobs]), total=len(fold_jobs)):
                res = await aws
                fold_results.append(res)
            return fold_results
        fold_results = asyncio.run(get_result(fold_jobs))
    else:
        fold_results = []
        for fold, (train_idx, val_idx) in enumerate(kf.split(train_val_idx)):
            run_train_fold = FoldTrain()
            job = run_train_fold(fold, train_mat_autoencoder, train_idx, val_idx, train_val_dataset, model_params_dir = model_params_dir, cfg =cfg, random_state = random_state)
            fold_results.append(job)
    # TEST
    executor = submitit.AutoExecutor(folder=str(Path("./logs") / "%j"))
    executor.update_parameters(
            timeout_min=240,
            slurm_partition="gpu_short",
            gpus_per_node=1,
            tasks_per_node=1,
            nodes=1
    )
    best_fold = get_best_fold(fold_results)
    job = executor.submit(test_mat_autoencoder, best_fold = best_fold, test_dataset =test_dataset, cfg = cfg, model_params_dir = model_params_dir,
                            recon_mat_dir = recon_mat_dir, device = device)
    output = job.result()
    return output
# -

if __name__ == "__main__":
    main()


