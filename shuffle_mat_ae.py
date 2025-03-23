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
from ContModeling.helper_classes import MatData, AETrain

print("Fetching device...")
torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Input to the train autoencoder function is train_dataset.dataset.matrices

# +
@hydra.main(config_path=".", config_name="shuffle_mat_ae_config")

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
        
    dataset_path = cfg.dataset.dataset_path
    targets = list(cfg.targets)
    synth_exp = cfg.synth_exp
    dataset = MatData(dataset_path, targets, synth_exp, reduced_mat=False, threshold=0)
    n_sub = len(dataset)
    indices = np.arange(n_sub)
    if cfg.dataset.name == "hcp":
        indices = indices[indices!=249]
    elif cfg.dataset.name == "abcd":
        indices = indices[indices!=863]
    
    train_ratios = list(cfg.train_ratio)
    # test_ratio = cfg.test_ratio
    
    # if cfg.external_test_mode: # deserves a separate function in the long run
    #     test_scanners = list(cfg.test_scanners)
    #     xr_dataset = xr.open_dataset(cfg.dataset_path)
    #     scanner_mask = np.sum([xr_dataset.isin(scanner).scanner.values for scanner in test_scanners],
    #                           axis = 0).astype(bool)
    #     test_idx = indices[scanner_mask]
    #     print("Size of test set: ", len(test_idx))
    #     train_val_idx = indices[~scanner_mask]
    #     print("Size of train set: ", len(train_val_idx))
    #     del xr_dataset
    # else:
    #     train_val_idx, test_idx = train_test_split(indices, test_size=test_ratio, random_state=random_state)
                
    multi_gpu = cfg.multi_gpu
    
    if multi_gpu:
        print("Using multi-gpu")
        log_folder = Path("logs")
        executor = submitit.AutoExecutor(folder=str(log_folder / "%j"))
        executor.update_parameters(
            timeout_min=240,
            slurm_partition="gpu_short",
            gpus_per_node=1,
            # tasks_per_node=1,
            # nodes=1
        )
        run_jobs = []
        with executor.batch():
            for seed in tqdm(list(cfg.random_seed), desc="Random Seed"):
                random_state = np.random.RandomState(seed)
                train_idx, test_idx = train_test_split(indices, test_size=cfg.test_ratio, random_state=random_state)
                
                for train_ratio in tqdm(train_ratios, desc="Training Size"):

                    train_size = int(len(train_idx) * train_ratio)
                    train_idx = random_state.choice(train_idx, train_size, replace=False)
                    
                    np.save(f"{results_dir}/train_idx_seed{seed}_train_ratio{train_ratio}.npy", train_idx)
                    np.save(f"{results_dir}/test_idx_seed{seed}_train_ratio{train_ratio}.npy", test_idx)
                    
                    train_run = AETrain()
                    job = executor.submit(train_run,
                                          train_mat_autoencoder,
                                          train_idx,
                                          test_idx,
                                          train_ratio,
                                          dataset,
                                          model_params_dir = model_params_dir,
                                          cfg =cfg,
                                          random_state = seed
                                         )
                    run_jobs.append(job)

        async def get_result(run_jobs):
            run_results = []
            for aws in tqdm(asyncio.as_completed([j.awaitable().result() for j in run_jobs]), total=len(run_jobs)):
                res = await aws
                run_results.append(res)
            return run_results
        run_results = asyncio.run(get_result(run_jobs))
    else:
        run_results = []
        for seed in tqdm(list(cfg.random_seed), desc="Random Seed"):
            random_state = np.random.RandomState(seed)
            train_idx, test_idx = train_test_split(indices, test_size=cfg.test_ratio, random_state=random_state)
                
            for train_ratio in tqdm(train_ratios, desc="Training Size"):
            
                train_size = int(len(train_idx) * train_ratio)
                train_idx = random_state.choice(train_idx, train_size, replace=False)
                
                np.save(f"{results_dir}/train_idx_seed{seed}_train_ratio{train_ratio}.npy", train_idx)
                np.save(f"{results_dir}/test_idx_seed{seed}_train_ratio{train_ratio}.npy", test_idx)
                        
                train_run = AETrain()
                job = train_run(train_mat_autoencoder,
                                    train_idx,
                                    test_idx,
                                    train_ratio,
                                    dataset,
                                    model_params_dir=model_params_dir,
                                    cfg =cfg,
                                    random_state=seed
                                    )
                run_results.append(job)
    # TEST
    # executor = submitit.AutoExecutor(folder=str(Path("./logs") / "%j"))
    # executor.update_parameters(
    #         timeout_min=240,
    #         slurm_partition="gpu_short",
    #         gpus_per_node=1,
    #         # tasks_per_node=1,
    #         # nodes=1
    # )
    # best_fold = get_best_fold(fold_results)
    # job = executor.submit(test_mat_autoencoder, 
    #                       train_ratio=train_ratio,
    #                       best_fold = best_fold,
    #                       test_dataset=test_dataset,
    #                       cfg = cfg,
    #                       model_params_dir = model_params_dir,
    #                       recon_mat_dir = recon_mat_dir, device = device)
    # output = job.result()
    return run_results
# -

if __name__ == "__main__":
    main()


