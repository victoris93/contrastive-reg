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
from ContModeling.model_testing import test_autoencoder
from ContModeling.utils import get_best_fold, mape_between_subjects, mean_correlations_between_subjects
from ContModeling.losses import LogEuclideanLoss, NormLoss
from ContModeling.models import MatAutoEncoder
from ContModeling.helper_classes import MatData

print("Fetching device...")
torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Input to the train autoencoder function is train_dataset.dataset.matrices
def train_autoencoder(fold, train_dataset, val_dataset, B_init_fMRI, cfg, model=None, device = device):

    wandb.init(project=cfg.project,
       group = f"train_fold_{fold}",
       mode = "offline",
       name=cfg.experiment_name,
       dir = cfg.output_dir)
    wandb.config.update(OmegaConf.to_container(cfg, resolve=True))
    
    input_dim_feat = cfg.input_dim_feat
    output_dim_feat = cfg.output_dim_feat
    batch_size = cfg.batch_size
    lr = cfg.lr
    weight_decay = cfg.weight_decay
    dropout_rate = cfg.dropout_rate
    num_epochs = cfg.num_epochs
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    if model is None:
        model = MatAutoEncoder(
            input_dim_feat,
            output_dim_feat,
            dropout_rate,
            cfg
        ).to(device)
        
    model.enc_mat1.weight = torch.nn.Parameter(B_init_fMRI.transpose(0,1))
    model.enc_mat2.weight = torch.nn.Parameter(B_init_fMRI.transpose(0,1))
    
    if cfg.loss_function == 'LogEuclidean':
        criterion = LogEuclideanLoss()
        optimizer_autoencoder = RiemannianAdam(model.parameters(), lr = lr, weight_decay = weight_decay)
    elif cfg.loss_function == 'Norm':
        criterion = NormLoss()
        optimizer_autoencoder = optim.Adam(model.parameters(), lr = lr, weight_decay = weight_decay)
    elif cfg.loss_function == 'MSE':
        criterion = nn.functional.mse_loss
        optimizer_autoencoder = optim.Adam(model.parameters(), lr = lr, weight_decay = weight_decay)
    else:
        raise ValueError("Unsupported loss function specified in config")
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer_autoencoder,
                                                     factor=0.1,
                                                     patience = cfg.scheduler_patience)
    
    loss_terms = []
    perf_metrics = []
        
    model.train()
    with tqdm(range(num_epochs), desc="Epochs", leave=False) as pbar:
        for epoch in pbar:
            recon_loss = 0
            loss_terms_batch = defaultdict(lambda:0)
            for features, targets in train_loader:
                
                optimizer_autoencoder.zero_grad()
                features = features.to(device)
                
                embedded_feat = model.encode_feat(features)
                reconstructed_feat = model.decode_feat(embedded_feat)
                
                loss = recon_loss + criterion(features,reconstructed_feat)
                loss.backward()
                optimizer_autoencoder.step()
                loss_terms_batch['loss'] += loss.item() / len(train_loader)
                
            model.eval()
            val_loss = 0
            val_mean_corr = 0
            val_mape = 0
            with torch.no_grad():
                for features, targets in val_loader:
                    features = features.to(device)

                    embedded_feat = model.encode_feat(features)
                    reconstructed_feat = model.decode_feat(embedded_feat)
                    
                    val_loss += criterion(features, reconstructed_feat)
                    val_mean_corr += mean_correlations_between_subjects(features, reconstructed_feat)
                    val_mape += mape_between_subjects(features, reconstructed_feat).item()
                
            val_loss /= len(val_loader)
            val_mean_corr /= len(val_loader)
            val_mape /= len(val_loader)
            
            wandb.log({
                "Fold" : fold,
                "Epoch" : epoch,
                "Loss/val" : val_loss.item(),
                "Metric/val_mean_corr" : val_mean_corr,
                "Metric/val_mape" : val_mape
            })
            
            loss_terms.append(('Validation', val_loss.item(), val_mean_corr, val_mape))
            
            scheduler.step(val_loss)
            if np.log10(scheduler._last_lr[0]) < -4:
                break

            pbar.set_postfix_str(f"Epoch {epoch} | Fold {fold} | Train Loss {loss:.02f} | Val Loss {val_loss:.02f} | Val Mean Corr {val_mean_corr:.02f} | Val MAPE {val_mape:.02f} | log10 lr {np.log10(scheduler._last_lr[0])}") # Train corr {train_mean_corr:.02f}| Train mape {train_mape:.02f}
            
    wandb.finish()
    print(loss_terms)
    
    return loss_terms, model.state_dict(), val_loss.item()


class FoldTrain(submitit.helpers.Checkpointable):
    def __init__(self):
        self.results = None

    def __call__(self, fold, train_idx, val_idx, train_dataset, model_params_dir, cfg, random_state=None, device=None, path: Path = None):

        if self.results is None:
            if device is None:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if not isinstance(random_state, np.random.RandomState):
                random_state = np.random.RandomState(random_state)
        
        self.fold = fold + 1
    
        input_dim_feat=cfg.input_dim_feat
        output_dim_feat=cfg.output_dim_feat


        print(f"Fold {self.fold}")
        fold_train_dataset = Subset(train_dataset, train_idx)
        fold_val_dataset = Subset(train_dataset, val_idx)
        
        train_features = torch.stack([fold_train_dataset[i][0] for i in range(len(fold_train_dataset))])
        mean_f = torch.mean(torch.tensor(train_features), dim=0).to(device)
        [D,V] = torch.linalg.eigh(mean_f,UPLO = "U")     
        B_init_fMRI = V[:,input_dim_feat-output_dim_feat:]
    
        loss_terms, trained_weights, val_loss = train_autoencoder(self.fold, fold_train_dataset, fold_val_dataset, B_init_fMRI, cfg = cfg)
        torch.save(trained_weights, f"{model_params_dir}/autoencoder_weights_fold{self.fold}.pth")
        
        self.results = {"fold": self.fold,
                        "val_loss": val_loss,
                        } 
        return self.results

    def checkpoint(self, *args, **kwargs):
        print("Checkpointing", flush=True)
        return super().checkpoint(*args, **kwargs)

    def save(self, path: Path):
        with open(path, "wb") as o:
            pickle.dump(self.results, o, pickle.HIGHEST_PROTOCOL)

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
    tensorboard_dir = os.path.join(results_dir, cfg.tensorboard_dir)
    os.makedirs(tensorboard_dir, exist_ok=True)
    
    random_state = np.random.RandomState(seed=42)
    
    dataset_path = cfg.dataset_path
    targets = list(cfg.targets)
    dataset = MatData(dataset_path, targets, threshold=0)
    train_val_idx, test_idx = train_test_split(np.arange(len(dataset)), test_size=0.2, random_state=random_state)
    train_val_dataset = Subset(dataset, train_val_idx)
    test_dataset = Subset(dataset, test_idx)
    np.save(f"{results_dir}/test_idx.npy", test_idx)
    
    kf = KFold(n_splits=cfg.kfolds, shuffle=True, random_state=random_state)
    multi_gpu = cfg.multi_gpu
    
    if multi_gpu:
        log_folder = Path("./logs")
        executor = submitit.AutoExecutor(folder=str(log_folder / "%j"))
        executor.update_parameters(
            timeout_min=120,
#             slurm_account="ftj@a100",
            slurm_partition="gpu_short",
            gpus_per_node=1,
            tasks_per_node=1,
            nodes=1,
            cpus_per_task=30
            # slurm_constraint="a100",
        )
        fold_jobs = []
        with executor.batch():
            for fold, (train_idx, val_idx) in enumerate(kf.split(train_val_idx)):
                run_train_fold = FoldTrain()
                job = executor.submit(run_train_fold, fold, train_idx, val_idx, train_val_dataset, model_params_dir = model_params_dir, cfg =cfg, random_state = random_state)
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
            job = run_train_fold(fold, train_idx, val_idx, train_val_dataset, random_state = random_state, model_params_dir = model_params_dir, cfg = cfg)
            fold_results.append(job)
    # TEST
    executor = submitit.AutoExecutor(folder=str(Path("./logs") / "%j"))
    executor.update_parameters(
        timeout_min=120,
#       slurm_account="ftj@a100",
        slurm_partition="gpu_short",
        gpus_per_node=1,
        tasks_per_node=1,
        nodes=1,
        cpus_per_task=30
        # slurm_constraint="a100",
    )
    best_fold = get_best_fold(fold_results)
    job = executor.submit(test_autoencoder, best_fold = best_fold, test_dataset =test_dataset, cfg = cfg, model_params_dir = model_params_dir,
                            recon_mat_dir = recon_mat_dir, device = device)
    output = job.result()
    print(output)
# -

if __name__ == "__main__":
    main()


