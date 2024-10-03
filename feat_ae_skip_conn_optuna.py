"""
Feature autoencoder with skipp connection to run with optuna sweeper plugin on hydra
"""
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
# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import (
    train_test_split, KFold
)
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset
from tqdm.auto import tqdm
from augmentations import augs, aug_args
import glob, os, shutil
from nilearn.datasets import fetch_atlas_schaefer_2018
import random
from geoopt.optim import RiemannianAdam
from torch.utils.tensorboard import SummaryWriter
import sys
# from viz_func import wandb_plot_corr

torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LogEuclideanLoss(nn.Module):
    def __init__(self):
        super(LogEuclideanLoss, self).__init__()

    def mat_batch_log(self, features):
        eps = 1e-6
        regularized_features = features + eps * \
            torch.eye(features.size(-1), device=features.device)
        Eigvals, Eigvecs = torch.linalg.eigh(regularized_features)
        Eigvals = torch.clamp(Eigvals, min=eps)
        log_eigvals = torch.diag_embed(torch.log(Eigvals))
        matmul1 = torch.matmul(log_eigvals, Eigvecs.transpose(-2, -1))
        matmul2 = torch.matmul(Eigvecs, matmul1)
        return matmul2

    def forward(self, features, recon_features):
        """
        Compute the Log-Euclidean distance between two batches of SPD matrices.

        Args:
            features: Tensor of shape [batch_size, n_parcels, n_parcels]
            recon_features: Tensor of shape [batch_size, n_parcels, n_parcels]

        Returns:
            A loss scalar.
        """
        device = features.device
        eye = torch.eye(features.size(-1), device=device)
        recon_features_diag = recon_features*(1-eye)+eye
        recon_features_diag = torch.round(recon_features_diag, decimals=3)

        log_features = self.mat_batch_log(features)
        log_recon_features = self.mat_batch_log(recon_features_diag)
        loss = torch.norm(log_features - log_recon_features,
                          dim=(-2, -1)).mean()
        return loss

class NormLoss(nn.Module):
    def __init__(self):
        super(NormLoss, self).__init__()
    
    def forward(self, features, recon_features):
        """
        Compute the Frobenius norm-based loss between two batches of matrices.

        Args:
            features: Tensor of shape [batch_size, n_parcels, n_parcels]
            recon_features: Tensor of shape [batch_size, n_parcels, n_parcels]
        
        Returns:
            A loss scalar.
        """
        loss = torch.norm(features - recon_features) ** 2
        return loss


class AutoEncoder(nn.Module):
    def __init__(
        self,
        input_dim_feat,
        output_dim_feat,
        B_init_fMRI,
        dropout_rate,
        cfg
    ):
        super(AutoEncoder, self).__init__()
        self.cfg = cfg
        # ENCODE MATRICES
        self.enc_mat1 = nn.Linear(in_features=input_dim_feat, out_features=output_dim_feat ,bias=False)
        self.enc_mat2 = nn.Linear(in_features=input_dim_feat, out_features=output_dim_feat, bias=False)
        self.enc_mat2.weight = torch.nn.Parameter(self.enc_mat1.weight)
        
        # DECODE MATRICES
        self.dec_mat1 = nn.Linear(in_features=output_dim_feat, out_features=input_dim_feat, bias=False)
        self.dec_mat2 = nn.Linear(in_features=output_dim_feat, out_features=input_dim_feat, bias=False)
        self.dec_mat1.weight = torch.nn.Parameter(self.enc_mat1.weight.transpose(0,1))
        self.dec_mat2.weight = torch.nn.Parameter(self.dec_mat1.weight)
        self.dropout = nn.Dropout(p=dropout_rate)
        
    def encode_feat(self, x):

        rect = self.cfg.ReEig
        z_n = self.enc_mat1(x)
        self.skip_enc_mat1 = z_n.detach().clone()

        c_hidd_fMRI = self.enc_mat2(z_n.transpose(1,2))
        if rect:
            reig = ReEig()
            c_hidd_fMRI = reig(c_hidd_fMRI)
        
        return c_hidd_fMRI
    
    def decode_feat(self,c_hidd_mat):

        skip_enc1 = self.cfg.skip_enc1
        z_n = self.dec_mat1(c_hidd_mat).transpose(1,2)

        if skip_enc1: # long skip conn
            z_n += self.skip_enc_mat1
        
        recon_mat = self.dec_mat2(z_n)

        return recon_mat

class ReEig(nn.Module):
    def __init__(self, epsilon=1e-4):
        super(ReEig, self).__init__()
        self.epsilon = epsilon

    def forward(self, X):
        D, V = torch.linalg.eigh(X)
        D = torch.clamp(D, min=self.epsilon)
        X_rectified = V @ torch.diag_embed(D) @ V.transpose(-2, -1)
        
        return X_rectified


class MatData(Dataset):
    def __init__(self, dataset_path, target_names, threshold=0):
        if not isinstance(target_names, list):
            target_names = [target_names]
        self.target_names = target_names
        self.threshold = threshold
        self.data_array = xr.open_dataset(dataset_path)
        self.matrices = self.data_array.to_array().squeeze().values.astype(np.float32)
        if threshold > 0:
            self.matrices = self.threshold_mat(self.matrices, self.threshold)
        self.matrices = torch.from_numpy(self.matrices).to(torch.float32)
        self.target = torch.from_numpy(np.array([self.data_array[target_name].values for target_name in self.target_names]).T).to(torch.float32)

        gc.collect()

    def threshold_mat(self, matrices, threshold): # as in Margulies et al. (2016)
        perc = np.percentile(np.abs(matrices), threshold, axis=2, keepdims=True)
        mask = np.abs(matrices) >= perc
        thresh_mat = matrices * mask
        return thresh_mat
    
    def __len__(self):
        return self.data_array.subject.__len__()
    
    def __getitem__(self, idx):
        matrix = self.matrices[idx]
        target = torch.from_numpy(np.array([self.data_array.sel(subject=idx)[target_name].values for target_name in self.target_names])).to(torch.float32)
        
    
    def __len__(self):
        return self.data_array.subject.__len__()
    
    def __getitem__(self, idx):
#         matrix = self.data_array.sel(subject = idx).to_array().values
#         if self.threshold > 0:
#             matrix = self.threshold_mat(matrix, self.threshold)
        matrix = self.matrices[idx]
        target = torch.from_numpy(np.array([self.data_array.sel(subject=idx)[target_name].values for target_name in self.target_names])).to(torch.float32)
        
        return matrix, target


def mean_absolute_percentage_error(y_true, y_pred):
    eps = 1e-6
    return torch.mean(torch.abs((y_true - y_pred)) / torch.abs(y_true)+eps) * 100


def mape_between_subjects(y_true, y_pred):
    eps = 1e-6
    mapes = []
    y_true = y_true.cpu().detach().numpy()  # Convert to NumPy array if using PyTorch tensor
    y_pred = y_pred.cpu().detach().numpy()  # Convert to NumPy array if using PyTorch tensor

    num_subjects = y_true.shape[0]
    matrix_size = y_true.shape[1]

    # Flatten upper triangle (excluding diagonal) for both y_true and y_pred
    upper_true = []
    upper_pred = []

    for subj in range(num_subjects):
        for i in range(matrix_size):
            for j in range(i + 1, matrix_size):
                true_val = y_true[subj, i, j]
                pred_val = y_pred[subj, i, j]

                # Add epsilon to denominator to avoid division by zero
                mape = np.abs((true_val - pred_val) / (true_val + eps)) * 100.0
                upper_true.append(true_val)
                upper_pred.append(pred_val)
                mapes.append(mape)

    # Calculate mean MAPE
    mean_mape = np.mean(mapes)

    return mean_mape


def mean_correlation(y_true, y_pred):
    correlations = []
    y_true = y_true.cpu().detach().numpy()
    y_pred = y_pred.cpu().detach().numpy()
    for i in range(y_true.shape[0]):
        corr, _ = spearmanr(y_true[i].flatten(), y_pred[i].flatten())
        correlations.append(corr)
    return np.mean(correlations)


def mean_correlations_between_subjects(y_true, y_pred):
    correlations = []
    y_true = y_true.cpu().detach().numpy()  # Convert to NumPy array if using PyTorch tensor
    y_pred = y_pred.cpu().detach().numpy()  # Convert to NumPy array if using PyTorch tensor
    
    num_subjects = y_true.shape[0]
    matrix_size = y_true.shape[1]
    
    # Flatten upper triangle (excluding diagonal) for both y_true and y_pred
    upper_true = []
    upper_pred = []
    
    for subj in range(num_subjects):
        for i in range(matrix_size):
            for j in range(i + 1, matrix_size):
                upper_true.append(y_true[subj, i, j])
                upper_pred.append(y_pred[subj, i, j])
    
    # Calculate Spearman correlation
    spearman_corr, _ = spearmanr(upper_true, upper_pred)
    correlations.append(spearman_corr)
    correlation = np.mean(correlations)
    
    return correlation


def mean_correlation(y_true, y_pred):
    correlations = []
    y_true = y_true.cpu().detach().numpy()
    y_pred = y_pred.cpu().detach().numpy()
    for i in range(y_true.shape[0]):
        corr, _ = spearmanr(y_true[i].flatten(), y_pred[i].flatten())
        correlations.append(corr)
    return np.mean(correlations)


#Input to the train autoencoder function is train_dataset.dataset.matrices
def train_autoencoder(fold, train_dataset, val_dataset, B_init_fMRI, cfg, model=None, device = device):
    
    input_dim_feat = cfg.input_dim_feat
    output_dim_feat = cfg.output_dim_feat
    batch_size = cfg.batch_size
    lr = cfg.model.lr
    print("lr", lr)
    weight_decay = cfg.weight_decay
    dropout_rate = cfg.dropout_rate
    num_epochs = cfg.num_epochs
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    if model is None:
        model = AutoEncoder(
            input_dim_feat,
            output_dim_feat,
            B_init_fMRI,
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
    
    tensorboard_dir = os.path.join(cfg.output_dir, cfg.experiment_name, cfg.tensorboard_dir)
    writer = SummaryWriter(log_dir=tensorboard_dir)
    writer.add_scalar('Fold', fold)
    
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
                writer.add_scalar('Loss/train', loss.item(), epoch)
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
            
            writer.add_scalar('Loss/val', val_loss.item(), epoch)
            writer.add_scalar('Metric/val_mean_corr', val_mean_corr, epoch)
            writer.add_scalar('Metric/val_mape', val_mape, epoch)
            writer.close()
            
            loss_terms.append(('Validation', val_loss.item(), val_mean_corr, val_mape))
            
            scheduler.step(val_loss)
            if np.log10(scheduler._last_lr[0]) < -4:
                break

            pbar.set_postfix_str(f"Epoch {epoch} | Fold {fold} | Train Loss {loss:.02f} | Val Loss {val_loss:.02f} | Val Mean Corr {val_mean_corr:.02f} | Val MAPE {val_mape:.02f} | log10 lr {np.log10(scheduler._last_lr[0])}") # Train corr {train_mean_corr:.02f}| Train mape {train_mape:.02f}

    model_weights = model.state_dict()
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
@hydra.main(config_path="conf_feat_ae_skip_conn_optuna", config_name="conf", version_base = None)

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
            slurm_partition="gpu-best",
            # gpus_per_node=1,
            # tasks_per_node=1,
            # nodes=1,
            cpus_per_task=40,
            #slurm_qos="qos_gpu-t3",
            # slurm_constraint="a100",
            #slurm_mem="10G",
            #slurm_additional_parameters={"requeue": True}
        )
        # srun -n 1  --verbose -A hjt@v100 -c 10 -C v100-32g   --gres=gpu:1 --time 5  python
        fold_jobs = []
        # module_purge = submitit.helpers.CommandFunction("module purge".split())
        # module_load = submitit.helpers.CommandFunction("module load pytorch-gpu/py3/2.0.1".split())
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
    folds = [fold_dict["fold"] for fold_dict in fold_results]
    val_losses = [fold_dict["val_loss"] for fold_dict in fold_results]
    best_fold = folds[val_losses.index(np.min(val_losses))]
    print("BEST FOLD IS: ", best_fold)
    
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)
    input_dim_feat = cfg.input_dim_feat
    output_dim_feat = cfg.output_dim_feat
    lr = cfg.model.lr

    weight_decay = cfg.weight_decay
    dropout_rate = cfg.dropout_rate

    model = AutoEncoder(
            input_dim_feat,
            output_dim_feat,
            torch.randn(input_dim_feat, output_dim_feat),
            dropout_rate,
            cfg
            ).to(device)
    
    model.load_state_dict(torch.load(f"{model_params_dir}/autoencoder_weights_fold{best_fold}.pth"))  # Load the best fold weights
    
    model.eval()
    test_loss = 0
    test_mean_corr = 0
    test_mape = 0
    
    wandb.init(project=cfg.project,  name=cfg.experiment_name)
    wandb.config.update(OmegaConf.to_container(cfg, resolve=True))
    
    if cfg.loss_function == 'LogEuclidean':
            criterion = LogEuclideanLoss()
    elif cfg.loss_function == 'Norm':
            criterion = NormLoss()
    elif cfg.loss_function == 'MSE':
            criterion = LogEuclideanLoss()
    else:
        raise ValueError("Unsupported loss function specified in config")

    writer = SummaryWriter(log_dir=cfg.tensorboard_dir)
    with torch.no_grad():
        for i, (features, targets) in enumerate(test_loader):

            features = features.to(device)

            embedded_feat = model.encode_feat(features)
            reconstructed_feat = model.decode_feat(embedded_feat)
            np.save(f'{recon_mat_dir}/recon_mat_fold{best_fold}_batch_{i+1}', reconstructed_feat.cpu().numpy())
            mape_mat = torch.abs((features - reconstructed_feat) / (features + 1e-10)) * 100
            np.save(f'{recon_mat_dir}/mape_mat_fold{best_fold}_batch_{i+1}', mape_mat.cpu().numpy())
            # wandb_plot_corr(wandb, features.cpu().numpy(), reconstructed_feat.cpu().numpy())

            test_loss += criterion(features,reconstructed_feat)
            test_mean_corr += mean_correlations_between_subjects(features, reconstructed_feat)
            test_mape += mape_between_subjects(features, reconstructed_feat).item()
            
            writer.add_scalar('Loss/test', test_loss.item())
            writer.add_scalar('Metric/test_mean_corr', test_mean_corr)
            writer.add_scalar('Metric/test_mape', test_mape)
            writer.close()
            # wandb.finish()

            
    test_loss /= len(test_loader)
    test_mean_corr /= len(test_loader)
    test_mape /= len(test_loader)
    print(f"Test Loss: {test_loss:.02f} | Test Mean Corr: {test_mean_corr:.02f} | Test MAPE: {test_mape:.02f}")
    
# -

if __name__ == "__main__":
    main()