"""
Python script for the feature autoencoder with a basic implementation of D'Souza's bilinear layer. 

"""


# %%
import math
import asyncio
import pickle
import sys
from pathlib import Path
import gc
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split, KFold
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm.auto import tqdm
from geoopt.optim import RiemannianAdam
from geoopt.manifolds import SymmetricPositiveDefinite
import yaml
import os
import hydra
from omegaconf import DictConfig, OmegaConf
from nilearn.plotting import plot_matrix
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from PIL import Image
from hydra.utils import get_original_cwd
import shutil
from nilearn import datasets
import tabulate

from torch.utils.tensorboard import SummaryWriter


torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# %%
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
        recon_features_diag = torch.round(recon_features, decimals=3)

        log_features = self.mat_batch_log(features)
        log_recon_features = self.mat_batch_log(recon_features_diag)
        loss = torch.norm(log_features - log_recon_features,
                          dim=(-2, -1)).mean()
        return loss

# %%


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

# %%


class AutoEncoder(nn.Module):
    def __init__(
        self,
        input_dim_feat,
        output_dim_feat,
        B_init_fMRI,
        dropout_rate
    ):
        super(AutoEncoder, self).__init__()

        self.enc1 = nn.Linear(in_features=input_dim_feat,
                              out_features=output_dim_feat, bias=False)
        self.enc2 = nn.Linear(in_features=input_dim_feat,
                              out_features=output_dim_feat, bias=False)
        self.enc2.weight = torch.nn.Parameter(self.enc1.weight)

        self.dec1 = nn.Linear(in_features=output_dim_feat,
                              out_features=input_dim_feat, bias=False)
        self.dec2 = nn.Linear(in_features=output_dim_feat,
                              out_features=input_dim_feat, bias=False)
        self.dec1.weight = torch.nn.Parameter(self.enc1.weight.transpose(0, 1))
        self.dec2.weight = torch.nn.Parameter(self.dec1.weight)

        self.dropout = nn.Dropout(p=dropout_rate)

    def encode_feat(self, x):
        z_n = self.enc1(x)
        c_hidd_fMRI = self.enc2(z_n.transpose(1, 2))
        return c_hidd_fMRI

    def decode_feat(self, embedding):
        z_n = (self.dec1(embedding)).transpose(1, 2)
        corr_n = self.dropout(self.dec2(z_n))
        return corr_n

# %%


class MatData(Dataset):
    def __init__(self, path_feat, path_targets, load=True):
        if load == True:
            self.matrices = np.load(
                path_feat, mmap_mode="r").astype(np.float32)
            self.matrices = torch.from_numpy(self.matrices).to(torch.float32)
            self.target = torch.tensor(pd.read_csv(path_targets).drop(columns=["Subject", "Unnamed: 0",
                                                                               'CardioMeasures_pulse_mean',
                                                                               'CardioMeasures_bp_sys_mean',
                                                                               'CardioMeasures_bp_dia_mean']).to_numpy(), dtype=torch.float32)
        else:
            self.matrices = torch.from_numpy(path_feat).to(torch.float32)
            self.target = torch.from_numpy(path_targets).to(torch.float32)

    def __len__(self):
        return len(self.matrices)

    def __getitem__(self, idx):
        matrix = self.matrices[idx]
        target = self.target[idx]
        return matrix, target


# %%
"""

Functions to compute various metrics : MAPE and correlation accross subjects.

"""


def mape_between_subjects(y_true, y_pred):
    eps = 1e-6
    mapes = []
    # Convert to NumPy array if using PyTorch tensor
    y_true = y_true.cpu().detach().numpy()
    # Convert to NumPy array if using PyTorch tensor
    y_pred = y_pred.cpu().detach().numpy()

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
                mape = (np.abs((true_val - pred_val)) /
                        (np.abs(true_val) + eps)) * 100.0
                upper_true.append(true_val)
                upper_pred.append(pred_val)
                mapes.append(mape)
    mean_mape = np.mean(mapes)

    return mean_mape


def mean_correlations_between_subjects(y_true, y_pred):
    correlations = []

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


# %%
"""

Functions to log visualizations and metrics into Tensorboard.

"""


def log_mape_between_subjects_and_region_rank(writer, y_true, y_pred, experiment_dir):
    eps = 1e-6

    num_subjects = y_true.shape[0]
    matrix_size = y_true.shape[1]

    # Calculating MAPE matrix
    mape_matrix = np.zeros((matrix_size, matrix_size))

    for i in range(matrix_size):
        for j in range(i + 1, matrix_size):
            mapes = []
            for subj in range(num_subjects):
                true_val = y_true[subj, i, j]
                pred_val = y_pred[subj, i, j]

                mape = (np.abs((true_val - pred_val)) /
                        (np.abs(true_val) + eps)) * 100.0
                mapes.append(mape)

            mean_mape = np.mean(mapes)
            mape_matrix[i, j] = mean_mape
            mape_matrix[j, i] = mean_mape

    # Log the ranking of predictions by region
    atlas = datasets.fetch_atlas_schaefer_2018(n_rois=100)
    atlas_labels = atlas['labels']
    row_sum = mape_matrix.sum(axis=1)
    df = pd.DataFrame({
        'Region': atlas_labels,
        'MAPE_Sum': row_sum
    })
    df['Rank'] = df['MAPE_Sum'].rank(method='min', ascending=False).astype(int)
    df_sorted = df.sort_values(by='Rank')
    df_final = df_sorted[['Rank', 'Region']]
    df_markdown = df_final.to_markdown(index=False)
    writer.add_text('Metrics/Region Ranking', df_markdown)
    df_final_path = os.path.join(experiment_dir, 'region_ranking.csv')
    df_final.to_csv(df_final_path, index=False)    
    # Log the MAPE matrix
    display = plot_matrix(mape_matrix, figure=(
        10, 8), vmin=0, vmax=300, colorbar=True, cmap='viridis')
    temp_file = f"temp_mape_matrix.png"
    display.figure.savefig(temp_file)
    img = Image.open(temp_file).convert('RGB')
    img = np.array(img).astype(np.float32) / 255.0  # Normalize to [0, 1]
    img_tensor = torch.tensor(img).permute(2, 0, 1)
    writer.add_image('Metrics/MAPE matrix', img_tensor)
    os.remove(temp_file)


def log_correlations_between_subjects(writer, y_true, y_pred):

    num_subjects = y_true.shape[0]
    matrix_size = y_true.shape[1]

    # Initialize matrix to store correlation values
    correlation_matrix = np.zeros((matrix_size, matrix_size))

    for i in range(matrix_size):
        for j in range(i + 1, matrix_size):
            true_vals = []
            pred_vals = []
            for subj in range(num_subjects):
                true_vals.append(y_true[subj, i, j])
                pred_vals.append(y_pred[subj, i, j])

            # Calculate Spearman correlation
            spearman_corr, _ = spearmanr(true_vals, pred_vals)
            correlation_matrix[i, j] = spearman_corr
            # Ensure the matrix is symmetric
            correlation_matrix[j, i] = spearman_corr

    display = plot_matrix(correlation_matrix, figure=(
        5, 4), vmin=-1, vmax=1, colorbar=True)
    temp_file = f"temp_corr_matrix.png"
    display.figure.savefig(temp_file)

    # Log the plot to TensorBoard
    img = Image.open(temp_file).convert('RGB')
    img = np.array(img).astype(np.float32) / 255.0  # Normalize to [0, 1]
    img_tensor = torch.tensor(img).permute(2, 0, 1)

    writer.add_image('Metrics/Correlation matrix', img_tensor)
    # Remove the temporary file
    os.remove(temp_file)

# %%


def train_autoencoder(train_dataset, val_dataset, B_init_fMRI, model=None, device=device, cfg=None, experiment_dir=None):
    input_dim_feat = 100
    output_dim_feat = cfg.output_dim_feat
    lr = cfg.learning_rate
    weight_decay = cfg.weight_decay
    dropout_rate = cfg.dropout_rate
    batch_size = cfg.batch_size
    num_epochs = cfg.num_epochs

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    if model is None:
        model = AutoEncoder(
            input_dim_feat,
            output_dim_feat,
            B_init_fMRI,
            dropout_rate
        ).to(device)

    model.enc1.weight = torch.nn.Parameter(B_init_fMRI.transpose(0, 1))
    model.enc2.weight = torch.nn.Parameter(B_init_fMRI.transpose(0, 1))

    if cfg.loss_function == 'LogEuclidean':
        criterion = LogEuclideanLoss()
        optimizer_autoencoder = RiemannianAdam(
            model.parameters(), lr=lr, weight_decay=weight_decay)
    elif cfg.loss_function == 'Norm':
        criterion = NormLoss()
        optimizer_autoencoder = optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError("Unsupported loss function specified in config")

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer_autoencoder, 100, 0.5, last_epoch=-1)
    loss_terms = []
    model.train()
    with tqdm(range(num_epochs), desc="Epochs", leave=False) as pbar:
        for epoch in pbar:
            optimizer_autoencoder.zero_grad()
            recon_loss = 0
            loss_terms_batch = defaultdict(lambda: 0)
            for features, targets in train_loader:

                features = features.to(device)
                embedded_feat = model.encode_feat(features)
                reconstructed_feat = model.decode_feat(embedded_feat)
                loss = recon_loss + criterion(features, reconstructed_feat)
                loss.backward()
                optimizer_autoencoder.step()
                scheduler.step()
                loss_terms_batch['loss'] += loss.item() / len(train_loader)

            model.eval()
            val_loss = 0
            val_mean_corr = 0
            val_mape = 0
            best_val_loss = 1e10
            best_weights = None
            with torch.no_grad():
                for features, targets in val_loader:
                    features = features.to(device)
                    targets = targets.to(device)
                    eye = torch.eye(features.size(-1), device=device)
                    embedded_feat = model.encode_feat(features)
                    reconstructed_feat = model.decode_feat(embedded_feat)
                    val_loss += criterion(features, reconstructed_feat)
                    val_mean_corr += mean_correlations_between_subjects(
                        features, reconstructed_feat)
                    val_mape += mape_between_subjects(
                        features, reconstructed_feat).item()
                    pbar.set_postfix_str(
                        f"Epoch {epoch} | Train Loss {loss:.02f} | Validation Loss: {val_loss:.02f}")

    val_loss /= len(val_loader)
    val_mean_corr /= len(val_loader)
    val_mape /= len(val_loader)

    loss_terms.append(('Validation', val_loss.item(), val_mean_corr, val_mape))

    # Save best model weights
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_weights = model.state_dict()

    if best_weights is not None:
        model.load_state_dict(best_weights)
    torch.save(model.state_dict(), os.path.join(experiment_dir,
               cfg.weight_dir, "best_autoencoder_weights.pth"))

    return loss_terms, model.state_dict(), val_loss.item()


# %%
@hydra.main(config_path=".", config_name="config_feature_autoencoder_basic")
def main(cfg: DictConfig):
    # Print configuration to track what is being used
    print(OmegaConf.to_yaml(cfg))

    experiment_dir = os.path.join(cfg.output_dir, cfg.experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(os.path.join(experiment_dir,
                cfg.tensorboard_dir), exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, cfg.weight_dir), exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, cfg.original_dir), exist_ok=True)
    os.makedirs(os.path.join(experiment_dir,
                cfg.reconstructed_dir), exist_ok=True)

    # Get the original working directory where Hydra started
    config_path = get_original_cwd()
    config_file = os.path.join(
        config_path, 'config_feature_autoencoder_basic.yaml')
    if os.path.exists(config_file):
        shutil.copy(config_file, os.path.join(experiment_dir,
                    'config_feature_autoencoder_basic.yaml'))
    else:
        print(f"Config file {config_file} does not exist.")

    path_feat = cfg.path_feat
    path_target = cfg.path_targets
    dataset = MatData(path_feat, path_target)
    train_idx, test_idx = train_test_split(
        np.arange(len(dataset)), test_size=0.2, random_state=42)
    train_dataset = Subset(dataset, train_idx)
    test_dataset = Subset(dataset, test_idx)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_models = []
    all_mape = []
    all_corr = []

    best_val_loss = 1e10

    for fold, (train_val_idx, val_idx) in enumerate(kf.split(train_dataset)):

        print(f"Fold {fold + 1}")

        train_data = Subset(train_dataset, train_val_idx)
        train_features = torch.stack(
            [train_data[i][0] for i in range(len(train_data))])
        mean_f = torch.mean(train_features, dim=0).to(device)
        [D, V] = torch.linalg.eigh(mean_f, UPLO="U")
        B_init_fMRI = V[:, cfg.input_dim_feat - cfg.output_dim_feat:]
        train_targets = torch.stack([train_data[i][1]
                                    for i in range(len(train_data))])

        val_data = Subset(train_dataset, val_idx)

        loss_terms, trained_weights, val_loss = train_autoencoder(
            train_data, val_data, B_init_fMRI, cfg=cfg, experiment_dir=experiment_dir)

        weights_path = os.path.join(
            experiment_dir, cfg.weight_dir, f"autoencoder_weights_fold_{fold + 1}.pth")
        torch.save(trained_weights, weights_path)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_fold = fold + 1

    # Evaluate on the test set using the best fold
    print(
        f"Loading weights from fold {best_fold} with lowest validation loss {best_val_loss:.02f}")
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    model = AutoEncoder(cfg.input_dim_feat, cfg.output_dim_feat,
                        B_init_fMRI, cfg.dropout_rate).to(device)
    best_weights_path = os.path.join(
        experiment_dir, cfg.weight_dir, f"autoencoder_weights_fold_{best_fold}.pth")
    model.load_state_dict(torch.load(best_weights_path))

    model.eval()
    test_loss = 0
    test_mean_corr = 0
    test_mape = 0
    original_matrices = []
    reconstructed_matrices = []
    tensorboard_dir = os.path.join(experiment_dir, cfg.tensorboard_dir)

    writer = SummaryWriter(log_dir=tensorboard_dir)
    if cfg.loss_function == 'LogEuclidean':
        criterion = LogEuclideanLoss()
    elif cfg.loss_function == 'Norm':
        criterion = NormLoss()
    else:
        raise ValueError("Unsupported loss function specified in config")

    with torch.no_grad():
        for features, targets in test_loader:
            features = features.to(device)
            eye = torch.eye(features.size(-1), device=device)
            embedded_feat = model.encode_feat(features)
            reconstructed_feat = model.decode_feat(embedded_feat)
            original_matrices.append(features.cpu().numpy())
            reconstructed_matrices.append(reconstructed_feat.cpu().numpy())
            test_loss += criterion(features, reconstructed_feat)
            test_mean_corr += mean_correlations_between_subjects(
                features, reconstructed_feat)
            test_mape += mape_between_subjects(features,
                                               reconstructed_feat).item()

    test_loss /= len(test_loader)
    test_mean_corr /= len(test_loader)
    test_mape /= len(test_loader)

    original_matrices = np.concatenate(original_matrices, axis=0)
    reconstructed_matrices = np.concatenate(reconstructed_matrices, axis=0)

    log_mape_between_subjects_and_region_rank(
        writer, original_matrices, reconstructed_matrices, experiment_dir)
    log_correlations_between_subjects(
        writer, original_matrices, reconstructed_matrices)

    writer.close()
    # Save original and reconstructed matrices
    original_path = os.path.join(
        experiment_dir, cfg.original_dir, f"Original.npy")
    reconstructed_path = os.path.join(
        experiment_dir, cfg.reconstructed_dir, f"Reconstructed.npy")
    np.save(original_path, original_matrices)
    np.save(reconstructed_path, reconstructed_matrices)

    print(
        f"Test Loss: {test_loss:.02f} | Test Mean Corr: {test_mean_corr:.02f} | Test MAPE: {test_mape:.02f}")


if __name__ == "__main__":
    main()
