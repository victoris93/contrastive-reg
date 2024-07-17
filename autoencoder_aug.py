# %%
import math
import asyncio
import submitit
import pickle
import sys
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

torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
class LogEuclideanLoss(nn.Module):
    def __init__(self):
        super(LogEuclideanLoss, self).__init__()
    
    def mat_batch_log(self, features):
        eps = 1e-6
        regularized_features = features + eps * torch.eye(features.size(-1), device=features.device)
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
        recon_features_diag = torch.round(recon_features, decimals = 3)
        
        
        log_features = self.mat_batch_log(features)
        log_recon_features = self.mat_batch_log(recon_features_diag)
        loss = torch.norm(log_features - log_recon_features, dim=(-2, -1)).mean()
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
        
        self.enc1 = nn.Linear(in_features=input_dim_feat, out_features=output_dim_feat,bias=False)
        self.enc2 = nn.Linear(in_features=input_dim_feat, out_features=output_dim_feat,bias=False)
        self.enc2.weight = torch.nn.Parameter(self.enc1.weight)
        
        self.dec1 = nn.Linear(in_features=output_dim_feat, out_features=input_dim_feat,bias=False)
        self.dec2 = nn.Linear(in_features=output_dim_feat, out_features=input_dim_feat,bias=False)
        self.dec1.weight = torch.nn.Parameter(self.enc1.weight.transpose(0,1))
        self.dec2.weight = torch.nn.Parameter(self.dec1.weight)
        self.dropout = nn.Dropout(p=dropout_rate)
        #self.elu = nn.ELU()
        #self.alpha = nn.Parameter(torch.ones(1))
    def encode_feat(self, x):
        z_n = self.enc1(x)
        c_hidd_fMRI = self.enc2(z_n.transpose(1,2))
        return c_hidd_fMRI
    
    def decode_feat(self,embedding):
        z_n = (self.dec1(embedding)).transpose(1,2)
        #corr_n = self.alpha*self.dropout((self.dec2(z_n)))
        corr_n = self.dec2(z_n)
        return corr_n
# %%
class MatData(Dataset):
    def __init__(self, path_feat, path_targets):
        self.matrices = np.load(path_feat, mmap_mode="r").astype(np.float32)
        self.matrices = torch.from_numpy(self.matrices).to(torch.float32)
        self.target = torch.tensor(pd.read_csv(path_targets).drop(columns=["Subject", "Unnamed: 0",
        'CardioMeasures_pulse_mean',
        'CardioMeasures_bp_sys_mean',
        'CardioMeasures_bp_dia_mean']).to_numpy(), dtype=torch.float32)

    def __len__(self):
        return len(self.matrices)

    def __getitem__(self, idx):
        matrix = self.matrices[idx]
        target = self.target[idx]
        return matrix, target
    
# %%
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
#Input to the train autoencoder function is train_dataset.dataset.matrices
def train_autoencoder(train_dataset, val_dataset, B_init_fMRI, model=None, device = device, num_epochs = 100, batch_size = 32):
    input_dim_feat = 100
    output_dim_feat = 15
    lr = 0.001
    weight_decay = 0
    lambda_0 = 1
    dropout_rate = 0
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    if model is None:
        model = AutoEncoder(
            input_dim_feat,
            output_dim_feat,
            B_init_fMRI,
            dropout_rate
        ).to(device)
        
    model.enc1.weight = torch.nn.Parameter(B_init_fMRI.transpose(0,1))
    model.enc2.weight = torch.nn.Parameter(B_init_fMRI.transpose(0,1))
    
    ae_criterion = LogEuclideanLoss().to(device)
    optimizer_autoencoder = optim.Adam(model.parameters(), lr = lr, weight_decay = weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer_autoencoder, 100, 0.5, last_epoch=-1)
    loss_terms = []
    model.train()
    with tqdm(range(num_epochs), desc="Epochs", leave=False) as pbar:
        for epoch in pbar:
            optimizer_autoencoder.zero_grad()
            recon_loss = 0
            loss_terms_batch = defaultdict(lambda:0)
            for features, targets in train_loader:
                
                features = features.to(device)
                
                embedded_feat = model.encode_feat(features)
                reconstructed_feat = model.decode_feat(embedded_feat)
                
                loss =  recon_loss + lambda_0*torch.norm((features-reconstructed_feat))**2#ae_criterion(features, reconstructed_feat)
                train_mean_corr = mean_correlations_between_subjects(features, reconstructed_feat)
                train_mape = mape_between_subjects(features, reconstructed_feat)
                #loss = ae_criterion(features, reconstructed_feat)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) 
                optimizer_autoencoder.step()
                scheduler.step()
                loss_terms_batch['loss'] += loss.item() / len(train_loader)
                pbar.set_postfix_str(f"Epoch {epoch} | Train Loss {loss:.02f} | Train corr {train_mean_corr:.02f} | Train mape {train_mape:.02f}")
                loss_terms.append((loss.item(), train_mean_corr, train_mape))

    model.eval()
    val_loss = 0
    val_mean_corr = 0
    val_mape = 0
    with torch.no_grad():
                    for features, targets in val_loader:
                        features = features.to(device)
                        targets = targets.to(device)

                        embedded_feat = model.encode_feat(features)
                        reconstructed_feat = model.decode_feat(embedded_feat)

                        #loss = ae_criterion(features, reconstructed_feat)
                        val_loss += lambda_0*torch.norm((features-reconstructed_feat))**2#ae_criterion(features, reconstructed_feat)#

                        val_mean_corr += mean_correlations_between_subjects(features, reconstructed_feat)
                        val_mape += mape_between_subjects(features, reconstructed_feat).item()

    val_loss /= len(val_loader)
    val_mean_corr /= len(val_loader)
    val_mape /= len(val_loader)

    print(f"Validation Loss: {val_loss:.02f} | Validation Mean Corr: {val_mean_corr:.02f} | Validation MAPE: {val_mape:.02f}")
    loss_terms.append(('Validation', val_loss.item(), val_mean_corr, val_mape))

    model_weights = model.state_dict()
    torch.save(model_weights, "weights/autoencoder_weights.pth")
    return loss_terms, model_weights

# %%
path_feat = "/data/parietal/store2/work/mrenaudi/contrastive-reg-3/conn_camcan_without_nan/stacked_mat.npy"
path_target = "/data/parietal/store2/work/mrenaudi/contrastive-reg-3/target_without_nan.csv"
dataset = MatData(path_feat, path_target)
train_idx, test_idx = train_test_split(np.arange(len(dataset)), test_size=0.2, random_state=42)
train_dataset = Subset(dataset, train_idx)
test_dataset = Subset(dataset, test_idx)

input_dim_feat = 100
output_dim_feat = 15
dropout_rate = 0
train_features = dataset.matrices[train_idx]
mean_f = torch.mean(torch.tensor(train_features), dim=0).to(device)
[D, V] = torch.linalg.eigh(mean_f, UPLO="U")
B_init_fMRI = V[:, input_dim_feat - output_dim_feat:]

kf = KFold(n_splits=5, shuffle=True, random_state=42)

# %%
for fold, (train_val_idx, val_idx) in enumerate(kf.split(train_dataset)):
    print(f"Fold {fold + 1}")
    train_data = Subset(train_dataset, train_val_idx)
    val_data = Subset(train_dataset, val_idx)
    
    # Train autoencoder
    loss_terms, trained_weights = train_autoencoder(train_data, val_data, B_init_fMRI)
    
    # Save fold results
    #with open(f"results/fold_{fold + 1}_loss_terms.pkl", "wb") as f:
        #pickle.dump(loss_terms, f)
    #torch.save(trained_weights, f"weights/autoencoder_weights_fold_{fold + 1}.pth")

# Evaluate on the test set
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
model = AutoEncoder(input_dim_feat, output_dim_feat, B_init_fMRI, dropout_rate).to(device)
model.load_state_dict(torch.load("weights/autoencoder_weights_fold_1.pth"))  # Load the best fold weights

model.eval()
test_loss = 0
test_mean_corr = 0
test_mape = 0

with torch.no_grad():
    for features, targets in test_loader:
        features = features.to(device)
        
        
        
        embedded_feat = model.encode_feat(features)
        reconstructed_feat = model.decode_feat(embedded_feat)
        
        test_loss += LogEuclideanLoss()(features, reconstructed_feat)
        test_mean_corr += mean_correlations_between_subjects(features, reconstructed_feat)
        test_mape += mape_between_subjects(features, reconstructed_feat).item()

test_loss /= len(test_loader)
test_mean_corr /= len(test_loader)
test_mape /= len(test_loader)

original_matrices = np.concatenate(features, axis=0)
reconstructed_matrices = np.concatenate(reconstructed_feat, axis=0)

# Save original and reconstructed matrices
np.save("original_matrices.npy", original_matrices)
np.save("reconstructed_matrices.npy", reconstructed_matrices)

print(f"Test Loss: {test_loss:.02f} | Test Mean Corr: {test_mean_corr:.02f} | Test MAPE: {test_mape:.02f}")

