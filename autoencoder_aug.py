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
from scipy.stats import pearsonr
from sklearn.model_selection import (
    train_test_split,
)
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset
from tqdm.auto import tqdm
from augmentations import augs, aug_args
import glob, os, shutil
from nilearn.datasets import fetch_atlas_schaefer_2018
import random
#from geoopt.optim import RiemannianAdam

torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
class LogEuclideanLoss(nn.Module):
    def __init__(self):
        super(LogEuclideanLoss, self).__init__()
    
    def mat_batch_log(self, features):
        
        Eigvals, Eigvecs = torch.linalg.eigh(features)
        Eigvals = torch.clamp(Eigvals, min=1e-6)
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
        B_init_fMRI
    ):
        super(AutoEncoder, self).__init__()
        
        self.enc1 = nn.Linear(in_features=input_dim_feat, out_features=output_dim_feat,bias=False)
        self.enc2 = nn.Linear(in_features=input_dim_feat, out_features=output_dim_feat,bias=False)
        #self.enc2.weight = torch.nn.Parameter(self.enc1.weight)
        
        self.dec1 = nn.Linear(in_features=output_dim_feat, out_features=input_dim_feat,bias=False)
        self.dec2 = nn.Linear(in_features=output_dim_feat, out_features=input_dim_feat,bias=False)
        #self.dec1.weight = torch.nn.Parameter(self.enc1.weight.transpose(0,1))
        #self.dec2.weight = torch.nn.Parameter(self.dec1.weight)
        
        if B_init_fMRI is not None:
            self.enc1.weight.data = B_init_fMRI.transpose(0,1)
            self.enc2.weight.data = B_init_fMRI.transpose(0,1)
            self.dec1.weight.data = B_init_fMRI
            self.dec2.weight.data = B_init_fMRI
        else:
            # Use default initialization (e.g., Xavier uniform)
            nn.init.xavier_uniform_(self.enc1.weight)
            nn.init.xavier_uniform_(self.enc2.weight)
            nn.init.xavier_uniform_(self.dec1.weight)
            nn.init.xavier_uniform_(self.dec2.weight)
    def encode_feat(self, x):
        z_n = self.enc1(x)
        c_hidd_fMRI = self.enc2(z_n.transpose(1,2))
        return c_hidd_fMRI
    
    def decode_feat(self,embedding):
        z_n = (self.dec1(embedding)).transpose(1,2)
        corr_n = (self.dec2(z_n))
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
#Input to the train autoencoder function is train_dataset.dataset.matrices
def train_autoencoder(train_dataset, B_init_fMRI, model=None, device = device, num_epochs = 1000, batch_size = 32):
    input_dim_feat = 100
    output_dim_feat = 25
    lr = 0.0001
    
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    if model is None:
        model = AutoEncoder(
            input_dim_feat,
            output_dim_feat,
            B_init_fMRI,
        ).to(device)
        
    #model.enc1.weight = torch.nn.Parameter(B_init_fMRI.transpose(0,1))
    #model.enc2.weight = torch.nn.Parameter(B_init_fMRI.transpose(0,1))
    
    ae_criterion = LogEuclideanLoss().to(device)
    optimizer_autoencoder = optim.Adam(model.parameters(), lr = lr)
    loss_terms = []
    with tqdm(range(num_epochs), desc="Epochs", leave=False) as pbar:
        for epoch in pbar:
            model.train()
            loss_terms_batch = defaultdict(lambda:0)
            for features, targets in train_loader:
                
                optimizer_autoencoder.zero_grad()
                features = features.to(device)
                targets = targets.to(device)
                
                embedded_feat = model.encode_feat(features)
                reconstructed_feat = model.decode_feat(embedded_feat)

                loss = ae_criterion(features, reconstructed_feat)
                loss.backward()
                optimizer_autoencoder.step()
                loss_terms_batch['loss'] += loss.item() / len(train_loader)
            loss_terms_batch['epoch'] = epoch
            pbar.set_postfix_str(
                f"Epoch {epoch} "
                f"| Loss {loss_terms_batch['loss']:.02f} "
            )
    model_weights = model.state_dict()
    torch.save(model_weights, "weights/autoencoder_weights.pth")
    return model_weights


# %%
path_feat = "/data/parietal/store2/work/mrenaudi/contrastive-reg-3/conn_camcan_without_nan/stacked_mat.npy"
path_target = "/data/parietal/store2/work/mrenaudi/contrastive-reg-3/target_without_nan.csv"
dataset = MatData(path_feat, path_target)
# %%
input_dim_feat=100
output_dim_feat=25
train_features = dataset.matrices
mean_f = torch.mean(torch.tensor(train_features), dim=0).to(device)
[D,V] = torch.linalg.eigh(mean_f,UPLO = "U")     
B_init_fMRI = V[:,input_dim_feat-output_dim_feat:]
# %%
trained_weights = train_autoencoder(dataset, B_init_fMRI)            


