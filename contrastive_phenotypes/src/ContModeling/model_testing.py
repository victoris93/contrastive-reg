# +
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
# import matplotlib.pyplot as plt
import numpy as np
# import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset
from tqdm.auto import tqdm
import glob, os, shutil
import random
from torch.utils.tensorboard import SummaryWriter
import sys

from .viz_func import wandb_plot_acc_vs_baseline, wandb_plot_test_recon_corr, wandb_plot_individual_recon
from .utils import mean_correlations_between_subjects, mape_between_subjects
from .losses import LogEuclideanLoss, NormLoss
from .models import MatAutoEncoder
from .helper_classes import MatData


# -

def test_autoencoder(best_fold, test_dataset, cfg, model_params_dir, recon_mat_dir, device):
    
    wandb.init(project=cfg.project,
        group = "ae_test",
        mode = "offline",
        name=cfg.experiment_name,
       dir = cfg.output_dir)
    wandb.config.update(OmegaConf.to_container(cfg, resolve=True))

    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)
    input_dim_feat = cfg.input_dim_feat
    output_dim_feat = cfg.output_dim_feat
    lr = cfg.lr
    weight_decay = cfg.weight_decay
    dropout_rate = cfg.dropout_rate

    model = AutoEncoder(
            input_dim_feat,
            output_dim_feat,
            torch.randn(input_dim_feat, output_dim_feat),
            dropout_rate,
            cfg
            ).to(device)
    
    model.load_state_dict(torch.load(f"{model_params_dir}/autoencoder_weights_fold{best_fold}.pth")) # Load the best fold weights
    
    model.eval()
    test_loss = 0
    test_mean_corr = 0
    test_mape = 0
    

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

            test_loss += criterion(features,reconstructed_feat)
            test_mean_corr += mean_correlations_between_subjects(features, reconstructed_feat)
            test_mape += mape_between_subjects(features, reconstructed_feat).item()
    
    wandb_plot_test_recon_corr(wandb, cfg.dataset_path, cfg.work_dir, cfg.experiment_name)
    wandb_plot_individual_recon(wandb, cfg.dataset_path, cfg.work_dir, cfg.experiment_name, 0)
    wandb_plot_individual_recon(wandb, cfg.dataset_path, cfg.work_dir, 'ae_loss_norm', 0)
    wandb_plot_acc_vs_baseline(wandb, cfg.dataset_path, cfg.work_dir, cfg.experiment_name, 'ae_loss_norm')
    
    wandb.finish()

            
    test_loss /= len(test_loader)
    test_mean_corr /= len(test_loader)
    test_mape /= len(test_loader)
    print(f"Test Loss: {test_loss:.02f} | Test Mean Corr: {test_mean_corr:.02f} | Test MAPE: {test_mape:.02f}")

# +
# test_full_model(test_dataset, cfg, model_params_dir, recon_mat_dir, wandb, device):