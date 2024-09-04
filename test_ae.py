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
from viz_func import wandb_plot_corr, wandb_plot_acc_vs_baseline, wandb_plot_test_recon_corr, wandb_plot_individual_recon
from utils_v import mean_correlations_between_subjects, mape_between_subjects
from losses import LogEuclideanLoss, NormLoss
from models import AutoEncoder
from helper_classes import MatData
from model_testing import test_autoencoder


@hydra.main(config_path=".", config_name="autoencoder_config")

def main(cfg: DictConfig):

    wandb.init(project=cfg.project, mode = "offline", name=cfg.experiment_name, dir = "/gpfs3/well/margulies/users/cpy397/contrastive-learning/results")
    wandb.config.update(OmegaConf.to_container(cfg, resolve=True))
    
    best_fold = int(cfg.best_fold)
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_idx_path = f"/gpfs3/well/margulies/users/cpy397/contrastive-learning/results/{cfg.experiment_name}/test_idx.npy"
    test_idx = np.load(test_idx_path)
    
    targets = list(cfg.targets)
    dataset = MatData(cfg.dataset_path, targets, threshold=0)
    test_dataset = Subset(dataset, test_idx)

    print("Testing model.\n", OmegaConf.to_yaml(cfg))
    results_dir = os.path.join(cfg.output_dir, cfg.experiment_name)
    os.makedirs(results_dir, exist_ok=True)
    recon_mat_dir = os.path.join(results_dir, cfg.reconstructed_dir)
    os.makedirs(recon_mat_dir, exist_ok=True)
    model_params_dir = os.path.join(results_dir, cfg.model_weight_dir)
    os.makedirs(model_params_dir, exist_ok=True)

    test_autoencoder(best_fold = best_fold, test_dataset =test_dataset, cfg = cfg, model_params_dir = model_params_dir,
                            recon_mat_dir = recon_mat_dir, wandb = wandb, device = device)

if __name__ == "__main__":
    main()

