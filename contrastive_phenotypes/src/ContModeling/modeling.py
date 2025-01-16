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
from nilearn.connectome import sym_matrix_to_vec, vec_to_sym_matrix
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
from .utils import mape_between_subjects, mean_correlations_between_subjects, save_embeddings, cauchy, gaussian_kernel
from .losses import LogEuclideanLoss, NormLoss, KernelizedSupCon
from .models import MatAutoEncoder, ReducedMatAutoEncoder, TargetDecoder
from .viz_func import load_mape, load_recon_mats, load_true_mats, wandb_plot_test_recon_corr, wandb_plot_individual_recon
from .helper_classes import MatData

SUPCON_KERNELS = {
    'cauchy': cauchy,
    'gaussian_kernel': gaussian_kernel,
    'None': None
    }
     
#Input to the train autoencoder function is train_dataset.dataset.matrices
def train_mat_autoencoder(fold, train_dataset, val_dataset, B_init_fMRI, cfg, device, model=None):

    wandb.init(project=cfg.project,
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
        
    model.train()
    with tqdm(range(num_epochs), desc="Epochs", leave=False) as pbar:
        for epoch in pbar:
            batch = 1
            loss_terms_batch = defaultdict(lambda:0)
            for features, _ in train_loader:
                
                optimizer_autoencoder.zero_grad()
                features = features.to(device)
                
                embedded_feat = model.encode_feat(features)
                reconstructed_feat = model.decode_feat(embedded_feat)
                
                loss = criterion(features,reconstructed_feat)
                loss.backward()
                        
                if cfg.clip_grad:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                if cfg.log_gradients:
                    for name, param in model.named_parameters():
                        wandb.log({
                            "Epoch": epoch,
                            "Batch": batch,
                            f"Gradient Norm/{name}": param.grad.norm().item()
                            })
                        
                optimizer_autoencoder.step()
                loss_terms_batch['loss'] += loss.item() / len(train_loader)
                batch += 1
                
            model.eval()
            val_loss = 0
            val_mean_corr = 0
            val_mape = 0

            with torch.no_grad():
                for features, _ in val_loader:
                    features = features.to(device)

                    embedded_feat = model.encode_feat(features)
                    save_embeddings(embedded_feat, "mat", test = False, cfg = cfg, fold = fold, epoch = epoch)
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

def train_reduced_mat_autoencoder(fold, train_dataset, val_dataset, cfg, device, model=None):
    wandb.init(project=cfg.project,
       mode = "offline",
       name=cfg.experiment_name,
       dir = cfg.output_dir)
    wandb.config.update(OmegaConf.to_container(cfg, resolve=True))
    
    input_dim_feat = cfg.input_dim_feat
    hidden_dim = cfg.hidden_dim
    output_dim_target = cfg.output_dim_target
    batch_size = cfg.batch_size
    lr = cfg.lr
    weight_decay = cfg.weight_decay
    dropout_rate = cfg.dropout_rate
    num_epochs = cfg.num_epochs
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    if model is None:
        model = ReducedMatAutoEncoder(
            input_dim_feat,
            hidden_dim,
            output_dim_target,
            dropout_rate,
            cfg
        ).to(device)

    kernel = SUPCON_KERNELS[cfg.SupCon_kernel]
    supcon_criterion = KernelizedSupCon(
        method="expw",
        temperature=cfg.supcon_temperature,
        base_temperature= cfg.supcon_base_temperature,
        reg_term = cfg.supcon_reg_term,
        kernel=kernel,
        krnl_sigma=cfg.supcon_sigma,
    )
    if cfg.loss_function == 'LogEuclidean':
        recon_criterion = LogEuclideanLoss()
        optimizer_autoencoder = RiemannianAdam(model.parameters(), lr = lr, weight_decay = weight_decay)
    elif cfg.loss_function == 'Norm':
        recon_criterion = NormLoss()
        optimizer_autoencoder = optim.Adam(model.parameters(), lr = lr, weight_decay = weight_decay)
    elif cfg.loss_function == 'MSE':
        recon_criterion = nn.functional.mse_loss
        optimizer_autoencoder = optim.Adam(model.parameters(), lr = lr, weight_decay = weight_decay)
    else:
        raise ValueError("Unsupported loss function specified in config")
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer_autoencoder,
                                                     factor=0.1,
                                                     patience = cfg.scheduler_patience)
    
    loss_terms = []
        
    model.train()
    with tqdm(range(num_epochs), desc="Epochs", leave=False) as pbar:
        for epoch in pbar:
            batch = 1
            loss_terms_batch = defaultdict(lambda:0)
            for features, targets in train_loader:
                
                optimizer_autoencoder.zero_grad()
                features = features.to(device)
                targets = targets.to(device)
                
                embedding, embedding_norm = model.embed_reduced_mat(features)
                reconstructed_reduced_mat = model.recon_reduced_mat(embedding)
                
                recon_loss = recon_criterion(features, reconstructed_reduced_mat) / 10_000
                supcon_loss = 100 * supcon_criterion(embedding_norm.unsqueeze(1), targets)[0]

                loss = supcon_loss + recon_loss
                loss.backward()
                        
                if cfg.clip_grad:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                if cfg.log_gradients:
                    for name, param in model.named_parameters():
                        wandb.log({
                            "Epoch": epoch,
                            "Batch": batch,
                            f"Gradient Norm/{name}": param.grad.norm().item()
                            })
                        
                optimizer_autoencoder.step()

                loss_terms_batch['loss'] += loss.item() / len(train_loader)
                batch += 1
                
            model.eval()
            val_loss = 0
            val_recon_losses = 0
            val_supcon_losses = 0
            val_mean_corr = 0
            val_mape = 0

            with torch.no_grad():
                for features, targets in val_loader:
                    features = features.to(device)
                    targets = targets.to(device)

                    embedding, embedding_norm = model.embed_reduced_mat(features)
                    save_embeddings(embedding, "reduced_mat_emb", test = False, cfg = cfg, fold = fold, epoch = epoch)
                    reconstructed_reduced_mat = model.recon_reduced_mat(embedding)
                    save_embeddings(reconstructed_reduced_mat, "recon_reduced_mat", test = False, cfg = cfg, fold = fold, epoch = epoch)
                    
                    recon_val_loss = recon_criterion(features, reconstructed_reduced_mat) / 1000
                    supcon_val_loss = 100 * supcon_criterion(embedding_norm.unsqueeze(1), targets)[0]
                    val_recon_losses += recon_val_loss
                    val_supcon_losses += supcon_val_loss
                    val_loss += (recon_val_loss + supcon_val_loss)
                    val_mean_corr += mean_correlations_between_subjects(features, reconstructed_reduced_mat)
                    val_mape += mape_between_subjects(features, reconstructed_reduced_mat).item()
            
            val_recon_losses /= len(val_loader)
            val_supcon_losses /= len(val_loader)
            val_loss /= len(val_loader)
            val_mean_corr /= len(val_loader)
            val_mape /= len(val_loader)
            
            wandb.log({
                "Fold" : fold,
                "Epoch" : epoch,
                "Loss/val" : val_loss.item(),
                "Loss/supcon_val" : val_supcon_losses.item(),
                "Loss/recon_val" : val_recon_losses.item(),
                "Metric/val_mean_corr" : val_mean_corr,
                "Metric/val_mape" : val_mape
            })
            
            loss_terms.append(('Validation', val_loss.item(), val_mean_corr, val_mape))
            
            scheduler.step(val_loss)
            if np.log10(scheduler._last_lr[0]) < -4:
                break

            pbar.set_postfix_str(f"Epoch {epoch} | Fold {fold} | Train Loss {loss:.02f} | Val ReconLoss {val_recon_losses:.02f} | Val SupConLoss {val_supcon_losses:.02f} | Val Mean Corr {val_mean_corr:.02f} | Val MAPE {val_mape:.02f} | log10 lr {np.log10(scheduler._last_lr[0])}") # Train corr {train_mean_corr:.02f}| Train mape {train_mape:.02f}
            
    wandb.finish()
    print(loss_terms)
    
    return loss_terms, model.state_dict(), val_loss.item()    
    
def test_mat_autoencoder(best_fold, test_dataset, cfg, model_params_dir, recon_mat_dir, device):
    
    wandb.init(project=cfg.project,
        mode = "offline",
        name=f"TEST_{cfg.experiment_name}",
        dir = cfg.output_dir,
        config = OmegaConf.to_container(cfg, resolve=True))

    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)
    input_dim_feat = cfg.input_dim_feat
    output_dim_feat = cfg.output_dim_feat
    dropout_rate = cfg.dropout_rate

    model = MatAutoEncoder(
            input_dim_feat,
            output_dim_feat,
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
            criterion = nn.functional.mse_loss
    else:
        raise ValueError("Unsupported loss function specified in config")

    with torch.no_grad():
        for i, (features, _) in enumerate(test_loader):

            features = features.to(device)

            embedded_feat = model.encode_feat(features)
            reconstructed_feat = model.decode_feat(embedded_feat)

            np.save(f'{recon_mat_dir}/recon_mat_fold{best_fold}_batch_{i+1}', reconstructed_feat.cpu().numpy())
            mape_mat = torch.abs((features - reconstructed_feat) / (features + 1e-10)) * 100
            np.save(f'{recon_mat_dir}/mape_mat_fold{best_fold}_batch_{i+1}', mape_mat.cpu().numpy())

            loss = criterion(features,reconstructed_feat)
            mean_corr = mean_correlations_between_subjects(features, reconstructed_feat)
            mape = mape_between_subjects(features, reconstructed_feat).item()

            test_loss += loss
            test_mean_corr += mean_corr
            test_mape += mape

            wandb.log({
                'Fold': best_fold,
                'Test Batch' : i+1,
                'Test | MAPE' : mape,
                'Test | Mean Corr' : mean_corr,
                'Test | Loss': loss,
                })
        
        recon_mat = load_recon_mats(cfg.experiment_name, cfg.work_dir, False)
        true_mat = load_true_mats(cfg.dataset_path, cfg.experiment_name, cfg.work_dir, False)
        mape_mat = load_mape(cfg.experiment_name, cfg.work_dir)
        test_idx_path = f"{cfg.output_dir}/{cfg.experiment_name}/test_idx.npy"
        test_idx = np.load(test_idx_path)

        wandb_plot_test_recon_corr(wandb, cfg.experiment_name, cfg.work_dir, recon_mat, true_mat, mape_mat)
        wandb_plot_individual_recon(wandb, cfg.experiment_name, cfg.work_dir, test_idx, recon_mat, true_mat, mape_mat, 0)

        
    wandb.finish()

            
    test_loss /= len(test_loader)
    test_mean_corr /= len(test_loader)
    test_mape /= len(test_loader)
    print(f"Test Loss: {test_loss:.02f} | Test Mean Corr: {test_mean_corr:.02f} | Test MAPE: {test_mape:.02f}")


def test_reduced_mat_autoencoder(best_fold, test_dataset, cfg, model_params_dir, recon_mat_dir, device):
    
    wandb.init(project=cfg.project,
        mode = "offline",
        name=f"TEST_{cfg.experiment_name}",
        dir = cfg.output_dir,
        config = OmegaConf.to_container(cfg, resolve=True))

    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)
    input_dim_feat = cfg.input_dim_feat
    hidden_dim = cfg.hidden_dim
    output_dim_target = cfg.output_dim_target
    dropout_rate = cfg.dropout_rate

    model = ReducedMatAutoEncoder(
            input_dim_feat,
            hidden_dim,
            output_dim_target,
            dropout_rate,
            cfg
            ).to(device)
    
    model.load_state_dict(torch.load(f"{model_params_dir}/autoencoder_weights_fold{best_fold}.pth"))
    
    model.eval()
    test_loss = 0
    test_mean_corr = 0
    test_mape = 0
    

    if cfg.loss_function == 'LogEuclidean':
            criterion = LogEuclideanLoss()
    elif cfg.loss_function == 'Norm':
            criterion = NormLoss()
    elif cfg.loss_function == 'MSE':
            criterion = nn.functional.mse_loss
    else:
        raise ValueError("Unsupported loss function specified in config")

    with torch.no_grad():
        for i, (features, _) in enumerate(test_loader):

            features = features.to(device)

            embedding, embedding_norm = model.embed_reduced_mat(features)
            reconstructed_reduced_mat = model.recon_reduced_mat(embedding)

            np.save(f'{recon_mat_dir}/recon_reduced_mat_fold{best_fold}_batch_{i+1}', reconstructed_reduced_mat.cpu().numpy())
            mape_mat = torch.abs((features - reconstructed_reduced_mat) / (features + 1e-10)) * 100
            mape_mat = vec_to_sym_matrix(mape_mat.cpu().numpy())
            np.save(f'{recon_mat_dir}/mape_reduced_mat_fold{best_fold}_batch_{i+1}', mape_mat)

            loss = criterion(features, reconstructed_reduced_mat)
            mean_corr = mean_correlations_between_subjects(features, reconstructed_reduced_mat)
            mape = mape_between_subjects(features, reconstructed_reduced_mat).item()

            test_loss += loss
            test_mean_corr += mean_corr
            test_mape += mape

            wandb.log({
                'Fold': best_fold,
                'Test Batch' : i+1,
                'Test | MAPE' : mape,
                'Test | Mean Corr' : mean_corr,
                'Test | Loss': loss,
                })
        
        recon_mat = load_recon_mats(cfg.experiment_name, cfg.work_dir, vectorize=False, reduced_mat=True)
        true_mat = load_true_mats(cfg.dataset_path, cfg.experiment_name, cfg.work_dir, reduced_mat=True)
        mape_mat = load_mape(cfg.experiment_name, cfg.work_dir)
        test_idx_path = f"{cfg.output_dir}/{cfg.experiment_name}/test_idx.npy"
        test_idx = np.load(test_idx_path)

        wandb_plot_test_recon_corr(wandb, cfg.experiment_name, cfg.work_dir, recon_mat, true_mat, mape_mat)
        wandb_plot_individual_recon(wandb, cfg.experiment_name, cfg.work_dir, test_idx, recon_mat, true_mat, mape_mat, 0)

        
    wandb.finish()

            
    test_loss /= len(test_loader)
    test_mean_corr /= len(test_loader)
    test_mape /= len(test_loader)
    print(f"Test Loss: {test_loss:.02f} | Test Mean Corr: {test_mean_corr:.02f} | Test MAPE: {test_mape:.02f}")