# from: https://github.com/EIDOSLAB/contrastive-brain-age-prediction/blob/master/src/util.py
import math
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
import random
import numpy as np
import os
# import wandb
from sklearn.decomposition import PCA
import torch.nn.functional as F
from pathlib import Path
from nilearn.connectome import sym_matrix_to_vec, vec_to_sym_matrix
from scipy.stats import pearsonr, spearmanr

def save_embeddings(embedding, emb_type, cfg, dataset_label, run_type, run_id, batch = None, train_ratio = None, epoch = None):
    embedding_dir = f"{cfg.output_dir}/{cfg.experiment_name}/{cfg.embedding_dir}"
    os.makedirs(embedding_dir, exist_ok=True)
    embedding_numpy = embedding.cpu().detach().numpy()
    emb_type = emb_type + '_'

    run_suffix = f"_{run_type}{run_id}"

    if batch is None:
        batch_suffix = ''
    else:
        batch_suffix = f"_batch{batch}"

    if train_ratio is None:
        train_ratio_suffix = ''
    else:
        train_ratio_suffix = f"_train_ratio{train_ratio}"

    if epoch is None:
        epoch_suffix = ''
    else:
        epoch_suffix = f"_epoch{epoch}"

    dataset_suffix = f"_{dataset_label}"

    save_path = f"{embedding_dir}/{emb_type}embeddings{epoch_suffix}{batch_suffix}{run_suffix}{train_ratio_suffix}{dataset_suffix}.npy"
    np.save(save_path, embedding_numpy)

def mean_correlations_between_subjects(y_true, y_pred):

    y_true = y_true.cpu().detach().numpy()
    y_pred = y_pred.cpu().detach().numpy()
    
    num_subjects = y_true.shape[0]

    lower_true = []
    lower_pred = []

    if len(y_true.shape) < 3:
        correlations, _ = spearmanr(y_true.flatten(), y_pred.flatten())
    else:
        for subj in range(num_subjects):
            L_pred = sym_matrix_to_vec(y_pred[subj], discard_diagonal=True)
            L_true = sym_matrix_to_vec(y_true[subj], discard_diagonal=True)
            lower_true.extend(L_true.tolist())
            lower_pred.extend(L_pred.tolist())
        correlations, _ = spearmanr(lower_true, lower_pred)
    
    correlation = np.mean(correlations)
    
    return correlation

def filter_nans_X(_X, indices = None):
    nan_idx_X = torch.isnan(_X).any(dim=(1,2))
    _X = _X[~nan_idx_X]
    if indices is not None:
        indices = indices[~nan_idx_X]
        
    return _X, indices

def filter_nans_y(_y, indices = None):
    nan_idx_y = torch.isnan(_y).squeeze()
    _y = _y[~nan_idx_y]
    if indices is not None:
        indices = indices[~nan_idx_y]

    return _y, indices

def filter_nans(_X, _y, indices = None, _z = None):
    if len(_y.shape) > 1:
        nan_idx_y = torch.isnan(_y).any(dim=1)
    else:
        nan_idx_y = torch.isnan(_y).squeeze()
        
    nan_idx_X = torch.isnan(_X).any(dim=(1,2))
    nan_idx = nan_idx_X | nan_idx_y

    _X = _X[~nan_idx]
    _y = _y[~nan_idx]

    if _z is not None:
        _z = _z[~nan_idx]

    if indices is not None:
        indices = indices[~nan_idx]

    return _X, _y, indices, _z
    
def mean_correlation(y_true, y_pred):
    correlations = []
    y_true = y_true.cpu().detach().numpy()
    y_pred = y_pred.cpu().detach().numpy()
    for i in range(y_true.shape[0]):
        corr, _ = spearmanr(y_true[i].flatten(), y_pred[i].flatten())
        correlations.append(corr)
    return np.mean(correlations)

def get_best_fold(train_fold_results):
    folds = [fold_dict["fold"] for fold_dict in train_fold_results]
    val_losses = [fold_dict["val_loss"] for fold_dict in train_fold_results]
    best_fold = folds[val_losses.index(np.min(val_losses))]
    print("BEST FOLD IS: ", best_fold)
    return best_fold

def mape_between_subjects(y_true, y_pred):
    eps = 1e-6
    y_true = y_true.cpu().detach().numpy()  # Convert to NumPy array if using PyTorch tensor
    y_pred = y_pred.cpu().detach().numpy()  # Convert to NumPy array if using PyTorch tensor

    num_subjects = y_true.shape[0]

    # Flatten upper triangle (excluding diagonal) for both y_true and y_pred
    lower_true = []
    lower_pred = []

    if len(y_true.shape) < 3:
        mean_mape = (np.abs((y_true.flatten() - y_pred.flatten()) / (y_true.flatten() + eps)) * 100.0).mean()
        print(y_true.flatten().shape)
        
    else:
        for subj in range(num_subjects):
            L_pred = sym_matrix_to_vec(y_pred[subj], discard_diagonal=True)
            L_true = sym_matrix_to_vec(y_true[subj], discard_diagonal=True)
            lower_true.extend(L_true.tolist())
            lower_pred.extend(L_pred.tolist())
        lower_true = np.array(lower_true)
        print(lower_true.shape)
        lower_pred = np.array(lower_pred)
        mean_mape = (np.abs((lower_true - lower_pred) / (lower_true + eps)) * 100.0).mean()
        

    return mean_mape

def mean_absolute_percentage_error(y_true, y_pred):
    eps = 1e-6
    return torch.mean(torch.abs((y_true - y_pred)) / torch.abs(y_true)+eps) * 100

def standardize(data, mean=None, std=None, epsilon = 1e-4): # any variable
    if mean is None:
        mean = np.mean(data, axis=0)
    if std is None:
        std = np.std(data, axis=0)+ epsilon
    return (data - mean)/std, mean, std

def standardize_target(target):
    eps = 0 # avoid division by zero
    target = (target - np.nanmin(target)) / (np.nanmax(target) - np.nanmin(target)) + eps
    return target

def standardize_dataset(dataset):
    features = torch.vstack([dataset[i][0] for i in range(len(dataset))])
    targets = torch.vstack([dataset[i][1] for i in range(len(dataset))])
    
    features_mean = features.mean(dim=0)
    features_std = features.std(dim=0)
    targets_mean = targets.mean(dim=0)
    targets_std = targets.std(dim=0)
    
    features_std[features_std == 0] = 1
    targets_std[targets_std == 0] = 1
    
    standardized_features = (features - features_mean) / features_std
    standardized_targets = (targets - targets_mean) / targets_std
    
    standardized_dataset = TensorDataset(standardized_features, standardized_targets)
    
    return standardized_dataset

def rbf(x, krnl_sigma):
    if len(x.shape) == 1:
        x = x.view(-1, 1)
    x = torch.cdist(x, x, p=2)
    return torch.exp(-(x ** 2) / (2 * (krnl_sigma ** 2)))

def gaussian_kernel(x, krnl_sigma):
    if x.shape[-1] > 1:
        x = torch.cdist(x, x)
    else:
        x = x - x.T
    return torch.exp(-(x**2) / (2 * (krnl_sigma**2))) / (
        math.sqrt(2 * torch.pi) * krnl_sigma
    )

def cauchy(x, krnl_sigma):
    if x.shape[-1] > 1:
        x = torch.cdist(x, x)
    else:
        x = x - x.T
    return 1.0 / (krnl_sigma * (x**2) + 1)