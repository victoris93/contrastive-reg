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
from scipy.stats import pearsonr, spearmanr


def save_embeddings(embedding, cfg, test = False, run = None, batch = None, fold = None, epoch = None):
    embedding_dir = f"{cfg.output_dir}/{cfg.experiment_name}/{cfg.embedding_dir}"
    os.makedirs(embedding_dir, exist_ok=True)
    embedding_numpy = embedding.cpu().detach().numpy()

    if batch is None:
        batch_suffix = ''
    else:
        batch_suffix = f"_batch{batch}"

    if fold is None:
        fold_suffix = ''
    else:
        fold_suffix = f"_fold{fold}"

    if run is None:
        run_suffix = ''
    else:
        run_suffix = f"_run{run}"

    if test:
        dataset_suffix = "_test"
    else:
        dataset_suffix = "_train"

    if epoch is None:
        epoch_suffix = ''
    else:
        epoch_suffix = f"_epoch{epoch}"

    save_path = f"{embedding_dir}/embeddings{epoch_suffix}{batch_suffix}{fold_suffix}{run_suffix}{dataset_suffix}.npy"
    np.save(save_path, embedding_numpy)

def mean_correlations_between_subjects(y_true, y_pred):
    correlations = []
    y_true = y_true.cpu().detach().numpy()
    y_pred = y_pred.cpu().detach().numpy()
    
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

def get_best_fold(train_fold_results):
    folds = [fold_dict["fold"] for fold_dict in train_fold_results]
    val_losses = [fold_dict["val_loss"] for fold_dict in train_fold_results]
    best_fold = folds[val_losses.index(np.min(val_losses))]
    print("BEST FOLD IS: ", best_fold)
    return best_fold

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

def mean_absolute_percentage_error(y_true, y_pred):
    eps = 1e-6
    return torch.mean(torch.abs((y_true - y_pred)) / torch.abs(y_true)+eps) * 100

def standardize(data, mean=None, std=None, epsilon = 1e-4): # any variable
    if mean is None:
        mean = np.mean(data, axis=0)
    if std is None:
        std = np.std(data, axis=0)+ epsilon
    return (data - mean)/std, mean, std


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

def gaussian_kernel(x, krnl_sigma):
    x = x - x.T
    return torch.exp(-(x**2) / (2 * (krnl_sigma**2))) / (
        math.sqrt(2 * torch.pi) * krnl_sigma
    )

def cauchy(x, krnl_sigma):
    x = x - x.T
    return 1.0 / (krnl_sigma * (x**2) + 1)

def multivariate_cauchy(x, krnl_sigma):
    x = torch.cdist(x, x)
    return 1.0 / (krnl_sigma * (x**2) + 1)
