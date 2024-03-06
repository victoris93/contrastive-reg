import torch
import torch.nn as nn
import torch.optim as optim
from cmath import isinf
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import pandas as pd
import math
from cmath import isinf


# def gaussian_kernel(x):
#     x = x - x.T
#     return torch.exp(-(x**2) / (2*(krnl_sigma**2))) / (math.sqrt(krnl_sigma*torch.pi)*1)

def gaussian_kernel(x, krnl_sigma = 0.5):
    x1 = x[:, :1] - x[:, :1].T
    x2 = x[:, 1:2] - x[:, 1:2].T
    x = x1**2 + x2**2
    return torch.exp(-x / (2*(krnl_sigma**2))) / (math.sqrt(krnl_sigma*torch.pi)*1)

# def rbf(x):
#         x = x - x.T
#         return torch.exp(-(x**2)/(2*(krnl_sigma**2)))

def rbf(X, krnl_sigma=0.5):
    x1 = x[:, :1] - x[:, :1].T
    x2 = x[:, 1:2] - x[:, 1:2].T
    x = x1**2 + x2**2
    return torch.exp(-x/(2*(krnl_sigma**2)))

# def cauchy(x):
#         x = x - x.T
#         return  1. / (krnl_sigma*(x**2) + 1)
    
def cauchy(X, krnl_sigma=0.5):
    x1 = x[:, :1] - x[:, :1].T
    x2 = x[:, 1:2] - x[:, 1:2].T
    x = x1**2 + x2**2
    return 1. / (krnl_sigma*x + 1)

class KernelizedSupCon(nn.Module):
    """Supervised contrastive loss: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR
    Based on: https://github.com/HobbitLong/SupContrast"""
    def __init__(self, method: str, temperature: float=0.07, contrast_mode: str='all',
                 base_temperature: float=0.07, kernel: callable=None, delta_reduction: str='sum'):
        super().__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.method = method
        self.kernel = kernel
        self.delta_reduction = delta_reduction

        if kernel is not None and method == 'supcon':
            raise ValueError('Kernel must be none if method=supcon')
        
        if kernel is None and method != 'supcon':
            raise ValueError('Kernel must not be none if method != supcon')

        if delta_reduction not in ['mean', 'sum']:
            raise ValueError(f"Invalid reduction {delta_reduction}")

    def __repr__(self):
        return f'{self.__class__.__name__} ' \
               f'(t={self.temperature}, ' \
               f'method={self.method}, ' \
               f'kernel={self.kernel is not None}, ' \
               f'delta_reduction={self.delta_reduction})'

    def forward(self, features, labels=None):
        """Compute loss for model. If `labels` is None, 
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_features]. 
            labels: ground truth of shape [bsz].
        Returns:
            A loss scalar.
        """
        device = features.device

        if len(features.shape) != 2:
            raise ValueError('`features` needs to be [bsz, n_feats], '
                            '2 dimensions are required')

        batch_size = features.shape[0]

        if labels is not None:
        #    labels = labels.view(-1, 1)
#             labels = labels.contiguous()
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            
            if self.kernel is None:
                mask = torch.eq(labels, labels.T).float()
            else:
                mask = self.kernel(labels)

        else:
            mask = torch.eye(batch_size, device=device)

        # compute similarity
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature
        )

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # apply mask for positive samples, excluding self-contrast cases
        inv_diagonal = torch.eye(batch_size, device=device) * -1 + 1

        # base case: alignment
        alignment = logits

        # base case for uniformity: exp(logits) * inverse diagonal mask
        uniformity = torch.exp(logits) * inv_diagonal

        # Compute uniformity based on the method
        if self.method == 'threshold':
            # irrelevant for now
            pass
        elif self.method == 'expw':
            # exp weight e^(s_j(1-w_j))
            uniformity = torch.exp(logits * (1 - mask)) * inv_diagonal

        uniformity = torch.log(uniformity.sum(1, keepdim=True))
        positive_mask = mask * inv_diagonal
        log_prob = alignment - uniformity  # log(alignment/uniformity)
        log_prob = (positive_mask * log_prob).sum(1) / positive_mask.sum(1)  # mean log-likelihood over positives

        # loss
        loss = - (self.temperature / self.base_temperature) * log_prob
        return loss.mean()

# class MatData(Dataset):
#     def __init__(self, path_mat, path_dm, target):
#         self.matrices = np.load(path_mat)
#         self.target = pd.read_csv(path_dm)[target].values
#     def __len__(self):
#         return len(self.matrices)
#     def __getitem__(self, idx):
#         matrix = self.matrices[idx]
#         target = self.target[idx]
#         matrix = torch.from_numpy(matrix).float()
#         target = torch.tensor(target, dtype=torch.float32)
#         return matrix, target
    

class MatData(Dataset):
    def __init__(self, path_feat, path_target):
        self.features = np.load(path_feat)
        self.target = np.load(path_target)
    def __len__(self):
        return len(self.features)
    def __getitem__(self, idx):
        features = self.features[idx]
        target = self.target[idx]
        features = torch.from_numpy(features).float()
        target = torch.from_numpy(target).float()
        return features, target

# class MLP(nn.Module):
#     def __init__(self, input_dim_feat, input_dim_target, hidden_dim_feat_1, hidden_dim_feat_2, hidden_dim_target_1, hidden_dim_target_2, output_dim):
#         super(MLP, self).__init__()
#         self.feat_mlp = nn.Sequential(
#             nn.Linear(input_dim_feat, hidden_dim_feat_1),
#             nn.BatchNorm1d(hidden_dim_feat_1),
#             nn.ReLU(), # add more layers?
#             nn.Linear(hidden_dim_feat_1, hidden_dim_feat_2),
#             nn.BatchNorm1d(hidden_dim_feat_2),
#             nn.ReLU(),
#             nn.Linear(hidden_dim_feat_2, output_dim)
#         )
#         self.target_mlp = nn.Sequential(
#             nn.Linear(input_dim_target, hidden_dim_target_1),
#             nn.BatchNorm1d(hidden_dim_target_1),
#             nn.ReLU(), # add more layers?
#             nn.Linear(hidden_dim_target_1, hidden_dim_target_2),
#             nn.BatchNorm1d(hidden_dim_target_2),
#             nn.ReLU(),
#             nn.Linear(hidden_dim_target_2, output_dim)
#         )

class MLP(nn.Module):
    def __init__(self, input_dim_feat, input_dim_target, hidden_dim_feat, hidden_dim_target, output_dim):
        super(MLP, self).__init__()
        self.feat_mlp = nn.Sequential(
            nn.Linear(input_dim_feat, hidden_dim_feat),
            nn.BatchNorm1d(hidden_dim_feat),
            nn.ReLU(), # add more layers?
            nn.Dropout(p=0.3),
            nn.Linear(hidden_dim_feat, output_dim)
        )
        self.target_mlp = nn.Sequential(
            nn.Linear(input_dim_target, hidden_dim_target),
            nn.BatchNorm1d(hidden_dim_target),
            nn.ReLU(), # add more layers?
            nn.Dropout(p=0.3),
            nn.Linear(hidden_dim_target, output_dim)
        )
    def forward(self, x, y):
        features = self.feat_mlp(x)
        targets = self.target_mlp(y)
        features = nn.functional.normalize(features, p=2, dim=1)
        targets = nn.functional.normalize(targets, p=2, dim=1)
        return features, targets