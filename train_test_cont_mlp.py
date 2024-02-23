import torch
import torch.nn as nn
import torch.optim as optim
from cmath import isinf
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import pandas as pd
import math
from utils_v import compute_age_mae_r2
from cmath import isinf
import torch.nn.functional as F

krnl_sigma = 1

def gaussian_kernel(x):
    x = x - x.T
    return torch.exp(-(x**2) / (2*(krnl_sigma**2))) / (math.sqrt(krnl_sigma*torch.pi)*1)

def rbf(x):
        x = x - x.T
        return torch.exp(-(x**2)/(2*(krnl_sigma**2)))

def cauchy(x):
        x = x - x.T
        return  1. / (krnl_sigma*(x**2) + 1)

# loss from: https://github.com/EIDOSLAB/contrastive-brain-age-prediction/blob/master/src/losses.py
# modified to accept input shape [bsz, n_feats]. In the age paper: [bsz, n_views, n_feats].
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
            labels = labels.view(-1, 1)
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

class MatData(Dataset):
    def __init__(self, path_mat, path_dm):
        self.matrices = np.load(path_mat)
        self.target = pd.read_csv(path_dm)['age'].values
    def __len__(self):
        return len(self.matrices)
    def __getitem__(self, idx):
        matrix = self.matrices[idx]
        target = self.target[idx]
        matrix = torch.from_numpy(matrix).float()
        target = torch.tensor(target, dtype=torch.float32)
        return matrix, target

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(), # add more layers?
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        features = self.network(x)
        normalized_features = nn.functional.normalize(features, p=2, dim=1)
        return normalized_features

# load data
participants = pd.read_csv('participants.csv')
vect_matrices = np.load("vectorized_matrices.npy")
dataset = MatData("vectorized_matrices.npy", "participants.csv")

total_size = len(dataset)
train_size = int(total_size * 0.8)
test_size = total_size - train_size
test_size = int(test_size)

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

input_dim = 499500 # vectorized mat, diagonal discarded
hidden_dim = 128
output_dim = 64

model = MLP(input_dim, hidden_dim, output_dim)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

criterion = KernelizedSupCon(method = 'expw', kernel = cauchy) # gaussian kernel returns nans for some reason
optimizer = optim.Adam(model.parameters(), lr=0.001)

model.train()
for epoch in range(100):
    batch_losses = []
    for batch_num, (mat, age) in enumerate(train_loader):
        optimizer.zero_grad()
        mat = mat.to(device).float()
        age = age.to(device)

        out_feat = model(mat)
        loss = criterion(out_feat,age)
        loss.backward()
        batch_losses.append(loss.item())
        optimizer.step()
        print('\tEpoch %d | Batch %d | Loss %6.2f' % (epoch, batch_num, loss.item()))
    batch_losses = np.array(batch_losses)
    np.save(f"losses/train_losses_batch{batch_num}_epoch{epoch}.npy", batch_losses)
    print('Epoch %d | Mean Loss %6.2f' % (epoch, sum(batch_losses)/len(batch_losses)))


test_losses = []
model.eval()
with torch.no_grad():
    total_loss = 0
    total_samples = 0
    for batch_num, (mat, age) in enumerate(test_loader):
        mat = mat.to(device).float()
        age = age.to(device)

        out_feat = model(mat)
        loss = criterion(out_feat, age)
        test_losses.append(loss.item())
        total_loss += loss.item() * mat.size(0)
        total_samples += mat.size(0)
    test_losses =np.array(test_losses)
    average_loss = total_loss / total_samples
    print('Mean Test Loss: %6.2f' % (average_loss))
    np.save(f"losses/test_losses_batch{batch_num}.npy", test_losses)

# estimate age from the embeddings and compute MAE
mae_train, r2_train, mae_test, r2_test = compute_age_mae_r2(model, train_loader, test_loader, device)
print(f"Train Age MAE: {mae_train}, Test Age MAE: {mae_test}.")
print(f"Train Age R2: {r2_train}, Test Age R2: {r2_test}.")