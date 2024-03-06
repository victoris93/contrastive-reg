import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from cmath import isinf
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split, KFold
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import pandas as pd
import math
from cmath import isinf
from utils_v import pca_labels, standardize_dataset, MatData

dataset = MatData("vectorized_matrices_la5c.npy", "hopkins_age.npy")
train_indices, test_indices = train_test_split(np.arange(len(dataset)), test_size = 0.2, random_state=42) #train_size = 5
train_dataset = Subset(dataset, train_indices)
test_dataset = Subset(dataset, test_indices)

def define_model(trial):
        
    dropout_rate = trial.suggest_float('dropout_rate',0.1, 0.5, step = 0.1)
    hidden_dim_feat = trial.suggest_int('hidden_dim_feat', 100, 1000, step = 200)
    input_dim_feat = 499500
    output_dim = 2


    class MLP(nn.Module):
        def __init__(self, input_dim_feat, hidden_dim_feat, output_dim):
            super(MLP, self).__init__()
            self.feat_mlp = nn.Sequential(
                nn.Linear(input_dim_feat, hidden_dim_feat),
                nn.BatchNorm1d(hidden_dim_feat),
                nn.ReLU(), # add more layers?
                nn.Dropout(p=dropout_rate),
                nn.Linear(hidden_dim_feat, output_dim)
            )

        def forward(self, x):
            features = self.feat_mlp(x)
            features = nn.functional.normalize(features, p=2, dim=1)
            return features
        
    return MLP(input_dim_feat, hidden_dim_feat, output_dim)
    
def objective(trial, train_loader, test_loader):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = define_model(trial)
    model.to(device)

    lr = trial.suggest_float('lr', 1e-5, 1e-1)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-1)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Training of the model.
    for epoch in range(10):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
        # Validation of the model.
        model.eval()
        correct = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(val_loader):
                data, target = data.cuda(), target.cuda()
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        accuracy = correct / len(val_loader.dataset)
        trial.report(accuracy, epoch)
        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.TrialPruned()
    return accuracy