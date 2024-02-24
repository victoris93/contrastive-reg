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
from utils_v import compute_age_mae_r2, save_model
from cmath import isinf
import torch.nn.functional as F
from helper_classes import MatData, KernelizedSupCon, MLP, cauchy, rbf, gaussian_kernel
from sklearn.model_selection import train_test_split, KFold

# load data
participants = pd.read_csv('participants.csv')
vect_matrices = np.load("vectorized_matrices.npy")
dataset = MatData("vectorized_matrices.npy", "participants.csv")

kf = KFold(n_splits=10, shuffle=True, random_state=42)
train_indices, test_indices = train_test_split(np.arange(len(dataset)), test_size=0.2, random_state=42)
train_dataset = Subset(dataset, train_indices)
test_dataset = Subset(dataset, test_indices)

input_dim = 499500 # vectorized mat, diagonal discarded
hidden_dim = 128
output_dim = 64

model = MLP(input_dim, hidden_dim, output_dim)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

criterion = KernelizedSupCon(method = 'expw', kernel = cauchy) # gaussian kernel returns nans for some reason
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 10-fold CV training. The model is reinitialized for each fold.
results_cv = []
best_mae = np.inf
best_r2 = 0

for fold, (train_idx, val_idx) in enumerate(kf.split(train_dataset)):
    print(f"Starting fold {fold}")

    train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
    val_subsampler = torch.utils.data.SubsetRandomSampler(val_idx)
    
    train_loader = DataLoader(train_dataset, batch_size=30, sampler=train_subsampler)
    val_loader = DataLoader(train_dataset, batch_size=30, sampler=val_subsampler)
    
    model = MLP(input_dim, hidden_dim, output_dim).to(device)
    criterion = KernelizedSupCon(method='expw', kernel=cauchy)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    num_epochs = 100
    
    for epoch in range(num_epochs):
        model.train()
        batch_losses = []
        for batch_num, (mat, age) in enumerate(train_loader):
            mat, age = mat.to(device), age.to(device)
            optimizer.zero_grad()
            out_feat = model(mat)
            loss = criterion(out_feat, age)
            loss.backward()
            batch_losses.append(loss.item())
            optimizer.step()
        print(f'Fold {fold} | Epoch {epoch} | Mean Loss {sum(batch_losses)/len(batch_losses)}')
            
    val_losses = []
    model.eval() 
    with torch.no_grad():
        total_loss = 0
        total_samples = 0
        for mat, age in val_loader:
            mat = mat.to(device).float()
            age = age.to(device)

            out_feat = model(mat)
            loss = criterion(out_feat, age)
            val_losses.append(loss.item())
            total_loss += loss.item() * mat.size(0)
            total_samples += mat.size(0)
        val_losses =np.array(val_losses)
        average_loss = total_loss / total_samples
    mae_train, r2_train, mae_val, r2_val = compute_age_mae_r2(model, train_loader, val_loader, device)
    if mae_val < best_mae and r2_val > best_r2:
        best_mae = mae_val
        best_r2 = r2_val
        save_model(model, fold, optimizer, f"best_model_cv.pt")
    results_cv.append([fold, mae_train, r2_train, mae_val, r2_val])


results_df = pd.DataFrame(results_cv, columns=['Fold', 'Train_MAE', 'Train_R2', 'Val_MAE', 'Val_R2'])
results_df.to_csv('cv_results.csv', index=False)

print("CV training complete.")
