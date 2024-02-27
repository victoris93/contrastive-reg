import torch
import torch.nn as nn
import torch.optim as optim
from cmath import isinf
import torch.nn.functional as F
import seaborn as sns
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import pandas as pd
import math
from utils_v import compute_target_score, estimate_target, save_model
from cmath import isinf
from sklearn.model_selection import train_test_split, KFold, LearningCurveDisplay, learning_curve
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_percentage_error, r2_score
from helper_classes import MatData, KernelizedSupCon, MLP, cauchy, rbf, gaussian_kernel

dataset = MatData("vectorized_matrices_la5c.npy", "hopkins.npy")
kf = KFold(n_splits=5, shuffle=True, random_state=42)

train_indices, test_indices = train_test_split(np.arange(len(dataset)), test_size=0.2, random_state=42)
train_dataset = Subset(dataset, train_indices)
test_dataset = Subset(dataset, test_indices)

input_dim_feat = 499500 # vectorized mat, diagonal discarded
input_dim_target = 58
hidden_dim_feat = 512
hidden_dim_target = 24
output_dim = 2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


results_cv = []
best_mae = np.inf
best_r2 = -np.inf
best_average_loss = np.inf
for fold, (train_idx, val_idx) in enumerate(kf.split(train_dataset)):
    print(f"Starting fold {fold}")

    train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
    val_subsampler = torch.utils.data.SubsetRandomSampler(val_idx)
    
    train_loader = DataLoader(train_dataset, batch_size=5, sampler=train_subsampler)
    val_loader = DataLoader(train_dataset, batch_size=5, sampler=val_subsampler)
    
    model = MLP(input_dim_feat, input_dim_target, hidden_dim_feat, hidden_dim_target, output_dim).to(device)
    criterion = KernelizedSupCon(method='expw', temperature = 1000, base_temperature = 0.1, kernel=gaussian_kernel)
    optimizer = optim.Adam(model.parameters(), lr=0.1)
    
    num_epochs = 1000
    
    for epoch in range(num_epochs):
        model.train()
        batch_losses = []
        for batch_num, (features, targets) in enumerate(train_loader):
            features, targets = features.to(device), targets.to(device)
            optimizer.zero_grad()
            out_feat, out_target = model(features, targets)
            loss = criterion(out_feat, out_target)
            loss.backward()
            batch_losses.append(loss.item())
            optimizer.step()
        print(f'Fold {fold} | Epoch {epoch} | Mean Loss {sum(batch_losses)/len(batch_losses)}')
            
    val_losses = []
    model.eval() 
    with torch.no_grad():
        total_loss = 0
        total_samples = 0
        for features, targets in val_loader:
            features = features.to(device).float()
            targets = targets.to(device)

            out_feat, out_target = model(features, targets)
            loss = criterion(out_feat, out_target)
            val_losses.append(loss.item())
            total_loss += loss.item() * features.size(0)
            total_samples += features.size(0)
        val_losses =np.array(val_losses)
        average_loss = total_loss / total_samples
#         if best_average_loss > average_loss:
#             best_average_loss = average_loss
#             save_model(model, fold, optimizer, f"best_model_hopkins_cv.pt")
    mae_train, mae_val = compute_target_score(model, train_loader, val_loader, device, 'mape')
    r2_train, r2_val = compute_target_score(model, train_loader, val_loader, device, 'r2')
    if mae_train < best_mae and r2_train > best_r2:
        best_mae = mae_train
        best_r2 = r2_train
        save_model(model, fold, optimizer, f"best_model_hopkins_cv.pt")
    results_cv.append([fold, mae_train, r2_train, mae_val, r2_val])

results_df = pd.DataFrame(results_cv, columns=['Fold', 'Train_MAPE', 'Train_R2', 'Val_MAPE', 'Val_R2'])
results_df.to_csv('cv_results_hopkins.csv', index=False)

# Testing on train

test_loader = DataLoader(test_dataset, batch_size=5, shuffle=False)
train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True)

model = MLP(input_dim_feat, input_dim_target, hidden_dim_feat, hidden_dim_target, output_dim)
model.load_state_dict(torch.load('best_model_hopkins_cv.pt')["model"])
criterion = KernelizedSupCon(method='expw', temperature = 1000, base_temperature = 0.1, kernel=gaussian_kernel)
optimizer = optim.Adam(model.parameters(), lr=0.001)
optimizer.load_state_dict(torch.load('best_model_hopkins_cv.pt')["optimizer"])

model.to(device)
test_losses = []
model.eval()
emb_features = []
emb_targets = []
with torch.no_grad():
    total_loss = 0
    total_samples = 0
    for batch_num, (features, targets) in enumerate(train_loader):
        features = features.to(device).float()
        targets = targets.to(device)

        out_feat, out_target = model(features, targets)
        emb_features.append(out_feat.cpu())
        emb_targets.append(out_target.cpu())
        loss = criterion(out_feat, out_target)
        test_losses.append(loss.item())
        total_loss += loss.item() * features.size(0)
        total_samples += features.size(0)
        
    test_losses =np.array(test_losses)
    average_loss = total_loss / total_samples
    print('Mean Test Loss: %6.2f' % (average_loss))

emb_features = np.row_stack(emb_features)
emb_targets = np.row_stack(emb_targets)
emb_features = pd.DataFrame(emb_features,columns = ["Dim_1", "Dim_2"])
emb_targets = pd.DataFrame(emb_targets,columns = ["Dim_1", "Dim_2"])
emb_features["sub"] = np.arange(1, len(emb_features) +1)
emb_targets["sub"] = np.arange(1, len(emb_targets) +1)
emb_features["Type"] = 'Features'
emb_targets["Type"] = 'Targets'
embeddings = pd.concat([emb_features, emb_targets])
embeddings.to_csv('embeddings_hopkins.csv', index=False)

mae_train, mae_test = compute_target_score(model, train_loader, test_loader, device, 'mae')
r2_train, r2_test = compute_target_score(model, train_loader, test_loader, device, 'r2')

print(f"Train MAE: {mae_train}, Test MAE: {mae_test}.")
print(f"Train R2: {r2_train}, Test R2: {r2_test}.")

