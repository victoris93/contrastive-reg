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
from helper_classes import MatData, CustomContrastiveLoss
from utils_v import standardize_dataset, compute_target_score
import logging
import json
import csv

def get_model_attribute(model, attribute_name):
    # Check if the model is wrapped with DataParallel
    if isinstance(model, nn.DataParallel):
        # Access attributes of the original model
        return getattr(model.module, attribute_name)
    else:
        # Access attributes directly
        return getattr(model, attribute_name)
    
def define_model(trial):
        
    dropout_rate = trial.suggest_float('dropout_rate',0.1, 0.5, step = 0.1)
    hidden_dim_feat = trial.suggest_int('hidden_dim_feat', 100, 1100, step = 200)
    hiddem_dim_target = 30
    input_dim_feat = 499500
    input_dim_target = 60
    output_dim = 2


    class MLP(nn.Module):
        def __init__(self, input_dim_feat, input_dim_target, hidden_dim_feat, hidden_dim_target, output_dim):
    #     def __init__(self, input_dim_feat, hidden_dim_feat, output_dim):
            super(MLP, self).__init__()
            self.hidden_dim_feat = hidden_dim_feat
            self.dropout_rate = dropout_rate
            self.feat_mlp = nn.Sequential(
                nn.Linear(input_dim_feat, self.hidden_dim_feat),
                nn.BatchNorm1d(self.hidden_dim_feat),
                nn.ReLU(), # add more layers?
                nn.Dropout(p=self.dropout_rate),
                nn.Linear(self.hidden_dim_feat, output_dim)
            )
            self.target_mlp = nn.Sequential(
                nn.Linear(input_dim_target, hidden_dim_target),
                nn.BatchNorm1d(hidden_dim_target),
                nn.ReLU(), # add more layers?
                nn.Dropout(p=self.dropout_rate),
                nn.Linear(hidden_dim_target, output_dim)
            )
            
        def forward(self, x, y):
            features = self.feat_mlp(x)
            targets = self.target_mlp(y)
            features = nn.functional.normalize(features, p=2, dim=1)
            targets = nn.functional.normalize(targets, p=2, dim=1)
            return features, targets
        
    return MLP(input_dim_feat, input_dim_target, hidden_dim_feat, hiddem_dim_target, output_dim)
    
def objective(trial, num_epochs):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_workers = 4 * torch.cuda.device_count() if torch.cuda.is_available() else -1

    model = define_model(trial)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs.")
        model = nn.DataParallel(model)

    model.to(device)

    sigma = trial.suggest_float('sigma', 0.1, 1.0, step = 0.1)
    lr = trial.suggest_float('lr', 1e-5, 1e-1)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-1)
    batch_size = 13
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'SGD', 'RMSprop'])
    if optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)

    dataset = MatData("vectorized_matrices_la5c.npy", "hopkins_covars.npy")
    train_indices, test_indices = train_test_split(np.arange(len(dataset)), test_size = 0.2, random_state=42) #train_size = 5
    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)
    
    standardized_train_dataset = standardize_dataset(train_dataset)
    std_train_loader = DataLoader(standardized_train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    standardized_test_dataset = standardize_dataset(test_dataset)
    std_test_loader = DataLoader(standardized_test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = CustomContrastiveLoss(sigma = sigma)

    for epoch in range(num_epochs):
        model.train()
        batch_losses = []
        for batch_num, (features, targets) in enumerate(std_train_loader):
            features, targets = features.to(device), targets.to(device)
            optimizer.zero_grad()
            out_feat, out_target = model(features, targets)
            loss = criterion(out_feat, out_target)
            loss.backward()
            batch_losses.append(loss.item())
            optimizer.step()
        # print(f'Epoch {epoch} | Mean Loss {sum(batch_losses)/len(batch_losses)}')

    model.eval()
    test_losses = []
    emb_features = [] # saving the embedded features for each batch
    emb_targets = []
    with torch.no_grad():
        total_loss = 0
        total_samples = 0
        for batch_num, (features, targets) in enumerate(std_test_loader):
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

        print(f'Trial {trial}: Mean Test Loss: %6.2f' % (average_loss))

    mape_train, mape_test = compute_target_score(model, std_train_loader, std_test_loader, device, 'mape')
    r2_train,  r2_test = compute_target_score(model, std_train_loader, std_test_loader, device, 'r2')

    print(f'MAPE on Train: {mape_train} | MAPE on Test: {mape_test}')
    print(f'R2 on Train: {r2_train} | R2 on Test: {r2_test}')

    if torch.cuda.device_count() > 1:
        hidden_dim_feat = get_model_attribute(model, 'hidden_dim_feat')
        dropout_rate = get_model_attribute(model, 'dropout_rate')
    else:
        hidden_dim_feat = model.hidden_dim_feat
        dropout_rate = model.dropout_rate
    with open('results/optim_results.csv', 'a') as f:
        # create the csv writer
        writer = csv.writer(f)
        # if the csv file is empty, write the header
        if f.tell() == 0:
            writer.writerow(['hidden_dim_feat',
                                'dropout_rate',
                                'sigma',
                                'lr',
                                'weight_decay',
                                'batch_size',
                                'optimizer_name',
                                'mean_test_loss',
                                'mape_test',
                                'r2_test'])
            
        writer.writerow([hidden_dim_feat,
                            dropout_rate,
                            sigma,
                            lr,
                            weight_decay,
                            batch_size,
                            optimizer_name,
                            average_loss,
                            mape_test,
                            r2_test])
            
    return average_loss, mape_test, r2_test


def main():
    db_file = "sqlite:///optuna_study.db"
    study = optuna.create_study(study_name="contrastive-optim",directions=['minimize', 'minimize', 'maximize'], storage=db_file, load_if_exists=True)
    study.optimize(lambda trial: objective(trial, num_epochs=100), n_trials=400)
    best_hyperparams = study.best_trial.params

    with open('results/best_hyperparameters.json', 'w') as f:
        json.dump(best_hyperparams, f)

    print("Optimization complete. The best hyperparameters are saved in 'best_hyperparameters.json'")

if __name__ == "__main__":
    main()
