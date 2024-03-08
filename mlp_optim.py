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
from utils_v import standardize_dataset
import json
import csv

dataset = MatData("vectorized_matrices_la5c.npy", "hopkins_covars.npy")
train_indices, test_indices = train_test_split(np.arange(len(dataset)), test_size = 0.2, random_state=42) #train_size = 5
train_dataset = Subset(dataset, train_indices)
test_dataset = Subset(dataset, test_indices)

standardized_train_dataset = standardize_dataset(train_dataset)
std_train_loader = DataLoader(standardized_train_dataset, batch_size=10, shuffle=True)

standardized_test_dataset = standardize_dataset(test_dataset)
std_test_loader = DataLoader(standardized_test_dataset, batch_size=10, shuffle=True)

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
    
def objective(trial, train_loader, test_loader, num_epochs):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = define_model(trial)
    model.to(device)
    sigma = trial.suggest_float('sigma', 0.1, 1.0, step = 0.1)
    lr = trial.suggest_float('lr', 1e-5, 1e-1)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-1)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    criterion = CustomContrastiveLoss(sigma = sigma)

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
        # print(f'Epoch {epoch} | Mean Loss {sum(batch_losses)/len(batch_losses)}')

    model.eval()
    test_losses = []
    emb_features = [] # saving the embedded features for each batch
    emb_targets = []
    with torch.no_grad():
        total_loss = 0
        total_samples = 0
        for batch_num, (features, targets) in enumerate(test_loader):
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
                                 'mean_test_loss'])
                
            writer.writerow([model.hidden_dim_feat,
                             model.dropout_rate,
                             sigma,
                             lr,
                             weight_decay,
                            average_loss])
    return average_loss


def main():
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, std_train_loader, std_test_loader,num_epochs=100), n_trials=300)
    best_hyperparams = study.best_trial.params

    with open('best_model/best_hyperparameters.json', 'w') as f:
        json.dump(best_hyperparams, f)

    print("Optimization complete. The best hyperparameters are saved in 'best_hyperparameters.json'")

if __name__ == "__main__":
    main()
