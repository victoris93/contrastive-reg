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
from sklearn.model_selection import train_test_split, KFold
from helper_classes import MatData, KernelizedSupCon, MLP, cauchy

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

participants = pd.read_csv('participants.csv')
vect_matrices = np.load("vectorized_matrices.npy")
dataset = MatData("vectorized_matrices.npy", "participants.csv")

kf = KFold(n_splits=10, shuffle=True, random_state=42)
train_indices, test_indices = train_test_split(np.arange(len(dataset)), test_size=0.2, random_state=42)
test_dataset = Subset(dataset, test_indices)
train_dataset = Subset(dataset, train_indices)
test_loader = DataLoader(test_dataset, batch_size=30, shuffle=False)
train_loader = DataLoader(train_dataset, batch_size=30, shuffle=True)
input_dim = 499500  # Adjust as per your model's input dimension
hidden_dim = 128  # Adjust as per your model
output_dim = 64  # Adjust as per your model

model = MLP(input_dim, hidden_dim, output_dim)
model.load_state_dict(torch.load('best_model_cv.pt')["model"])
criterion = KernelizedSupCon(method='expw', kernel=cauchy)
optimizer = optim.Adam(model.parameters(), lr=0.001)
optimizer.load_state_dict(torch.load('best_model_cv.pt')["optimizer"])

model.to(device)

test_losses = []
model.eval()
with torch.no_grad():
    print('==> Testing...')
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
print('==> Computing age MAE and R2...')
_, _, mae_test, r2_test = compute_age_mae_r2(model, train_loader, test_loader, device)
print(f"Mean Test Loss: {average_loss} | Test Age MAE: {mae_test} | Test Age R2: {r2_test}.")