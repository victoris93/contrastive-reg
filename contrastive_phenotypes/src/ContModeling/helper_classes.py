import torch
import torch.nn as nn
import torch.optim as optim
from cmath import isinf
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import os
import sys
import pandas as pd
import math
from cmath import isinf
import gc
from collections import defaultdict
import xarray as xr


class MatData(Dataset):
    def __init__(self, dataset_path, target_names, threshold=0):
        if not isinstance(target_names, list):
            target_names = [target_names]
        self.target_names = target_names
        self.threshold = threshold
        self.data_array = xr.open_dataset(dataset_path)
        self.matrices = self.data_array.to_array().squeeze().values.astype(np.float32)
        if threshold > 0:
            self.matrices = self.threshold_mat(self.matrices, self.threshold)
        self.matrices = torch.from_numpy(self.matrices).to(torch.float32)
        self.target = torch.from_numpy(np.array([self.data_array[target_name].values for target_name in self.target_names]).T).to(torch.float32)

        gc.collect()

    def threshold_mat(self, matrices, threshold): # as in Margulies et al. (2016)
        perc = np.percentile(np.abs(matrices), threshold, axis=2, keepdims=True)
        mask = np.abs(matrices) >= perc
        thresh_mat = matrices * mask
        return thresh_mat
    
    def __len__(self):
        return self.data_array.subject.__len__()
    
    def __getitem__(self, idx):
        matrix = self.matrices[idx]
        target = torch.from_numpy(np.array([self.data_array.sel(subject=idx)[target_name].values for target_name in self.target_names])).to(torch.float32)
        
    
    def __len__(self):
        return self.data_array.subject.__len__()
    
    def __getitem__(self, idx):
        matrix = self.matrices[idx]
        target = torch.from_numpy(np.array([self.data_array.sel(subject=idx)[target_name].values for target_name in self.target_names])).to(torch.float32)
        
        return matrix, target

