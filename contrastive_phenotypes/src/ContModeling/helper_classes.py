import torch
import torch.nn as nn
import torch.optim as optim
from cmath import isinf
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from nilearn.connectome import sym_matrix_to_vec, vec_to_sym_matrix
import numpy as np
import os
from sklearn.decomposition import PCA
import sys
import pandas as pd
import math
from cmath import isinf
import gc
from collections import defaultdict
import xarray as xr
import submitit
import pickle


class AETrain(submitit.helpers.Checkpointable):
    def __init__(self):
        self.results = None

    def __call__(self,train_func, train_idx, val_idx, train_ratio, dataset, model_params_dir, cfg, fold=None, random_state=None, device=None, path = None):

        if self.results is None:
            if device is None:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if not isinstance(random_state, np.random.RandomState):
                seed = random_state
                random_state = np.random.RandomState(seed)
                
        if fold is not None:
            self.run_type = "fold"
            self.run_id = fold
        else:
            self.run_type = "seed"
            self.run_id = seed
            
        print(f"Run Type {self.run_type}, Run ID {self.run_id}")
        fold_train_dataset = Subset(dataset, train_idx)
        fold_val_dataset = Subset(dataset, val_idx)

        input_dim_feat=cfg.input_dim_feat
        output_dim_feat=cfg.output_dim_feat
        
        train_features = fold_train_dataset.dataset.matrices[fold_train_dataset.indices]
        mean_f = torch.mean(torch.tensor(train_features), dim=0).to(device)
        [D,V] = torch.linalg.eigh(mean_f,UPLO = "U")     
        B_init_fMRI = V[:,input_dim_feat-output_dim_feat:]

        loss_terms, trained_weights, val_loss = train_func(fold_train_dataset,
                                                           fold_val_dataset,
                                                           B_init_fMRI,
                                                           self.run_type,
                                                           self.run_id,
                                                           cfg,
                                                           device)


        torch.save(trained_weights, f"{model_params_dir}/autoencoder_weights_{self.run_type}{self.run_id}_train_ratio{train_ratio}.pth")
        
        self.results = {self.run_type: self.run_id,
                        "val_loss": val_loss,
                        } 
        return self.results

    def checkpoint(self, *args, **kwargs):
        print("Checkpointing", flush=True)
        return super().checkpoint(*args, **kwargs)

    def save(self, path):
        with open(path, "wb") as o:
            pickle.dump(self.results, o, pickle.HIGHEST_PROTOCOL)


class MatData(Dataset):
    def __init__(self, dataset_path, target_names, synth_exp=False, reduced_mat=False, vectorize=False, threshold=0):
        if not isinstance(target_names, list):
            target_names = [target_names]
        self.target_names = target_names
        self.threshold = threshold
        self.data_array = xr.open_dataset(dataset_path)
        self.pca_features = None
        if reduced_mat:
            self.matrices = self.data_array.reduced_matrices.values.astype(np.float32)
        else:
            self.matrices = self.data_array.matrices.values.astype(np.float32)

        self.intra_network_conn = torch.from_numpy(self.data_array.intra_network_conn.values).to(torch.float32)
        self.inter_network_conn = torch.from_numpy(self.data_array.inter_network_conn.values).to(torch.float32)

        if vectorize:
            self.matrices = sym_matrix_to_vec(self.matrices, discard_diagonal=True)
            
        self.targets = np.array([self.data_array[target_name].values for target_name in self.target_names]).T

        if threshold > 0:
            self.matrices = self.threshold_mat()

        if synth_exp:
            print("Simulating effect")
            self.matrices = self.simulate_effect(self.matrices, self.targets)

        self.matrices = torch.from_numpy(self.matrices).to(torch.float32)
        self.targets = torch.from_numpy(self.targets).to(torch.float32)

        gc.collect()

    def threshold_mat(self, matrices, threshold): # as in Margulies et al. (2016)
        perc = np.percentile(np.abs(matrices), threshold, axis=2, keepdims=True)
        mask = np.abs(matrices) >= perc
        thresh_mat = matrices * mask
        return thresh_mat
    
    def compute_pca(self, pca_obj=None):
        vec_features = sym_matrix_to_vec(self.matrices, discard_diagonal=True)
        if pca_obj is None:
            pca_obj = PCA(n_components=64).fit(vec_features)

        pca_features = pca_obj.transform(vec_features)
        self.pca_features = pca_features
        return pca_obj
    
    def simulate_effect(self, matrices, targets):
        # hypothesis: positive connectivity is stronger
        # when IQ is higher
        # standardize lables
        targets_std = (targets - 100) / 15 * 0.1

        beta = 5
        effect = beta * targets_std
        effect_matrices = np.zeros_like(matrices)

        for idx, effect_matrix in enumerate(effect_matrices):
            pos_conn_idx = np.where(matrices[idx] > 1)
            effect_matrix[pos_conn_idx] = effect[idx]
            effect_matrices[idx] = effect_matrix

        sim_matrices = matrices + effect_matrices
        return sim_matrices

    def __len__(self):
        return self.data_array.index.__len__()
    
    def __getitem__(self, idx):
        matrix = self.matrices[idx]
        target = self.targets[idx]
        pca_feature = None
        if self.pca_features is not None:
            pca_feature = self.pca_features[idx]
        intra_network_conn_vect = self.intra_network_conn[idx]
        inter_network_conn_vect = self.inter_network_conn[idx]
        
        return matrix, pca_feature, target, intra_network_conn_vect, inter_network_conn_vect
    