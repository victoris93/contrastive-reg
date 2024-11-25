import numpy as np
from nilearn.connectome import sym_matrix_to_vec, vec_to_sym_matrix
from scipy.linalg import pinv, diagsvd
from sklearn.utils.extmath import randomized_svd
from itertools import combinations
import torch.nn as nn
import torch
from copy import copy
import math
import random
from nilearn import datasets


# +

class GPC(nn.Module):
    def __init__(self, in_dim, out_dim, node):
        super(GPC, self).__init__()
        self.out_dim = out_dim
        self.in_dim = in_dim
        self.node = node
        self.conv = nn.Conv2d(self.in_dim, self.out_dim, (1, self.node))
        nn.init.normal_(self.conv.weight, std=math.sqrt(2/(self.node*self.in_dim+self.node*self.out_dim)))
        
    def forward(self, x):
        x = x.view(1, 1, self.node, self.node)
        batchsize = x.shape[0]
        
        x_c = self.conv(x)
        x_C = x_c.expand(batchsize, self.out_dim, self.node, self.node)
        x_R = x_C.permute(0,1,3,2)
        x = x_C+x_R
        
        return x


# -

def gpc(matrix, in_dim, out_dim, node):
    gpc_mat = torch.from_numpy(copy(matrix)).float()
    gpc = GPC(in_dim, out_dim, node)
    gpc_mat = gpc(gpc_mat).squeeze()
    gpc_mat = gpc_mat.detach().numpy()
    mat_aug = matrix + gpc_mat
    
    return mat_aug


def threshMat(conn, thresh):
    perc = np.percentile(np.abs(conn), thresh, axis=1)  # Calculate the percentile along each matrix
    mask = np.abs(conn) > perc[:, None]
    thresh_mat = conn * mask
    return thresh_mat, mask, perc

def random_threshold_augmentation(matrices, threshold, bound = 1): # as in Margulies et al. (2016)
    
    thresh_mat, mask, perc = threshMat(matrices, threshold)
    random_values = np.random.uniform(-1/bound,1/bound, matrices.shape) * perc
    random_values_masked = random_values * (1-mask)
    
    mat = thresh_mat + random_values_masked
    
    return mat

def flipping_threshold_augmentation(matrix, threshold, hemisphere_size=None):
    matrix_thresholded, _, _ = threshMat(matrix, threshold)
    
    if hemisphere_size is None:
        hemisphere_size = 500
        
    reg_indices_left = np.arange(0, hemisphere_size)
    reg_indices_right = np.arange(hemisphere_size, 2*hemisphere_size)
    
    edge_indices_left = list(combinations(reg_indices_left, 2))
    edge_indices_right = list(combinations(reg_indices_right, 2))
    
    flip_index_pairs = []
    for edge_left, edge_right in zip(edge_indices_left, edge_indices_right):
        if (matrix_thresholded[edge_left] == 0) & (matrix_thresholded[edge_right] == 0):
            flip_index_pairs.append([edge_left, edge_right])
            
    flipped_matrix = flip_edges(matrix, flip_index_pairs)
            

    return flipped_matrix

def SVD_augmentation(X, n_components, n_iter, noise_factor, random_state=42):
    X_pinv = pinv(X)
    U, Sigma, VT = randomized_svd(X_pinv, n_components=n_components, n_iter=n_iter, random_state=random_state)
    Sigma += noise_factor * np.random.normal(size=Sigma.shape)
    original_Sigma = 1 / Sigma
    original_U = np.transpose(U)
    original_V = np.transpose(VT)
    X_augmented = np.dot(original_V, np.dot(np.diag(original_Sigma), original_U))
    return X_augmented

def flip_edges(matrix, edge_idx, vectorize_mat=False):
    
    if not isinstance(edge_idx[0], type([])):
        edge_idx = [edge_idx]
    flipped_matrix = matrix.copy()
    for i, _ in enumerate(edge_idx):
        edge_pair = edge_idx[i]
        indices_left = edge_pair[0]
        indices_right = edge_pair[1]
        edge_left = matrix[indices_left]
        edge_right = matrix[indices_right]
            
        flipped_matrix[indices_left] = edge_right
        flipped_matrix[indices_right] = edge_left
        
    if vectorize_mat:
        # Assuming 'sym_matrix_to_vec' is a function you have that vectorizes a symmetric matrix
        flipped_matrix = sym_matrix_to_vec(flipped_matrix, discard_diagonal=True)
        
        
    return flipped_matrix


def extract_label_idx(substring):
    schaefer_1000 = datasets.fetch_atlas_schaefer_2018(n_rois=1000, yeo_networks=7, resolution_mm=1)
    labels = schaefer_1000['labels']
    label_indices = [i for i, label in enumerate(labels) if substring.encode() in label]

    return label_indices

def mixup(matrix, training_matrices, threshold):
    thresh_mat, mask, perc = threshMat(copy(matrix), threshold)
    thresholded_indices = np.where(~mask)
    random_mat = random.choice(training_matrices)
    matrix[thresholded_indices] = random_mat[thresholded_indices]
    return matrix

#if mode = extract, extracts a network a set other values to 0, else if mode = eliminate, 
#sets the values of the chosen network to 0 and keeps the other values
def isolate_network(matrix, string, mode = "extract"):
    label_indices = extract_label_idx(string)
    mask = np.zeros_like(matrix, dtype=bool)
    mask[:, label_indices] = True
    mask[label_indices, :]= True
    if mode == "extract" : 
        matrix[~mask]=0
    elif mode == "eliminate" : 
        matrix[mask]=0
    return matrix


# +

augs = {
    "random_threshold_augmentation": random_threshold_augmentation,
    "flipping_threshold_augmentation": flipping_threshold_augmentation,
    "SVD_augmentation": SVD_augmentation,
    "gpc": gpc,
    "isolate_network": isolate_network,
    "mixup": mixup
    
    }
# -

aug_args = {
    "random_threshold_augmentation": {"threshold": 60,"bound" : 1},
    "flipping_threshold_augmentation": {"threshold": 60, "hemisphere_size": None},
    "SVD_augmentation": {"n_components": 10, "n_iter": 5, "noise_factor": 0.01, "random_state":42},
    "gpc" : {"in_dim": 1, "out_dim": 1, 'node' : 200},
    "isolate_network" : {"string" : "Default"},
    "mixup" : {"threshold": 60}
    
    }