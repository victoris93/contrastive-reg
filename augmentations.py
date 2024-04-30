import numpy as np
from nilearn.connectome import sym_matrix_to_vec
from scipy.linalg import pinv, diagsvd
from sklearn.utils.extmath import randomized_svd
import random


def random_threshold(matrices, threshold, bound = 1): # as in Margulies et al. (2016)
    
    perc = np.percentile(np.abs(matrices), threshold, axis=2, keepdims=True)
    mask = np.abs(matrices) >= perc
    thresh_mat = matrices * mask
    
    random_values = np.random.uniform(-1/bound,1/bound, matrices.shape) * perc
    random_values_masked = random_values * (1-mask)
    
    mat = thresh_mat + random_values_masked
    
    return mat

def flipping_threshold_augmentation(features, hemisphere_size=None):
    threshold = np.quantile(features, 0.95)
    features_thresholded = np.where(features < threshold, 0, features)
    zero_indices = np.argwhere(features_thresholded == 0)
    if hemisphere_size is None:
        hemisphere_size = features.shape[0] // 2
    if zero_indices.size > 0:
        selected_idx = random.choice(zero_indices)
        selected_pairs = []
        region1 = selected_idx[0]
        opposite_region = region1 + hemisphere_size if region1 < hemisphere_size else region1 - hemisphere_size
        if features_thresholded[opposite_region, selected_idx[1]] == 0:
            selected_pairs.append([region1, opposite_region])
        if selected_pairs:
            flipped_matrices = flip_edge_between_regions(features, selected_pairs, vectorize_mat=False)
            return flipped_matrices
    return features

def SVD_augmentation(X, n_components, n_iter, noise_factor, random_state=42):
    X_pinv = pinv(X)
    U, Sigma, VT = randomized_svd(X_pinv, n_components=n_components, n_iter=n_iter, random_state=random_state)
    Sigma += noise_factor * np.random.normal(size=Sigma.shape)
    original_Sigma = 1 / Sigma
    original_U = np.transpose(U)
    original_V = np.transpose(VT)
    X_augmented = np.dot(original_V, np.dot(np.diag(original_Sigma), original_U))
    X_augmented = np.expand_dims(X_augmented, axis=0)

    return X_augmented

def flip_edge_between_regions(matrix, regions, vectorize_mat=True):
    if not isinstance(regions[0], list):
        regions = [regions]
    flipped_matrices = []
    
    for pair in regions:
        new_matrix = matrix.copy()
        hemisphere_size = matrix.shape[0] // 2
        
        # Determine the indices for flipping
        for index1, index2 in [pair]:
            is_left_hemisphere1 = index1 < hemisphere_size
            is_left_hemisphere2 = index2 < hemisphere_size
            
            # Calculate the opposite indices in the other hemisphere
            #opposite_index1 = index1 + (-1 if is_left_hemisphere1 else 1) * hemisphere_size
            #opposite_index2 = index2 + (-1 if is_left_hemisphere2 else 1) * hemisphere_size
            opposite_index1 = index1 + (1 if is_left_hemisphere1 else -1) * hemisphere_size
            opposite_index2 = index2 + (1 if is_left_hemisphere2 else -1) * hemisphere_size

            # Flip the connectivity edge between the specified pairs across hemispheres
            new_matrix[index1, opposite_index2], new_matrix[index2, opposite_index1] = \
                new_matrix[index2, opposite_index1], new_matrix[index1, opposite_index2]
            new_matrix[opposite_index2, index1], new_matrix[opposite_index1, index2] = \
                new_matrix[opposite_index1, index2], new_matrix[opposite_index2, index1]
        
        if vectorize_mat:
            # Assuming 'sym_matrix_to_vec' is a function you have that vectorizes a symmetric matrix
            new_matrix = sym_matrix_to_vec(new_matrix, discard_diagonal=True)
        
        flipped_matrices.append(new_matrix)
        
        return np.array(flipped_matrices)

augs = {
    "random_threshold_augmentation": random_threshold_augmentation,
    "flipping_threshold_augmentation": flipping_threshold_augmentation,
    "SVD_augmentation": SVD_augmentation,
    "flip_edge_between_regions": flip_edge_between_regions
}
