import numpy as np
from nilearn.connectome import sym_matrix_to_vec, vec_to_sym_matrix
from scipy.linalg import pinv, diagsvd
from sklearn.utils.extmath import randomized_svd

def threshMat(conn, thresh):
    perc = np.percentile(np.abs(conn), thresh, axis=1)  # Calculate the percentile along each matrix
    mask = np.abs(conn) > perc[:, None]
    thresh_mat = conn * mask
    return thresh_mat, perc

def random_threshold_augmentation(features, threshold):
    features_thresholded, threshold = threshMat(features, threshold)
    random_values = np.random.uniform(0, threshold, features.shape)
    augmented_features = np.where(features_thresholded == 0, random_values, features_thresholded)
#     norm = np.linalg.norm(augmented_features)
#     normalized_features = augmented_features / norm if norm != 0 else augmented_features
    return augmented_features

def flipping_threshold_augmentation(features, threshold, max_attempts, hemisphere_size=None):
    features_thresholded, _ = threshMat(features, threshold)
    zero_indices = np.argwhere(features_thresholded == 0)
    print(zero_indices.shape)
    
    if hemisphere_size is None:
        hemisphere_size = features.shape[0] // 2
    
    attempt = 0
    max_attempts = max_attempts  
    
    while attempt < max_attempts:
            selected_idx = zero_indices[np.random.choice(zero_indices.shape[0])]
            region1 = selected_idx[0]
            opposite_region = region1 + hemisphere_size if region1 < hemisphere_size else region1 - hemisphere_size
            
            if features_thresholded[opposite_region, selected_idx[1]] == 0:
                selected_pairs = [[region1, opposite_region]]
                flipped_matrices = flip_edge_between_regions(features, selected_pairs)
            else:
                print(f"Attempt {attempt}: no valid pair found.")
                flipped_matrices = None
            attempt += 1

    return flipped_matrices

def SVD_augmentation(X, n_components, n_iter, noise_factor, random_state=42):
    X_pinv = pinv(X)
    U, Sigma, VT = randomized_svd(X_pinv, n_components=n_components, n_iter=n_iter, random_state=random_state)
    Sigma += noise_factor * np.random.normal(size=Sigma.shape)
    original_Sigma = 1 / Sigma
    original_U = np.transpose(U)
    original_V = np.transpose(VT)
    X_augmented = np.dot(original_V, np.dot(np.diag(original_Sigma), original_U))
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
}

aug_args = {
    "random_threshold_augmentation": {"threshold": 60},
    "flipping_threshold_augmentation": {"threshold": 60, "max_attempts" : 100, "hemisphere_size": None},
    "SVD_augmentation": {"n_components": 10, "n_iter": 5, "noise_factor": 0.01, "random_state":42}
}