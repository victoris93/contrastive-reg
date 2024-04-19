import numpy as np
import pandas as pd
import sys
import os

def compute_kernel_similarity_matrix(augmented_matrices, original_matrices, krnl_sigma):
    # Vectorized computation of kernel values between two sets of matrices
    augmented_flat = augmented_matrices.reshape(augmented_matrices.shape[0], -1)
    original_flat = original_matrices.reshape(original_matrices.shape[0], -1)
    
    # Compute differences using broadcasting
    diff = augmented_flat[:, np.newaxis, :] - original_flat[np.newaxis, :, :]
    
    # Apply Gaussian kernel to each difference vector
    kernel_values = gaussian_kernel(diff, krnl_sigma)
    
    # Sum along the last dimension to get a single similarity score per pair
    return kernel_values.sum(axis=2)

def l2_normalize(matrix):
    norm = np.linalg.norm(matrix, axis=(1, 2), keepdims=True)
    return matrix / norm

def augment_sample(matrix, threshold_quantile=0.8):
    threshold = np.quantile(np.abs(matrix), threshold_quantile)
    features_thresholded = np.where(np.abs(matrix) > threshold, matrix, np.zeros_like(matrix))
    return features_thresholded

def gaussian_kernel(x, krnl_sigma):
    return np.exp(-(x**2) / (2 * (krnl_sigma**2))) / (np.sqrt(2 * np.pi) * krnl_sigma)

# Load data
path_matrix = "path/to/matrix.npy"
path_participants = "path/to/participants.csv"
matrix = np.load(path_matrix)
participants = pd.read_csv(path_participants)
krnl_sigma = 2

sorted_indices = participants['age'].argsort()
participants_sorted = participants.iloc[sorted_indices].reset_index(drop=True)
matrix_sorted = matrix[sorted_indices]

matrix_normalized = l2_normalize(matrix_sorted)
matrix_augmented = np.array([augment_sample(sample) for sample in matrix_normalized])

# Receive the index from the command line
index = int(sys.argv[1])

# Compute kernel values for a single sample
kernel_values_single = np.zeros((1, 2 * len(matrix_normalized)))
kernel_values_single[0, :len(matrix_normalized)] = compute_kernel_similarity_matrix(matrix_augmented[index:index+1, :, :], matrix_normalized, krnl_sigma)
kernel_values_single[0, len(matrix_normalized):] = compute_kernel_similarity_matrix(matrix_normalized[index:index+1, :, :], matrix_normalized, krnl_sigma)


output_dir = "/data/parietal/store2/work/mrenaudi/contrastive-reg-1/empty_header_large_data.csv"
output_file = "empty_header_large_data.csv"
output_path = os.path.join(output_dir, output_file)

# Ensure the directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Append results to the CSV, writing header only if the file doesn't exist
header = not os.path.exists(output_path)  # True if file does not exist, False otherwise
pd.DataFrame(kernel_values_single).to_csv(output_path, mode='a', header=header, index=False)

print(f"Kernel values appended to {output_path}")
