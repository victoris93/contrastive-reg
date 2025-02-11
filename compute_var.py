import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from nilearn import datasets
import xarray as xr
from tqdm import tqdm
import sys


def get_network_indices(network_labels):
    network_indices = dict.fromkeys(network_labels)
    network_labels = np.array(network_labels)
    for key in network_indices.keys():
        network_indices[key] = np.arange(len(network_labels))[network_labels == key]
    return network_indices

def replace_with_network(label, network_labels):
    for network in network_labels:
        if network in label:
            return network
    return label

def mean_conn_var_network(matrix, network_indices):
    mean_conn_var_network = dict.fromkeys(network_indices.keys())
    for network in network_indices.keys():
        indices = network_indices[network]
        net_mean_var = matrix[indices].mean()
        mean_conn_var_network[network] = net_mean_var
    return mean_conn_var_network

atlas_labels = datasets.fetch_atlas_schaefer_2018()['labels']
atlas_labels = [label.decode('utf-8') for label in atlas_labels]
network_labels = ['Vis', 'SomMot', 'DorsAttn', 'SalVentAttn', 'Limbic', 'Cont', 'Default']
network_labels = [replace_with_network(label, network_labels) for label in atlas_labels]
network_indices = get_network_indices(network_labels)

root = "/data/parietal/store/work/vshevche"
run = 0
exp_name = sys.argv[1]
dataset = sys.argv[2]

data =  xr.open_dataset(f'{root}/data/hcp_kong_400parcels.nc')

exp_dir = f'{root}/results/{exp_name}'
predictions = pd.read_csv(f'{exp_dir}/pred_results.csv')
predictions = predictions[(predictions["dataset"] == dataset) & (predictions["train_ratio"] == 1) & (predictions["model_run"] == 0)]
sub_idx = predictions.indices.values
matrices = data.isel(index = sub_idx).matrices.values

ref_mat = matrices[0]
ref_mat_network_conn = np.fromiter(mean_conn_var_network(ref_mat, network_indices).values(), dtype=float)
network_conn_diff = []

for mat in matrices:
    mean_network_conn = np.fromiter(mean_conn_var_network(mat, network_indices).values(), dtype=float)
    diff = np.abs(ref_mat_network_conn - mean_network_conn)
    network_conn_diff.append(diff)

network_conn = np.array(network_conn_diff)
np.save(f"{exp_dir}/embeddings/mean_conn_network_{dataset}.npy", network_conn)
print(f"Variance of connectivity matrices for {exp_name} is computed and saved.")
print(f"{exp_dir}/embeddings/mean_conn_network_{dataset}.npy")