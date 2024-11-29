import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import xarray as xr
from tqdm import tqdm
import sys

def nn(exp_name, dataset, n_neighbors):

    root = "/data/parietal/store2/work/mrenaudi/contrastive-reg-3"
    run = 0
    exp_name = exp_name
    dataset = dataset

    exp_dir = f'{root}/results/{exp_name}'
    predictions = pd.read_csv(f'{exp_dir}/pred_results.csv')
    predictions = predictions[(predictions["dataset"] == dataset) & (predictions["train_ratio"] == 1)]
    sub_idx = predictions.indices.values
    embeddings = np.load(f"{exp_dir}/embeddings/joint_embeddings_run0_{dataset}.npy")
    n_neighbors = int(n_neighbors)

    neigh = NearestNeighbors(n_neighbors=n_neighbors, radius=1, algorithm='brute', metric = 'cosine', p = 2).fit(embeddings)
    distances, neighbors = neigh.kneighbors(embeddings, return_distance = True)
    data =  xr.open_dataset(f'{root}/dataset_400parcels_2.nc')

    neigh_conn_var = []
    #neigh_conn_dvar_ddist = []

    for neighborhood_idx, neighborhood in enumerate(neighbors):
        var = 0
        dist_neigh = distances[neighborhood_idx]
        neigh_idx = sub_idx[neighborhood]
        neigh_matrices = data.matrices.isel(subject = neigh_idx).values
        centroid = neigh_matrices[0]
        neigh_conn_dvar_ddist = []
        for mat_idx, mat in enumerate(neigh_matrices):
            if mat_idx == 0:
                continue
            var_shift = (centroid - mat) ** 2
            var += var_shift
            sub_dvar_ddist = var_shift / dist_neigh[mat_idx]
            neigh_conn_dvar_ddist.append(sub_dvar_ddist)
        
        neigh_conn_var.append(var / len(neighborhood[1:]))
        np.save(f"{exp_dir}/embeddings/neigh_conn_dvar_ddist_{dataset}_{n_neighbors}_neigh_idx{neighborhood_idx}.npy", np.array(neigh_conn_dvar_ddist))

    neigh_conn_var = np.array(neigh_conn_var)
    np.save(f"{exp_dir}/embeddings/neigh_conn_var_{dataset}_{n_neighbors}.npy", neigh_conn_var)