import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import xarray as xr
from tqdm import tqdm
import sys

root = "/gpfs3/well/margulies/users/cpy397/contrastive-learning"
run = 0
exp_name = sys.argv[1]
dataset = sys.argv[2]

exp_dir = f'{root}/results/{exp_name}'
predictions = pd.read_csv(f'{exp_dir}/pred_results.csv')
predictions = predictions[(predictions["dataset"] == dataset) & (predictions["train_ratio"] == 1)]
sub_idx = predictions.indices.values
embeddings = np.load(f"{exp_dir}/embeddings/joint_embeddings_run0_{dataset}.npy")
n_neighbors = int(sys.argv[3])

neigh = NearestNeighbors(n_neighbors=n_neighbors, radius=1, algorithm='brute', metric = 'cosine', p = 2).fit(embeddings)
distances, neighbors = neigh.kneighbors(embeddings, return_distance = True)
data =  xr.open_dataset(f'{root}/ABCD/abcd_dataset_400parcels_1.nc')
mean_mat = data.matrices.values.mean(axis = 0)

neigh_conn_var = []
neigh_conn_dvar_ddist = []

for neighborhood_idx, neighborhood in enumerate(tqdm(neighbors)):
    neigh_idx = sub_idx[neighborhood]
    matrices = data.matrices.isel(subject = neigh_idx).values
    var = np.var(matrices, mean = mean_mat, axis = 0)
    neigh_conn_var.append(var)
    dvar_ddist = []
    for mat_idx, mat in enumerate(matrices):
        sub_dvar_ddist = np.zeros_like(matrices[0])
        if mat_idx != 0:
            centroid_var = np.var(matrices[0][None, :], mean = mean_mat, axis = 0)
            neigh_var = np.var(mat[None, :], mean = mean_mat, axis = 0)
            sub_dvar_ddist = np.abs(centroid_var - neigh_var)/distances[neighborhood_idx][mat_idx]
        dvar_ddist.append(np.mean(sub_dvar_ddist))
    neigh_conn_dvar_ddist.append(dvar_ddist)
neigh_conn_var = np.array(neigh_conn_var)
neigh_conn_dvar_ddist = np.array(neigh_conn_dvar_ddist)

np.save(f"{exp_dir}/embeddings/neigh_conn_var_{dataset}.npy", neigh_conn_var)
np.save(f"{exp_dir}/embeddings/neigh_conn_dvar_ddist_mean_{dataset}.npy", neigh_conn_dvar_ddist)
