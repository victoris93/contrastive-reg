import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import xarray as xr
from tqdm import tqdm
import sys

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

vars = []
for idx in sub_idx:
    if idx != 0:
        mat = data.matrices.isel(index = idx).values
        var_shift = (ref_mat - mat) ** 2
        vars.append(var_shift)

vars = np.array(vars)
np.save(f"{exp_dir}/embeddings/conn_var_{dataset}.npy", vars)
print(f"Variance of connectivity matrices for {exp_name} is computed and saved.")
print(f"{exp_dir}/embeddings/conn_var_{dataset}.npy")