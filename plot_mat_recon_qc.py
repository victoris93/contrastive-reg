import numpy as np
import torch
import sys
import matplotlib.pyplot as plt
import xarray as xr
from nilearn import plotting
from ContModeling.models import PhenoProj
from omegaconf import DictConfig, OmegaConf
from hydra import initialize, compose


mat_qc_dir = sys.argv[1]
data_path = sys.argv[2]
idx = int(sys.argv[3])


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
with initialize(version_base=None, config_path="."):
    cfg = compose(config_name='main_model_config.yaml')
    print(OmegaConf.to_yaml(cfg))

dataset = xr.open_dataset(data_path)

# MODEL DIMS
input_dim_feat = cfg.input_dim_feat
input_dim_target = cfg.input_dim_target
hidden_dim = cfg.hidden_dim
output_dim_target = cfg.output_dim_target
output_dim_feat = cfg.output_dim_feat

results_path = cfg.output_dir

# TRAINING PARAMS
lr = cfg.lr

dropout_rate = cfg.dropout_rate
weight_decay = cfg.weight_decay

model = PhenoProj(
    input_dim_feat,
    input_dim_target,
    hidden_dim,
    output_dim_target,
    output_dim_feat,
    dropout_rate,
    cfg
).to(device)

state_dict = torch.load(f"{results_path}/{cfg.experiment_name}/saved_models/model_weights_run0.pth", map_location=device)

true_mat = dataset.matrices.isel(subject = idx).values
true_mat = torch.Tensor(true_mat).to(torch.float32).to(device)

model.eval()
enc_mat = model.matrix_ae.encode_feat(true_mat)
dec_mat = model.matrix_ae.decode_feat(enc_mat)

mape = torch.abs(true_mat - dec_mat)/true_mat * 100
resid = true_mat - dec_mat

true_mat = true_mat.cpu().detach().numpy()
mape = mape.cpu().detach().numpy()
resid = resid.cpu().detach().numpy()


idx = np.where(test_idx == mat_idx)[0][0]
recon = recon_mat[idx]
mape = mape_mat[idx]
true = dataset.matrices.isel(subject = mat_idx).values

# plotting

fig, axes = plt.subplots(1, 4, figsize=(36, 7))

plotting.plot_matrix(true_mat,
axes = axes[0],
vmax = 1.,
vmin = -1.,
)

plotting.plot_matrix(recon,
axes = axes[1],
vmax = 1.,
vmin = -1.,        
)

plotting.plot_matrix(resid,
axes = axes[2],
vmax = 1.,
vmin = -1.,   
)

plotting.plot_matrix(mape,
axes = axes[3],
vmax = 100, vmin=0,
cmap='viridis'
)
plt.savefig(f"{mat_qc_dir}/individual_recon_idx{idx}.png", format = "png", dpi = 300, bbox_inches='tight')

