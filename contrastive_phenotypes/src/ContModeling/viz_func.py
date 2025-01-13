import pandas as pd
import nilearn as nl
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from nilearn.connectome import vec_to_sym_matrix, sym_matrix_to_vec
from nilearn import plotting
import nilearn as nl
import seaborn as sns
import numpy as np
import torch
import xarray as xr
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset
from PIL import Image
from scipy.linalg import issymmetric
import os
import re
from tqdm import tqdm


def replace_with_network(label, network_labels):
    for network in network_labels:
        if network in label:
            return network
    return label

atlas_labels = nl.datasets.fetch_atlas_schaefer_2018()['labels']
atlas_labels = [label.decode('utf-8') for label in atlas_labels]
NETWORK_LABELS = ['Vis', 'SomMot', 'DorsAttn', 'SalVentAttn', 'Limbic', 'Cont', 'Default']
NETWORK_LABELS = [replace_with_network(label, NETWORK_LABELS) for label in atlas_labels]

def plot_cog(data, cog_score, title):
    plt.figure(figsize=(10, 6))
    ax = sns.violinplot(data=data, x="train_ratio", y=cog_score, hue="dataset", hue_order=['train', 'test'], width = 0.8, scale = 'count', split = True) #, width = 2, gap = 0.01

    for patch in ax.collections:
        patch.set_alpha(0.4)

    sns.pointplot(x='train_ratio', y=cog_score, hue='dataset', data=data.groupby(['train_ratio', 'dataset'], as_index=False)[cog_score].median(), ax=ax, hue_order=['train', 'test'], markers="_")
    # ax.set_yticks(np.arange(0, 50, 5))
    #set x axis limit to 100
    # ax.set_ylim(-5, 50)
    #plt.axhline(10, c='r')
    #plt.axhline(5, c='g', linestyle='--')
    plt.ylabel("MAPE")
    # plt.axhline(0, c='k')
    # plt.suptitle(f"Training set ratio 20%, 20 experiments per size, thresh =  {threshold}%, FlippedEdge Aug")
    plt.suptitle(title)

    plt.grid()


def replace_with_network(label, network_labels):
    for network in network_labels:
        if network in label:
            return network
    return label


def mape_cog(csv, cog_score):
    file = pd.read_csv(csv)
    file = file[["train_ratio", "experiment", "dataset", cog_score]]
    file[cog_score]= file[cog_score]#*100
    return file


def plot_loss(csv, title):
    loss_j = pd.read_csv(csv)
    plt.figure(figsize=(10, 6))
    sns.pointplot(x='train_ratio', y='loss', data=loss_j.groupby(['train_ratio'], as_index=False)['loss'].median(), markers="_", label = "loss")
    sns.pointplot(x='train_ratio', y='target_decoding', data=loss_j.groupby(['train_ratio'], as_index=False)['target_decoding'].median(), markers="_", label='target decoding')
    sns.pointplot(x='train_ratio', y='kernel_feature', data=loss_j.groupby(['train_ratio'], as_index=False)['kernel_feature'].median(), markers="_", label = "kernel_feature")
    sns.pointplot(x='train_ratio', y='kernel_target', data=loss_j.groupby(['train_ratio'], as_index=False)['kernel_target'].median(), markers="_", label = "kernel_target")
    sns.pointplot(x='train_ratio', y='joint_embedding', data=loss_j.groupby(['train_ratio'], as_index=False)['joint_embedding'].median(), markers="_", label = "joint_embedding")
    #sns.pointplot(x='train_ratio', y='feature_decoding', data=loss_j.groupby(['train_ratio'], as_index=False)['feature_decoding'].median(), markers="_", label = "feature_decoding")

    plt.grid()
    plt.legend(title=title)
    plt.show()


def combine_images(image_paths, save_to):

    images = [Image.open(image_path) for image_path in image_paths]

    total_width = sum(image.width for image in images)
    max_height = max(image.height for image in images)

    combined_image = Image.new("RGB", (total_width, max_height))

    x_offset = 0
    for image in images:
        combined_image.paste(image, (x_offset, 0))
        x_offset += image.width

    combined_image.save(save_to)


def compute_batch_elementwise_correlation(true, recon):
    batch_size, rows, cols = true.shape
    correlations = np.zeros((rows, cols))

    flat_true = true.reshape(batch_size, -1)
    flat_recon = recon.reshape(batch_size, -1)
    
    for i in range(rows * cols):
        correlations[i // cols, i % cols] = spearmanr(flat_true[:, i], flat_recon[:, i])[0]
        
    np.fill_diagonal(correlations, 1.0)


    return correlations


def load_recon_mats(exp_name, work_dir, vectorize, is_full_model=False, reduced_mat=False, run_num=None):
    recon_path_suffix = ""
    if is_full_model:
        recon_path_suffix = f"_run{run_num}"
    mat_type_suffix = "recon_"
    mat_type_suffix = mat_type_suffix + "reduced_mat" if reduced_mat else mat_type_suffix + "mat"
    
    exp_dir = f"{work_dir}/results/{exp_name}"
    recon_mat_dir = f"{exp_dir}/recon_mat"
    recon_mat_files = [file for file in os.listdir(recon_mat_dir) if mat_type_suffix in file and recon_path_suffix in file]
    recon_mat_files = sorted(recon_mat_files, key=lambda x: int(re.search(r'batch_(\d+)', x)[1]))
    recon_paths = [os.path.join(recon_mat_dir, file) for file in recon_mat_files]
    recon_mat = np.concatenate([np.load(path) for path in recon_paths])
    
    for i in range(recon_mat.shape[0]):
        np.fill_diagonal(recon_mat[i], 1.0)
    
    if vectorize:
        recon_mat = sym_matrix_to_vec(recon_mat, discard_diagonal = True)
    return recon_mat

def load_true_mats(data_path, exp_name, work_dir, vectorize=False, is_full_model=False, run_num=None):
    recon_path_suffix = ""
    if is_full_model:
        recon_path_suffix = f"_run{run_num}"
    mat_type_suffix = "recon_"

    test_idx_path = f"{work_dir}/results/{exp_name}/test_idx{recon_path_suffix}.npy"
    test_idx = np.load(test_idx_path)
    dataset = xr.open_dataset(data_path)
    true_mat = dataset.matrices.isel(subject = test_idx).values
    if vectorize:
        true_mat = sym_matrix_to_vec(true_mat, discard_diagonal = True)
    return true_mat

def load_mape(exp_name, work_dir, is_full_model = False, run_num = None):
    recon_path_suffix = ""
    if is_full_model:
        recon_path_suffix = f"_run{run_num}"
    exp_dir = f"{work_dir}/results/{exp_name}"
    recon_mat_dir = f"{exp_dir}/recon_mat"
    mape_mat_files = sorted([file for file in os.listdir(recon_mat_dir) if f"mape_mat{recon_path_suffix}" in file])
    mape_mat_files = sorted(mape_mat_files, key=lambda x: int(re.search(r'batch_(\d+)', x)[1]))
    mape_paths = [os.path.join(recon_mat_dir, file) for file in mape_mat_files]
    mape_mat = np.concatenate([np.load(path) for path in mape_paths])
    return mape_mat

def get_corr_data(exp_name, recon_mat, true_mat, network_labels = NETWORK_LABELS):

    corr_mat_pred = compute_batch_elementwise_correlation(true_mat, recon_mat)

    corr_data = {
    'correlation': [],
    'network': [],
    'experiment': exp_name
    }
    for i, network in enumerate(network_labels):
        corr_data['correlation'].extend(corr_mat_pred[i])
        corr_data['network'].extend([network]*corr_mat_pred.shape[1])
        
    corr_data = pd.DataFrame(corr_data)
    return corr_data

def compile_test_corrs(mats_exp1, mats_exp2): # give tuples (true, recon)
    corr_data_1 = get_corr_data(mats_exp1[0], mats_exp1[1])
    corr_data_2 = get_corr_data(mats_exp2[0], mats_exp2[1])
    corr_data = pd.concat([corr_data_1, corr_data_2])
    return corr_data

def wandb_plot_test_recon_corr(wandb, exp_name, work_dir, recon_mat, true_mat, mape_mat, is_full_model = False, run_num = None):
    suffix =''
    if is_full_model:
        suffix = f"_run{run_num}"

    fig_path = f"{work_dir}/results/figures/test_recon_corr_{exp_name}{suffix}.png"

    corr_mat_pred = compute_batch_elementwise_correlation(true_mat, recon_mat)
    
    mean_corr = corr_mat_pred.mean()
    median_mape = np.median(mape_mat)
    
    np.fill_diagonal(corr_mat_pred, 1.0)

    fig = plotting.plot_matrix(corr_mat_pred,
    title=f"Corr(True, Recon) | Exp {exp_name}",
                     grid = False,
                     vmax = 1.,
                     vmin = -1.
    )

    plt.text(-12, 0.02, f'mean_corr = {mean_corr:.2f}', color='black', ha='right', va='bottom', fontsize=12, transform=plt.gca().transAxes,
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    plt.text(-10.5, 0.09, f'median_mape = {median_mape:.2f}', color='black', ha='right', va='bottom', fontsize=12, transform=plt.gca().transAxes,
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()
    wandb.log({f"Corr(True, Recon) | All Test | {exp_name}": wandb.Image(fig_path)})


def wandb_plot_individual_recon(wandb, exp_name, work_dir, test_idx, recon_mat, true_mat, mape_mat, sub_idx, is_full_model = False, run_num = None):
    suffix =''
    if is_full_model:
        suffix = f"_run{run_num}"

    sub_idx_in_test = test_idx[sub_idx]
    recon = recon_mat[sub_idx]
    mape = np.abs(mape_mat[sub_idx])
    true = true_mat[sub_idx]
    residual = true - recon

    fig, axes = plt.subplots(1, 4, figsize=(36, 7))

    plotting.plot_matrix(true,
    axes = axes[0],
    title=f"True Mat | Exp {exp_name} idx {sub_idx_in_test}",
    vmax = 1.,
    vmin = -1.
    )

    plotting.plot_matrix(recon,
    axes = axes[1],
    title=f"Recon Mat | Exp {exp_name} idx {sub_idx_in_test}",
    vmax = 1.,
    vmin = -1.
    )

    plotting.plot_matrix(residual,
    axes = axes[2],
    title=f"Risiduals | Exp {exp_name} idx {sub_idx_in_test}",
    vmax = 1.,
    vmin = -1.
    )

    plotting.plot_matrix(mape,
    axes = axes[3],
    title=f"MAPE | Exp {exp_name} idx {sub_idx_in_test}",
    vmax = 100, vmin=0
    )
    plt.tight_layout()

    fig_path = f"{work_dir}/results/figures/individual_recon_{exp_name}{suffix}_idx_{sub_idx_in_test}.png"
    plt.savefig(fig_path)
    plt.close(fig)
    wandb.log({f"Individual Reconstructions | {exp_name}": wandb.Image(fig_path)})


def wandb_plot_acc_vs_baseline(wandb, exp_name, work_dir, data_path, baseline_exp_name):
    
    fig_path = f"{work_dir}/results/figures/corr_violinplot_{exp_name}_vs_{baseline_exp_name}.png"
    
    corr_data = compile_test_corrs(data_path, work_dir, exp_name, baseline_exp_name)
    
    sns.set(style="whitegrid")
    fig = plt.figure(figsize=(15, 8))
    sns.violinplot(data=corr_data, x="network", y="correlation", hue="experiment", split=True, inner="quart", width = 1., dodge = True, palette = 'hls')
    plt.legend(loc='lower right')
    plt.ylim(0.4, 1)
    
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close(fig)

    wandb.log({"Correlation Distributions": wandb.Image(fig_path)})


