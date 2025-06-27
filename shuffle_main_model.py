import math
import wandb
import xarray as xr
import asyncio
import submitit
import pickle
import sys
from pathlib import Path
import gc
from collections import defaultdict
from nilearn.connectome import sym_matrix_to_vec, vec_to_sym_matrix
import numpy as np
import pandas as pd
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from scipy.stats import spearmanr
from sklearn.model_selection import (
    train_test_split,
    KFold
)
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset
from tqdm.auto import tqdm
# from augmentations import augs, aug_args
import glob, os, shutil
from nilearn.datasets import fetch_atlas_schaefer_2018
import random
from sklearn.preprocessing import MinMaxScaler

from ContModeling.utils import (
    gaussian_kernel,
    cauchy,
    rbf,
    standardize_target,
    save_embeddings,
    filter_nans
)
from ContModeling.losses import LogEuclideanLoss, NormLoss, KernelizedSupCon, OutlierRobustMSE
from ContModeling.models import PhenoProj
from ContModeling.helper_classes import MatData
from ContModeling.viz_func import (
    wandb_plot_acc_vs_baseline,
    wandb_plot_test_recon_corr,
    wandb_plot_individual_recon
)

torch.cuda.empty_cache()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EMB_LOSSES ={
    'Norm': NormLoss(),
    'LogEuclidean': LogEuclideanLoss(),
    'MSE': nn.functional.mse_loss,
    'MAE': nn.functional.l1_loss,
    'MSERobust': OutlierRobustMSE(),
    'Huber': nn.HuberLoss(),
    'cosine': nn.functional.cosine_embedding_loss,
}

SUPCON_KERNELS = {
    'cauchy': cauchy,
    'gaussian_kernel': gaussian_kernel,
    'rbf': rbf,
    'None': None
    }


class ModelRun(submitit.helpers.Checkpointable):
    def __init__(self):
        self.results = None
        self.embeddings = None

    def __call__(self, train, train_idx, val_idx, train_ratio, dataset, cfg, fold=None, random_state=None, device=None, save_model = True, path: Path = None):
        if self.results is None:
            if device is None:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                print(f"Device {device}, ratio {train_ratio}", flush=True)
            if not isinstance(random_state, np.random.RandomState):
                seed = random_state
                random_state = np.random.RandomState(seed)
            
            recon_mat_dir = os.path.join(cfg.output_dir, cfg.experiment_name, cfg.reconstructed_dir)
            os.makedirs(recon_mat_dir, exist_ok=True)
    
            predictions = {}
            losses = []
            if fold is not None:
                self.run_type = "fold"
                self.run_id = fold
                test_type='validation'
            else:
                self.run_type = "seed"
                self.run_id = seed
                test_type='test'

            print(f"Run Type {self.run_type}, Run ID {self.run_id}")
            
            self.embeddings = {'train': [], test_type: []}

            if cfg.full_model_pretrained:
                print(f"Loading val indices from a pretrain exp {cfg.full_model_pretrained_exp}, {self.run_type} {self.run_id}, train ratio {train_ratio}...")
                train_idx = np.load(f"{cfg.output_dir}/{cfg.full_model_pretrained_exp}/train_idx_{self.run_type}{self.run_id}_train_ratio{train_ratio}.npy")
                val_idx = np.load(f"{cfg.output_dir}/{cfg.full_model_pretrained_exp}/{test_type}_idx_{self.run_type}{self.run_id}_train_ratio{train_ratio}.npy")
            elif cfg.mat_ae_pretrained:
                print(f"Loading val indices from a pretrain exp {cfg.mat_ae_pretrained_exp}, {self.run_type} {self.run_id}, train ratio {train_ratio}...")
                train_idx = np.load(f"{cfg.output_dir}/{cfg.mat_ae_pretrained_exp}/train_idx_{self.run_type}{self.run_id}_train_ratio{train_ratio}.npy")
                val_idx = np.load(f"{cfg.output_dir}/{cfg.mat_ae_pretrained_exp}/{test_type}_idx_{self.run_type}{self.run_id}_train_ratio{train_ratio}.npy")

            train_dataset = Subset(dataset, train_idx)
            val_dataset = Subset(dataset, val_idx)
            
            train_features = train_dataset.dataset.matrices[train_dataset.indices]

            pca_obj = train_dataset.compute_pca()
            val_dataset.compute_pca(pca_obj)

            if cfg.standardize_target:
                target_names = dataset.target_names
                train_targets = train_dataset.dataset.targets[train_dataset.indices]
                for i, target_name in enumerate(target_names):
                    print(f"Standardizing target {target_name} (min: {np.nanmin(train_targets[:, i])}, max: {np.nanmax(train_targets[:, i])}) to [0, 1] in train")
                    train_targets[:, i] = standardize_target(train_targets[:, i])
                
                test_targets = val_dataset.dataset.targets[val_dataset.indices]
                for i, target_name in enumerate(target_names):
                    print(f"Standardizing target {target_name} (min: {np.nanmin(test_targets[:, i])}, max: {np.nanmax(test_targets[:, i])}) to [0, 1] in test")
                    test_targets[:, i] = standardize_target(test_targets[:, i])

                train_dataset.dataset.targets[train_dataset.indices] = train_targets

            input_dim_feat =cfg.input_dim_feat
            output_dim_feat = cfg.output_dim_feat

            ## Weight initialization for bilinear layer
            mean_f = torch.mean(train_features, dim=0).to(device)
            
            [D,V] = torch.linalg.eigh(mean_f,UPLO = "U")
            B_init_fMRI = V[:,input_dim_feat-output_dim_feat:]

            loss_terms, model = train(train_ratio,
                                      train_dataset,
                                      val_dataset,
                                      B_init_fMRI,
                                      self.run_type,
                                      self.run_id,
                                      cfg,
                                      device=device)
            
            loss_terms["train_ratio"] = train_ratio
            loss_terms[self.run_type] = self.run_id
            losses.append(loss_terms)
            
            wandb.init(project=cfg.project,
                mode = "offline",
                name=f"{test_type}_{cfg.experiment_name}_{self.run_type}{self.run_id}_train_ratio_{train_ratio}",
                dir = cfg.output_dir,
                config = OmegaConf.to_container(cfg, resolve=True))
            
            embedding_dir = os.path.join(cfg.output_dir, cfg.experiment_name, cfg.embedding_dir)
            os.makedirs(embedding_dir, exist_ok=True)

            model.eval()
            with torch.no_grad():
                for label, d, d_indices in (('train', train_dataset, train_idx), (test_type, val_dataset, val_idx)):

                    is_test = True
                    if label == 'train':
                        is_test = False
                    
                    X, X_pca, y, _, _ = zip(*d)
                    X = torch.stack(X)
                    X_pca = torch.stack(X_pca)
                    y = torch.stack(y)
                    X, y, d_indices, _ = filter_nans(X, y, d_indices)
                    X = X.to(device)
                    X_pca = X_pca.to(device)
                    y = y.to(device)

                    X_embedded = model.encode_features(X)
                    X_embedded = X_embedded.cpu().numpy()
                    X_embedded = torch.tensor(sym_matrix_to_vec(X_embedded)).to(torch.float32).to(device)
                    X_emb_reduced, X_emb_reduced_norm = model.encode_reduced_mat(X_embedded)
                    X_emb_reduced_norm = X_emb_reduced_norm + nn.functional.normalize(X_pca, p=2, dim=1)
                    
                    np.save(f"{cfg.output_dir}/{cfg.experiment_name}/{label}_idx_{self.run_type}{self.run_id}_train_ratio{train_ratio}.npy", d_indices)

                    if is_test:
                        inv_feat_embedding = model.decode_reduced_mat(X_emb_reduced).detach().cpu().numpy()
                        inv_feat_embedding = vec_to_sym_matrix(inv_feat_embedding)
                        inv_feat_embedding = torch.tensor(inv_feat_embedding).to(torch.float32).to(device)
                        recon_mat = model.decode_features(inv_feat_embedding)
                        mape_mat = torch.abs((X - recon_mat) / (X + 1e-10)) * 100
                        
                        wandb_plot_test_recon_corr(wandb,
                                                   cfg.experiment_name,
                                                   cfg.work_dir,
                                                   recon_mat.cpu().numpy(),
                                                   X.cpu().numpy(), 
                                                   mape_mat.cpu().numpy(),
                                                   self.run_type,
                                                   self.run_id
                                                  )
                        wandb_plot_individual_recon(wandb,
                                                    cfg.experiment_name,
                                                    cfg.work_dir,
                                                    d_indices,
                                                    recon_mat.cpu().numpy(),
                                                    X.cpu().numpy(),
                                                    mape_mat.cpu().numpy(),
                                                    0,
                                                    self.run_type,
                                                    self.run_id
                                                   )

                        np.save(f'{recon_mat_dir}/recon_mat_{self.run_type}{self.run_id}_train_ratio{train_ratio}', recon_mat.cpu().numpy())
                        np.save(f'{recon_mat_dir}/mape_mat_{self.run_type}{self.run_id}_train_ratio{train_ratio}', mape_mat.cpu().numpy())

                    y_pred = model.decode_targets(X_emb_reduced_norm)

                    save_embeddings(X_embedded, "mat", cfg, label, self.run_type, self.run_id, train_ratio=train_ratio)
                    save_embeddings(X_emb_reduced_norm, "joint", cfg, label, self.run_type, self.run_id, train_ratio=train_ratio)

                    if is_test:
                        epsilon = 1e-8
                        mape =  100 * torch.mean(torch.abs((y - y_pred)) / torch.abs((y + epsilon))).item()
                        corr =  spearmanr(y.cpu().numpy().flatten(), y_pred.cpu().numpy().flatten())[0]

                        wandb.log({
                            self.run_type: self.run_id,
                            "Train ratio": train_ratio,
                            'Val | Target MAPE/val' : mape,
                            'Val | Target Corr/val': corr,
                            'Val | Train ratio' : train_ratio
                            })
            
                    predictions[(train_ratio, self.run_id, label)] = (y.cpu().numpy(), y_pred.cpu().numpy(), d_indices)
                    for i, idx in enumerate(d_indices):
                        self.embeddings[label].append({
                            'index': idx,
                            'joint_embedding': X_emb_reduced[i].cpu().numpy()
                        })
            wandb.finish()
            
            self.results = (losses, predictions, self.embeddings)

        if save_model:
            saved_models_dir = os.path.join(cfg.output_dir, cfg.experiment_name, cfg.model_weight_dir)
            os.makedirs(saved_models_dir, exist_ok=True)
            torch.save(model.state_dict(), f"{saved_models_dir}/model_weights_{self.run_type}{self.run_id}_train_ratio{train_ratio}.pth")

        return self.results
        
    def checkpoint(self, *args, **kwargs):
        print("Checkpointing", flush=True)
        return super().checkpoint(*args, **kwargs)

    def checkpoint(self, *args, **kwargs):
        print("Checkpointing", flush=True)
        return super().checkpoint(*args, **kwargs)

def train(train_ratio,
          train_dataset,
          val_dataset,
          B_init_fMRI,
          run_type,
          run_id,
          cfg, model=None, device=device):
    print(f"Start training...{run_type} {run_id}")

    # MODEL DIMS
    input_dim_feat = cfg.input_dim_feat
    input_dim_target = cfg.input_dim_target
    hidden_dim = cfg.hidden_dim
    output_dim_target = cfg.output_dim_target
    output_dim_feat = cfg.output_dim_feat
    kernel = SUPCON_KERNELS[cfg.SupCon_kernel]
    
    # TRAINING PARAMS
    lr = cfg.lr
    batch_size = cfg.batch_size
    dropout_rate = cfg.dropout_rate
    weight_decay = cfg.weight_decay
    num_epochs = cfg.num_epochs

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    if model is None:
        model = PhenoProj(
            input_dim_feat,
            input_dim_target,
            hidden_dim,
            output_dim_target,
            output_dim_feat,
            dropout_rate,
            cfg
        ).to(device)

    if cfg.full_model_pretrained:
        print(f"Loading pretrained FULL model, {run_type} {run_id}, train ratio {train_ratio}...")
        state_dict = torch.load(f"{cfg.output_dir}/{cfg.full_model_pretrained_exp}/saved_models/model_weights_{run_type}{run_id}_train_ratio{train_ratio}.pth")
        model.load_state_dict(state_dict)
    elif cfg.mat_ae_pretrained:
        print(f"Loading pretrained MatrixAutoencoder, {run_type} {run_id}, train ratio {train_ratio}...")
        state_dict = torch.load(f"{cfg.output_dir}/{cfg.mat_ae_pretrained_exp}/saved_models/autoencoder_weights_{run_type}{run_id}_train_ratio{train_ratio}.pth")
        model.matrix_ae.load_state_dict(state_dict)
    else:
        model.matrix_ae.enc_mat1.weight = torch.nn.Parameter(B_init_fMRI.transpose(0,1))
        model.matrix_ae.enc_mat2.weight = torch.nn.Parameter(B_init_fMRI.transpose(0,1))   
        
    if cfg.mat_ae_enc_freeze:
        print("Freezing weights for mat encoding...")
        for param in model.matrix_ae.enc_mat1.parameters():
            param.requires_grad = False
        for param in model.matrix_ae.enc_mat2.parameters():
            param.requires_grad = False

    if cfg.mat_ae_dec_freeze:
        print("Freezing weights for mat decoding...")
        for param in model.matrix_ae.dec_mat1.parameters():
            param.requires_grad = False
        for param in model.matrix_ae.dec_mat2.parameters():
            param.requires_grad = False
    
    if cfg.reduced_mat_ae_enc_freeze:
        print("Freezing weights for reduced mat encoding...")
        for param in model.reduced_matrix_ae.reduced_mat_to_embed.parameters():
            param.requires_grad = False

    if cfg.reduced_mat_ae_dec_freeze:
        print("Freezing weights for reduced mat decoding...")
        for param in model.reduced_matrix_ae.embed_to_reduced_mat.parameters():
            param.requires_grad = False
    
    if cfg.target_dec_freeze:
        print("Freezing TargetDecoder...")
        for param in model.target_dec.parameters():
            param.requires_grad = False

    criterion_pft = KernelizedSupCon(
        method="yaware",
        temperature=cfg.pft_temperature,
        base_temperature= cfg.pft_base_temperature,
        reg_term = cfg.pft_reg_term,
        kernel=kernel,
        krnl_sigma=cfg.pft_sigma,
    )

    criterion_ptt = KernelizedSupCon(
        method="yaware",
        temperature=cfg.ptt_temperature,
        base_temperature= cfg.ptt_base_temperature,
        reg_term = cfg.ptt_reg_term,
        kernel=kernel,
        krnl_sigma=cfg.ptt_sigma,
    )
    feature_autoencoder_crit = EMB_LOSSES[cfg.feature_autoencoder_crit]
    target_decoding_crit = EMB_LOSSES[cfg.target_decoding_crit]

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr, weight_decay=weight_decay,
    #                             momentum=0.9)
    scheduler = StepLR(optimizer, step_size=40, gamma=0.1)
    loss_terms = []
    validation = []

    gc.collect()

    wandb.init(project=cfg.project,
        mode = "offline",
        name=f"{cfg.experiment_name}_{run_type}{run_id}_train_ratio_{train_ratio}",
        dir = cfg.output_dir,
        config = OmegaConf.to_container(cfg, resolve=True))
    
    with tqdm(range(num_epochs), desc="Epochs", leave=False) as pbar:
        for epoch in pbar:
            model.train()

            if cfg.reduced_mat_ae_pretrained:
                model.reduced_matrix_ae.eval()
            if cfg.reduced_mat_ae_enc_freeze:
                model.reduced_matrix_ae.reduced_mat_to_embed.eval()
            if cfg.reduced_mat_ae_dec_freeze:
                model.reduced_matrix_ae.embed_to_reduced_mat.eval()
            if cfg.target_dec_freeze:
                model.target_dec.eval()
                
            loss_terms_batch = defaultdict(lambda:0)

            for features, pca_features, targets, inter_network_conn, _ in train_loader:
                loss = 0
                
                optimizer.zero_grad()

                features, targets, _, inter_network_conn = filter_nans(features, targets, _z=inter_network_conn)

                features = features.to(device)
                targets = targets.to(device)
                inter_network_conn = inter_network_conn.to(device)

                ## FEATURE ENCODING == MATRIX REDUCTION
                embedded_feat = model.encode_features(features)
                
                ## VECTORIZE REDUCED MATRIX
                embedded_feat_vectorized = sym_matrix_to_vec(embedded_feat.detach().cpu().numpy())
                embedded_feat_vectorized = torch.tensor(embedded_feat_vectorized).to(torch.float32).to(device)

                ## EMBEDDING OF THE REDUCED MATRIX
                reduced_mat_embedding, reduced_mat_embedding_norm = model.encode_reduced_mat(embedded_feat_vectorized)
                ## ADD PCA
                reduced_mat_embedding_norm = reduced_mat_embedding_norm + nn.functional.normalize(pca_features, p=2, dim=1)
                out_target_decoded = model.decode_targets(reduced_mat_embedding_norm)

                ## RECONSTRUCT REDUCED MATRIX FROM EMBEDDING AND THE FULL MATRIX FROM REDUCED
                recon_reduced_mat = model.decode_reduced_mat(reduced_mat_embedding)

                if not cfg.reduced_mat_ae_dec_freeze:
                    reduced_mat_recon_loss = feature_autoencoder_crit(embedded_feat_vectorized, recon_reduced_mat) / 1000
                    loss += reduced_mat_recon_loss

                recon_reduced_mat = vec_to_sym_matrix(recon_reduced_mat.detach().cpu().numpy())
                recon_reduced_mat = torch.tensor(recon_reduced_mat).to(torch.float32).to(device)

                reconstructed_feat = model.decode_features(recon_reduced_mat)

                ## LOSS: TARGET DECODING FROM TARGET EMBEDDING
                if cfg.target_decoding_crit == 'Huber' and cfg.huber_delta != 'None':
                    target_decoding_crit = nn.HuberLoss(delta = cfg.huber_delta)

                if not cfg.reduced_mat_ae_enc_freeze:
                    ## KERNLIZED LOSS: MAT embedding vs targets
                    kernel_embedded_target_loss, _ = criterion_ptt(reduced_mat_embedding_norm.unsqueeze(1), targets)
                    loss += kernel_embedded_target_loss
                    
                    if cfg.network_loss:
                        kernel_embedded_network_loss, _ = criterion_pft(reduced_mat_embedding_norm.unsqueeze(1), inter_network_conn)
                        loss += kernel_embedded_network_loss

                if not cfg.mat_ae_enc_freeze or not cfg.mat_ae_dec_freeze:
                    feature_autoencoder_loss = feature_autoencoder_crit(features, reconstructed_feat) / 1000
                    loss += feature_autoencoder_loss

                if not cfg.target_dec_freeze:
                    target_decoding_from_reduced_emb_loss = target_decoding_crit(targets, out_target_decoded)
                    loss += target_decoding_from_reduced_emb_loss

                loss.backward()

                if cfg.clip_grad:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                if cfg.log_gradients:
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            wandb.log({
                                "Epoch": epoch,
                                run_type: run_id,
                                "Train ratio": train_ratio,
                                f"Gradient Norm/{name}": param.grad.norm().item()
                            })  

                optimizer.step()

                loss_terms_batch['loss'] = loss.item() / len(features)

                if not cfg.reduced_mat_ae_enc_freeze:
                    loss_terms_batch['kernel_embedded_target_loss'] = kernel_embedded_target_loss.item() / len(features)
                    if cfg.network_loss:
                        loss_terms_batch['kernel_embedded_network_loss'] = kernel_embedded_network_loss.item() / len(features)

                    wandb.log({
                        'Epoch': epoch,
                        run_type: run_id,
                        "Train ratio": train_ratio,
                        'kernel_embedded_target_loss': loss_terms_batch['kernel_embedded_target_loss'],
                    })

                    if cfg.network_loss:
                        wandb.log({
                            'Epoch': epoch,
                            run_type: run_id,
                            "Train ratio": train_ratio,
                            'kernel_embedded_network_loss': loss_terms_batch['kernel_embedded_network_loss'],
                        })

                if not cfg.reduced_mat_ae_dec_freeze:
                    loss_terms_batch['reduced_mat_recon_loss'] = reduced_mat_recon_loss.item() / len(features)
                    wandb.log({
                        'Epoch': epoch,
                        run_type: run_id,
                        "Train ratio": train_ratio,
                        'reduced_mat_recon_loss': loss_terms_batch['reduced_mat_recon_loss'],
                    })
                
                if not cfg.target_dec_freeze:
                    loss_terms_batch['target_decoding_loss'] = target_decoding_from_reduced_emb_loss.item() / len(features)
                    wandb.log({
                        'Epoch': epoch,
                        run_type: run_id,
                        "Train ratio": train_ratio,
                        'target_decoding_loss': loss_terms_batch['target_decoding_loss'],
                    })

                # loss_terms_batch['direction_reg_target_loss'] = direction_reg_target.item() / len(features)
                
                if not cfg.mat_ae_enc_freeze or not cfg.mat_ae_dec_freeze:
                    loss_terms_batch['feature_autoencoder_loss'] = feature_autoencoder_loss.item() / len(features)
                    wandb.log({
                        'Epoch': epoch,
                        run_type: run_id,
                        "Train ratio": train_ratio,
                        'feature_autoencoder_loss': loss_terms_batch['feature_autoencoder_loss'],
                    })
                
                wandb.log({
                    'Epoch': epoch,
                    run_type: run_id,
                    "Train ratio": train_ratio,
                    'total_loss': loss_terms_batch['loss'],
                })

            loss_terms_batch['epoch'] = epoch
            loss_terms.append(loss_terms_batch)

            model.eval()
            mape_batch = 0
            corr_batch = 0
            with torch.no_grad():
                for features, pca_features, targets, _, _ in val_loader:

                    features, targets, _, _ = filter_nans(features, targets)
                    
                    features, targets = features.to(device), targets.to(device)                    
                    reduced_mat = model.encode_features(features)
                    
                    reduced_mat = torch.tensor(sym_matrix_to_vec(reduced_mat.detach().cpu().numpy())).to(torch.float32).to(device)
                    embedding, embedding_norm = model.encode_reduced_mat(reduced_mat)
                    embedding_norm = embedding_norm + nn.functional.normalize(pca_features, p=2, dim=1)
                    out_target_decoded = model.decode_targets(embedding_norm)
                    
                    epsilon = 1e-8

                    mape =  torch.mean(torch.abs((targets - out_target_decoded)) / torch.abs((targets + epsilon))) * 100
                    if torch.isnan(mape):
                        mape = torch.tensor(0.0)
                    
                    corr =  spearmanr(targets.cpu().numpy().flatten(), out_target_decoded.cpu().numpy().flatten())[0]
                    if np.isnan(corr):
                        corr = 0.0
                        
                    mape_batch += mape.item()
                    corr_batch += corr

                mape_batch = mape_batch/len(val_loader)
                corr_batch = corr_batch/len(val_loader)
                validation.append(mape_batch)

            wandb.log({
                run_type: run_id,
                "Train ratio": train_ratio,
                'Target MAPE/val' : mape_batch,
                'Target Corr/val': corr_batch,
                })
            
            scheduler.step()
            if np.log10(scheduler._last_lr[0]) < -7:
                break

            pbar.set_postfix_str(
                f"Epoch {epoch} "
                f"| Loss {loss_terms[-1]['loss']:.02f} "
                f"| val Target MAPE {mape_batch:.02f}"
                f"| val Target Corr {corr_batch:.02f} "
                f"| log10 lr {np.log10(scheduler._last_lr[0])}"
            )
    wandb.finish()
    loss_terms = pd.DataFrame(loss_terms)
    return loss_terms, model

@hydra.main(config_path=".", config_name="shuffle_main_model_config")
def main(cfg: DictConfig):

    print("Experiment: ", cfg.experiment_name)
    print("Dataset: ", cfg.dataset.name)
    results_dir = os.path.join(cfg.output_dir, cfg.experiment_name)
    os.makedirs(results_dir, exist_ok=True)

    dataset_path = cfg.dataset.dataset_path

    if isinstance(cfg.targets, str):
        targets =[cfg.targets]
    else:
        targets = list(cfg.targets)
        
    dataset = MatData(dataset_path, targets, synth_exp = cfg.synth_exp, threshold=cfg.mat_threshold)
    n_sub = len(dataset)
    indices = np.arange(n_sub)

    if cfg.dataset.name == "hcp":
        indices = indices[indices!=249]
    elif cfg.dataset.name == "abcd":
        indices = indices[indices!=863]
        
    multi_gpu = cfg.multi_gpu
    train_ratios = list(cfg.train_ratio)
    
    if multi_gpu:
        print("Using multi-gpu")
        log_folder = Path("logs")
        executor = submitit.AutoExecutor(folder=str(log_folder / "%j"))
        executor.update_parameters(
            timeout_min=120,
            slurm_partition="gpu_short",
            gpus_per_node=1,
            tasks_per_node=1,
            nodes=1
        )
        run_jobs = []

        with executor.batch():
            for seed in tqdm(list(cfg.random_seed), desc="Random Seed"):
                random_state = np.random.RandomState(seed)
                train_idx, test_idx = train_test_split(indices, test_size=cfg.test_ratio, random_state=random_state)
                
                for train_ratio in tqdm(train_ratios, desc="Training Size"):

                    train_size = int(len(train_idx) * train_ratio)
                    train_idx = random_state.choice(train_idx, train_size, replace=False)
                    
                    np.save(f"{results_dir}/train_idx_seed{seed}_train_ratio{train_ratio}.npy", train_idx)
                    np.save(f"{results_dir}/test_idx_seed{seed}_train_ratio{train_ratio}.npy", test_idx)
                    
                    run_model = ModelRun()
                    job = executor.submit(run_model,
                                          train,
                                          train_idx,
                                          test_idx,
                                          train_ratio,
                                          dataset,
                                          cfg,
                                          random_state=seed,
                                          device=None)
                    run_jobs.append(job)

        async def get_result(run_jobs):
            run_results = []
            for aws in tqdm(asyncio.as_completed([j.awaitable().result() for j in run_jobs]), total=len(run_jobs)):
                res = await aws
                run_results.append(res)
            return run_results
        run_results = asyncio.run(get_result(run_jobs))

    else:
        run_results = []
        
        for seed in tqdm(list(cfg.random_seed), desc="Random Seed"):
            random_state = np.random.RandomState(seed)
            train_idx, test_idx = train_test_split(indices, test_size=cfg.test_ratio, random_state=random_state)
            
            for train_ratio in tqdm(train_ratios, desc="Training Size"):

                train_size = int(len(train_idx) * train_ratio)
                train_idx = random_state.choice(train_idx, train_size, replace=False)
                
                np.save(f"{results_dir}/train_idx_seed{seed}_train_ratio{train_ratio}.npy", train_idx)
                np.save(f"{results_dir}/test_idx_seed{seed}_train_ratio{train_ratio}.npy", test_idx)
                
                run_model = ModelRun()
                job = run_model(train,
                                train_idx,
                                test_idx,
                                train_ratio,
                                dataset,
                                cfg,
                                random_state=seed,
                                device=None)
                run_results.append(job)

    losses, predictions, embeddings = zip(*run_results)

    prediction_metrics = predictions[0]
    for prediction in predictions[1:]:
        prediction_metrics.update(prediction)

    pred_results = []
    for k, v in prediction_metrics.items():
        true_targets, predicted_targets, indices = v
        
        true_targets_dict = {"train_ratio": [k[0]] * len(true_targets),
                             "seed":[k[1]] * len(true_targets),
                             "dataset":[k[2]] * len(true_targets)
                            }
        predicted_targets_dict = {"indices": indices}
        
        for i, target in enumerate(targets):
            true_targets_dict[target] = true_targets[:, i]
            predicted_targets_dict[f"{target}_pred"] = predicted_targets[:, i]
            
            
        true_targets = pd.DataFrame(true_targets_dict)
        predicted_targets = pd.DataFrame(predicted_targets_dict)
        
        pred_results.append(pd.concat([true_targets, predicted_targets], axis = 1))
    pred_results = pd.concat(pred_results)
    pred_results.to_csv(f"{results_dir}/pred_results.csv", index=False)

    prediction_mape_by_element = []
    for k, v in prediction_metrics.items():
        true_targets, predicted_targets, indices = v
        
        mape_by_element = np.abs(true_targets - predicted_targets) / (np.abs(true_targets)+1e-10)
        
        for i, mape in enumerate(mape_by_element):
            prediction_mape_by_element.append(
                {
                    'train_ratio': k[0],
                    'seed': k[1],
                    'dataset': k[2],
                    'mape': mape
                }
            )

    df = pd.DataFrame(prediction_mape_by_element)
    df = pd.concat([df.drop('mape', axis=1), df['mape'].apply(pd.Series)], axis=1)
    df.columns = ['train_ratio', 'seed', 'dataset'] + targets
    df= df.groupby(['train_ratio', 'seed', 'dataset']).agg('mean').reset_index()
    df.to_csv(f"{results_dir}/mape.csv", index = False)

if __name__ == "__main__":
    main()



