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
from scipy.stats import spearmanr
from sklearn.model_selection import (
    train_test_split,
)
import yaml
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset
from tqdm.auto import tqdm
from ContModeling.augmentations import augs, aug_args
import glob, os, shutil
from nilearn.datasets import fetch_atlas_schaefer_2018
import random
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, PowerTransformer
import optuna
from ContModeling.utils import gaussian_kernel, cauchy, standardize, save_embeddings
from ContModeling.losses import LogEuclideanLoss, NormLoss, KernelizedSupCon, OutlierRobustMSE
from ContModeling.models import PhenoProj
from ContModeling.helper_classes import MatData
from ContModeling.viz_func import wandb_plot_acc_vs_baseline, wandb_plot_test_recon_corr, wandb_plot_individual_recon

torch.cuda.empty_cache()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EMB_LOSSES ={
    'Norm': NormLoss(),
    'LogEuclidean': LogEuclideanLoss(),
    'MSE': nn.functional.mse_loss,
    'MSERobust': OutlierRobustMSE(),
    'Huber': nn.HuberLoss(),
    'cosine': nn.functional.cosine_embedding_loss,
}

SUPCON_KERNELS = {
    'cauchy': cauchy,
    'gaussian_kernel': gaussian_kernel,
    'None': None
    }


class ModelRun(submitit.helpers.Checkpointable):
    def __init__(self):
        self.results = None
        self.embeddings = None

    def __call__(self, train, test_size, indices, train_ratio, run_size, run, dataset, cfg, random_state=None, device=None, save_model = True, path: Path = None):
        if self.results is None:
            if device is None:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                print(f"Device {device}, ratio {train_ratio}", flush=True)
            if not isinstance(random_state, np.random.RandomState):
                random_state = np.random.RandomState(random_state)

            augmentations = cfg.augmentations

            recon_mat_dir = os.path.join(cfg.output_dir, cfg.experiment_name, cfg.reconstructed_dir)
            os.makedirs(recon_mat_dir, exist_ok=True)
    
            predictions = {}
            autoencoder_features = {}
            losses = []
            self.embeddings = {'train': [], 'test': []}
            self.run = run

            if cfg.mat_ae_pretrained:
                print("Loading test indices from the pretraining experiment...")
                test_indices = np.load(f"{cfg.output_dir}/{cfg.pretrained_mat_ae_exp}/test_idx.npy")
                train_indices = np.setdiff1d(indices, test_indices)
            elif cfg.external_test_mode:
                test_scanners = list(cfg.test_scanners)
                xr_dataset = xr.open_dataset(cfg.dataset_path)
                scanner_mask = np.sum([xr_dataset.isin(scanner).scanner.values for scanner in test_scanners],
                                    axis = 0).astype(bool)
                test_indices = indices[scanner_mask]
                train_indices = indices[~scanner_mask]
                del xr_dataset
            else:
                run_indices = random_state.choice(indices, run_size, replace=False)
                train_indices, test_indices = train_test_split(run_indices, test_size=test_size, random_state=random_state)
                
            train_dataset = Subset(dataset, train_indices)
            test_dataset = Subset(dataset, test_indices)

            train_features = train_dataset.dataset.matrices[train_dataset.indices]
            train_targets = train_dataset.dataset.target[train_dataset.indices].numpy()
            #std_train_targets, mean, std= standardize(train_targets)
            #scaler = PowerTransformer(method='box-cox', standardize=True).fit(train_targets)
            #train_targets = scaler.transform(train_targets)
            #train_targets, scalers = scale_targets_independently(train_targets)
            #train_targets = torch.nn.functional.normalize(torch.tensor(train_targets).to(device)).cpu().numpy()
            #scaler = PowerTransformer(method='yeo-johnson').fit(train_targets)
            #train_targets = scaler.transform(train_targets)
            train_targets = np.log1p(train_targets+1)
            
            input_dim_feat =cfg.input_dim_feat
            output_dim_feat = cfg.output_dim_feat

            ## Weight initialization for bilinear layer
            mean_f = torch.mean(train_features, dim=0).to(device)
            [D,V] = torch.linalg.eigh(mean_f,UPLO = "U")
            B_init_fMRI = V[:,input_dim_feat-output_dim_feat:] 
            test_features= test_dataset.dataset.matrices[test_dataset.indices].numpy()
            test_targets = test_dataset.dataset.target[test_dataset.indices].numpy()
            test_targets = np.log1p(test_targets+1)
            #test_targets = scaler.transform(test_targets)
            #test_targets = transform_targets_independently(test_targets, scalers)
            #test_targets = torch.nn.functional.normalize(torch.tensor(test_targets).to(device)).cpu().numpy()

            ### Augmentation
#             if augmentations != 'None':
# #                 aug_params = {}
#                 if not isinstance(augmentations, list):
#                     augmentations = [augmentations]
#                 n_augs = len(augmentations)
#                 vect_train_features = sym_matrix_to_vec(train_features, discard_diagonal=True)
#                 n_samples = len(train_dataset)
#                 n_features = vect_train_features.shape[-1]
#                 new_train_features = np.zeros((n_samples + n_samples * n_augs, 1, n_features))
#                 new_train_features[:n_samples, 0, :] = vect_train_features

#                 for i, aug in enumerate(augmentations):
#                     transform = augs[aug]
#                     transform_args = aug_args[aug]
# #                     aug_params[aug] = transform_args # to save later in the metrics df

#                     num_aug = i + 1
#                     aug_features = np.array([transform(sample, **transform_args) for sample in train_features])
#                     aug_features = sym_matrix_to_vec(aug_features, discard_diagonal=True)

#                     new_train_features[n_samples * num_aug: n_samples * (num_aug + 1), 0, :] = aug_features

#                 train_features = new_train_features
#                 train_targets = np.concatenate([train_targets]*(n_augs + 1), axis=0)
            
            train_dataset = TensorDataset(train_features, torch.from_numpy(train_targets).to(torch.float32))
            test_dataset = TensorDataset(torch.from_numpy(test_features).to(torch.float32), torch.from_numpy(test_targets).to(torch.float32))

            loss_terms, model = train(run, train_ratio, train_dataset, test_dataset, B_init_fMRI, cfg, device=device)
            losses.append(loss_terms.eval("train_ratio = @train_ratio").eval("run = @run"))

            #mean = torch.tensor(mean).to(device) #do we need this?
            #std  = torch.tensor(std).to(device)

            wandb.init(project=cfg.project,
                mode = "offline",
                name=f"TEST_{cfg.experiment_name}_run{run}_train_ratio_{train_ratio}",
                dir = cfg.output_dir,
                config = OmegaConf.to_container(cfg, resolve=True))
            
            embedding_dir = os.path.join(cfg.output_dir, cfg.experiment_name, cfg.embedding_dir)
            os.makedirs(embedding_dir, exist_ok=True)

            model.eval()
            with torch.no_grad():
                train_dataset = Subset(dataset, train_indices)
                train_features = train_dataset.dataset.matrices[train_dataset.indices].numpy()
                train_targets = train_dataset.dataset.target[train_dataset.indices].numpy()
                train_targets = np.log1p(train_targets+1)
                #train_targets = scaler.transform(train_targets)
                train_dataset = TensorDataset(torch.from_numpy(train_features).to(torch.float32), torch.from_numpy(train_targets).to(torch.float32))
                #std_train_targets,_,_ = standardize(train_targets)
                

                for label, d, d_indices in (('train', train_dataset, train_indices), ('test', test_dataset, test_indices)):
                    is_test = True
                    if label == 'train':
                        is_test = False
                    
                    X, y = zip(*d)
                    X = torch.stack(X).to(device)
                    y = torch.stack(y).to(device)
                    X_embedded, y_embedded = model.forward(X, y)
                                        
                    if label == 'test' and train_ratio == 1.0:
                        np.save(f'{recon_mat_dir}/test_idx_run{run}',d_indices)
                        recon_mat = model.decode_features(X_embedded)
                        mape_mat = torch.abs((X - recon_mat) / (X + 1e-10)) * 100
                        
                        wandb_plot_test_recon_corr(wandb, cfg.experiment_name, cfg.work_dir, recon_mat.cpu().numpy(), X.cpu().numpy(), mape_mat.cpu().numpy(), True, run)
                        wandb_plot_individual_recon(wandb, cfg.experiment_name, cfg.work_dir, d_indices, recon_mat.cpu().numpy(), X.cpu().numpy(), mape_mat.cpu().numpy(), 0, True, run)

                        np.save(f'{recon_mat_dir}/recon_mat_run{run}', recon_mat.cpu().numpy())
                        np.save(f'{recon_mat_dir}/mape_mat_run{run}', mape_mat.cpu().numpy())

                    X_embedded = X_embedded.cpu().numpy()
                    X_embedded = torch.tensor(sym_matrix_to_vec(X_embedded, discard_diagonal=True)).to(torch.float32).to(device)
                    X_emb_reduced = model.transfer_embedding(X_embedded).to(device)
                    y_pred = model.decode_targets(X_emb_reduced)
                    # y_pred = []
                    # for target_index in range(y.size(1)):
                    #     # Decode the specific target using the reduced feature embedding
                    #     out_single_target_decoded = model.decode_targets(X_emb_reduced, target_index)
                        
                    #     # Extract the decoded target for the current target_index
                    #     out_target_decoded = out_single_target_decoded[:, 0]
                    #     y_pred.append(out_target_decoded.unsqueeze(1))
                    # # Stack the decoded targets (assuming you want them to be in the same shape as targets)
                    # y_pred = torch.cat(y_pred, dim=1)
                    #y_pred = inverse_scale_targets_independently(y_pred.cpu().numpy(), scalers)
                    #y_pred = scaler.inverse_transform(y_pred.cpu().numpy())
                    y_pred = np.exp(y_pred.cpu().numpy())-1
                    # np.save(f'{recon_mat_dir}/y_pred_run{run}', y_pred)
                    y_pred = torch.tensor(y_pred).to(device)
                    #y = scaler.inverse_transform(y.cpu().numpy())
                    y = np.exp(y.cpu().numpy())-1
                    # np.save(f'{recon_mat_dir}/y_run{run}', y)
                    y = torch.tensor(y).to(device)
                    
                    save_embeddings(X_embedded, "mat", cfg, is_test, run)
                    save_embeddings(X_emb_reduced, "joint", cfg, is_test, run)

                    if label == 'test':
                        epsilon = 1e-8
                        mape =  100 * torch.mean(torch.abs(y - y_pred) / torch.abs((y + epsilon))).item()
                        corr =  spearmanr(y.cpu().numpy().flatten(), y_pred.cpu().numpy().flatten())[0]

                        wandb.log({
                            'Run': run,
                            'Test | Target MAPE/val' : mape,
                            'Test | Target Corr/val': corr,
                            'Test | Train ratio' : train_ratio
                            })
            
                    predictions[(train_ratio, run, label)] = (y.cpu().numpy(), y_pred.cpu().numpy(), d_indices)
                    for i, idx in enumerate(d_indices):
                        self.embeddings[label].append({
                            'index': idx,
                            'target_embedded': y_embedded[i].cpu().numpy(),
                            'feature_embedded': X_emb_reduced[i].cpu().numpy()
                        })
            wandb.finish()
            
            self.results = (losses, predictions, self.embeddings, mape, corr)

        if save_model:
            saved_models_dir = os.path.join(cfg.output_dir, cfg.experiment_name, cfg.model_weight_dir)
            os.makedirs(saved_models_dir, exist_ok=True)
            torch.save(model.state_dict(), f"{saved_models_dir}/model_weights_run{run}.pth")

        return self.results

    def checkpoint(self, *args, **kwargs):
        print("Checkpointing", flush=True)
        return super().checkpoint(*args, **kwargs)
        
def train(run, train_ratio, train_dataset, test_dataset, B_init_fMRI, cfg, model=None, device=device):
    print("Start training...")

    augmentations = cfg.augmentations
    # MODEL DIMS
    input_dim_feat = cfg.input_dim_feat
    input_dim_target = cfg.input_dim_target
    hidden_dim = cfg.hidden_dim
    output_dim_target = cfg.output_dim_target
    output_dim_feat = cfg.output_dim_feat
    kernel = SUPCON_KERNELS[cfg.SupCon_kernel]
    num_targets = cfg.num_targets
    
    # TRAINING PARAMS
    lr = cfg.lr
    batch_size = cfg.batch_size
    dropout_rate = cfg.dropout_rate
    weight_decay = cfg.weight_decay
    num_epochs = cfg.num_epochs

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    #mean= torch.tensor(mean).to(device)
    #std = torch.tensor(std).to(device)
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

    if cfg.mat_ae_pretrained:
        print("Loading pretrained MatrixAutoencoder...")
        state_dict = torch.load(f"{cfg.output_dir}/{cfg.pretrained_mat_ae_exp}/saved_models/autoencoder_weights_fold{cfg.best_mat_ae_fold}.pth")
        model.matrix_ae.load_state_dict(state_dict)
    else:
        model.matrix_ae.enc_mat1.weight = torch.nn.Parameter(B_init_fMRI.transpose(0,1))
        model.matrix_ae.enc_mat2.weight = torch.nn.Parameter(B_init_fMRI)
    
    if cfg.target_ae_pretrained:
        print("Loading pretrained TargetAutoencoder...")
        state_dict = torch.load(f"{cfg.output_dir}/{cfg.pretrained_target_ae_exp}/saved_models/autoencoder_weights_fold{cfg.best_target_ae_fold}.pth")
        model.target_ae.load_state_dict(state_dict)

    criterion_pft = KernelizedSupCon(
        method="expw",
        temperature=cfg.pft_temperature,
        base_temperature= cfg.pft_base_temperature,
        reg_term = cfg.reg_term,
        kernel=kernel,
        krnl_sigma_univar=cfg.pft_sigma_univar,
        krnl_sigma_multivar=cfg.pft_sigma_multivar,
    )
    
    feature_autoencoder_crit = EMB_LOSSES[cfg.feature_autoencoder_crit]
    target_decoding_crit = EMB_LOSSES[cfg.target_decoding_crit]

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, factor=0.1, patience = cfg.scheduler_patience)

    loss_terms = []
    validation_mape = []
    validation_corr = []
    autoencoder_features = []
    
    

    gc.collect()
    
    wandb.init(project=cfg.project,
        mode = "offline",
        name=f"{cfg.experiment_name}_run{run}_train_ratio_{train_ratio}",
        dir = cfg.output_dir,
        config = OmegaConf.to_container(cfg, resolve=True))

    with tqdm(range(num_epochs), desc="Epochs", leave=False) as pbar:
        for epoch in pbar:
            model.train()

            loss_terms_batch = defaultdict(lambda:0)
            for features, targets in train_loader:
                
                optimizer.zero_grad()
                features = features.to(device)
                targets = targets.to(device)

                ## FEATURE ENCODING
                embedded_feat = model.encode_features(features)
                ## FEATURE DECODING
                if not cfg.mat_ae_pretrained:
                    reconstructed_feat = model.decode_features(embedded_feat)
                    ## FEATURE DECODING LOSS
                    feature_autoencoder_loss = feature_autoencoder_crit(features, reconstructed_feat) / 10_000
                
                if augmentations != 'None':
#                 aug_params = {}
                    # if not isinstance(augmentations, list):
                    #     augmentations = [augmentations]
                    # if isinstance(augmentations, list):
                    #     # Flatten any nested lists in augmentations
                    #     augmentations = [aug if isinstance(aug, str) else aug[0] for aug in augmentations]
                    # else:
                    #     augmentations = [augmentations]
                    n_augs = len(augmentations)
                    vect_embedded_features = sym_matrix_to_vec(embedded_feat.detach().cpu().numpy(), discard_diagonal=True)
                    n_samples = len(embedded_feat)
                    n_features = vect_embedded_features.shape[-1]
                    #new_embedded_features = np.zeros((n_samples + n_samples * n_augs, 1, n_features))
                    new_embedded_features = np.zeros((n_samples + n_samples * n_augs, n_features))
                    #new_embedded_features[:n_samples, 0, :] = vect_embedded_features
                    new_embedded_features[:n_samples, :] = vect_embedded_features
                    for i, aug in enumerate(augmentations):
                        transform = augs[aug]
                        transform_args = aug_args[aug]
    #                     aug_params[aug] = transform_args # to save later in the metrics df

                        num_aug = i + 1
                        aug_features = np.array([transform(sample, **transform_args) for sample in embedded_feat.detach().cpu().numpy()])
                        aug_features = sym_matrix_to_vec(aug_features, discard_diagonal=True)

                        #new_embedded_features[n_samples * num_aug: n_samples * (num_aug + 1), 0, :] = aug_features
                        new_embedded_features[n_samples * num_aug: n_samples * (num_aug + 1),:] = aug_features

                    embedded_feat = torch.tensor(new_embedded_features).to(device)
                    embedded_feat = embedded_feat.float()
                    targets = torch.cat([targets]*(n_augs + 1), axis=0).to(device)
                
                
                ## REDUCED FEAT TO TARGET EMBEDDING
                # embedded_feat_vectorized = sym_matrix_to_vec(embedded_feat.detach().cpu().numpy(), discard_diagonal = True)
                # embedded_feat_vectorized = torch.tensor(embedded_feat_vectorized).to(device)
                # reduced_feat_embedding = model.transfer_embedding(embedded_feat_vectorized)
                reduced_feat_embedding = model.transfer_embedding(embedded_feat)
                
                ##KERNELIZED LOSS : MAT embeddings vs MAT vectorized
                # feat_vectorized = sym_matrix_to_vec(features.detach().cpu().numpy(), discard_diagonal = True)
                # feat_vectorized = torch.tensor(feat_vectorized).to(device)
                # kernel_embedding_loss, direction_reg_embedding = criterion_pft(embedded_feat_vectorized.unsqueeze(1), feat_vectorized)
                # kernel_embedding_loss = 100 * kernel_embedding_loss
                # direction_reg_embedding = 100*direction_reg_embedding
                ## TARGET DECODING FROM MAT EMBEDDING
                out_target_decoded = model.decode_targets(reduced_feat_embedding)

                ## KERNLIZED LOSS: MAT embedding vs targets
                kernel_embedded_feature_loss, direction_reg = criterion_pft(reduced_feat_embedding.unsqueeze(1), targets)
                kernel_embedded_feature_loss = 100 * kernel_embedded_feature_loss 
                direction_reg = 100 * direction_reg
                
                ## SECOND KERNELIZED LOSS : Reduced MAT embeddings vs MAT vectorized
                # kernel_feature_loss, direction_reg_reduced_embedding = criterion_pft(reduced_feat_embedding.unsqueeze(1), feat_vectorized)
                # kernel_feature_loss = 100* kernel_feature_loss
                # direction_reg_reduced_embedding = 100* direction_reg_reduced_embedding
               
                ## LOSS: TARGET DECODING FROM TARGET EMBEDDING
                if cfg.target_decoding_crit == 'Huber' and cfg.huber_delta != 'None':
                    target_decoding_crit = nn.HuberLoss(delta = cfg.huber_delta)
                
                target_decoding_from_reduced_emb_loss = target_decoding_crit(targets, out_target_decoded) / 100
                # target_decoding_from_reduced_emb_loss = 0
                # print("targets", targets.shape)
                # for target_index in range(targets.size(1)):
                #     target = targets[:, target_index]
                #     print("target", target.shape)
                #     # Decode the specific target using the appropriate decoder
                #     out_target_decoded = model.decode_targets(reduced_feat_embedding, target_index)
                #     print("out_target_decoded", out_target_decoded.shape)
                #     # Calculate loss for this specific target
                #     target_loss = target_decoding_crit(target, out_target_decoded) / 100
                #     target_decoding_from_reduced_emb_loss += target_loss

                ## SUM ALL LOSSES
                loss = kernel_embedded_feature_loss + target_decoding_from_reduced_emb_loss #+ kernel_feature_loss #+ kernel_feature_loss #+ kernel_embedding_loss
                # print(kernel_embedded_feature_loss, kernel_embedded_feature_loss.type, target_decoding_from_reduced_emb_loss, target_decoding_from_reduced_emb_loss.type, direction_reg, direction_reg.type)

                if not cfg.mat_ae_pretrained:
                    loss += feature_autoencoder_loss

                loss.backward()

                if cfg.clip_grad:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                if cfg.log_gradients:
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            wandb.log({
                                "Epoch": epoch,
                                f"Gradient Norm/{name}": param.grad.norm().item()
                                })  

                optimizer.step()

                loss_terms_batch['loss'] = loss.item() / len(features)
                loss_terms_batch['kernel_embedded_feature_loss'] = kernel_embedded_feature_loss.item() / len(features)
                loss_terms_batch['target_decoding_from_reduced_emb_loss'] = target_decoding_from_reduced_emb_loss.item() / len(features)
                loss_terms_batch['direction_reg_loss'] = direction_reg.item() / len(features)
                
                if not cfg.mat_ae_pretrained:
                    loss_terms_batch['feature_autoencoder_loss'] = feature_autoencoder_loss.item() / len(features)
                    wandb.log({
                        'Epoch': epoch,
                        'feature_autoencoder_loss': loss_terms_batch['feature_autoencoder_loss']
                    })
                
                wandb.log({
                    'Epoch': epoch,
                    'Run': run,
                    'total_loss': loss_terms_batch['loss'],
                    'kernel_embedded_feature_loss': loss_terms_batch['kernel_embedded_feature_loss'],
                    'direction_reg_loss': loss_terms_batch['direction_reg_loss'],
                    'target_decoding_from_reduced_emb_loss': loss_terms_batch['target_decoding_from_reduced_emb_loss']
                })

            loss_terms_batch['epoch'] = epoch
            loss_terms.append(loss_terms_batch)

            model.eval()
            mape_batch = 0
            corr_batch = 0
            with torch.no_grad():
                for (features, targets) in test_loader:
                    
                    features, targets = features.to(device), targets.to(device)                    
                    out_feat = model.encode_features(features)
                    out_feat = torch.tensor(sym_matrix_to_vec(out_feat.detach().cpu().numpy(), discard_diagonal = True)).float().to(device)
                    transfer_out_feat = model.transfer_embedding(out_feat)
                    out_target_decoded = model.decode_targets(transfer_out_feat)
                    # all_decoded_targets = []
                    # for target_index in range(targets.size(1)):
                    #     # Decode the specific target using the reduced feature embedding
                    #     out_single_target_decoded = model.decode_targets(transfer_out_feat, target_index)
                        
                    #     # Extract the decoded target for the current target_index
                    #     out_target_decoded = out_single_target_decoded[:, 0]
                    #     all_decoded_targets.append(out_target_decoded.unsqueeze(1))

                    # # Stack the decoded targets (assuming you want them to be in the same shape as targets)
                    # all_decoded_targets = torch.cat(all_decoded_targets, dim=1)
                    #out_target_decoded_denormalized = inverse_scale_targets_independently(out_target_decoded.cpu().numpy(), scalers)
                    #targets_denormalized = inverse_scale_targets_independently(targets.cpu().numpy(), scalers)
                    epsilon = 1e-6
                    mape =  torch.mean(torch.abs((targets - out_target_decoded)) / torch.abs((targets + epsilon))) * 100
                    corr =  spearmanr(targets.cpu().numpy().flatten(), out_target_decoded.cpu().numpy().flatten())[0]
                    mape_batch+=mape.item()
                    corr_batch += corr

                mape_batch = mape_batch/len(test_loader)
                corr_batch = corr_batch/len(test_loader)
                validation_mape.append(mape_batch)
                validation_corr.append(corr_batch)

            wandb.log({
                'Target MAPE/val' : mape_batch,
                'Target Corr/val': corr_batch,
                })
            
            scheduler.step(mape_batch)
            if np.log10(scheduler._last_lr[0]) < -4:
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



@hydra.main(config_path=".", config_name="main_model_config_optuna")
def main(cfg: DictConfig):

    results_dir = os.path.join(cfg.output_dir, cfg.experiment_name)
    os.makedirs(results_dir, exist_ok=True)

    random_state = np.random.RandomState(seed=42)

    dataset_path = cfg.dataset_path
    targets = list(cfg.targets)
    test_ratio = cfg.test_ratio

    dataset = MatData(dataset_path, targets, synth_exp = cfg.synth_exp, threshold=cfg.mat_threshold)
    n_sub = len(dataset)
    test_size = int(test_ratio * n_sub)
    indices = np.arange(n_sub)
    n_runs = cfg.n_runs
    multi_gpu = cfg.multi_gpu
    train_ratio = cfg.train_ratio
    
    def objective(trial):
        # Suggest hyperparameters using the trial object
        lr = trial.suggest_loguniform("lr", 1e-5, 1e-2)
        output_dim_target = trial.suggest_categorical("output_dim_target", [128, 256, 512])
        hidden_dim = trial.suggest_categorical("hidden_dim", [50, 100, 200])
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
        weight_decay = trial.suggest_loguniform("weight_decay", 1e-6, 1e-3)
        dropout_rate = trial.suggest_uniform("dropout_rate", 0.0, 0.5)
        pft_base_temperature = trial.suggest_uniform("pft_base_temperature", 0.01, 1.0)
        pft_temperature = pft_base_temperature
        ptt_base_temperature = trial.suggest_uniform("ptt_base_temperature", 0.01, 1.0)
        ptt_temperature = ptt_base_temperature
        reg_term = trial.suggest_loguniform("reg_term", 1e-4, 1e-1)
        
        # Update config with trial parameters
        cfg.lr = lr
        cfg.output_dim_target = output_dim_target
        cfg.hidden_dim = hidden_dim
        cfg.batch_size = batch_size
        cfg.weight_decay = weight_decay
        cfg.dropout_rate = dropout_rate
        cfg.pft_base_temperature = pft_base_temperature
        cfg.pft_temperature = pft_temperature
        cfg.ptt_base_temperature = ptt_base_temperature
        cfg.ptt_temperature = ptt_temperature
        cfg.reg_term = reg_term

        dataset = MatData(dataset_path, targets, synth_exp = cfg.synth_exp, threshold=cfg.mat_threshold)
        n_sub = len(dataset)
        test_size = int(test_ratio * n_sub)
        indices = np.arange(n_sub)
        n_runs = cfg.n_runs
        multi_gpu = cfg.multi_gpu
        train_ratio = cfg.train_ratio

        if multi_gpu:
            print("Using multi-gpu")
            log_folder = Path("logs")
            executor = submitit.AutoExecutor(folder=str(log_folder / "%j"))
            executor.update_parameters(
                timeout_min=120,
                slurm_partition="gpu-best",
                gpus_per_node=1,
                tasks_per_node=1,
                nodes=1
                #slurm_constraint="v100-32g",
            )
            run_jobs = []
        
            with executor.batch():
                train_size = int(n_sub * (1 - test_ratio) * train_ratio)
                run_size = test_size + train_size
                for run in tqdm(range(n_runs)):
                    run_model = ModelRun()
                    job = executor.submit(run_model, train, test_size, indices, train_ratio, run_size, run, dataset, cfg, random_state=random_state, device=None)
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
            train_size = int(n_sub * (1 - test_ratio) * train_ratio)
            run_size = test_size + train_size
            for run in tqdm(range(n_runs), desc="Model Run"):
                run_model = ModelRun()
                job = run_model(train, test_size, indices, train_ratio, run_size, run, dataset, cfg, random_state=random_state, device=None)
                run_results.append(job)

        losses, predictions, embeddings, mape, corr = zip(*run_results)
        
        prediction_metrics = predictions[0]
        for prediction in predictions[1:]:
            prediction_metrics.update(prediction)

        pred_results = []
        for k, v in prediction_metrics.items():
            true_targets, predicted_targets, indices = v
            
            true_targets_dict = {"train_ratio": [k[0]] * len(true_targets),
                                "model_run":[k[1]] * len(true_targets),
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
                        'model_run': k[1],
                        'dataset': k[2],
                        'mape': mape
                    }
                )

        df = pd.DataFrame(prediction_mape_by_element)
        df = pd.concat([df.drop('mape', axis=1), df['mape'].apply(pd.Series)], axis=1)
        df.columns = ['train_ratio', 'model_run', 'dataset'] + targets
        df= df.groupby(['train_ratio', 'model_run', 'dataset']).agg('mean').reset_index()
        df.to_csv(f"{results_dir}/mape.csv", index = False)
        
        # Calculate the optimization metric
        avg_mape = np.mean(mape)
        avg_corr = np.mean(corr)

        # Return the optimization objective (minimize MAPE, maximize correlation)
        trial.set_user_attr("avg_corr", avg_corr)  # Optional: log additional metrics
        return avg_corr
    
    study = optuna.create_study(direction="maximize")  # Change to "maximize" if optimizing correlation
    study.optimize(objective, n_trials=cfg.optuna.n_trials)

    best_trials = study.best_trials

    # Directory to save configurations
    best_configs_dir = os.path.join(results_dir, "best_configs")
    os.makedirs(best_configs_dir, exist_ok=True)

    for i, trial in enumerate(best_trials):
        best_config_path = os.path.join(best_configs_dir, f"best_config_trial_{i}.yaml")
        with open(best_config_path, "w") as f:
            yaml.dump(trial.params, f, default_flow_style=False)

    print(f"Best configurations saved in {best_configs_dir}")
        
        

if __name__ == "__main__":
    main()