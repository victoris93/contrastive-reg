import math
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
from scipy.stats import pearsonr
from sklearn.model_selection import (
    train_test_split,
)
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset
from tqdm.auto import tqdm
# from augmentations import augs, aug_args
import glob, os, shutil
from nilearn.datasets import fetch_atlas_schaefer_2018
import random
from torch.utils.tensorboard import SummaryWriter

torch.cuda.empty_cache()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MLP(nn.Module):
    def __init__(
        self,
        input_dim_feat,
        input_dim_target,
        hidden_dim_feat,
        output_dim_target,
        output_dim_feat,
        dropout_rate,
    ):
        super(MLP, self).__init__()
        
        self.output_dim_feat = output_dim_feat
        A = np.random.rand(self.output_dim_feat, self.output_dim_feat)
        A = (A + A.T) / 2
        self.vectorized_feat_emb_dim = len(sym_matrix_to_vec(A, discard_diagonal = True))

        # ENCODE MATRICES
        self.enc_mat1 = nn.Linear(in_features=input_dim_feat, out_features= output_dim_feat,bias=False)
        self.enc_mat2 = nn.Linear(in_features=input_dim_feat, out_features=output_dim_feat, bias=False)
        self.enc_mat2.weight = torch.nn.Parameter(self.enc_mat1.weight)
        
        # DECODE MATRICES
        self.dec_mat1 = nn.Linear(in_features=output_dim_feat, out_features=input_dim_feat, bias=False)
        self.dec_mat2 = nn.Linear(in_features=output_dim_feat, out_features=input_dim_feat, bias=False)
        self.dec_mat1.weight = torch.nn.Parameter(self.enc_mat1.weight.transpose(0,1))
        self.dec_mat2.weight = torch.nn.Parameter(self.dec_mat1.weight)

        self.target_mlp = nn.Sequential(
            nn.Linear(input_dim_target, hidden_dim_feat),
            nn.BatchNorm1d(hidden_dim_feat),
            nn.ELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim_feat, output_dim_target),
        )
        self.init_weights(self.target_mlp)

        self.decode_target = nn.Sequential(
            nn.Linear(output_dim_target, hidden_dim_feat),
            nn.BatchNorm1d(hidden_dim_feat),
            nn.ELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim_feat, input_dim_target),
            
        )
        self.init_weights(self.decode_target)
        
        self.feat_to_target_embedding = nn.Sequential(
            nn.Linear(self.vectorized_feat_emb_dim, hidden_dim_feat),
            nn.BatchNorm1d(hidden_dim_feat),
            nn.ELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim_feat, 2),
            
        )
        self.init_weights(self.feat_to_target_embedding)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0.0)

    def encode_feat(self, x):
        z_n = self.enc_mat1(x)
        c_hidd_mat = self.enc_mat2(z_n.transpose(1,2)) # the right dims for transpose?
        return c_hidd_mat

    def decode_feat(self,c_hidd_mat):
        z_n = self.dec_mat1(c_hidd_mat).transpose(1,2)
        recon_mat = self.dec_mat2(z_n)
        return recon_mat
    
    def transform_targets(self, y):
        targets = self.target_mlp(y)
        targets = nn.functional.normalize(targets, p=2, dim=1)
        return targets

    def decode_targets(self, embedding):
        return self.decode_target(embedding)
    
    def transfer_embedding(self, embedding):
        return self.feat_to_target_embedding(embedding)

    def forward(self, x, y):
        x_embedding = self.encode_feat(x)
        y_embedding = self.transform_targets(y)
        return x_embedding, y_embedding

class MatData(Dataset):
    def __init__(self, dataset_path, target_names, threshold=THRESHOLD):
        if not isinstance(target_names, list):
            target_names = [target_names]
        self.target_names = target_names
        self.threshold = threshold
        self.data_array = xr.open_dataset(dataset_path)
        self.matrices = self.data_array.to_array().squeeze().values.astype(np.float32)
        if threshold > 0:
            self.matrices = self.threshold_mat(self.matrices, self.threshold)
        self.matrices = torch.from_numpy(self.matrices).to(torch.float32)
        self.target = torch.from_numpy(np.array([self.data_array[target_name].values for target_name in self.target_names]).T).to(torch.float32)

        gc.collect()

    def threshold_mat(self, matrices, threshold): # as in Margulies et al. (2016)
        perc = np.percentile(np.abs(matrices), threshold, axis=2, keepdims=True)
        mask = np.abs(matrices) >= perc
        thresh_mat = matrices * mask
        return thresh_mat
    
    def __len__(self):
        return self.data_array.subject.__len__()
    
    def __getitem__(self, idx):
        matrix = self.matrices[idx]
        target = torch.from_numpy(np.array([self.data_array.sel(subject=idx)[target_name].values for target_name in self.target_names])).to(torch.float32)
        
        return matrix, target

class LogEuclideanLoss(nn.Module):
    def __init__(self):
        super(LogEuclideanLoss, self).__init__()

    def mat_batch_log(self, features):
        eps = 1e-6
        regularized_features = features + eps * \
            torch.eye(features.size(-1), device=features.device)
        Eigvals, Eigvecs = torch.linalg.eigh(regularized_features)
        Eigvals = torch.clamp(Eigvals, min=eps)
        log_eigvals = torch.diag_embed(torch.log(Eigvals))
        matmul1 = torch.matmul(log_eigvals, Eigvecs.transpose(-2, -1))
        matmul2 = torch.matmul(Eigvecs, matmul1)
        return matmul2

    def forward(self, features, recon_features):
        """
        Compute the Log-Euclidean distance between two batches of SPD matrices.

        Args:
            features: Tensor of shape [batch_size, n_parcels, n_parcels]
            recon_features: Tensor of shape [batch_size, n_parcels, n_parcels]

        Returns:
            A loss scalar.
        """
        device = features.device
        eye = torch.eye(features.size(-1), device=device)
        recon_features_diag = recon_features*(1-eye)+eye
        recon_features_diag = torch.round(recon_features_diag, decimals=3)

        log_features = self.mat_batch_log(features)
        log_recon_features = self.mat_batch_log(recon_features_diag)
        loss = torch.norm(log_features - log_recon_features,
                          dim=(-2, -1)).mean()
        return loss

class NormLoss(nn.Module):
    def __init__(self):
        super(NormLoss, self).__init__()
    
    def forward(self, features, recon_features):
        """
        Compute the Frobenius norm-based loss between two batches of matrices.

        Args:
            features: Tensor of shape [batch_size, n_parcels, n_parcels]
            recon_features: Tensor of shape [batch_size, n_parcels, n_parcels]
        
        Returns:
            A loss scalar.
        """
        loss = torch.norm(features - recon_features) ** 2
        return loss

class KernelizedSupCon(nn.Module):
    """Supervised contrastive loss: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR
    Based on: https://github.com/HobbitLong/SupContrast"""

    def __init__(
        self,
        method: str,
        temperature: float = 0.07,
        contrast_mode: str = "all",
        base_temperature: float = 0.07,
        krnl_sigma: float = 1.0,
        kernel: callable = None,
        delta_reduction: str = "sum",
    ):
        super().__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.method = method
        self.kernel = kernel
        self.krnl_sigma = krnl_sigma
        self.delta_reduction = delta_reduction

        if kernel is not None and method == "supcon":
            raise ValueError("Kernel must be none if method=supcon")

        if kernel is None and method != "supcon":
            raise ValueError("Kernel must not be none if method != supcon")

        if delta_reduction not in ["mean", "sum"]:
            raise ValueError(f"Invalid reduction {delta_reduction}")

    def __repr__(self):
        return (
            f"{self.__class__.__name__} "
            f"(t={self.temperature}, "
            f"method={self.method}, "
            f"kernel={self.kernel is not None}, "
            f"delta_reduction={self.delta_reduction})"
        )

    def forward(self, features, labels=None):
        """Compute loss for model. If `labels` is None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, n_features].
                input has to be rearranged to [bsz, n_views, n_features] and labels [bsz],
            labels: ground truth of shape [bsz].
        Returns:
            A loss scalar.
        """
        device = features.device

        if len(features.shape) != 3:
            raise ValueError(
                "`features` needs to be [bsz, n_views, n_feats],"
                "3 dimensions are required"
            )

        batch_size = features.shape[0]
        n_views = features.shape[1]

        if labels is None:
            mask = torch.eye(batch_size, device=device)

        else:
            #labels = labels.view(-1, 1)
            #if labels.shape[0] != batch_size:
            #    raise ValueError("Num of labels does not match num of features")

            if self.kernel is None:
                mask = torch.eq(labels, labels.T)
            else:
                mask = self.kernel(labels, krnl_sigma=self.krnl_sigma)

        view_count = features.shape[1]
        features = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == "one":
            features = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == "all":
            features = features
            anchor_count = view_count
        else:
            raise ValueError("Unknown mode: {}".format(self.contrast_mode))

        # Tile mask
        mask = mask.repeat(anchor_count, view_count)

        # Inverse of torch-eye to remove self-contrast (diagonal)
        inv_diagonal = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * n_views, device=device).view(-1, 1),
            0,
        )

        # compute similarity
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T), self.temperature
        )

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        alignment = logits

        # base case is:
        # - supcon if kernel = none
        # - y-aware is kernel != none
        uniformity = torch.exp(logits) * inv_diagonal

        if self.method == "threshold":
            repeated = mask.unsqueeze(-1).repeat(
                1, 1, mask.shape[0]
            )  # repeat kernel mask

            delta = (mask[:, None].T - repeated.T).transpose(
                1, 2
            )  # compute the difference w_k - w_j for every k,j
            delta = (delta > 0.0).float()

            # for each z_i, repel only samples j s.t. K(z_i, z_j) < K(z_i, z_k)
            uniformity = uniformity.unsqueeze(-1).repeat(1, 1, mask.shape[0])

            if self.delta_reduction == "mean":
                uniformity = (uniformity * delta).mean(-1)
            else:
                uniformity = (uniformity * delta).sum(-1)

        elif self.method == "expw":
            # exp weight e^(s_j(1-w_j))
            uniformity = torch.exp(logits * (1 - mask)) * inv_diagonal

        uniformity = torch.log(uniformity.sum(1, keepdim=True))

        # positive mask contains the anchor-positive pairs
        # excluding <self,self> on the diagonal
        positive_mask = mask * inv_diagonal

        log_prob = (
            alignment - uniformity
        )  # log(alignment/uniformity) = log(alignment) - log(uniformity)
        log_prob = (positive_mask * log_prob).sum(1) / positive_mask.sum(
            1
        )  # compute mean of log-likelihood over positive

        # loss
        loss = -(self.temperature / self.base_temperature) * log_prob
        return loss.mean()

def gaussian_kernel(x, krnl_sigma):
    x = x - x.T
    return torch.exp(-(x**2) / (2 * (krnl_sigma**2))) / (
        math.sqrt(2 * torch.pi) * krnl_sigma
    )

def cauchy(x, krnl_sigma):
    x = x - x.T
    return 1.0 / (krnl_sigma * (x**2) + 1)

def multivariate_cauchy(x, krnl_sigma):
    x = torch.cdist(x, x)
    return 1.0 / (krnl_sigma * (x**2) + 1)

EMB_LOSSES ={
    'Norm': NormLoss(),
    'LogEuclidean': LogEuclideanLoss(),
    'MSE': nn.functional.mse_loss,
    'cosine': nn.functional.cosine_embedding_loss,
}

SUPCON_KERNELS = {
    'multivariate_cauchy': multivariate_cauchy,
    'cauchy': cauchy,
    'gaussian_kernel': gaussian_kernel
    
}

def train(experiment, train_ratio, train_dataset, test_dataset, mean, std, B_init_fMRI, cfg, model=None, device=device):

    # MODEL DIMS
    input_dim_feat = cfg.input_dim_feat
    input_dim_target = cfg.input_dim_target
    hidden_dim_feat = cfg.hidden_dim_feat
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
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    mean= torch.tensor(mean).to(device)
    std = torch.tensor(std).to(device)
    if model is None:
        model = MLP(
            input_dim_feat,
            input_dim_target,
            hidden_dim_feat,
            output_dim_target,
            output_dim_feat,
            dropout_rate,
        ).to(device)

    model.enc_mat1.weight = torch.nn.Parameter(B_init_fMRI.transpose(0,1))
    model.enc_mat2.weight = torch.nn.Parameter(B_init_fMRI.transpose(0,1))

    criterion_pft = KernelizedSupCon(
        method="expw", temperature=cfg.pft_temperature, base_temperature= cfg.pft_base_temperature, kernel=kernel, krnl_sigma=cfg.pft_sigma
    )
    criterion_ptt = KernelizedSupCon(
        method="expw", temperature=cfg.ptt_temperature, base_temperature=cfg.ptt_base_temperature, kernel=kernel, krnl_sigma=cfg.ptt_sigma
    )
    feature_autoencoder_crit = EMB_LOSSES[cfg.feature_autoencoder_crit]
    joint_embedding_crit = EMB_LOSSES[cfg.joint_embedding_crit]
    target_decoding_crit = EMB_LOSSES[cfg.target_decoding_crit]
    target_decoding_from_reduced_emb_crit = EMB_LOSSES[cfg.target_decoding_from_reduced_emb_crit]


    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, factor=0.1, patience = cfg.scheduler_patience)

    loss_terms = []
    validation = []
    autoencoder_features = []
    
    torch.cuda.empty_cache()
    gc.collect()

    tensorboard_dir = os.path.join(cfg.output_dir,cfg.experiment_name, cfg.tensorboard_dir)
    os.makedirs(tensorboard_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=cfg.tensorboard_dir)
    writer.add_scalar('Experiment', experiment + 1)
    writer.add_scalar('Train_ratio', train_ratio)

    with tqdm(range(num_epochs), desc="Epochs", leave=False) as pbar:
        for epoch in pbar:
            model.train()
            loss_terms_batch = defaultdict(lambda:0)
            for features, targets in train_loader:
                
                optimizer.zero_grad()
                features = features.to(device)
                targets = targets.to(device)
                #target_destandardized = targets*std+mean
                
                ## FEATURE ENCODING
                embedded_feat = model.encode_feat(features)
                ## FEATURE DECODING
                reconstructed_feat = model.decode_feat(embedded_feat)
                ## FEATURE DECODING LOSS
                feature_autoencoder_loss = feature_autoencoder_crit(features, reconstructed_feat)

                ## REDUCED FEAT TO TARGET EMBEDDING
                embedded_feat_vectorized = sym_matrix_to_vec(embedded_feat.detach().cpu().numpy(), discard_diagonal = True)
                embedded_feat_vectorized = torch.tensor(embedded_feat_vectorized).to(device)
                reduced_feat_embedding = model.transfer_embedding(embedded_feat_vectorized)                
                out_target = model.transform_targets(targets)

                if cfg.joint_embedding_crit == 'cosine':
                    joint_embedding_loss = 100 * joint_embedding_crit(reduced_feat_embedding,
                                                                      out_target,
                                                                      torch.ones(out_target.shape[0]).to(device)
                                                                      )
                else:
                    joint_embedding_loss = 100 * joint_embedding_crit(reduced_feat_embedding,
                                                                      out_target
                                                                      )
                
                kernel_embedded_feature_loss = criterion_pft(reduced_feat_embedding.unsqueeze(1), targets)

                ## TARGET DECODING FROM TARGET EMBEDDING
                out_target_decoded = model.decode_target(out_target)               

                ## KERNLIZED LOSS: target embedding vs targets
                kernel_embedded_target_loss = criterion_ptt(out_target.unsqueeze(1), targets)
                
                ## LOSS: TARGET DECODING FROM TARGET EMBEDDING
                target_decoding_loss = 10*target_decoding_crit(targets, out_target_decoded)

                ## TARGET DECODING FROM THE REDUCED FEATURE EMBEDDING
                target_decoded_from_reduced_emb = model.decode_target(reduced_feat_embedding)

                ## LOSS: TARGET DECODING FROM THE REDUCED FEATURE EMBEDDING
                target_decoding_from_reduced_emb_loss = 100*target_decoding_from_reduced_emb_crit(targets, target_decoded_from_reduced_emb)

                ## SUM ALL LOSSES
                loss = feature_autoencoder_loss + kernel_embedded_feature_loss + kernel_embedded_target_loss + joint_embedding_loss + target_decoding_loss + target_decoding_from_reduced_emb_loss

                loss.backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                
                loss_terms_batch['loss'] += loss.item() / len(train_loader)
                writer.add_scalar('Loss/loss', loss_terms_batch['loss'], epoch)

                loss_terms_batch['feature_autoencoder_loss'] += feature_autoencoder_loss.item() / len(train_loader)
                writer.add_scalar('Loss/feature_autoencoder_loss', loss_terms_batch['feature_autoencoder_loss'], epoch)

                loss_terms_batch['kernel_embedded_feature_loss'] += kernel_embedded_feature_loss.item() / len(train_loader)
                writer.add_scalar('Loss/kernel_embedded_feature_loss', loss_terms_batch['kernel_embedded_feature_loss'], epoch)

                loss_terms_batch['kernel_embedded_target_loss'] += kernel_embedded_target_loss.item() / len(train_loader)
                writer.add_scalar('Loss/kernel_embedded_target_loss', loss_terms_batch['kernel_embedded_target_loss'], epoch)

                loss_terms_batch['joint_embedding_loss'] += joint_embedding_loss.item() / len(train_loader)
                writer.add_scalar('Loss/joint_embedding_loss', loss_terms_batch['joint_embedding_loss'], epoch)

                loss_terms_batch['target_decoding_loss'] += target_decoding_loss.item() / len(train_loader)
                writer.add_scalar('Loss/target_decoding_loss', loss_terms_batch['target_decoding_loss'], epoch)

                loss_terms_batch['target_decoding_from_reduced_emb_loss'] += target_decoding_from_reduced_emb_loss.item() / len(train_loader)
                writer.add_scalar('Loss/target_decoding_from_reduced_emb_loss', loss_terms_batch['target_decoding_from_reduced_emb_loss'], epoch)

            loss_terms_batch['epoch'] = epoch
            loss_terms.append(loss_terms_batch)

            model.eval()
            mape_batch = 0
            with torch.no_grad():
                for (features, targets) in test_loader:
                    
                    features, targets = features.to(device), targets.to(device)                    
                    out_feat = model.encode_feat(features)
                    out_feat = torch.tensor(sym_matrix_to_vec(out_feat.detach().cpu().numpy(), discard_diagonal = True)).float().to(device)
                    transfer_out_feat = model.transfer_embedding(out_feat)
                    out_target_decoded = model.decode_target(transfer_out_feat)
                    
                    epsilon = 1e-8
                    mape =  torch.mean(torch.abs((targets - out_target_decoded)) / torch.abs((targets + epsilon))) * 100
                    mape_batch+=mape.item()
                    writer.add_scalar('MAPE/val', mape_batch, epoch)
                    writer.close()

                validation.append(mape_batch)
            scheduler.step(mape_batch)
            if np.log10(scheduler._last_lr[0]) < -4:
                break


            pbar.set_postfix_str(
                f"Epoch {epoch} "
                f"| Loss {loss_terms[-1]['loss']:.02f} "
                f"| val MAPE {validation[-1]:.02f}"
                f"| log10 lr {np.log10(scheduler._last_lr[0])}"
            )
    loss_terms = pd.DataFrame(loss_terms)
    print(loss_terms[['loss','kernel_embedded_feature_loss', 'kernel_embedded_target_loss']])
    return loss_terms, model

def standardize(data, mean=None, std=None, epsilon = 1e-4):
    if mean is None:
        mean = np.mean(data, axis=0)
    if std is None:
        std = np.std(data, axis=0)+ epsilon
    return (data - mean)/std, mean, std

class Experiment(submitit.helpers.Checkpointable):
    def __init__(self):
        self.results = None
        self.embeddings = None 

    def __call__(self, train, test_size, indices, train_ratio, experiment_size, experiment, dataset, cfg, random_state=None, device=None, path: Path = None):
        if self.results is None:
            if device is None:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                print(f"Device {device}, ratio {train_ratio}", flush=True)
            if not isinstance(random_state, np.random.RandomState):
                random_state = np.random.RandomState(random_state)

            augmentations = cfg.augmentation

            recon_mat_dir = os.path.join(cfg.output_dir, cfg.experiment_name, cfg.reconstructed_dir)
            os.makedirs(recon_mat_dir, exist_ok=True)
    
            predictions = {}
            autoencoder_features = {}
            losses = []
            self.embeddings = {'train': [], 'test': []}  # Initialize embeddings dictionary

            experiment_indices = random_state.choice(indices, experiment_size, replace=False)
            train_indices, test_indices = train_test_split(experiment_indices, test_size=test_size, random_state=random_state)
            train_dataset = Subset(dataset, train_indices)
            test_dataset = Subset(dataset, test_indices)

            train_features = train_dataset.dataset.matrices[train_dataset.indices]
            train_targets = train_dataset.dataset.target[train_dataset.indices].numpy()
            train_targets, mean, std= standardize(train_targets)

            ## Weight initialization for bilinear layer
            input_dim_feat =cfg.input_dim_feat
            output_dim_feat = cfg.output_dim_feat
            mean_f = torch.mean(train_features, dim=0).to(device)
            [D,V] = torch.linalg.eigh(mean_f,UPLO = "U")
            B_init_fMRI = V[:,input_dim_feat-output_dim_feat:] 
            test_features= test_dataset.dataset.matrices[test_dataset.indices].numpy()
            test_targets = test_dataset.dataset.target[test_dataset.indices].numpy()
            test_targets = (test_targets-mean)/std

            ### Augmentation
            if augmentations != 'None':
#                 aug_params = {}
                if not isinstance(augmentations, list):
                    augmentations = [augmentations]
                n_augs = len(augmentations)
                vect_train_features = sym_matrix_to_vec(train_features, discard_diagonal=True)
                n_samples = len(train_dataset)
                n_features = vect_train_features.shape[-1]
                new_train_features = np.zeros((n_samples + n_samples * n_augs, 1, n_features))
                new_train_features[:n_samples, 0, :] = vect_train_features

                for i, aug in enumerate(augmentations):
                    transform = augs[aug]
                    transform_args = aug_args[aug]
#                     aug_params[aug] = transform_args # to save later in the metrics df

                    num_aug = i + 1
                    aug_features = np.array([transform(sample, **transform_args) for sample in train_features])
                    aug_features = sym_matrix_to_vec(aug_features, discard_diagonal=True)

                    new_train_features[n_samples * num_aug: n_samples * (num_aug + 1), 0, :] = aug_features

                train_features = new_train_features
                train_targets = np.concatenate([train_targets]*(n_augs + 1), axis=0)
            
            train_dataset = TensorDataset(train_features, torch.from_numpy(train_targets).to(torch.float32))
            test_dataset = TensorDataset(torch.from_numpy(test_features).to(torch.float32), torch.from_numpy(test_targets).to(torch.float32))
            test_dataset = TensorDataset(torch.from_numpy(test_features).to(torch.float32), torch.from_numpy(test_targets).to(torch.float32))

            loss_terms, model = train(experiment, train_ratio, train_dataset, test_dataset,mean, std, B_init_fMRI, cfg, device=device)
            losses.append(loss_terms.eval("train_ratio = @train_ratio").eval("experiment = @experiment"))
            mean = torch.tensor(mean).to(device)
            std  = torch.tensor(std).to(device)
            model.eval()
            with torch.no_grad():
                train_dataset = Subset(dataset, train_indices)
                train_features = train_dataset.dataset.matrices[train_dataset.indices].numpy()
                train_targets = train_dataset.dataset.target[train_dataset.indices].numpy()
                train_targets,_,_ = standardize(train_targets)
                train_dataset = TensorDataset(torch.from_numpy(train_features).to(torch.float32), torch.from_numpy(train_targets).to(torch.float32))
                for label, d, d_indices in (('train', train_dataset, train_indices), ('test', test_dataset, test_indices)):
                    X, y = zip(*d)
                    X = torch.stack(X).to(device)
                    y = torch.stack(y).to(device)
                    X_embedded = model.encode_feat(X)
                    y_embedded = model.transform_targets(y)
                                        
                    if label == 'test' and train_ratio == 1.0:
                        recon_mat = model.decode_feat(X_embedded)
                        mape_mat = torch.abs((X - recon_mat) / (X + 1e-10)) * 100
                        np.save(f'{recon_mat_dir}/recon_mat_exp{experiment}', recon_mat.cpu().numpy())
                        np.save(f'{recon_mat_dir}/mape_mat_exp{experiment}', mape_mat.cpu().numpy())
                        
                    X_embedded = X_embedded.cpu().numpy()
                    X_embedded = torch.tensor(sym_matrix_to_vec(X_embedded, discard_diagonal=True)).to(torch.float32).to(device)
                    X_emb_reduced = model.transfer_embedding(X_embedded).to(device)
                    y_pred = model.decode_target(X_emb_reduced)
                    print(y_pred[:3])
                    
                    
                    predictions[(train_ratio, experiment, label)] = (y.cpu().numpy(), y_pred.cpu().numpy(), d_indices)
                    for i, idx in enumerate(d_indices):
                        self.embeddings[label].append({
                            'index': idx,
                            'target_embedded': y_embedded[i].cpu().numpy(),
                            'feature_embedded': X_emb_reduced[i].cpu().numpy()
                        })
                    
            self.results = (losses, predictions, self.embeddings)
            
        if path:
            self.save(path)
        
        return self.results

    def checkpoint(self, *args, **kwargs):
        print("Checkpointing", flush=True)
        return super().checkpoint(*args, **kwargs)

    def save(self, path: Path):
        with open(path, "wb") as o:
            pickle.dump(self.results, o, pickle.HIGHEST_PROTOCOL)

@hydra.main(config_path=".", config_name="main_model_config")

def main(cfg: DictConfig):

    results_dir = os.path.join(cfg.output_dir, cfg.experiment_name)
    os.makedirs(results_dir, exist_ok=True)
    tensorboard_dir = os.path.join(results_dir, cfg.tensorboard_dir)
    os.makedirs(tensorboard_dir, exist_ok=True)

    random_state = np.random.RandomState(seed=42)

    dataset_path = cfg.dataset_path
    targets = list(cfg.targets)
    test_ratio = cfg.test_ratio

    dataset = MatData(dataset_path, targets, threshold=cfg.mat_threshold)
    n_sub = len(dataset)
    test_size = int(test_ratio * n_sub)
    indices = np.arange(n_sub)
    experiments = cfg.experiments
    multi_gpu = cfg.multi_gpu

    if multi_gpu:
        log_folder = Path("logs")
        executor = submitit.AutoExecutor(folder=str(log_folder / "%j"))
        executor.update_parameters(
            timeout_min=120,
            slurm_partition="gpu_short",
            gpus_per_node=1,
            tasks_per_node=1,
            nodes=1,
            cpus_per_task=30
            #slurm_constraint="v100-32g",

        )
        experiment_jobs = []

        with executor.batch():
            for train_ratio in tqdm(np.linspace(.1, 1., 5)):
                train_size = int(n_sub * (1 - test_ratio) * train_ratio)
                experiment_size = test_size + train_size
                for experiment in tqdm(range(experiments)):
                    run_experiment = Experiment()
                    job = executor.submit(run_experiment, train, test_size, indices, train_ratio, experiment_size, experiment, dataset, cfg, random_state=random_state, device=None)
                    experiment_jobs.append(job)

        async def get_result(experiment_jobs):
            experiment_results = []
            for aws in tqdm(asyncio.as_completed([j.awaitable().result() for j in experiment_jobs]), total=len(experiment_jobs)):
                res = await aws
                experiment_results.append(res)
            return experiment_results
        experiment_results = asyncio.run(get_result(experiment_jobs))

    else:
        experiment_results = []
        
        for train_ratio in tqdm(np.linspace(.1, 1., 5), desc="Training Size"):
            train_size = int(n_sub * (1 - test_ratio) * train_ratio)
            experiment_size = test_size + train_size
            for experiment in tqdm(range(experiments), desc="Experiment"):
                run_experiment = Experiment()
                job = run_experiment(train, test_size, indices, train_ratio, experiment_size, experiment, dataset, cfg, random_state=random_state, device=None)
                experiment_results.append(job)

    losses, predictions, embeddings = zip(*experiment_results)

    prediction_metrics = predictions[0]
    for prediction in predictions[1:]:
        prediction_metrics.update(prediction)

    pred_results = []
    for k, v in prediction_metrics.items():
        true_targets, predicted_targets, indices = v
        
        true_targets_dict = {"train_ratio": [k[0]] * len(true_targets),
                             "experiment":[k[1]] * len(true_targets),
                             "dataset":[k[2]] * len(true_targets)
                            }
        predicted_targets_dict = {"indices": indices}
        
        for i, target is targets:
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
                    'experiment': k[1],
                    'dataset': k[2],
                    'mape': mape
                }
            )

    df = pd.DataFrame(prediction_mape_by_element)
    df = pd.concat([df.drop('mape', axis=1), df['mape'].apply(pd.Series)], axis=1)
    df.columns = ['train_ratio', 'experiment', 'dataset'] + targets
    df= df.groupby(['train_ratio', 'experiment', 'dataset']).agg('mean').reset_index()
    df.to_csv(f"{results_dir}/mape.csv", index = False)

    embedding_data = []
    for experiment_embedding in embeddings:
        for key, values in experiment_embedding.items():
            for value in values:
                embedding_data.append({
                    'dataset': key,
                    'index': value['index'],
                    'target_embedded': value['target_embedded'],
                    'feature_embedded': value['feature_embedded']
                })

    embedding_df = pd.DataFrame(embedding_data)
    embedding_df.to_csv(f"{results_dir}/embeddings.csv", index=False)

if __name__ == "__main__":
    main()
