import math
import asyncio
import submitit
import pickle
import sys
from pathlib import Path
import gc
from collections import defaultdict
from nilearn.connectome import sym_matrix_to_vec
# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import pearsonr
from sklearn.model_selection import (
    train_test_split,
)
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset
from tqdm.auto import tqdm
from augmentations import augs, aug_args
import glob, os, shutil
from nilearn.datasets import fetch_atlas_schaefer_2018
import random

torch.cuda.empty_cache()
multi_gpu = True

fmri_data_path = '/gpfs3/well/margulies/projects/ABCD/fmriresults01/abcd-mproc-release5'

# data_path = Path('~/research/data/victoria_mat_age/data_mat_age_demian').expanduser()
# -

# THRESHOLD = float(sys.argv[1])
THRESHOLD = 0
SELECTED_REGIONS = None #[b'7Networks_RH_Vis_2', b'7Networks_LH_DorsAttn_Post_1']
FUNCTION = None #'deactivate_selected_regions'
AUGMENTATION = None

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

        self.feat_mlp = nn.Sequential(
            nn.BatchNorm1d(input_dim_feat),
            nn.Linear(input_dim_feat, hidden_dim_feat),
            nn.BatchNorm1d(hidden_dim_feat),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim_feat, hidden_dim_feat),
            nn.BatchNorm1d(hidden_dim_feat),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim_feat, hidden_dim_feat),
            nn.BatchNorm1d(hidden_dim_feat),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim_feat, output_dim_feat),
        )
        self.init_weights(self.feat_mlp)
        
        self.decode_feat = nn.Sequential(
            nn.BatchNorm1d(output_dim_feat),
            nn.Linear(output_dim_feat, hidden_dim_feat),
            nn.BatchNorm1d(hidden_dim_feat),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim_feat, hidden_dim_feat),
            nn.BatchNorm1d(hidden_dim_feat),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim_feat, hidden_dim_feat),
            nn.BatchNorm1d(hidden_dim_feat),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim_feat, input_dim_feat),
        )
        self.init_weights(self.decode_feat)

        # Xavier initialization for target MLP
        self.target_mlp = nn.Sequential(
            #nn.BatchNorm1d(input_dim_target),
            nn.Linear(input_dim_target, hidden_dim_feat),
            nn.BatchNorm1d(hidden_dim_feat),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim_feat, hidden_dim_feat),
            nn.BatchNorm1d(hidden_dim_feat),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim_feat, output_dim_target)
        )
        self.init_weights(self.target_mlp)

        self.decode_target = nn.Sequential(
            nn.Linear(output_dim_target, hidden_dim_feat),
            nn.BatchNorm1d(hidden_dim_feat),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim_feat, hidden_dim_feat),
            nn.BatchNorm1d(hidden_dim_feat),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim_feat, hidden_dim_feat),
            nn.BatchNorm1d(hidden_dim_feat),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim_feat, input_dim_target)
        )
        self.init_weights(self.decode_target)
        
        self.feat_to_target_embedding = nn.Sequential(
            nn.Linear(output_dim_feat, hidden_dim_feat),
            nn.BatchNorm1d(hidden_dim_feat),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim_feat, hidden_dim_feat),
            nn.BatchNorm1d(hidden_dim_feat),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim_feat, output_dim_target)
            
        )

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0.0)

    def transform_feat(self, x):
        features = self.feat_mlp(x)
        features = nn.functional.normalize(features, p=2, dim=1)
        return features
    
    def transform_targets(self, y):
        targets = self.target_mlp(y)
        targets = nn.functional.normalize(targets, p=2, dim=1)
        return targets

    def decode_targets(self, embedding):
        return self.decode_target(embedding)
    
    def decode_feats(self,embedding):
        return self.decode_feat(embedding)
    
    def transfer_embedding(self, embedding):
        return self.feat_to_target_embedding(embedding)

    def forward(self, x, y):
        x_embedding = self.transform_feat(x)
        y_embedding = self.transform_targets(y)
        return x_embedding, y_embedding

# class MatData(Dataset):
#     def __init__(self, path_feat, path_targets, threshold=THRESHOLD, selected_regions = SELECTED_REGIONS, function_to_use=FUNCTION):
#         # self.matrices = np.load(path_feat, mmap_mode="r")
#         self.matrices = np.load(path_feat, mmap_mode="r").astype(np.float32)
#         self.target = torch.tensor(pd.read_csv(path_targets).drop(columns=["Subject", "Unnamed: 0",
#         'CardioMeasures_pulse_mean',
#         'CardioMeasures_bp_sys_mean',
#         'CardioMeasures_bp_dia_mean']).to_numpy(), dtype=torch.float32)

#         #self.target = torch.tensor(pd.read_csv(path_targets)["BentonFaces_total","Cattell_total"].to_numpy(), dtype=torch.float32)
#         if threshold > 0:
#             self.matrices = self.threshold(self.matrices, threshold)
#             #self.matrices = torch.from_numpy(self.matrices).to(torch.float32)
#         gc.collect()

#         if selected_regions is not None:
#             self.matrices = self.process_selected_regions(self.matrices, selected_regions, function_to_use)
#             self.matrices = torch.from_numpy(self.matrices).to(torch.float32)
#         else: 
#             self.matrices = torch.from_numpy(self.matrices).to(torch.float32)
#         gc.collect()

#     def process_selected_regions(self, matrix, selected_regions, function_to_use):
#         if function_to_use == 'deactivate_selected_regions':
#             return self.deactivate_selected_regions(matrix, selected_regions)
#         elif function_to_use == 'deactivate_not_selected_regions':
#             return self.deactivate_not_selected_regions(matrix, selected_regions)
#         elif function_to_use == 'replace_selected_regions':
#             return self.replace_selected_regions(matrix, selected_regions)
#         elif function_to_use == 'replace_selected_regions_2':
#             return self.replace_selected_regions_2(matrix, selected_regions)
#         else:
#             raise ValueError(f"Unknown function_to_use: {function_to_use}")

#     def deactivate_selected_regions(self, matrix, selected_regions):
#         atlas_data = fetch_atlas_schaefer_2018(n_rois=100, yeo_networks=7, resolution_mm=1)
#         atlas_labels = atlas_data.labels
#         for label in selected_regions:
#             parcel_index = np.where(atlas_labels == label)[0][0]
#             matrix[:, parcel_index, :] = 0
#             matrix[:, :, parcel_index] = 0
#         return matrix

#     def deactivate_not_selected_regions(self, matrix, selected_regions):
#         atlas_data = fetch_atlas_schaefer_2018(n_rois=100, yeo_networks=7, resolution_mm=1)
#         atlas_labels = atlas_data.labels

#         indices_to_keep = [np.where(atlas_labels == label)[0][0] for label in selected_regions]

#         deactivated_matrix = np.zeros_like(matrix)

#         for idx in indices_to_keep:
#             deactivated_matrix[:, idx, :] = matrix[:, idx, :]
#             deactivated_matrix[:, :, idx] = matrix[:, :, idx]

#         return deactivated_matrix

#     def replace_selected_regions(self, matrix, selected_regions):
#         atlas_data = fetch_atlas_schaefer_2018(n_rois=100, yeo_networks=7, resolution_mm=1)
#         atlas_labels = atlas_data.labels

#         indices_to_replace = [np.where(atlas_labels == label)[0][0] for label in selected_regions]

#         modified_matrix = np.copy(matrix)
#         num_samples = matrix.shape[0]

#         for sample_idx in range(num_samples):
#             random_sample_idx = random.choice([i for i in range(num_samples) if i != sample_idx])
#             for idx in indices_to_replace:
#                 modified_matrix[sample_idx, idx, :] = matrix[random_sample_idx, idx, :]
#                 modified_matrix[sample_idx, :, idx] = matrix[random_sample_idx, :, idx]

#         return modified_matrix

#     def replace_selected_regions_2(self, matrix, selected_regions):
#         atlas_data = fetch_atlas_schaefer_2018(n_rois=100, yeo_networks=7, resolution_mm=1)
#         atlas_labels = atlas_data.labels

#         indices_to_replace = [np.where(atlas_labels == label)[0][0] for label in selected_regions]

#         modified_matrix = np.copy(matrix)
#         num_samples = matrix.shape[0]

#         for sample_idx in range(num_samples):
#             for idx in indices_to_replace:
#                 random_sample_idx = random.choice([i for i in range(num_samples) if i != sample_idx])
#                 modified_matrix[sample_idx, idx, :] = matrix[random_sample_idx, idx, :]
#                 modified_matrix[sample_idx, :, idx] = matrix[random_sample_idx, :, idx]

#         return modified_matrix

#     def threshold(self, matrices, threshold):
#         perc = np.percentile(np.abs(matrices), threshold, axis=2, keepdims=True)
#         mask = np.abs(matrices) >= perc
#         thresh_mat = matrices * mask
#         return thresh_mat

#     def __len__(self):
#         return len(self.matrices)

#     def __getitem__(self, idx):
#         matrix = self.matrices[idx]
#         target = self.target[idx]
#         return matrix, target

class MatData(Dataset):
    def __init__(self, path_feat, path_targets, target_name, threshold=THRESHOLD):
        # self.matrices = np.load(path_feat, mmap_mode="r")
        self.matrices = np.load(path_feat, mmap_mode="r").astype(np.float32)
        self.target = torch.tensor(
                pd.read_csv(path_targets, sep = '\t')[target_name].values,
                dtype=torch.float32
        )
        if threshold > 0:
            self.matrices = self.threshold(self.matrices, threshold)
        self.matrices = torch.from_numpy(self.matrices).to(torch.float32)
        gc.collect()

    def threshold(self, matrices, threshold): # as in Margulies et al. (2016)
        perc = np.percentile(np.abs(matrices), threshold, axis=2, keepdims=True)
        mask = np.abs(matrices) >= perc
        thresh_mat = matrices * mask
        return thresh_mat

    def __len__(self):
        return len(self.matrices)
    def __getitem__(self, idx):
        matrix = self.matrices[idx]
        target = self.target[idx]
        return matrix, target

# loss from: https://github.com/EIDOSLAB/contrastive-brain-age-prediction/blob/master/src/losses.py
# modified to accept input shape [bsz, n_feats]. In the age paper: [bsz, n_views, n_feats].
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
    return torch.exp(-(x**2) / (2 * (krnl_sigma**2))) / (math.sqrt(2 * torch.pi))


def gaussian_kernel_original(x, krnl_sigma):
    x = x - x.T
    return torch.exp(-(x**2) / (2 * (krnl_sigma**2))) / (
        math.sqrt(2 * torch.pi) * krnl_sigma
    )


def cauchy(x, krnl_sigma):
    x = x - x.T
    return 1.0 / (krnl_sigma * (x**2) + 1)

def multivariate_cauchy(x, krnl_sigma):
    #x = normalize(x)
    x = torch.cdist(x, x)
    return 1.0 / (krnl_sigma * (x**2) + 1)



def train(train_dataset, test_dataset, mean, std, model=None, device=device, kernel=cauchy, num_epochs=300, batch_size=32):
    input_dim_feat = 79800
    input_dim_target = 2
    # the rest is arbitrary
    hidden_dim_feat = 1000
    
    
    output_dim_target = 50
    output_dim_feat = 500
    
    lr = 0.01  # too low values return nan loss
    kernel = multivariate_cauchy
    batch_size = 32  # too low values return nan loss
    dropout_rate = 0
    weight_decay = 0

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

    criterion_pft = KernelizedSupCon(
        method="expw", temperature=0.01, base_temperature=0.01, kernel=kernel, krnl_sigma=1/50
    )
    criterion_ptt = KernelizedSupCon(
        method="expw", temperature=0.01, base_temperature=0.01, kernel=kernel, krnl_sigma=1/50
    )
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, factor=0.1)

    loss_terms = []
    validation = []
    
    torch.cuda.empty_cache()
    gc.collect()

    with tqdm(range(num_epochs), desc="Epochs", leave=False) as pbar:
        for epoch in pbar:
            model.train()
            loss_terms_batch = defaultdict(lambda:0)
            for features, targets in train_loader:
                
                bsz = targets.shape[0]
                n_views = features.shape[1]
                n_feat = features.shape[-1]
                
                optimizer.zero_grad()
                features = features.view(bsz * n_views, n_feat)
                features = features.to(device)
                targets = targets.to(device)
                #target_destandardized = targets*std+mean
                ##JOINT EMBEDDING
                out_feat, out_target = model(features, torch.cat(n_views*[targets], dim=0))
                #print("out_target", out_target.shape)
                out_feat_reduced = model.transfer_embedding(out_feat)
                #print("out_feat_reduced", out_feat_reduced.shape)
                #print("out_feat", out_feat)
                #print("out target", out_target)
                #out_feat_reduced_squeezed = out_feat.squeeze()
                #print("out_feat_reduced_squeezed", out_feat_reduced_squeezed.shape)
                #joint_embedding = 10*nn.functional.mse_loss(out_feat, out_target)
                joint_embedding = 100 * nn.functional.cosine_embedding_loss(out_feat_reduced, out_target, torch.ones(out_feat_reduced.shape[0]).to(device))
                
                
                ##FEATURE DECODING
                out_feat_decoded = model.decode_feat(out_feat)
                feature_decoding = 10*nn.functional.mse_loss(torch.cat(n_views*[features]), out_feat_decoded)
                
                ##KERNEL FEATURE
                out_feat = torch.split(out_feat, [bsz]*n_views, dim=0)
                out_feat = torch.cat([f.unsqueeze(1) for f in out_feat], dim=1)
                kernel_feature = criterion_pft(out_feat, targets)
            
                
                ##KERNEL TARGET
                out_target_decoded = model.decode_target(out_target)
                #out_target_decoded_destandardized = out_target_decoded*std+mean
                #print("out target decoded",out_target_decoded)
                #cosine_target = torch.ones(len(out_target), device=device)                
                out_target = torch.split(out_target, [bsz]*n_views, dim=0)
                out_target = torch.cat([f.unsqueeze(1) for f in out_target], dim=1)
                kernel_target = criterion_ptt(out_target, targets)
                
                ##TARGET DECODING
                #print("target", targets.shape)
                #print("target modified", torch.cat(n_views*[targets], dim=0).shape)
                #print("out_feat_reduced", out_feat_reduced.shape)
                #target_pred = model.decode_target(out_feat_reduced)
                target_decoding = 10*nn.functional.mse_loss(torch.cat(n_views*[targets], dim=0), out_target_decoded)

                loss = kernel_feature + kernel_target + joint_embedding + target_decoding
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                loss_terms_batch['loss'] += loss.item() / len(train_loader)
                loss_terms_batch['kernel_feature'] += kernel_feature.item() / len(train_loader)
                loss_terms_batch['kernel_target'] += kernel_target.item() / len(train_loader)
                loss_terms_batch['joint_embedding'] += joint_embedding.item() / len(train_loader)
                loss_terms_batch['target_decoding'] += target_decoding.item() / len(train_loader)
                loss_terms_batch['feature_decoding'] += feature_decoding.item() / len(train_loader)
            loss_terms_batch['epoch'] = epoch
            loss_terms.append(loss_terms_batch)

            model.eval()
            mae_batch = 0
            with torch.no_grad():
                for (features, targets) in test_loader:
                    bsz = targets.shape[0]
                    n_views = 1
                    n_feat = features.shape[-1]
                    
                    if len(features.shape) > 2:
                        n_views = features.shape[1]
                        features = features.view(bsz * n_views, n_feat)
                    features, targets = features.to(device), targets.to(device)
                    targets = targets*std - mean
                    
                    out_feat = model.transform_feat(features)
                    
                    out_target_decoded = model.decode_target(model.transfer_embedding(out_feat))
                    out_target_decoded = out_target_decoded*std-mean
                    

                    mae_batch += (targets - out_target_decoded).abs().mean() / len(test_loader)
                validation.append(mae_batch.item())
            scheduler.step(mae_batch)
            if np.log10(scheduler._last_lr[0]) < -4:
                break


            pbar.set_postfix_str(
                f"Epoch {epoch} "
                f"| Loss {loss_terms[-1]['loss']:.02f} "
                f"| val MAE {validation[-1]:.02f}"
                f"| log10 lr {np.log10(scheduler._last_lr[0])}"
            )
    loss_terms = pd.DataFrame(loss_terms)
    print("loss_terms", loss_terms)
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

    def __call__(self, train, test_size, indices, train_ratio, experiment_size, experiment, dataset, augmentations = AUGMENTATION, random_state=None, device=None, path: Path = None):
        if self.results is None:
            if device is None:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                print(f"Device {device}, ratio {train_ratio}", flush=True)
            if not isinstance(random_state, np.random.RandomState):
                random_state = np.random.RandomState(random_state)

            # if dataset is None:
            #     print("Loading data", flush=True)
            #     dataset = MatData(
            #         data_path / "vectorized_matrices.npy",
            #         data_path / "participants.csv",
            #         "age",
            #         threshold=threshold
            #     )

            # print("Data loaded", flush=True)
            predictions = {}
            losses = []
            self.embeddings = {'train': [], 'test': []}  # Initialize embeddings dictionary

            experiment_indices = random_state.choice(indices, experiment_size, replace=False)
            train_indices, test_indices = train_test_split(experiment_indices, test_size=test_size, random_state=random_state)
            train_dataset = Subset(dataset, train_indices)
            test_dataset = Subset(dataset, test_indices)
            ### Augmentation
            train_features = train_dataset.dataset.matrices[train_dataset.indices].numpy()
            train_targets = train_dataset.dataset.target[train_dataset.indices].numpy()
            train_targets, mean, std= standardize(train_targets)
            
            
            test_features= test_dataset.dataset.matrices[test_dataset.indices].numpy()
            test_targets = test_dataset.dataset.target[test_dataset.indices].numpy()
            test_targets = (test_targets-mean)/std
            
            if augmentations is not None:
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
            else:
                train_features = sym_matrix_to_vec(train_features, discard_diagonal=True)
                train_features = np.expand_dims(train_features, axis = 1)
            
            train_dataset = TensorDataset(torch.from_numpy(train_features).to(torch.float32), torch.from_numpy(train_targets).to(torch.float32))
            test_features = sym_matrix_to_vec(test_features, discard_diagonal=True)
            test_dataset = TensorDataset(torch.from_numpy(test_features).to(torch.float32), torch.from_numpy(test_targets).to(torch.float32))

            loss_terms, model = train(train_dataset, test_dataset,mean, std, device=device)
            losses.append(loss_terms.eval("train_ratio = @train_ratio").eval("experiment = @experiment"))
            mean = torch.tensor(mean).to(device)
            std  = torch.tensor(std).to(device)
            model.eval()
            with torch.no_grad():
                train_dataset = Subset(dataset, train_indices)
                train_features = train_dataset.dataset.matrices[train_dataset.indices].numpy()
                train_targets = train_dataset.dataset.target[train_dataset.indices].numpy()
                train_targets,_,_ = standardize(train_targets)
                train_features = np.array([sym_matrix_to_vec(i, discard_diagonal=True) for i in train_features])
                train_dataset = TensorDataset(torch.from_numpy(train_features).to(torch.float32), torch.from_numpy(train_targets).to(torch.float32))
                for label, d, d_indices in (('train', train_dataset, train_indices), ('test', test_dataset, test_indices)):
                    X, y = zip(*d)
                    X = torch.stack(X).to(device)
                    y = torch.stack(y).to(device)
                    X_embedded = model.transform_feat(X)
                    y_embedded = model.transform_targets(y)
                    y = y*std + mean
                    y_pred = model.decode_target(model.transfer_embedding(X_embedded))*std + mean
                    predictions[(train_ratio, experiment, label)] = (y.cpu().numpy(), y_pred.cpu().numpy(), d_indices)
                    for i, idx in enumerate(d_indices):
                        self.embeddings[label].append({
                            'index': idx,
                            'target_embedded': y_embedded[i].cpu().numpy(),
                            'feature_embedded': X_embedded[i].cpu().numpy(),
#                             'y': y.cpu().numpy(),
#                             'y_pred': y_pred.cpu().numpy(),
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

path_feat = "./ABCD/abcd_matrices.npy"
path_target = f"{fmri_data_path}/participants.tsv"
random_state = np.random.RandomState(seed=42)
# dataset = MatData(path_feat, path_target, threshold=THRESHOLD,
#     selected_regions = SELECTED_REGIONS, function_to_use = FUNCTION)
dataset = MatData(path_feat, path_target, ['cbcl_scr_syn_internal_r', 'cbcl_scr_syn_external_r'], threshold=THRESHOLD)
n_sub = len(dataset)
test_ratio = .2
test_size = int(test_ratio * n_sub)
indices = np.arange(n_sub)
experiments = 30

if multi_gpu:
    log_folder = Path("log_folder")
    executor = submitit.AutoExecutor(folder=str(log_folder / "%j"))
    executor.update_parameters(
        timeout_min=120,
        slurm_partition="gpu_short",
        gpus_per_node=1,
        tasks_per_node=1,
        nodes=1,
        cpus_per_task=30
        #slurm_qos="qos_gpu-t3",
        #slurm_constraint="v100-32g",
        #slurm_mem="10G",
        #slurm_additional_parameters={"requeue": True}
    )
    # srun -n 1  --verbose -A hjt@v100 -c 10 -C v100-32g   --gres=gpu:1 --time 5  python
    experiment_jobs = []
    # module_purge = submitit.helpers.CommandFunction("module purge".split())
    # module_load = submitit.helpers.CommandFunction("module load pytorch-gpu/py3/2.0.1".split())
    with executor.batch():
        for train_ratio in tqdm(np.linspace(.1, 1., 5)):
            train_size = int(n_sub * (1 - test_ratio) * train_ratio)
            experiment_size = test_size + train_size
            for experiment in tqdm(range(experiments)):
                run_experiment = Experiment()
                job = executor.submit(run_experiment, train, test_size, indices, train_ratio, experiment_size, experiment, dataset, augmentations = AUGMENTATION, random_state=random_state, device=None)
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
            job = run_experiment(train,  test_size, indices, train_ratio, experiment_size, experiment, dataset, augmentations = AUGMENTATION, random_state=random_state, device=None)
            experiment_results.append(job)
    


losses, predictions, embeddings = zip(*experiment_results)
# print("losses: ", losses)
# print("predicitons", predictions)
prediction_metrics = predictions[0]
for prediction in predictions[1:]:
    prediction_metrics.update(prediction)

pred_results = []
for k, v in prediction_metrics.items():
    true_targets, predicted_targets, indices = v
    true_targets = pd.DataFrame({"train_ratio": [k[0]] * len(true_targets),
                                 "experiment":[k[1]] * len(true_targets),
                                 "dataset":[k[2]] * len(true_targets),
                                 "cbcl_scr_syn_internal_r": true_targets[:, 0],
                                 "cbcl_scr_syn_external_r": true_targets[:, 1]
                                })
    predicted_targets = pd.DataFrame({"cbcl_scr_syn_internal_r_pred": predicted_targets[:, 0],
                                 "cbcl_scr_syn_external_r_pred": predicted_targets[:, 1],
                                 "indices": indices})
    pred_results.append(pd.concat([true_targets, predicted_targets], axis = 1))
pred_results = pd.concat(pred_results)
pred_results.to_csv(f"results/multivariate/pred_results.csv", index=False)

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
#embeddings = pd.DataFrame(embeddings, columns = ["X", "y"])
#loss  = pd.DataFrame(losses)
#loss.to_csv(f"results/multivariate/loss_test_1_lr_000001.csv", index=True)
df = pd.concat([df.drop('mape', axis=1), df['mape'].apply(pd.Series)], axis=1)
df.columns = ['train_ratio', 'experiment', 'dataset', 
            'cbcl_scr_syn_internal_r',
            'cbcl_scr_syn_external_r'
            ]
df= df.groupby(['train_ratio', 'experiment', 'dataset']).agg('mean').reset_index()

#embeddings.to_csv(f"results/multivariate/embeddings_test_1", index=True)

df.to_csv(f"results/multivariate/less_deep_decoding.csv", index=True)

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
embedding_df.to_csv(f"results/multivariate/embedding_less_deep_decoding.csv", index=False)


flat_losses = [df for sublist in losses for df in sublist]

# Concatenate all DataFrames into one single DataFrame
all_losses_df = pd.concat(flat_losses, ignore_index=True)

# Define the file path for saving the concatenated DataFrame
all_losses_file_path = "results/multivariate/loss_less_deep_decoding.csv"

# Save concatenated DataFrame to CSV
all_losses_df.to_csv(all_losses_file_path, index=False)


