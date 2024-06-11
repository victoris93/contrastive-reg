# %%
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

# %%
# data_path = Path('~/research/data/victoria_mat_age/data_mat_age_demian').expanduser()
# -

# %%
# THRESHOLD = float(sys.argv[1])
THRESHOLD = 0
# AUGMENTATION = sys.argv[1]

# %%
AUGMENTATION = None

REGION_LABELS_TO_DEACTIVATE = None#[b'7Networks_RH_Vis_2', b'7Networks_LH_DorsAttn_Post_1',b'7Networks_RH_Vis_3',b'7Networks_LH_Vis_2']
REGION_LABELS_NOT_TO_DEACTIVATE = None#[b'7Networks_RH_Vis_2', b'7Networks_LH_DorsAttn_Post_1']
SELECTED_REGIONS = [b'7Networks_RH_Vis_2', b'7Networks_LH_DorsAttn_Post_1', b'7Networks_RH_Vis_6', b'7Networks_RH_Vis_1', b'7Networks_RH_Vis_3',b'7Networks_LH_Vis_2', b'7Networks_LH_Vis_4', b'7Networks_LH_Vis_7']
# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
class MLP(nn.Module):
    def __init__(
        self,
        input_dim_feat,
        input_dim_target,
        hidden_dim_feat,
        output_dim,
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
            nn.Linear(hidden_dim_feat, output_dim),
        )
        self.init_weights(self.feat_mlp)

        # Xavier initialization for target MLP
        self.target_1_mlp = nn.Sequential(
            nn.BatchNorm1d(input_dim_target),
            nn.Linear(input_dim_target, hidden_dim_feat),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim_feat, output_dim)
        )
        self.init_weights(self.target_1_mlp)

        
        self.target_2_mlp = nn.Sequential(
            nn.BatchNorm1d(input_dim_target),
            nn.Linear(input_dim_target, hidden_dim_feat),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim_feat, output_dim)
        )
        self.init_weights(self.target_2_mlp)

        self.decode_target_1 = nn.Sequential(
            nn.Linear(output_dim, hidden_dim_feat),
            nn.BatchNorm1d(hidden_dim_feat),
            nn.ReLU(),
            nn.Linear(hidden_dim_feat, input_dim_target)
        )
        self.init_weights(self.decode_target_1)
        
        self.decode_target_2 = nn.Sequential(
            nn.Linear(output_dim, hidden_dim_feat),
            nn.BatchNorm1d(hidden_dim_feat),
            nn.ReLU(),
            nn.Linear(hidden_dim_feat, input_dim_target)
        )
        self.init_weights(self.decode_target_2)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0.0)

    def transform_feat(self, x):
        features = self.feat_mlp(x)
        features = nn.functional.normalize(features, p=2, dim=1)
        return features
    
    def transform_target_1(self, y):
        target_1 = self.target_1_mlp(y)
        target_1 = nn.functional.normalize(target_1, p=2, dim=1)
        return target_1
    
    def transform_target_2(self, y):
        target_2 = self.target_2_mlp(y)
        target_2 = nn.functional.normalize(target_2, p=2, dim=1)
        return target_2

    def decode_targets_1(self, y_embedding):
        decoded_1 = self.decode_target_1(y_embedding)
        return decoded_1
    
    def decode_targets_2(self, z_embedding):
        decoded_2 = self.decode_target_2(z_embedding)
        return decoded_2

    def forward(self, x, y, z):
        x_embedding = self.transform_feat(x)
        y_embedding = self.transform_target_1(y)
        z_embedding = self.transform_target_2(z)
        return x_embedding, y_embedding, z_embedding

# %%
class MatData(Dataset):
    def __init__(self, path_feat, path_targets, target_name_1, target_name_2, threshold=THRESHOLD, region_label_to_deactivate = REGION_LABELS_TO_DEACTIVATE, region_label_not_to_deactivate = REGION_LABELS_NOT_TO_DEACTIVATE, selected_regions = SELECTED_REGIONS):
        # self.matrices = np.load(path_feat, mmap_mode="r")
        self.matrices = np.load(path_feat, mmap_mode="r").astype(np.float32)
        self.target_1 = torch.tensor(
            np.expand_dims(
                pd.read_csv(path_targets)[target_name_1].values, axis=1
            ),
            dtype=torch.float32)
        self.target_2 = torch.tensor(
            np.expand_dims(
                pd.read_csv(path_targets)[target_name_2].values, axis=1
            ),
            dtype=torch.float32)
        
        if threshold > 0:
            self.matrices = self.threshold(self.matrices, threshold)
            self.matrices = torch.from_numpy(self.matrices).to(torch.float32)
        gc.collect()
        
        if region_label_to_deactivate is not None :
            self.matrices = self.deactivate_selected_regions(self.matrices, region_label_to_deactivate)
            self.matrices = torch.from_numpy(self.matrices).to(torch.float32)
        gc.collect()
        
        if region_label_not_to_deactivate is not None : 
            self.matrices = self.deactivate_not_selected_regions(self.matrices, region_label_not_to_deactivate)
            self.matrices = torch.from_numpy(self.matrices).to(torch.float32)
        gc.collect()
        
        if selected_regions is not None : 
            self.matrices = self.replace_selected_regions(self.matrices, selected_regions)
            self.matrices = torch.from_numpy(self.matrices).to(torch.float32)
        gc.collect()
            
        #self.matrices = torch.from_numpy(self.matrices).to(torch.float32)
        
    def deactivate_selected_regions(self, matrix, region_label_to_deactivate):
        atlas_data = fetch_atlas_schaefer_2018(n_rois=100, yeo_networks=7, resolution_mm=1)
        atlas_labels = atlas_data.labels
        for label in region_label_to_deactivate : 
            parcel_index = np.where(atlas_labels == label)[0][0]
            matrix[:, parcel_index, :] = 0
            matrix[:, :, parcel_index] = 0
        return matrix
        
    def deactivate_not_selected_regions(self, matrix, region_label_not_to_deactivate):
        atlas_data = fetch_atlas_schaefer_2018(n_rois=100, yeo_networks=7, resolution_mm=1)
        atlas_labels = atlas_data.labels

        # Get the indices of the regions to keep
        indices_to_keep = [np.where(atlas_labels == label)[0][0] for label in region_label_not_to_deactivate]

        # Create a matrix of zeros with the same shape as the input matrix
        deactivated_matrix = np.zeros_like(matrix)

        # Fill the deactivated matrix with values from the rows and columns corresponding to the indices to keep
        for idx in indices_to_keep:
            deactivated_matrix[:, idx, :] = matrix[:, idx, :]
            deactivated_matrix[:, :, idx] = matrix[:, :, idx]

        return deactivated_matrix
    
    def replace_selected_regions(self, matrix, selected_regions):
        atlas_data = fetch_atlas_schaefer_2018(n_rois=100, yeo_networks=7, resolution_mm=1)
        atlas_labels = atlas_data.labels

        # Get the indices of the selected regions
        indices_to_replace = [np.where(atlas_labels == label)[0][0] for label in selected_regions]

        # Create a copy of the matrix to avoid modifying the original one
        modified_matrix = np.copy(matrix)

        num_samples = matrix.shape[0]
        
        for sample_idx in range(num_samples):
            # Randomly select another sample index
            random_sample_idx = random.choice([i for i in range(num_samples) if i != sample_idx])

            for idx in indices_to_replace:
                # Replace the values in the selected regions
                modified_matrix[sample_idx, idx, :] = matrix[random_sample_idx, idx, :]
                modified_matrix[sample_idx, :, idx] = matrix[random_sample_idx, :, idx]

        return modified_matrix

    def threshold(self, matrices, threshold): # as in Margulies et al. (2016)
        perc = np.percentile(np.abs(matrices), threshold, axis=2, keepdims=True)
        mask = np.abs(matrices) >= perc
        thresh_mat = matrices * mask
        return thresh_mat

    def __len__(self):
        return len(self.matrices)
    def __getitem__(self, idx):
        matrix = self.matrices[idx]
        target_1 = self.target_1[idx]
        target_2 = self.target_2[idx]
        return matrix, target_1, target_2

# %%
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
            labels = labels.view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError("Num of labels does not match num of features")

            if self.kernel is None:
                mask = torch.eq(labels, labels.T)
                #mask = torch.eq(labels.unsqueez(1), labels.unsqueeze(2))
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


# %%
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


# %%

def train(train_dataset, test_dataset, model=None, device=device, kernel=cauchy, num_epochs=100, batch_size=32):
    input_dim_feat = 4950
    # the rest is arbitrary
    hidden_dim_feat = 1000
    input_dim_target = 1
    output_dim = 2
    

    num_epochs = 100

    lr = 0.1  # too low values return nan loss
    kernel = cauchy
    batch_size = 32  # too low values return nan loss
    dropout_rate = 0
    weight_decay = 0

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    if model is None:
        model = MLP(
            input_dim_feat,
            input_dim_target,
            hidden_dim_feat,
            output_dim,
            dropout_rate=dropout_rate,
        ).to(device)

    criterion_pft = KernelizedSupCon(
        method="expw", temperature=0.03, base_temperature=0.03, kernel=kernel, krnl_sigma=1
    )
    criterion_ptt = KernelizedSupCon(
        method="expw", temperature=0.03, base_temperature=0.03, kernel=kernel, krnl_sigma=1
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
            for features, target_1, target_2 in train_loader:
                
                bsz = target_1.shape[0]
                n_views = features.shape[1]
                n_feat = features.shape[-1]
                
                optimizer.zero_grad()
                features = features.view(bsz * n_views, n_feat)
                features = features.to(device)
                target_1 = target_1.to(device)
                target_2 = target_2.to(device)
                out_feat, out_target_1, out_target_2 = model(features, torch.cat(n_views*[target_1], dim=0), torch.cat(n_views*[target_2], dim=0))
                out_feat_squeezed = out_feat.squeeze()

                joint_embedding_1 = 100 * nn.functional.cosine_embedding_loss(out_feat_squeezed, out_target_1, torch.ones(out_feat_squeezed.shape[0]).to(device))
                joint_embedding_2 = 100 * nn.functional.cosine_embedding_loss(out_feat_squeezed, out_target_2, torch.ones(out_feat_squeezed.shape[0]).to(device))
                joint_embedding_3 = 100 * nn.functional.cosine_embedding_loss(out_target_1, out_target_2, torch.ones(out_target_1.shape[0]).to(device))
                
                out_feat = torch.split(out_feat, [bsz]*n_views, dim=0)
                out_feat = torch.cat([f.unsqueeze(1) for f in out_feat], dim=1)
                out_feat.to(device)
                kernel_feature_1 = criterion_pft(out_feat, target_1)
                kernel_feature_2 = criterion_pft(out_feat, target_2)

                out_target_1_decoded = model.decode_targets_1(out_target_1)
                out_target_2_decoded = model.decode_targets_2(out_target_2)

                out_target_1 = torch.split(out_target_1, [bsz]*n_views, dim=0)
                out_target_1 = torch.cat([f.unsqueeze(1) for f in out_target_1], dim=1)
                
                out_target_2 = torch.split(out_target_2, [bsz]*n_views, dim=0)
                out_target_2 = torch.cat([f.unsqueeze(1) for f in out_target_2], dim=1)
                
                kernel_target_1 = criterion_ptt(out_target_1, target_1)
                kernel_target_2 = criterion_ptt(out_target_2, target_2)
                #joint_embedding = 1000 * nn.functional.cosine_embedding_loss(out_feat, out_target, cosine_target)
                target_decoding_1 = .1 * nn.functional.mse_loss(torch.cat(n_views*[target_1], dim=0), out_target_1_decoded)
                target_decoding_2 = .1 * nn.functional.mse_loss(torch.cat(n_views*[target_2], dim=0), out_target_2_decoded)


                loss = kernel_feature_1 + kernel_feature_2 + kernel_target_1 + kernel_target_2 + joint_embedding_1+ joint_embedding_2 + joint_embedding_3 + target_decoding_1 + target_decoding_2
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                loss_terms_batch['loss'] += loss.item() / len(train_loader)
                loss_terms_batch['kernel_feature_1'] += kernel_feature_1.item() / len(train_loader)
                loss_terms_batch['kernel_feature_2'] += kernel_feature_2.item() / len(train_loader)
                loss_terms_batch['kernel_target_1'] += kernel_target_1.item() / len(train_loader)
                loss_terms_batch['kernel_target_2'] += kernel_target_2.item() / len(train_loader)
                loss_terms_batch['joint_embedding_1'] += joint_embedding_1.item() / len(train_loader)
                loss_terms_batch['joint_embedding_2'] += joint_embedding_2.item() / len(train_loader)
                loss_terms_batch['joint_embedding_3'] += joint_embedding_3.item() / len(train_loader)
                loss_terms_batch['target_decoding_1'] += target_decoding_1.item() / len(train_loader)
                loss_terms_batch['target_decoding_2'] += target_decoding_1.item() / len(train_loader)
            loss_terms_batch['epoch'] = epoch
            loss_terms.append(loss_terms_batch)

            model.eval()
            mae_batch_1 = 0
            mae_batch_2 = 0
            mae_batch_total = 0
            with torch.no_grad():
                for (features, target_1, target_2) in test_loader:
                    bsz = target_1.shape[0]
                    n_views = 1
                    n_feat = features.shape[-1]
                    
                    if len(features.shape) > 2:
                        n_views = features.shape[1]
                        features = features.view(bsz * n_views, n_feat)
                        
                    features, target_1, target_2 = features.to(device), target_1.to(device), target_2.to(device)
                    
                    out_feat = model.transform_feat(features)
                    out_target_1_decoded = model.decode_targets_1(out_feat)
                    out_target_2_decoded = model.decode_targets_2(out_feat)
                    
                    mae_batch_1 += (target_1 - out_target_1_decoded).abs().mean() / len(test_loader)
                    mae_batch_2 += (target_2 - out_target_2_decoded).abs().mean() / len(test_loader)
                    mae_batch_total += mae_batch_1 + mae_batch_2
                    
                validation.append(mae_batch_total.item())
            scheduler.step(mae_batch_total)
            if np.log10(scheduler._last_lr[0]) < -4:
                break


            pbar.set_postfix_str(
                f"Epoch {epoch} "
                f"| Loss {loss_terms[-1]['loss']:.02f} "
                f"| val MAE {validation[-1]:.02f}"
                f"| log10 lr {np.log10(scheduler._last_lr[0])}"
            )
    loss_terms = pd.DataFrame(loss_terms)
    return loss_terms, model
# %%
class Experiment(submitit.helpers.Checkpointable):
    def __init__(self):
        self.results = None

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
            predictions_1 = {}
            predictions_2 = {}
            losses = []
            experiment_indices = random_state.choice(indices, experiment_size, replace=False)
            train_indices, test_indices = train_test_split(experiment_indices, test_size=test_size, random_state=random_state)
            train_dataset = Subset(dataset, train_indices)
            test_dataset = Subset(dataset, test_indices)
            
            ### Augmentation
            train_features = train_dataset.dataset.matrices[train_dataset.indices].numpy()
            train_target_1 = train_dataset.dataset.target_1[train_dataset.indices].numpy()
            train_target_2 = train_dataset.dataset.target_2[train_dataset.indices].numpy()

            test_features= test_dataset.dataset.matrices[test_dataset.indices].numpy()
            test_target_1 = test_dataset.dataset.target_1[test_dataset.indices].numpy()
            test_target_2 = test_dataset.dataset.target_2[test_dataset.indices].numpy()

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
                train_target_1 = np.concatenate([train_target_1]*(n_augs + 1), axis=0)
                train_target_2 = np.concatenate([train_target_2]*(n_augs + 1), axis=0)
            else:
                train_features = sym_matrix_to_vec(train_features, discard_diagonal=True)
                train_features = np.expand_dims(train_features, axis = 1)
            
            train_dataset = TensorDataset(torch.from_numpy(train_features).to(torch.float32), torch.from_numpy(train_target_1).to(torch.float32), torch.from_numpy(train_target_2).to(torch.float32))
            test_features = sym_matrix_to_vec(test_features, discard_diagonal=True)
            test_dataset = TensorDataset(torch.from_numpy(test_features).to(torch.float32), torch.from_numpy(test_target_1).to(torch.float32), torch.from_numpy(test_target_2).to(torch.float32))

            loss_terms, model = train(train_dataset, test_dataset, device=device)
            losses.append(loss_terms.eval("train_ratio = @train_ratio").eval("experiment = @experiment"))
            model.eval()
            with torch.no_grad():
                train_dataset = Subset(dataset, train_indices)
                train_features = train_dataset.dataset.matrices[train_dataset.indices].numpy()
                
                train_target_1 = train_dataset.dataset.target_1[train_dataset.indices].numpy()
                train_target_2 = train_dataset.dataset.target_2[train_dataset.indices].numpy()
                train_features = np.array([sym_matrix_to_vec(i, discard_diagonal=True) for i in train_features])
                train_dataset = TensorDataset(torch.from_numpy(train_features).to(torch.float32), torch.from_numpy(train_target_1).to(torch.float32),torch.from_numpy(train_target_2).to(torch.float32))
                for label, d, d_indices in (('train', train_dataset, train_indices), ('test', test_dataset, test_indices)):
                    X, y, z = zip(*d)
                    X = torch.stack(X).to(device)
                    y = torch.stack(y).to(device)
                    z = torch.stack(z).to(device)
                    y_pred = model.decode_targets_1(model.transform_feat(X))
                    z_pred = model.decode_targets_2(model.transform_feat(X))
                    predictions_1[(train_ratio, experiment, label)] = (y.cpu().numpy(), y_pred.cpu().numpy(), d_indices)
                    predictions_2[(train_ratio, experiment, label)] = (z.cpu().numpy(), z_pred.cpu().numpy(), d_indices)

            self.results = (losses, predictions_1, predictions_2)

        if path:
            self.save(path)
        
        return self.results

    def checkpoint(self, *args, **kwargs):
        print("Checkpointing", flush=True)
        return super().checkpoint(*args, **kwargs)

    def save(self, path: Path):
        with open(path, "wb") as o:
            pickle.dump(self.results, o, pickle.HIGHEST_PROTOCOL)
            

# %%
cognitive_scores = ["BentonFaces_total", "CardioMeasures_pulse_mean", "CardioMeasures_bp_sys_mean", 
                    "CardioMeasures_bp_dia_mean", "Cattell_total", "EkmanEmHex_pca1", "EkmanEmHex_pca1_expv", 
                    "FamousFaces_details", "Hotel_time", "PicturePriming_baseline_acc", 
                    "PicturePriming_baseline_rt", "PicturePriming_priming_prime", "PicturePriming_priming_target", 
                    "Proverbs", "RTchoice", "RTsimple", "Synsem_prop_error", "Synsem_RT", "TOT", 
                    "VSTMcolour_K_mean", "VSTMcolour_K_precision", "VSTMcolour_K_doubt", "VSTMcolour_MSE"]

path_feat = "/data/parietal/store2/work/mrenaudi/contrastive-reg-3/conn_camcan_without_nan/stacked_mat.npy"
path_target = "/data/parietal/store2/work/mrenaudi/contrastive-reg-3/target_without_nan.csv"
random_state = np.random.RandomState(seed=42)
dataset = MatData(path_feat, path_target, "BentonFaces_total" , "Cattell_total", threshold=THRESHOLD)
n_sub = len(dataset)
test_ratio = .2
test_size = int(test_ratio * n_sub)
indices = np.arange(n_sub)
experiments = 20

# %% ## Training
if multi_gpu:
    log_folder = Path("log_folder")
    executor = submitit.AutoExecutor(folder=str(log_folder / "%j"))
    executor.update_parameters(
        timeout_min=120,
        slurm_partition="gpu-best",
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

# %%
losses, predictions_1, predictions_2 = zip(*experiment_results)

# %%
def calculate_variance_explained(predicted, actual):
    return 100 * (1 - np.var(predicted - actual) / np.var(actual))


# %%
prediction_metrics_1 = predictions_1[0]
for prediction in predictions_1[1:]:
    prediction_metrics_1.update(prediction)
    
prediction_mape_1 = [
    k + ((np.abs(v[0] - v[1]) / np.abs(v[0])).mean(),)
    for k, v in prediction_metrics_1.items()
]
prediction_var_1 = [
    k + (calculate_variance_explained(v[1], v[0]),)
    for k, v in prediction_metrics_1.items()
]

prediction_metrics_2 = predictions_2[0]
for prediction in predictions_2[1:]:
    prediction_metrics_2.update(prediction)
    
prediction_mape_2 = [
    k + ((np.abs(v[0] - v[1]) / np.abs(v[0])).mean(),)
    for k, v in prediction_metrics_2.items()
]

prediction_var_2 = [
    k + (calculate_variance_explained(v[1], v[0]),)
    for k, v in prediction_metrics_2.items()
]


prediction_mape_1 = pd.DataFrame(prediction_mape_1, columns=["train ratio", "experiment", "dataset", "MAPE_1"])
prediction_var_1 = pd.DataFrame(prediction_var_1, columns=["train ratio", "experiment", "dataset", "var_explained_1"])
prediction_mape_2 = pd.DataFrame(prediction_mape_2, columns=["train ratio", "experiment", "dataset", "MAPE_2"])
prediction_var_2 = pd.DataFrame(prediction_var_1, columns=["train ratio", "experiment", "dataset", "var_explained_2"])

prediction_mape_1["train size"] = (prediction_mape_1["train ratio"] * len(dataset) * (1 - test_ratio)).astype(int)
prediction_var_1["train size"] = (prediction_var_1["train ratio"] * len(dataset) * (1 - test_ratio)).astype(int)

prediction_mape_2["train size"] = (prediction_mape_2["train ratio"] * len(dataset) * (1 - test_ratio)).astype(int)
prediction_var_2["train size"] = (prediction_var_2["train ratio"] * len(dataset) * (1 - test_ratio)).astype(int)

# if AUGMENTATION is not None:
#     prediction_metrics["aug_args"] = str(aug_args)
prediction_mape_1.to_csv(f"results/prediction_mape_1_replacing_ffa_r_8.csv", index=False)
prediction_var_1.to_csv(f"results/prediction_var_1_replacing_ffa_r_8.csv", index=False)

prediction_mape_2.to_csv(f"results/prediction_mape_2_replacing_ffa_r_8.csv", index=False)
prediction_var_2.to_csv(f"results/prediction_var_2_replacing_ffa_r_8.csv", index=False)