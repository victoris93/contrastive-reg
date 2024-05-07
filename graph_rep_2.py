#!/usr/bin/env python
# coding: utf-8

# In[61]:


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
from augmentations import augs


# In[30]:


torch.cuda.empty_cache()
multi_gpu = True


# In[31]:


THRESHOLD = 0
AUGMENTATION = None


# In[32]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device


# ## Loading data

# In[33]:


path_feat = "/data/parietal/store/work/dwassermann/data/victoria_mat_age/matrices.npy"
path_target = "/data/parietal/store/work/dwassermann/data/victoria_mat_age/data_mat_age_demian/participants.csv"


# In[8]:


import os
# Load the dataset
matrices = np.load(path_feat)

# Create a folder to store the matrices if it doesn't exist
output_folder = "matrices"
os.makedirs(output_folder, exist_ok=True)

# Iterate over each matrix and save it to disk
for i, matrix in enumerate(matrices):
    filename = os.path.join(output_folder, f"matrix_{i}.npy")
    np.save(filename, matrix)
    print(f"Matrix {i} saved as {filename}")


# In[43]:


from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn.functional as F
from torch_geometric.nn import Sequential as graph_sequential


# ## Modele

# In[44]:


class GCN(torch.nn.Module):
    def __init__(self, input_dim_feat, input_dim_target, hidden_dim_feats, output_dim, dropout_rate, lr, weight_decay):
        
        super(GCN, self).__init__()
        
        self.feat_gcn = graph_sequential('x, edge_index', [
            (GCNConv(input_dim_feat, hidden_dim_feats), 'x, edge_index -> x'),
            nn.ReLU(),
            (GCNConv(hidden_dim_feats, hidden_dim_feats), 'x, edge_index -> x'),
            nn.ReLU(),
            (GCNConv(hidden_dim_feats, output_dim), 'x, edge_index -> x')
        ])
        #self.init_weights(self.feat_gcn)
        
        self.target_mlp = nn.Sequential(
            nn.BatchNorm1d(input_dim_target),
            nn.Linear(input_dim_target, hidden_dim_feats),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim_feats, output_dim)
        )
        #self.init_weights(self.target_mlp)
        
        self.decode_target = nn.Sequential(
            nn.Linear(output_dim, hidden_dim_feats),
            nn.BatchNorm1d(hidden_dim_feats),
            nn.ReLU(),
            nn.Linear(hidden_dim_feats, input_dim_target)
        )
        #self.init_weights(self.decode_target)
    
    
        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, GCNConv):
            # GCNConv initializes its weights during initialization
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
                    

    
    def transform_feat(self, x, edge_index):
        features = self.feat_gcn(x, edge_index)
        pooling = global_mean_pool(features, batch=None)
        pooling = nn.functional.normalize(features, p=2, dim=1)
        return pooling
    
    def transform_targets(self, y):
        targets = self.target_mlp(y)
        targets = nn.functional.normalize(targets, p=2, dim=1)
        return targets
 
    def decode_targets(self, embedding):
        return self.decode_target(embedding)

        
    def forward(self, x, y, edge_index):
        x_embedding = self.transform_feat(x, edge_index)
        y_embedding = self.transform_targets(y)
        return x_embedding, y_embedding
    
    def initialize_optimizer(self, lr, weight_decay):
        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        return optimizer


# ## GraphDataBatch

# In[51]:


import networkx as nx

def get_edge_indices(matrix):
    # Convert the matrix to a graph
    graph = nx.from_numpy_array(matrix)

    # Get the edge indices
    edge_indices = np.array(graph.edges())

    return edge_indices


# In[84]:


from torch_geometric.data import Data as graph_data
from torch_geometric.data import Dataset as graph_dataset


class GraphDataBatch(graph_dataset):
    def __init__(self, features_list, targets, augmentations):
        self.features_list = features_list
        self.targets = targets
        self.augmentations = augmentations

    def __len__(self):
        return len(self.features_list)

    def __getitem__(self, idx):
        features = self.features_list#[idx]
        target = self.targets#[idx]
        data_list = []

        for anchor in features:
            edge_index_anchor = get_edge_indices(anchor)
            data = graph_data(x=torch.FloatTensor(anchor), y=torch.FloatTensor(target), edge_index=torch.LongTensor(edge_index_anchor))
            

            data_list.append(data)

            if self.augmentations is not None:
                        transform = augs[self.augmentations]
                        aug_sample = np.array(transform(anchor))
                        edge_index_aug = get_edge_indices(aug_sample)
                        data_aug = graph_data(x=torch.FloatTensor(aug_sample), y=torch.FloatTensor(target), edge_index=torch.LongTensor(edge_index_aug))
                        data_list.append(data_aug)

        return data_list



# ## Loss

# In[12]:


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


# In[13]:


def gaussian_kernel(x, krnl_sigma):
    x = x - x.T
    return torch.exp(-(x**2) / (2 * (krnl_sigma**2))) / (math.sqrt(2 * torch.pi))

def gaussian_kernel_on_similarity_matrix(x, krnl_sigma):
    return torch.exp(-(x**2) / (2 * (krnl_sigma**2))) / (math.sqrt(2 * torch.pi))



def gaussian_kernel_original(x, krnl_sigma):
    x = x - x.T
    return torch.exp(-(x**2) / (2 * (krnl_sigma**2))) / (
        math.sqrt(2 * torch.pi) * krnl_sigma
    )


def cauchy(x, krnl_sigma):
    x = x - x.T
    return 1.0 / (krnl_sigma * (x**2) + 1)


# ## Train

# In[14]:


from torch_geometric.loader import DataLoader as graph_dataloader


# In[82]:


def train(train_dataset, test_dataset, model=None, device=device, kernel=cauchy, num_epochs=100, batch_size=32):
    input_dim_feat = 1000
    # the rest is arbitrary
    hidden_dim_feat = 500
    input_dim_target = 1
    output_dim = 2

    num_epochs = 100

    lr = 0.1  # too low values return nan loss
    kernel = cauchy
    batch_size = 32  # too low values return nan loss
    dropout_rate = 0
    weight_decay = 0

    train_loader = graph_dataloader(train_dataset, batch_size=batch_size, shuffle=True)
    first_batch = next(iter(train_loader))
    first_sample = first_batch[0]
    print("Shape of the first sample:", first_sample.shape)
    
    test_loader = graph_dataloader(test_dataset, batch_size=batch_size, shuffle=True)

    if model is None:
        model = GCN(
            input_dim_feat,
            input_dim_target,
            hidden_dim_feat,
            output_dim,
            dropout_rate=dropout_rate,
            lr = lr,
            weight_decay = weight_decay
        ).to(device)
    
    model.init_weights()   
    criterion_pft = KernelizedSupCon(
    method="expw", temperature=0.03, base_temperature=0.03, kernel=kernel, krnl_sigma=1
    )
    
    criterion_ptt = KernelizedSupCon(
        method="expw", temperature=0.03, base_temperature=0.03, kernel=kernel, krnl_sigma=1
    )
    
    optimizer = model.initialize_optimizer(lr=lr, weight_decay=weight_decay)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, factor=0.1)

    loss_terms = []
    validation = []

    torch.cuda.empty_cache()
    gc.collect()
    
    with tqdm(range(num_epochs), desc="Epochs", leave=False) as pbar:
        for epoch in pbar:
            model.train()
            loss_terms_batch = defaultdict(lambda:0)
            for features, targets, edge_indices in train_loader:
                
                bsz = targets.shape[0]
                n_views = features.shape[1]
                n_feat = features.shape[-1]
                
                optimizer.zero_grad()
                features = features.view(bsz * n_views, n_feat)
                features = features.to(device)              
                targets = targets.to(device)
                
                out_feat, out_target = model(features, torch.cat(n_views*[targets], dim=0), edge_indices)
                
                joint_embedding = nn.functional.mse_loss(out_feat, out_target)
                
                out_feat = torch.split(out_feat, [bsz]*n_views, dim=0)
                out_feat = torch.cat([f.unsqueeze(1) for f in out_feat], dim=1)
                kernel_feature = criterion_pft(out_feat, targets)

                out_target_decoded = model.decode_target(out_target)
                #cosine_target = torch.ones(len(out_target), device=device)                
                out_target = torch.split(out_target, [bsz]*n_views, dim=0)
                out_target = torch.cat([f.unsqueeze(1) for f in out_target], dim=1)
        
                kernel_target = criterion_ptt(out_target, targets)
                #joint_embedding = 1000 * nn.functional.cosine_embedding_loss(out_feat, out_target, cosine_target)
                target_decoding = .1 * nn.functional.mse_loss(torch.cat(n_views*[targets], dim=0), out_target_decoded)

                loss = kernel_feature + kernel_target + joint_embedding + target_decoding
                loss.backward()
                                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                loss_terms_batch['loss'] += loss.item() / len(train_loader)
                loss_terms_batch['kernel_feature'] += kernel_feature.item() / len(train_loader)
                loss_terms_batch['kernel_target'] += kernel_target.item() / len(train_loader)
                loss_terms_batch['joint_embedding'] += joint_embedding.item() / len(train_loader)
                loss_terms_batch['target_decoding'] += target_decoding.item() / len(train_loader)
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
                    
                    out_feat = model.transform_feat(features)
                    out_target_decoded = model.decode_target(out_feat)
                    
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
    return loss_terms, model


# ## Experiment

# In[50]:


class Experiment(submitit.helpers.Checkpointable):
    def __init__(self):
        self.results = None

    def __call__(self, train, test_size, indices, train_ratio, experiment_size, experiment, folder_path, file_names, targets, random_state=None, device=None, path: Path = None):
        if self.results is None:
            if device is None:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                print(f"Device {device}, ratio {train_ratio}", flush=True)
            if not isinstance(random_state, np.random.RandomState):
                random_state = np.random.RandomState(random_state)
                
            predictions = {}
            losses = []
            experiment_indices = random_state.choice(indices, experiment_size, replace=False)
            train_indices, test_indices = train_test_split(experiment_indices, test_size=test_size, random_state=random_state)
            
            ## Creation of the train dataset
            train_features = [np.load(os.path.join(folder_path, file_names[i])) for i in train_indices]
            train_targets = targets[train_indices]
            train_dataset = GraphDataBatch(train_features, train_targets, augmentations = AUGMENTATION)
            
            #Creation of the test dataset
            test_features = [np.load(os.path.join(folder_path, file_names[i])) for i in test_indices]
            test_targets = targets[test_indices]
            test_dataset = GraphDataBatch(test_features, test_targets, augmentations = None)
            
            loss_terms, model = train(train_dataset, test_dataset, model = None, device=device)
            losses.append(loss_terms.eval("train_ratio = @train_ratio").eval("experiment = @experiment"))
            model.eval()
            with torch.no_grad():
                
                train_features = [np.load(os.path.join(folder_path, file_names[i])) for i in train_indices]
                train_targets = targets.iloc[train_indices]
                train_dataset = GraphDataBatch(train_features, train_targets, augmentations = AUGMENTATION)
                
                test_features = [np.load(os.path.join(folder_path, file_names[i])) for i in test_indices]
                test_targets = targets.iloc[test_indices]
                test_dataset = GraphDataBatch(test_features, test_targets, augmentations = None)
                    
                
                for label, d, d_indices in (('train', train_dataset, train_indices), ('test', test_dataset, test_indices)):
                    X, y = zip(*d)
                    X = torch.stack(X).to(device)
                    y = torch.stack(y).to(device)
                    y_pred = model.decode_target(model.transform_feat(X))
                    predictions[(train_ratio, experiment, label)] = (y.cpu().numpy(), y_pred.cpu().numpy(), d_indices)

            self.results = (losses, predictions)

        if path:
            self.save(path)
        
        return self.results

    def checkpoint(self, *args, **kwargs):
        print("Checkpointing", flush=True)
        return super().checkpoint(*args, **kwargs)
    
    def save(self, path: Path):
        with open(path, "wb") as o:
            pickle.dump(self.results, o, pickle.HIGHEST_PROTOCOL)


# ## Test
# 

# In[36]:


random_state = np.random.RandomState(seed=42)
#dataset = GraphData(path_feat, path_target, "age")
n_sub = 936
test_ratio = .2
test_size = int(test_ratio * n_sub)
indices = np.arange(n_sub)
experiments = 1
folder_path = "/data/parietal/store2/work/mrenaudi/contrastive-reg-3/matrices"
file_names = os.listdir(folder_path)
target_name = "age"
targets = np.expand_dims(
                pd.read_csv(path_target)[target_name].values, axis=1
            )


# In[85]:


# %% ## Training
if multi_gpu:
    log_folder = Path("log_folder")
    executor = submitit.AutoExecutor(folder=str(log_folder / "%j"))
    executor.update_parameters(
        #timeout_min=120,
        slurm_partition="gpu-best",
        gpus_per_node=1,
        tasks_per_node=1,
        nodes=1,
        cpus_per_task=8
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
                job = executor.submit(run_experiment, train, test_size, indices, train_ratio, experiment_size, experiment, folder_path, file_names, targets, random_state=random_state, device=None)
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
            job = run_experiment(train,  test_size, indices, train_ratio, experiment_size, experiment, folder_path, file_names, targets, random_state=random_state, device=None)
            experiment_results.append(job)


# ## Debugging

# In[55]:


experiment_indices = random_state.choice(indices, experiment_size, replace=False)

train_indices, test_indices = train_test_split(experiment_indices, test_size=test_size, random_state=random_state)
            
## Creation of the train dataset
train_features = [np.load(os.path.join(folder_path, file_names[i])) for i in train_indices]
train_targets = targets[train_indices]
train_dataset = GraphDataBatch(train_features, train_targets, augmentations = AUGMENTATION)
            


# In[79]:


train_features[0]


# In[72]:


print(train_features[0].shape)


# In[73]:


graph = nx.from_numpy_array(train_features[0])
    


# In[76]:


edge_indices = np.array(graph.edges())


# In[77]:


edge_indices


# In[78]:


torch.LongTensor(edge_indices)


# In[64]:


features = self.features_list[idx]
target = self.targets[idx]
data_list = []

for anchor in features:
            edge_index_anchor = get_edge_indices(anchor)
            data = graph_data(x=torch.FloatTensor(anchor), y=torch.FloatTensor(target), edge_index=torch.LongTensor(edge_index_anchor))
            print("x", data.x.shape)
            print("y", data.y.shape)
            print("edge_index", data.edge_index.shape)

            data_list.append(data)

if self.augmentations is not None:
                        transform = augs[augmentations]
                        aug_sample = np.array(transform(anchor))
                        edge_index_aug = get_edge_indices(aug_sample)
                        data_aug = graph_data(x=torch.FloatTensor(aug_sample), y=torch.FloatTensor(target), edge_index=torch.LongTensor(edge_index_aug))
                        data_list.append(data_aug)

return data_list


# In[ ]:





