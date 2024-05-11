# %%
import math
import asyncio
import submitit
import pickle
import sys
import os
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
import torch.nn.functional as F
import torch.optim as optim
from scipy.stats import pearsonr
from sklearn.model_selection import (
    train_test_split,
)
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset
from tqdm.auto import tqdm
from augmentations import augs, aug_args
import re

torch.cuda.empty_cache()
multi_gpu = True

# %%
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Dataset as graph_dataset
from torch_geometric.data import Data as graph_data
from os import listdir
from os.path import isfile, join
from torch_geometric.nn import Sequential as graph_sequential
from torch_geometric.loader import DataLoader as graph_dataloader

# %%
# data_path = Path('~/research/data/victoria_mat_age/data_mat_age_demian').expanduser()
# -

# %%
# THRESHOLD = float(sys.argv[1])
# AUGMENTATION = sys.argv[1]

# %%
AUGMENTATION = None

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%

class GCN(torch.nn.Module):
    def __init__(self, input_dim_feat, input_dim_target, hidden_dim_feats, output_dim, dropout_rate, lr, weight_decay):
        
        super().__init__()
        
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
    
    def initialize_optimizer(self, lr, weight_decay):
        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        return optimizer
    
    def forward(self, x, y, edge_index):
        x_embedding = self.transform_feat(x, edge_index)
        y_embedding = self.transform_targets(y)
        return x_embedding, y_embedding

# %%
class GraphData(graph_dataset):
    def __init__(self, path_feat, path_target, target_name, indices, augmentations = None):
        
        self.path_feat = path_feat
        self.path_target = path_target
        self.augmentations = augmentations
        
        self.features = np.array([np.load(i) for i in self.raw_file_names[indices]])
        self.targets = np.expand_dims(
                pd.read_csv(path_target)[target_name].values[indices], axis=1
            )
        self.features, self.graphs, self.targets = self.process(self.features, self.targets)
        self.features = sym_matrix_to_vec(self.features, discard_diagonal = True)
            
        self.targets = torch.FloatTensor(self.targets)
        self.features = torch.FloatTensor(self.features)
        gc.collect()

    @property
    def raw_file_names(self):
        raw_file_names = [join(self.path_feat, f) for f in listdir(self.path_feat) if isfile(join(self.path_feat, f))]
        raw_file_names = sorted(raw_file_names, key=lambda x: int(re.search(r'\d+', x).group()))
        raw_file_names = np.array(raw_file_names)
        return raw_file_names

    @property
    def show_indices(self):
        return self.indices
    
    def process(self, features, targets):
        ### AUGMENTATIONS
        n_augs = 0
        if self.augmentations is not None:
            aug_samples = []
            if not isinstance(self.augmentations, list):
                self.augmentations = [self.augmentations]
                
            n_augs = len(self.augmentations)
            for func in self.augmentations:
                transform = augs[func]
                transform_args = aug_args[func]
                for sample in features:
                    aug_features = transform(sample, **transform_args)
                    aug_samples.append(aug_features)
                        
            aug_samples = np.array(aug_samples)
            features = np.concatenate([features, aug_samples], axis=0)
        targets = np.concatenate([targets]*(n_augs + 1), axis=0)
        targets = np.array(targets)
#         targets = torch.FloatTensor(targets)
        ### GRAPH CONSTRUCTION
        graphs = []
        for sample in tqdm(features, desc="Processing samples"):
            graph = self.make_graph(sample)
            graphs.append(graph)
        return features, graphs, targets
    
    def get_edge_indices(self, feat_sample):
#         idx_upper_tri = np.triu_indices_from(feat_sample, k=1)  # k=1 excludes the diagonal
#         source_nodes = idx_upper_tri[0]
#         target_nodes = idx_upper_tri[1]
        
        source_nodes, target_nodes = np.nonzero(feat_sample)
        edge_indices = np.vstack((source_nodes, target_nodes))
        return edge_indices

    def make_graph(self, feat_sample):
        edge_index = self.get_edge_indices(feat_sample)
        graph = graph_data(x=torch.FloatTensor(feat_sample), edge_index=torch.LongTensor(edge_index).contiguous())
        return graph
    
    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx], self.features[idx], self.targets[idx]

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
    return torch.exp(-(x**2) / (2 * (krnl_sigma**2))) / (
        math.sqrt(2 * torch.pi) * krnl_sigma
    )



def gaussian_kernel_on_similarity_matrix(x, krnl_sigma):
    return torch.exp(-(x**2) / (2 * (krnl_sigma**2))) / (math.sqrt(2 * torch.pi))


def cauchy(x, krnl_sigma):
    x = x - x.T
    return 1.0 / (krnl_sigma * (x**2) + 1)


# %%
def train(train_dataset, test_dataset, model=None, device=device, kernel=cauchy, num_epochs=100, batch_size=32):
    input_dim_feat = 1000
    # the rest is arbitrary
    hidden_dim_feat = 500
    input_dim_target = 1
    output_dim = 2
    lr = 0.1  # too low values return nan loss
    dropout_rate = 0
    weight_decay = 0

    train_loader = graph_dataloader(train_dataset, batch_size=batch_size, shuffle=True)
    # first_batch = next(iter(train_loader))
    # first_sample = first_batch[0]
    # print("Shape of the first sample:", first_sample.shape)
    
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
            for graphs, features, targets in train_loader:
                
                print(type(graphs), features.shape, targets.shape)
                graphs = graphs.to(device)
                nodes = graphs.x
                edge_index = graphs.edge_index
                
#                 bsz = targets.shape[0]
                optimizer.zero_grad()

                features = features.to(device)              
                targets = targets.to(device)
                
                out_feat, out_target = model(nodes, targets, edge_index)
                
                joint_embedding = nn.functional.mse_loss(out_feat, out_target)
                kernel_feature = criterion_pft(out_feat.unsqueeze(1), targets)

                out_target_decoded = model.decode_target(out_target)
#                 cosine_target = torch.ones(len(out_target), device=device)                
        
                kernel_target = criterion_ptt(out_target.unsqueeze(1), targets)
#                 joint_embedding = 1000 * nn.functional.cosine_embedding_loss(out_feat, out_target, cosine_target)
                target_decoding = .1 * nn.functional.mse_loss(targets, out_target_decoded)

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
# %%
class Experiment(submitit.helpers.Checkpointable):
    def __init__(self):
        self.results = None

    def __call__(self, train, test_size, indices, train_ratio, experiment_size, experiment, feat_path, target_path, target_name, random_state, device = None, augmentations = AUGMENTATION, save_path: Path = None):
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
            
            ## Create train & test datasets
            train_dataset = GraphData(feat_path, target_path, target_name, train_indices, augmentations = augmentations)
            test_dataset = GraphData(feat_path, target_path, target_name, test_indices, augmentations = None)
            
            loss_terms, model = train(train_dataset, test_dataset, model = None, device=device)
            losses.append(loss_terms.eval("train_ratio = @train_ratio").eval("experiment = @experiment"))

            model.eval()
            with torch.no_grad():
                train_dataset = GraphData(feat_path, target_path, train_indices, augmentations = None)
                for label, d, d_indices in (('train', train_dataset, train_indices), ('test', test_dataset, test_indices)):
                    X, y = zip(*d)
                    X = torch.stack(X).to(device)
                    y = torch.stack(y).to(device)
                    y_pred = model.decode_target(model.transform_feat(X))
                    predictions[(train_ratio, experiment, label)] = (y.cpu().numpy(), y_pred.cpu().numpy(), d_indices)

            self.results = (losses, predictions)

        if save_path:
            self.save(save_path)
        
        return self.results

    def checkpoint(self, *args, **kwargs):
        print("Checkpointing", flush=True)
        return super().checkpoint(*args, **kwargs)
    
    def save(self, path: Path):
        with open(path, "wb") as o:
            pickle.dump(self.results, o, pickle.HIGHEST_PROTOCOL)
# %%
random_state = np.random.RandomState(seed=42)
#dataset = GraphData(path_feat, path_target, "age")
n_sub = 936
test_ratio = .2
test_size = int(test_ratio * n_sub)
indices = np.arange(n_sub)
experiments = 2
feat_path = './matrices'
target_path = 'participants.csv'
target_name = 'age'
# folder_path = "./matrices"
# file_names = os.listdir(folder_path)
# target_name = "age"
# targets = np.expand_dims(
#                 pd.read_csv(path_target)[target_name].values, axis=1
#             )

# %% ## Training
# %% ## Training
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
                job = executor.submit(run_experiment, train, test_size, indices, train_ratio, experiment_size, experiment, feat_path, target_path, target_name, random_state, device)
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
            job = run_experiment(train,  test_size, indices, train_ratio, experiment_size, experiment, feat_path, target_path, target_name, random_state, device)
            experiment_results.append(job)

# %%
losses, predictions = zip(*experiment_results)

# %%
prediction_metrics = predictions[0]
for prediction in predictions[1:]:
    prediction_metrics |= prediction
prediction_metrics = [
    k + (np.abs(v[0]-v[1]).mean(),)
    for k, v in prediction_metrics.items()
]
prediction_metrics = pd.DataFrame(prediction_metrics, columns=["train ratio", "experiment", "dataset", "MAE"])
prediction_metrics["train size"] = (prediction_metrics["train ratio"] * len(dataset) * (1 - test_ratio)).astype(int)
# if AUGMENTATION is not None:
#     prediction_metrics["aug_args"] = str(aug_args)
prediction_metrics.to_csv(f"results/prediction_metrics_augmentations.csv", index=False)
