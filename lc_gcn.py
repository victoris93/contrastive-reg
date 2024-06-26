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
import torch_geometric as tg
from copy import copy

torch.cuda.empty_cache()

multi_gpu = True

from torch_geometric.nn import GCNConv, global_mean_pool, Linear
from torch_geometric.data import Dataset as graph_dataset
from torch_geometric.data import Data as graph_data
from os import listdir
from os.path import isfile, join
from torch_geometric.nn import Sequential as graph_sequential
from torch_geometric.loader import DataLoader as graph_dataloader
from torch_geometric.utils import dense_to_sparse
from torch_geometric.transforms.line_graph import LineGraph
from torch_geometric.transforms import ToUndirected
from torch_geometric.nn.norm import BatchNorm
from torch_geometric.utils import dropout

THRESHOLD = 0#int(sys.argv[1])

AUGMENTATION = None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class graph_MLP(torch.nn.Module):
    def __init__(self, input_dim_feat, input_dim_target, hidden_dim_feat, output_dim, dropout_rate, lr, weight_decay):
        super().__init__()

        self.feat_mlp = graph_sequential('x, edge_index', [
            (BatchNorm(input_dim_feat), 'x -> x'),
            (Linear(input_dim_feat, hidden_dim_feat), 'x -> x'),
            nn.ReLU(),
#             nn.Dropout(p=dropout_rate),
            (Linear(hidden_dim_feat, output_dim), 'x -> x')
        ])

        self.target_mlp = nn.Sequential(
            nn.BatchNorm1d(input_dim_target),
            nn.Linear(input_dim_target, hidden_dim_feat),
            nn.ReLU(),
#             nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim_feat, output_dim)
        )

        self.decode_target = nn.Sequential(
            nn.Linear(output_dim, hidden_dim_feat),
            nn.BatchNorm1d(hidden_dim_feat),
            nn.ReLU(),
            nn.Linear(hidden_dim_feat, input_dim_target)
        )
        
        self.double() # attempt to fix File "/.local/lib/python3.9/site-packages/torch/nn/functional.py", line 2478, in batch_norm: return torch.batch_norm(: "RuntimeError: expected scalar type Double but found Float"
        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def transform_feat(self, x, edge_index, batch):
        features_1 = self.feat_mlp(x, edge_index)
        features_2 = global_mean_pool(features_1, batch)
        features_normalized = nn.functional.normalize(features_2, p=2, dim=1)
        return features_normalized

    def transform_target(self, y):
        targets = self.target_mlp(y)
        targets_normalized = nn.functional.normalize(targets, p=2, dim=1)
        return targets_normalized

    def decode_targets(self, embedding):
        return self.decode_target(embedding)

    def forward(self, x, y, edge_index, batch):
        
        x_embedding = self.transform_feat(x, edge_index, batch)

        y_embedding = self.transform_target(y)

        return x_embedding, y_embedding

    def initialize_optimizer(self, lr, weight_decay):
        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        return optimizer

class GraphData(graph_dataset):
    def __init__(self, path_feat, path_target, target_name, indices, to_line_graph=True, augmentations = None, threshold = 0, preprocess = True):
        
        self.path_feat = path_feat
        self.preproc_dir =join(self.path_feat, 'preprocessed') 
        self.indices = indices
        self.path_target = path_target
        self.augmentations = augmentations
        self.preprocess = preprocess
        self.to_line_graph = to_line_graph

        self.features = np.array([np.load(i) for i in self.raw_file_names[self.indices]])
        
        no_diag_features = [] # setting the diagonal to 0
        for feature in self.features:
            no_diag_feature = np.copy(feature)
            np.fill_diagonal(no_diag_feature, 0)
            no_diag_features.append(no_diag_feature)

        self.features = no_diag_features
        
        if threshold > 0:
            self.features = self._threshold(self.features, threshold)
            
        self.targets = np.expand_dims(
                pd.read_csv(path_target)[target_name].values[self.indices], axis=1
            )
    
        
        self.features, self.targets = self._preprocess(self.features, self.targets)
        gc.collect()

    @property
    def raw_file_names(self):
        raw_file_names = [join(self.path_feat, f) for f in listdir(self.path_feat) if isfile(join(self.path_feat, f))]
        raw_file_names = sorted(raw_file_names, key=lambda x: int(re.search(r'\d+', x).group()))
        raw_file_names = np.array(raw_file_names)
        return raw_file_names
    
    @property
    def preprocessed_file_names(self):
        if not os.path.exists(self.preproc_dir):
            os.makedirs(self.preproc_dir)
        preprocessed_file_names = [join(self.preproc_dir, f'xfeat_{idx}.pkl') for idx in self.indices]
        return preprocessed_file_names
    
    @property
    def show_indices(self):
        return self.indices
    
    @property
    def edge_idx_path(self):
        edge_idx_path = join(self.preproc_dir, 'edge_index.pkl')
        return edge_idx_path
    
    def _preprocess(self, features, targets):
        n_augs = 0
        if self.augmentations is not None:
            aug_samples = []
            if not isinstance(self.augmentations, list):
                self.augmentations = [self.augmentations]
                
            n_augs = len(self.augmentations)

        if self.preprocess:
            if self.augmentations is not None:
                ### AUGMENTATIONS
                print("preprocessing")
                for func in self.augmentations:
                    transform = augs[func]
                    transform_args = aug_args[func]
                    for sample in features:
                        aug_features = transform(sample, **transform_args)

                        aug_samples.append(aug_features)

                aug_samples = np.array(aug_samples)
                features = np.concatenate([features, aug_samples], axis=0)
                
            ### GRAPH CONSTRUCTION
            graphs = []
            xfeatures = []
            for sample in tqdm(features, desc="Processing samples"):
                graph, xfeat = self._make_graph(sample)
                graphs.append(graph)
                xfeatures.append(xfeat)
            self.save_preprocessed_features(graphs)
            print("Preprocessed node features saved.")
        else:
            xfeatures = self.load_preprocessed_featres()
        
        targets = np.concatenate([targets]*(n_augs + 1), axis=0)
        targets = np.array(targets)
        targets = torch.FloatTensor(targets)

        return xfeatures, targets
    
    def _threshold(self, matrices, threshold): # as in Margulies et al. (2016)
        perc = np.percentile(np.abs(matrices), threshold, axis=2, keepdims=True)
        mask = np.abs(matrices) >= perc
        thresh_mat = matrices * mask
        return thresh_mat
    
    def get_edge_indices(self, feat_sample):
#         idx_upper_tri = np.triu_indices_from(feat_sample, k=1)  # k=1 excludes the diagonal
#         source_nodes = idx_upper_tri[0]
#         target_nodes = idx_upper_tri[1]
        
        source_nodes, target_nodes = np.nonzero(feat_sample)
        edge_indices = np.vstack((source_nodes, target_nodes))
        return edge_indices

    def _make_graph(self, feat_sample):
        feat_sample = torch.tensor(feat_sample)
        num_nodes = feat_sample.size(0)
        edge_index = torch.triu_indices(*feat_sample.shape, offset=1)
        edge_attr = feat_sample[edge_index[0], edge_index[1]]
        graph = graph_data(edge_index=edge_index, edge_attr=edge_attr, num_nodes=num_nodes)
        ToUndir = ToUndirected()
        graph = ToUndir.forward(graph)
        graph = LineGraph()(copy(graph))
        print("Is line graph directed? ", graph.is_directed())
        return graph, graph.x
    
    def save_preprocessed_features(self, graphs):
        edge_index_path = join(self.preproc_dir, 'edge_index.pkl')
        for i, graph in enumerate(graphs):
            if not os.path.exists(edge_index_path):
                    graph_index = graph.edge_index
                    with open(edge_index_path, 'wb') as f:
                        pickle.dump(graph_index, f)
            xfeat = graph.x
            xfeat_path = self.preprocessed_file_names[i]
            with open(xfeat_path, 'wb') as f:
                pickle.dump(xfeat, f)

    def load_preprocessed_featres(self):
        xfeatures = []
        for xfeat_path in self.preprocessed_file_names:
            with open(xfeat_path, 'rb') as f:
                xfeat = pickle.load(f)
            xfeatures.append(xfeat)
#             graph = graph_data(edge_index=edge_index, x=xfeat, num_nodes=len(xfeat))
        return xfeatures
    
    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

def create_graph(node_features, edge_index, num_nodes):
    if node_features.dim() == 1:
        node_features = node_features.unsqueeze(1)
    return graph_data(x=node_features.to(torch.float64), edge_index=edge_index, num_nodes = num_nodes)

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

def train(train_dataset, test_dataset, edge_index_path, model=None, device=device, kernel=cauchy, num_epochs=100, batch_size=32):
    input_dim_feat = 1
    # the rest is arbitrary
    hidden_dim_feat = 64
   
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
        model = graph_MLP(
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
    
    with open(edge_index_path, 'rb') as f:
        edge_index = pickle.load(f)
    
    edge_index = edge_index.to(device, dtype=torch.long) # we store edge index only once
    
    with tqdm(range(num_epochs), desc="Epochs", leave=False) as pbar:
        for epoch in pbar:
            model.train()
            loss_terms_batch = defaultdict(lambda:0)
            for xfeat_list, targets in train_loader:
                targets = targets.to(device, dtype = float)
                data_list = []
                for features in xfeat_list:
                    features = features.to(device)
                    graph = create_graph(features, edge_index, len(features))
                    data_list.append(copy(graph))

                graph_batch = tg.data.Batch.from_data_list(data_list)
                batch = graph_batch.batch
                batch_xfeat = graph_batch.x
                batch_edge_index = graph_batch.edge_index
                
#                 bsz = targets.shape[0]
                optimizer.zero_grad()
                out_feat, out_target = model(batch_xfeat, targets, batch_edge_index, batch)
                
#                 out_feat, out_target = model(batch_xfeat, targets, batch_edge_index, batch)
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
                for xfeat_list, targets in test_loader:
                    data_list_test = []
                    targets = targets.to(device, dtype = float)
                    # print("targets test", targets.shape)
                    for features in xfeat_list:
                        features = features.to(device)
                        graph = create_graph(features, edge_index, len(features))
                        data_list_test.append(copy(graph))

                    graph_batch = tg.data.Batch.from_data_list(data_list_test)
                    batch = graph_batch.batch
                    batch_xfeat = graph_batch.x
                    batch_edge_index = graph_batch.edge_index
                    
                    out_feat = model.transform_feat(batch_xfeat, batch_edge_index, batch)
                    # print("out_feat", out_feat.shape)
                    out_target_decoded = model.decode_target(out_feat)
                    # print("out_target_decoded", out_target_decoded.shape)
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
    
    del edge_index
    torch.cuda.empty_cache()
    
    return loss_terms, model

class Experiment(submitit.helpers.Checkpointable):
    def __init__(self):
        self.results = None

    def __call__(self, train, test_size, indices, train_ratio, experiment_size, experiment, feat_path, target_path, target_name, threshold, random_state=None, device=None, path: Path = None):
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
            
            train_dataset = GraphData(feat_path, target_path, target_name, train_indices, to_line_graph=True, threshold = threshold,augmentations = AUGMENTATION, preprocess = False)
            test_dataset = GraphData(feat_path, target_path, target_name, test_indices,to_line_graph=True, threshold = threshold, augmentations = None, preprocess = False)
            
            edge_index_path = train_dataset.edge_idx_path
            loss_terms, model = train(train_dataset, test_dataset, model = None, device=device, edge_index_path = edge_index_path)
            losses.append(loss_terms.eval("train_ratio = @train_ratio").eval("experiment = @experiment"))
            
            
            
            model.eval()
            with torch.no_grad():
                
                train_dataset = GraphData(feat_path, target_path,target_name, train_indices, to_line_graph=True, threshold = threshold,augmentations = None, preprocess = False)
                train_loader = graph_dataloader(train_dataset, batch_size=32, shuffle=False)
                test_loader = graph_dataloader(test_dataset, batch_size=32, shuffle=False)
                
                with open(edge_index_path, 'rb') as f:
                    edge_index = pickle.load(f)

                edge_index = edge_index.to(device, dtype=torch.long)
                
                for label, loader, d_indices in (('train', train_loader, train_indices), ('test', test_loader, test_indices)):
                    preds = []
                    y = []
                    for xfeat_list, targets in loader:
                        targets = targets.to(device)
                        data_list = []
                        for features in xfeat_list:
                            features = features.to(device)
                            graph = create_graph(features, edge_index, len(features))
                            data_list.append(copy(graph))
                        
                        graph_batch = tg.data.Batch.from_data_list(data_list)
                        batch = graph_batch.batch
                        batch_xfeat = graph_batch.x
                        batch_edge_index = graph_batch.edge_index

                        y_pred = model.decode_target(model.transform_feat(batch_xfeat, batch_edge_index, batch))
                        preds.extend(y_pred)
                        y.extend(targets)
                    preds = torch.concat(preds)
                    y = torch.concat(y)
                    predictions[(train_ratio, experiment, label)] = (y.cpu().numpy(),preds.cpu().detach().numpy(), d_indices)

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

random_state = np.random.RandomState(seed=42)
#dataset = GraphData(path_feat, path_target, "age")
n_sub = 936
test_ratio = .2
test_size = int(test_ratio * n_sub)
indices = np.arange(n_sub)
experiments = 20
path_feat = "./matrices/schaefer400"
path_target = "participants.csv"
target_name = 'age'

# graph_dataset = GraphData(path_feat, path_target, target_name, [1, 2, 3,4], preprocess = False)

# edge_index_path = graph_dataset.edge_idx_path

# loader = graph_dataloader(graph_dataset, batch_size = 2)

# with open(edge_index_path, 'rb') as f:
#     edge_index = pickle.load(f)

# edge_index = edge_index.to(device, dtype=torch.long)

# for xfeat_list, targets in loader:
#     data_list = []
#     for features in xfeat_list:
#         features.to(device)
#         graph = create_graph(features, edge_index, len(features))
#         data_list.append(copy(graph))

# graph.edge_index

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
#         mem_gb=187
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
                job = executor.submit(run_experiment, train, test_size, indices, train_ratio, experiment_size, experiment, path_feat, path_target, target_name, THRESHOLD, random_state, device)
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
            job = run_experiment(train,  test_size, indices, train_ratio, experiment_size, experiment, path_feat, path_target, target_name, THRESHOLD, random_state, device)
            experiment_results.append(job)

losses, predictions = zip(*experiment_results)

prediction_metrics = predictions[0]
for prediction in predictions[1:]:
    prediction_metrics |= prediction
prediction_metrics = [
    k + (np.abs(v[0]-v[1]).mean(),)
    for k, v in prediction_metrics.items()
]
prediction_metrics = pd.DataFrame(prediction_metrics, columns=["train ratio", "experiment", "dataset", "MAE"])
prediction_metrics["train size"] = (prediction_metrics["train ratio"] * 936 * (1 - test_ratio)).astype(int)
# if AUGMENTATION is not None:
#     prediction_metrics["aug_args"] = str(aug_args)
prediction_metrics.to_csv(f"results/prediction_metrics_graph_thresh{THRESHOLD}.csv", index=False)