import sys
import numpy as np
import pandas as pd
import torch
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Data as graph_data
import os
import pickle
import gc
from tqdm import tqdm
from os import listdir, makedirs
from os.path import isfile, join
import re
from torch_geometric.data import Dataset as graph_dataset
from copy import copy
from torch_geometric.transforms.line_graph import LineGraph


feat_path = "./matrices"
target_path = "/data/parietal/store/work/dwassermann/data/victoria_mat_age/data_mat_age_demian/participants.csv"
target_name = "age"

index = int(sys.argv[1])

class GraphData(graph_dataset):
    def __init__(self, path_feat, path_target, target_name, indices, to_line_graph=True, augmentations = None, threshold = 0, processed_file_paths=None):
        
        self.path_feat = path_feat
        self.path_target = path_target
        self.augmentations = augmentations
        self.to_line_graph = to_line_graph
        
        self.processed_file_paths = processed_file_paths

        self.features = np.array([np.load(i) for i in self.raw_file_names[indices]])
        
        if threshold > 0:
            self.features = self._threshold(self.features, threshold)
            
        self.targets = np.expand_dims(
                pd.read_csv(path_target)[target_name].values[indices], axis=1
            )

        self.graphs, self.targets = self.process(self.features, self.targets)
        self.targets = torch.FloatTensor(self.targets)
        
        if self.processed_file_paths is None:
            self.graphs, self.targets = self.process(self.features, self.targets)
            self.save_preprocessed_graphs()
        else:
            self.load_preprocessed_graphs()
        
        self.targets = torch.FloatTensor(self.targets)
        
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
        return graphs, targets
    
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

    def make_graph(self, feat_sample):
        #edge_index = self.get_edge_indices(feat_sample)
        feat_sample = torch.tensor(feat_sample)
        num_nodes = feat_sample.size(0)
        x = torch.ones(num_nodes,1)
        edge_index, edge_attr = dense_to_sparse(feat_sample)
        graph = graph_data(x=x, edge_index=edge_index, edge_attr = edge_attr)
        if self.to_line_graph:
            graph = self._to_line_graph(copy(graph))
        return graph
    
    def _to_line_graph(self,graph):
        lgf = LineGraph()
        line_graph = lgf.forward(copy(graph))
        return line_graph
    
    def save_preprocessed_graphs(self):
        os.makedirs('./preprocessed', exist_ok=True)
        self.processed_file_paths = []
        for idx, graph in enumerate(self.graphs):
            file_path = f'./preprocessed/graph_{idx}.pkl'
            with open(file_path, 'wb') as f:
                pickle.dump(graph, f)
            self.processed_file_paths.append(file_path)

    def load_preprocessed_graphs(self):
        self.graphs = []
        for file_path in self.processed_file_paths:
            with open(file_path, 'rb') as f:
                graph = pickle.load(f)
            self.graphs.append(graph)
    
    
    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx], self.targets[idx]
    
    
    
graph_data_instance = GraphData(feat_path, target_path, target_name, [index])

print(f'Processed graph for index {index}')