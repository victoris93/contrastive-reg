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
from augmentations import augs


torch.cuda.empty_cache()
multi_gpu = True
# %%
# data_path = Path('~/research/data/victoria_mat_age/data_mat_age_demian').expanduser()
# -


# %%
THRESHOLD = 0
AUGMENTATION = None
# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# %%
node = 1000

class GPC(nn.Module):

	def __init__(self, in_dim, out_dim):
		super(GPC, self).__init__()
		self.out_dim = out_dim
		self.conv = nn.Conv2d(in_dim, out_dim, (1, node))
		nn.init.normal_(self.conv.weight, std=math.sqrt(2/(node*in_dim+node*out_dim)))

	def forward(self, x):
		batchsize = x.shape[0]

		x_c = self.conv(x)
		x_C = x_c.expand(batchsize, self.out_dim, node, node)
		x_R = x_C.permute(0,1,3,2)
		x = x_C+x_R

		return x

class GPC_MLP(nn.Module):
    def __init__(
        self,
        input_dim_feat,
        input_dim_target,
        hidden_dim_feat,
        output_dim,
        dropout_rate,
    ):
        super(GPC_MLP, self).__init__()

        # Xavier initialization for feature MLP
        # self.feat_mlp = nn.Sequential(
        #     nn.BatchNorm1d(input_dim_feat),
        #     nn.Linear(input_dim_feat, hidden_dim_feat),
        #     nn.BatchNorm1d(hidden_dim_feat),
        #     nn.ReLU(),
        #     nn.Dropout(p=dropout_rate),
        #     nn.Linear(hidden_dim_feat, hidden_dim_feat),
        #     nn.BatchNorm1d(hidden_dim_feat),
        #     nn.ReLU(),
        #     nn.Dropout(p=dropout_rate),
        #     nn.Linear(hidden_dim_feat, output_dim),
        # )
        self.gpc = GPC(1, 1)

        self.feat_mlp =nn.Sequential(
            nn.BatchNorm1d(input_dim_feat),
            nn.Linear(input_dim_feat, hidden_dim_feat),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim_feat, output_dim)
        )
        self.init_weights(self.feat_mlp)

        # Xavier initialization for target MLP
        self.target_mlp = nn.Sequential(
            nn.BatchNorm1d(input_dim_target),
            nn.Linear(input_dim_target, hidden_dim_feat),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim_feat, output_dim)
        )
        self.init_weights(self.target_mlp)

        self.decode_target = nn.Sequential(
            nn.Linear(output_dim, hidden_dim_feat),
            nn.BatchNorm1d(hidden_dim_feat),
            nn.ReLU(),
            nn.Linear(hidden_dim_feat, input_dim_target)
        )
        self.init_weights(self.decode_target)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0.0)

    def transform_feat(self, x):
        print("X: ", x.shape)
        gpc_features = self.gpc(x)
        print("gpc_features: ", gpc_features.shape)
        gpc_features = torch.from_numpy(sym_matrix_to_vec(gpc_features.detach().cpu().numpy(), discard_diagonal=True)).float().to(device)
        x_vect = torch.from_numpy(sym_matrix_to_vec(x.detach().cpu().numpy(), discard_diagonal=True)).float().to(device)
        feat = gpc_features + x_vect
        print("feat: ", feat.shape)
        features = self.feat_mlp(feat.squeeze(1))
        print("features: ", features.shape)
        features = nn.functional.normalize(features, p=2, dim=1)
        return features
    
    def transform_targets(self, y):
        targets = self.target_mlp(y)
        targets = nn.functional.normalize(targets, p=2, dim=1)
        return targets
 
    def decode_targets(self, embedding):
        return self.decode_target(embedding)

    def forward(self, x, y):
       x_embedding = self.transform_feat(x)
       y_embedding = self.transform_targets(y)
       return x_embedding, y_embedding
# %%
class MatData(Dataset):
    def __init__(self, path_feat, path_targets, target_name, threshold=0):
        # self.matrices = np.load(path_feat, mmap_mode="r")
        self.matrices = np.load(path_feat, mmap_mode="r").astype(np.float32)
        no_diag_matrices = []
        for matrix in self.matrices:
            no_diag_feature = np.copy(matrix)
            np.fill_diagonal(no_diag_feature, 0)
            no_diag_matrices.append(no_diag_feature)
        self.matrices = np.array(no_diag_matrices)

        self.target = torch.tensor(
            np.expand_dims(
                pd.read_csv(path_targets)[target_name].values, axis=1
            ),
            dtype=torch.float32
        )
        if threshold > 0:
            self.matrices = self.threshold(self.matrices, threshold)
        # if threshold > 0: 
        #     thrs = np.quantile(np.abs(self.matrices), q=threshold, axis=1, keepdims=True)
        #     self.matrices = self.matrices * (np.abs(self.matrices) >= thrs)
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
    input_dim_feat = 499500
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
        model = GPC_MLP(
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
            for features, targets in train_loader:
                bsz = targets.shape[0]
                n_views = features.shape[1]
                n_feat = features.shape[-1]
                
                optimizer.zero_grad()
                features = features.to(device)
                targets = targets.to(device)
                out_feat, out_target = model(features, targets)
                joint_embedding = nn.functional.mse_loss(out_feat, out_target)
                kernel_feature = criterion_pft(out_feat.unsqueeze(1), targets.unsqueeze(1))

                out_target_decoded = model.decode_target(out_target)

                kernel_target = criterion_ptt(out_target.unsqueeze(1), targets.unsqueeze(1))
                #joint_embedding = 1000 * nn.functional.cosine_embedding_loss(out_feat, out_target, cosine_target)
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
                        
                    features, targets = features.to(device), targets.to(device)
                    out_feat = model.transform_feat(features)
                    out_target_decoded = model.decode_target(out_feat.squeeze(1))
                    
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

    def __call__(self, train, test_size, indices, train_ratio, experiment_size, experiment, dataset, random_state=None, device=None, path: Path = None):
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
            experiment_indices = random_state.choice(indices, experiment_size, replace=False)
            train_indices, test_indices = train_test_split(experiment_indices, test_size=test_size, random_state=random_state)
            train_dataset = Subset(dataset, train_indices)
            test_dataset = Subset(dataset, test_indices)
            ### Augmentation
            n_views = 1
            train_features = train_dataset.dataset.matrices[train_dataset.indices].numpy()
            train_targets = train_dataset.dataset.target[train_dataset.indices].numpy()

            test_features= test_dataset.dataset.matrices[test_dataset.indices].numpy()
            test_targets = test_dataset.dataset.target[test_dataset.indices].numpy()

            if AUGMENTATION is not None:
                transform = augs[AUGMENTATION]
                aug_features = np.array([transform(sample) for sample in train_features])

                train_features = sym_matrix_to_vec(train_features, discard_diagonal=True)
                aug_features = sym_matrix_to_vec(train_features, discard_diagonal=True)

                n_views = n_views + aug_features.shape[1]
                n_features = train_features.shape[-1]
                n_samples = len(train_dataset)

                new_train_features = np.zeros((n_samples, n_views, n_features))
                new_train_features[:, 0, :] = train_features
                new_train_features[:, 1:, :] = aug_features
            else:

                train_features = np.expand_dims(train_features, axis = 1)
                test_features = np.expand_dims(test_features, axis = 1)

            train_dataset = TensorDataset(torch.from_numpy(train_features).to(torch.float32), torch.from_numpy(train_targets).to(torch.float32))
            test_dataset = TensorDataset(torch.from_numpy(test_features).to(torch.float32), torch.from_numpy(test_targets).to(torch.float32))

            loss_terms, model = train(train_dataset, test_dataset, device=device)
            losses.append(loss_terms.eval("train_ratio = @train_ratio").eval("experiment = @experiment"))
            model.eval()
            
            with torch.no_grad():
                for label, d, d_indices in (('train', train_dataset, train_indices), ('test', test_dataset, test_indices)):
                    X, y = zip(*d)
                    X = torch.stack(X).to(device)
                    y = torch.stack(y).to(device)
                    y_pred = model.decode_target(model.transform_feat(X).squeeze(1))
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
# %%
random_state = np.random.RandomState(seed=42)
dataset = MatData("matrices.npy", "participants.csv", "age", threshold=THRESHOLD)
n_sub = len(dataset)
test_ratio = .2
test_size = int(test_ratio * n_sub)
indices = np.arange(n_sub)
experiments = 20
# %%
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
                job = executor.submit(run_experiment, train, test_size, indices, train_ratio, experiment_size, experiment, dataset, random_state=random_state, device=None)
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
            job = run_experiment(train,  test_size, indices, train_ratio, experiment_size, experiment, dataset, random_state=random_state, device=None)
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

prediction_metrics.to_csv(f"results/prediction_metrics_GPC_MLP.csv", index=False)
