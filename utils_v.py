# from: https://github.com/EIDOSLAB/contrastive-brain-age-prediction/blob/master/src/util.py
import math
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
import random
import numpy as np
import os
# import wandb
from sklearn.decomposition import PCA
import torch.nn.functional as F
from pathlib import Path
from scipy.stats import pearsonr, spearmanr

def mean_correlations_between_subjects(y_true, y_pred):
    correlations = []
    y_true = y_true.cpu().detach().numpy()  # Convert to NumPy array if using PyTorch tensor
    y_pred = y_pred.cpu().detach().numpy()  # Convert to NumPy array if using PyTorch tensor
    
    num_subjects = y_true.shape[0]
    matrix_size = y_true.shape[1]
    
    # Flatten upper triangle (excluding diagonal) for both y_true and y_pred
    upper_true = []
    upper_pred = []
    
    for subj in range(num_subjects):
        for i in range(matrix_size):
            for j in range(i + 1, matrix_size):
                upper_true.append(y_true[subj, i, j])
                upper_pred.append(y_pred[subj, i, j])
    
    # Calculate Spearman correlation
    spearman_corr, _ = spearmanr(upper_true, upper_pred)
    correlations.append(spearman_corr)
    correlation = np.mean(correlations)
    
    return correlation

def mean_correlation(y_true, y_pred):
    correlations = []
    y_true = y_true.cpu().detach().numpy()
    y_pred = y_pred.cpu().detach().numpy()
    for i in range(y_true.shape[0]):
        corr, _ = spearmanr(y_true[i].flatten(), y_pred[i].flatten())
        correlations.append(corr)
    return np.mean(correlations)

def get_best_fold(train_fold_results):
    folds = [fold_dict["fold"] for fold_dict in train_fold_results]
    val_losses = [fold_dict["val_loss"] for fold_dict in train_fold_results]
    best_fold = folds[val_losses.index(np.min(val_losses))]
    print("BEST FOLD IS: ", best_fold)
    return best_fold

def mape_between_subjects(y_true, y_pred):
    eps = 1e-6
    mapes = []
    y_true = y_true.cpu().detach().numpy()  # Convert to NumPy array if using PyTorch tensor
    y_pred = y_pred.cpu().detach().numpy()  # Convert to NumPy array if using PyTorch tensor

    num_subjects = y_true.shape[0]
    matrix_size = y_true.shape[1]

    # Flatten upper triangle (excluding diagonal) for both y_true and y_pred
    upper_true = []
    upper_pred = []

    for subj in range(num_subjects):
        for i in range(matrix_size):
            for j in range(i + 1, matrix_size):
                true_val = y_true[subj, i, j]
                pred_val = y_pred[subj, i, j]

                # Add epsilon to denominator to avoid division by zero
                mape = np.abs((true_val - pred_val) / (true_val + eps)) * 100.0
                upper_true.append(true_val)
                upper_pred.append(pred_val)
                mapes.append(mape)

    # Calculate mean MAPE
    mean_mape = np.mean(mapes)

    return mean_mape

def standardize_dataset(dataset):
    features = torch.vstack([dataset[i][0] for i in range(len(dataset))])
    targets = torch.vstack([dataset[i][1] for i in range(len(dataset))])
    
    features_mean = features.mean(dim=0)
    features_std = features.std(dim=0)
    targets_mean = targets.mean(dim=0)
    targets_std = targets.std(dim=0)
    
    features_std[features_std == 0] = 1
    targets_std[targets_std == 0] = 1
    
    standardized_features = (features - features_mean) / features_std
    standardized_targets = (targets - targets_mean) / targets_std
    
    standardized_dataset = TensorDataset(standardized_features, standardized_targets)
    
    return standardized_dataset

def pca_labels(dataset, n_components):
    features = torch.vstack([dataset[i][0] for i in range(len(dataset))])
    targets = torch.vstack([dataset[i][1] for i in range(len(dataset))])

    targets_np = targets.numpy()
    pca = PCA(n_components=n_components)
    reduced_labels_np = pca.fit_transform(targets_np)
    reduced_labels = torch.tensor(reduced_labels_np, dtype=torch.float32)
    reduced_labels = nn.functional.normalize(reduced_labels, p=2, dim=1)
    dataset = TensorDataset(features, reduced_labels)

    return dataset

class NViewTransform:
    """Create N augmented views of the same image"""
    def __init__(self, transform, N):
        self.transform = transform
        self.N = N

    def __call__(self, x):
        return [self.transform(x) for _ in range(self.N)]

def arg2bool(val):
    if isinstance(val, bool):
        return val
    
    elif isinstance(val, str):
        if val == "true":
            return True
        
        if val == "false":
            return False
    
    val = int(val)
    assert val == 0 or val == 1
    return val == 1

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class MAE():
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.outputs = []
        self.targets = []
        self.avg = np.inf
    
    def update(self, outputs, targets):
        self.outputs.append(outputs.detach())
        self.targets.append(targets.detach())
        self.avg = F.l1_loss(torch.cat(self.outputs, 0), torch.cat(self.targets, 0))

class Accuracy():
    def __init__(self, topk=(1,)):
        self.reset()
        self.topk = topk
    
    def reset(self):
        self.outputs = []
        self.targets = []
        self.avg = np.inf
    
    def update(self, outputs, targets):
        self.outputs.append(outputs.detach())
        self.targets.append(targets.detach())
        self.avg = accuracy(torch.cat(self.outputs, 0), torch.cat(self.targets, 0), self.topk)

def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=True)

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return res

def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(seed)

def save_model(model, cv_fold, optimizer, save_file):
    print('==> Saving...')
    state_dict = model.state_dict()
    if torch.cuda.device_count() > 1:
        state_dict = model.module.state_dict()

    state = {
        'model': state_dict,
        'optimizer': optimizer.state_dict(),
        'cv_fold': cv_fold
    }
    torch.save(state, save_file)
    del state

def adjust_learning_rate(args, optimizer, epoch):
    lr = args.lr
    if args.lr_decay == 'cosine':
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
    if args.warm and epoch <= args.warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / \
            (args.warm_epochs * total_batches)
        lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

@torch.no_grad()
def gather_feats_targets(model, dataloader, device):
    features = []
    targets = []

    model.eval()
    for idx, (feat, target) in enumerate(dataloader):
        if len(feat.shape) > 2:
            feat = feat[:, 0, :]
        feat = feat.to(device)
        target = target.to(device)
        out_feat, out_target= model(feat, target)
        features.append(out_feat)
        targets.append(out_target)
    
    return torch.cat(features, 0).cpu().numpy(), torch.cat(targets, 0).cpu().numpy()
