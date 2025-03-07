from cmath import isinf
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from sklearn.preprocessing import MinMaxScaler

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
        reg_term: float = 1.0,
        krnl_sigma: float = 1.0,
        kernel: callable = None,
        delta_reduction: str = "sum",
    ):
        super().__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.reg_term = reg_term
        self.method = method
        self.kernel = kernel
        self.krnl_sigma = krnl_sigma
        self.delta_reduction = delta_reduction

        if kernel is not None and method == "supcon":
            raise ValueError("Kernel must be none if method=supcon")

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
    
    def direction_reg(self, features): # reg term is gamma in Mohan et al. 2020
        feat_mask = self.kernel(features, krnl_sigma=self.krnl_sigma)
        direction_reg = self.reg_term * feat_mask
        return direction_reg

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

            # if self.kernel is None:
            #     scaler = MinMaxScaler()
            #     mask = -torch.cdist(labels, labels)
            #     mask = scaler.fit_transform(mask.cpu().numpy())
            #     mask = torch.tensor(mask, device=device).to(torch.float64)
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
        # print("anchor_dot_contrast", anchor_dot_contrast)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        # print("logits_max", logits_max)
        logits = anchor_dot_contrast - logits_max.detach()
        # print("logits", logits)

        alignment = logits

        # base case is:
        # - supcon if kernel = none
        # - y-aware is kernel != none
        uniformity = torch.exp(logits) * inv_diagonal
        # print("uniformity", uniformity)

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
            uniformity = (torch.exp(logits * (1 - mask)) + 1e-6) * inv_diagonal

        # print("uniformity", uniformity)

        uniformity = torch.log(uniformity.sum(1, keepdim=True))
        # print("uniformity", uniformity)

        # positive mask contains the anchor-positive pairs
        # excluding <self,self> on the diagonal
        positive_mask = mask  * inv_diagonal
        # print("positive_mask", positive_mask)
        direction_reg = torch.abs((self.direction_reg(features) * inv_diagonal - positive_mask).sum(1))

        log_prob = (
            alignment - uniformity # this is not in the formula
        )  # log(alignment/uniformity) = log(alignment) - log(uniformity)
        # print("log_prob", log_prob)
        log_prob = (positive_mask * log_prob).sum(1) / positive_mask.sum(
            1
        )  # compute mean of log-likelihood over positive

        # print("log_prob", log_prob.mean())
        # loss

        loss = -(self.temperature / self.base_temperature) * log_prob

        return loss.mean(), direction_reg.mean()


class OutlierRobustMSE(nn.Module):
    def __init__(self, lmbd = 0.5):
        super(OutlierRobustMSE, self).__init__()
        self.lmbd = lmbd

    def forward(self, targets, pred):
        targets_standardized = torch.abs((targets - 100) / 15)
        base_mse = nn.functional.mse_loss(targets, pred)
        penalty = torch.exp(1 + targets_standardized).sum() * (1 - 1/(torch.abs(targets-pred)/targets).sum())
        mse_with_outlier_penalty = base_mse + penalty
        return mse_with_outlier_penalty


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


