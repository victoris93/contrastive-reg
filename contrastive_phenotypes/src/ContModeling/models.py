import torch
import torch.nn as nn
from nilearn.connectome import sym_matrix_to_vec, vec_to_sym_matrix
import numpy as np

class PhenoProj(nn.Module):
    def __init__(
        self,
        input_dim_feat,
        input_dim_target,
        hidden_dim,
        output_dim_target,
        output_dim_feat,
        dropout_rate,
        cfg
    ):
        super(PhenoProj, self).__init__()

        self.input_dim_feat = input_dim_feat
        self.input_dim_target = input_dim_target
        self.hidden_dim = hidden_dim
        self.output_dim_target = output_dim_target
        self.output_dim_feat = output_dim_feat
        self.dropout_rate = dropout_rate
        self.cfg = cfg

        self.matrix_ae = MatAutoEncoder(input_dim_feat,
                                        output_dim_feat,
                                        dropout_rate,
                                        cfg)
        
        self.target_ae = TargetAutoEncoder(input_dim_target,
                                           hidden_dim,
                                           output_dim_target,
                                           dropout_rate,
                                           cfg)

        A = np.random.rand(output_dim_feat, output_dim_feat)
        A = (A + A.T) / 2
        self.vectorized_feat_emb_dim = len(sym_matrix_to_vec(A, discard_diagonal = True))

        self.init_weights(self.target_ae.encode_target)
        self.init_weights(self.target_ae.decode_target)
        
        self.feat_to_target_embedding = nn.Sequential(
            nn.Linear(self.vectorized_feat_emb_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, output_dim_target),
            
        )
        self.init_weights(self.feat_to_target_embedding)

    def init_weights(self, layer):
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0.0)

    def encode_features(self, x):
        return self.matrix_ae.encode_feat(x)
    
    def decode_features(self, embedding):
        return self.matrix_ae.decode_feat(embedding)
    
    def encode_targets(self, y):
        return self.target_ae.encode_targets(y)

    def decode_targets(self, embedding):
        return self.target_ae.decode_targets(embedding)
    
    def transfer_embedding(self, feat_embedding): # note that the feat embedding was vectorized
        feat_embedding_transfer = self.feat_to_target_embedding(feat_embedding)
        feat_embedding_transfer = nn.functional.normalize(feat_embedding_transfer, p=2, dim=1)
        return feat_embedding_transfer

    def forward(self, x, y):
        x_embedding = self.encode_features(x)
        y_embedding = self.encode_targets(y)
        return x_embedding, y_embedding


class MatAutoEncoder(nn.Module):
    def __init__(
        self,
        input_dim_feat,
        output_dim_feat,
        dropout_rate,
        cfg
    ):
        super(MatAutoEncoder, self).__init__()
        self.cfg = cfg
        # ENCODE MATRICES
        self.enc_mat1 = nn.Linear(in_features=input_dim_feat, out_features=output_dim_feat ,bias=False)
        self.enc_mat2 = nn.Linear(in_features=input_dim_feat, out_features=output_dim_feat, bias=False)
        self.enc_mat2.weight = torch.nn.Parameter(self.enc_mat1.weight) # Here weights are B_init_MRI.T
        
        # DECODE MATRICES
        self.dec_mat1 = nn.Linear(in_features=output_dim_feat, out_features=input_dim_feat, bias=False)
        self.dec_mat2 = nn.Linear(in_features=output_dim_feat, out_features=input_dim_feat, bias=False)
        self.dec_mat1.weight = torch.nn.Parameter(self.enc_mat1.weight.transpose(0,1)) # Here weights are B_init_MRI
        self.dec_mat2.weight = torch.nn.Parameter(self.dec_mat1.weight) # Here weights are B_init_MRI
        self.dropout = nn.Dropout(p=dropout_rate)
        
    def encode_feat(self, x):
        rect = self.cfg.ReEig
        z_n = self.enc_mat1(x)
        self.skip_enc_mat1 = z_n.detach().clone()
        c_hidd_fMRI = self.enc_mat2(z_n.permute(0, 2, 1))

        if rect:
            reig = ReEig()
            c_hidd_fMRI = reig(c_hidd_fMRI)
        
        return c_hidd_fMRI
    
    def decode_feat(self,c_hidd_mat):
        skip_enc1 = self.cfg.skip_enc1
        z_n = self.dec_mat1(c_hidd_mat).permute(0, 2, 1)

        if skip_enc1: # long skip conn
            z_n += self.skip_enc_mat1
        
        recon_mat = self.dec_mat2(z_n)
        return recon_mat


class TargetAutoEncoder(nn.Module):
    def __init__(
        self,
        input_dim_target,
        hidden_dim,
        output_dim_target,
        dropout_rate,
        cfg # just in case
    ):
        super(TargetAutoEncoder, self).__init__()

        self.encode_target = nn.Sequential(
            nn.Linear(input_dim_target, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, output_dim_target),
        )

        self.decode_target = nn.Sequential(
            nn.Linear(output_dim_target, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, input_dim_target),
        )

    def encode_targets(self, y):
        target_embedding = self.encode_target(y)
        target_embedding = nn.functional.normalize(target_embedding, p=2, dim=1) # do we want to do this?
        return target_embedding

    def decode_targets(self, target_embedding):
        return self.decode_target(target_embedding)


class ReEig(nn.Module):
    def __init__(self, epsilon=1e-4):
        super(ReEig, self).__init__()
        self.epsilon = epsilon

    def forward(self, X):
        D, V = torch.linalg.eigh(X)
        D = torch.clamp(D, min=self.epsilon)
        X_rectified = V @ torch.diag_embed(D) @ V.transpose(-2, -1)
        
        return X_rectified
