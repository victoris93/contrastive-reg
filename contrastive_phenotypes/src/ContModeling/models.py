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
        
        self.reduced_matrix_ae = ReducedMatAutoEncoder(output_dim_feat,
                                                       hidden_dim,
                                                       output_dim_target,
                                                       dropout_rate,
                                                       cfg)
        

        self.target_dec = TargetDecoder(input_dim_target,
                                           hidden_dim,
                                           output_dim_target,
                                           dropout_rate,
                                           cfg)
        

        A = np.random.rand(output_dim_feat, output_dim_feat)
        A = (A + A.T) / 2
        self.vectorized_feat_emb_dim = len(sym_matrix_to_vec(A))        

    def init_weights(self, layer):
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0.0)

    def encode_features(self, x):
        return self.matrix_ae.encode_feat(x)
    
    def decode_features(self, embedding):
        return self.matrix_ae.decode_feat(embedding)
    
    def encode_reduced_mat(self, feat_embedding): # note that the feat embedding was vectorized
        embedding, embedding_norm = self.reduced_matrix_ae.embed_reduced_mat(feat_embedding)
        return embedding, embedding_norm

    def decode_reduced_mat(self, embedding): # invert the embedding from target space to feature space to reconstruct the matrix
        recon_reduced_mat = self.reduced_matrix_ae.recon_reduced_mat(embedding)
        return recon_reduced_mat
    
    def decode_targets(self, embedding):
        return self.target_dec.decode_targets(embedding)
        
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
        # nn.init.xavier_uniform_(self.enc_mat1.weight)
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
        
class ReducedMatAutoEncoder(nn.Module):
    def __init__(
        self,
        output_dim_feat,
        hidden_dim,
        output_dim_target,
        dropout_rate,
        cfg
    ):
        super(ReducedMatAutoEncoder, self).__init__()
        self.cfg = cfg
        self.skip_conn = self.cfg.skip_conn
        
        A = np.random.rand(output_dim_feat, output_dim_feat)
        A = (A + A.T) / 2
        self.vectorized_feat_emb_dim = len(sym_matrix_to_vec(A))
        
        self.reduced_mat_to_embed = nn.Sequential(
            nn.Linear(self.vectorized_feat_emb_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ELU(),
            nn.Dropout(p=dropout_rate),

            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ELU(),
            nn.Dropout(p=dropout_rate),

            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ELU(),
            nn.Dropout(p=dropout_rate),

            nn.Linear(hidden_dim, output_dim_target) 
        )
        
        self.init_weights(self.reduced_mat_to_embed)
        
        self.embed_to_reduced_mat = nn.Sequential(
            nn.Linear(output_dim_target, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ELU(),
            nn.Dropout(p=dropout_rate),

            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ELU(),
            nn.Dropout(p=dropout_rate),

            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ELU(),
            nn.Dropout(p=dropout_rate),

            nn.Linear(hidden_dim, self.vectorized_feat_emb_dim)
        )

        self.init_weights(self.embed_to_reduced_mat)
    
    def init_weights(self, layer):
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0.0)
        
    def embed_reduced_mat(self, reduced_mat):
        if self.skip_conn:
            self.original_reduced_mat = reduced_mat
        feat_embedding = self.reduced_mat_to_embed(reduced_mat)
        feat_embedding_norm = nn.functional.normalize(feat_embedding, p=2, dim=1)
        return feat_embedding, feat_embedding_norm

    def recon_reduced_mat(self, feat_embedding):
        recon_reduced_mat = self.embed_to_reduced_mat(feat_embedding)
        if self.skip_conn:
            recon_reduced_mat = self.original_reduced_mat + recon_reduced_mat
        return recon_reduced_mat

    def forward(self, reduced_mat):
        feat_embedding, feat_embedding_norm = self.reduced_mat_to_embed(reduced_mat)
        recon_reduced_mat = self.recon_reduced_mat(feat_embedding)
        return recon_reduced_mat, feat_embedding, feat_embedding_norm

class TargetDecoder(nn.Module):
    def __init__(
        self,
        input_dim_target,
        hidden_dim,
        output_dim_target,
        dropout_rate,
        cfg # just in case
    ):
        super(TargetDecoder, self).__init__()

        self.decode_target = nn.Sequential(
            nn.Linear(output_dim_target, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ELU(),
            nn.Dropout(p=dropout_rate),

            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ELU(),
            nn.Dropout(p=dropout_rate),

            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ELU(),
            nn.Dropout(p=dropout_rate),

            nn.Linear(hidden_dim, input_dim_target),
        )
        
        self.init_weights(self.decode_target)

    def init_weights(self, layer):
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0.0)

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

