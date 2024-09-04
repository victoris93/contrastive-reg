import torch
import torch.nn as nn


class AutoEncoder(nn.Module):
    def __init__(
        self,
        input_dim_feat,
        output_dim_feat,
        B_init_fMRI,
        dropout_rate,
        cfg
    ):
        super(AutoEncoder, self).__init__()
        self.cfg = cfg
        # ENCODE MATRICES
        self.enc_mat1 = nn.Linear(in_features=input_dim_feat, out_features=output_dim_feat ,bias=False)
        self.enc_mat2 = nn.Linear(in_features=input_dim_feat, out_features=output_dim_feat, bias=False)
        self.enc_mat2.weight = torch.nn.Parameter(self.enc_mat1.weight)
        
        # DECODE MATRICES
        self.dec_mat1 = nn.Linear(in_features=output_dim_feat, out_features=input_dim_feat, bias=False)
        self.dec_mat2 = nn.Linear(in_features=output_dim_feat, out_features=input_dim_feat, bias=False)
        self.dec_mat1.weight = torch.nn.Parameter(self.enc_mat1.weight.transpose(0,1))
        self.dec_mat2.weight = torch.nn.Parameter(self.dec_mat1.weight)
        self.dropout = nn.Dropout(p=dropout_rate)
        
    def encode_feat(self, x):

        rect = self.cfg.ReEig
        z_n = self.enc_mat1(x)
        self.skip_enc_mat1 = z_n.detach().clone()

        c_hidd_fMRI = self.enc_mat2(z_n.transpose(1,2))
        if rect:
            reig = ReEig()
            c_hidd_fMRI = reig(c_hidd_fMRI)
        
        return c_hidd_fMRI
    
    def decode_feat(self,c_hidd_mat):

        skip_enc1 = self.cfg.skip_enc1
        z_n = self.dec_mat1(c_hidd_mat).transpose(1,2)

        if skip_enc1: # long skip conn
            z_n += self.skip_enc_mat1
        
        recon_mat = self.dec_mat2(z_n)

        return recon_mat


class ReEig(nn.Module):
    def __init__(self, epsilon=1e-4):
        super(ReEig, self).__init__()
        self.epsilon = epsilon

    def forward(self, X):
        D, V = torch.linalg.eigh(X)
        D = torch.clamp(D, min=self.epsilon)
        X_rectified = V @ torch.diag_embed(D) @ V.transpose(-2, -1)
        
        return X_rectified
