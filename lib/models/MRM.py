from core.config import cfg
import os
import numpy as np
import torch
import os.path as osp
import torch.nn as nn
import torch.nn.functional as F
import copy,math
from models.ChebConv import ChebConv, _ResChebGC
from models.GraFormer import MultiHeadedAttention, GraphNet, GraAttenLayer, MLP_SE, PositionwiseFeedForward



BASE_DATA_DIR = cfg.DATASET.BASE_DATA_DIR
# SMPL_MEAN_PARAMS = osp.join(BASE_DATA_DIR, 'smpl_mean_params.npz')
SMPL_MEAN_vertices = osp.join(BASE_DATA_DIR, 'smpl_mean_vertices.npy')


class Meshnet(nn.Module):
    def __init__(self, num_joints=17, embed_dim=128, num_layers=5,
                 n_head=4, dropout=0.25):
        super(Meshnet, self).__init__()
        self.num_joints = num_joints
        self.embed_dim = embed_dim
        self.n_layers = num_layers

        init_vertices = torch.from_numpy(np.load(SMPL_MEAN_vertices)).unsqueeze(0)
        self.register_buffer('init_vertices', init_vertices)

        self.up_feature = nn.Linear(embed_dim, embed_dim)
        self.up_linear = nn.Linear(689 * 2, embed_dim)

        #_mlpse_layers = []
        _mlp_layers = []
        _attention_layer = []

        c = copy.deepcopy
        attn = MultiHeadedAttention(n_head, embed_dim)
        gcn = GraphNet(in_features=embed_dim, out_features=embed_dim, n_pts=self.num_joints + 15)

        for i in range(num_layers):
            #_mlpse_layers.append(MLP_SE(embed_dim, self.num_joints + 15, embed_dim))
            _mlp_layers.append(PositionwiseFeedForward(128, 256, 0.25))
            _attention_layer.append(GraAttenLayer(embed_dim, c(attn), c(gcn), dropout))

        #self.mlpse_layers = nn.ModuleList(_mlpse_layers)
        self.mlp_layers = nn.ModuleList(_mlp_layers)
        self.atten_layers = nn.ModuleList(_attention_layer)

        self.gelu = nn.GELU()
        self.norm = nn.LayerNorm(embed_dim)

        self.d_conv = nn.Conv1d(self.num_joints + 15, 3, kernel_size=3, padding=1)
        self.d_linear = nn.Linear(self.embed_dim, 6890)

        self.mask = torch.ones(1, 1, self.num_joints + 15, dtype=torch.bool).cuda()


    def forward(self, feature):
        B = feature.shape[0]
        feature = feature.view(B, self.num_joints, -1)
        feature = self.up_feature(feature)

        init_vertices = self.init_vertices.expand(B, -1, -1)
        mean_smpl = self.up_linear(init_vertices.transpose(1,2).reshape(B, 15, -1))

        x = torch.cat((feature, mean_smpl), dim=1)

        for i in range(self.n_layers):
            x = self.atten_layers[i](x, self.mask)
            x = self.mlp_layers[i](x)

        x = self.norm(x)

        ####### after mesh regression module, x_out [B, J+T, emb_dim]
        x_out = x
        x_out = self.d_linear(x_out)
        x_out = self.gelu(x_out)
        x_out = self.d_conv(x_out).transpose(1, 2)
        x_out = x_out + init_vertices

        return x_out

def get_model(num_joint=17, embed_dim=128, num_layers=4, n_head=4, dropout=0.25):
    model = Meshnet(num_joint, embed_dim, num_layers, n_head, dropout)
    return model

