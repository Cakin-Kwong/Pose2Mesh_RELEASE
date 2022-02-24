from core.config import cfg
import os
import numpy as np
import torch
import os.path as osp
import torch.nn as nn
import torch.nn.functional as F
import copy,math
from models.ChebConv import ChebConv, _ResChebGC
from models.GraFormer import MultiHeadedAttention, GraphNet, GraAttenLayer



BASE_DATA_DIR = cfg.DATASET.BASE_DATA_DIR
# SMPL_MEAN_PARAMS = osp.join(BASE_DATA_DIR, 'smpl_mean_params.npz')
SMPL_MEAN_vertices = osp.join(BASE_DATA_DIR, 'smpl_mean_vertices.npy')


class MLP_SE(nn.Module):
    def __init__(self, in_features, in_channel, hidden_features=None):
        super().__init__()
        self.in_channel = in_channel
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, in_features)

        self.fc_down1 = nn.Linear(in_features*in_channel, in_channel)
        self.fc_down2 = nn.Linear(in_channel, 2*in_channel)
        self.fc_down3 = nn.Linear(2*in_channel, in_channel)
        self.sigmoid = nn.Sigmoid()

        self.act = nn.GELU()

    def forward(self, x):
        B = x.shape[0]
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        ####up_stream
        x1 = x
        ### down_stream
        x2 = x.view(B,-1)
        x2 = self.fc_down1(x2).view(B,1,-1)
        x2 = self.act(x2)
        x2 = self.fc_down2(x2)
        x2 = self.act(x2)
        x2 = self.fc_down3(x2)
        x2 = self.sigmoid(x2)
        #### out
        x = ((x1.transpose(1,2))*x2).transpose(1,2)
        return x


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



        _mlpse_layers = []
        _attention_layer = []

        c = copy.deepcopy
        attn = MultiHeadedAttention(n_head, embed_dim)
        gcn = GraphNet(in_features=embed_dim, out_features=embed_dim, n_pts=self.num_joints + 15)

        for i in range(num_layers):
            _mlpse_layers.append(MLP_SE(embed_dim, self.num_joints + 15, embed_dim))
            _attention_layer.append(GraAttenLayer(embed_dim, c(attn), c(gcn), dropout))

        self.mlpse_layers = nn.ModuleList(_mlpse_layers)
        self.atten_layers = nn.ModuleList(_attention_layer)

        self.relu = nn.ReLU()
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
            x = self.mlpse_layers[i](x)

        x = self.norm(x)

        ####### after mesh regression module, x_out [B, J+T, emb_dim]
        x_out = x
        x_out = self.d_linear(x_out)
        x_out = self.relu(x_out)
        x_out = self.d_conv(x_out).transpose(1, 2)
        x_out = x_out + init_vertices

        return x_out

def get_model(num_joint=17, embed_dim=128, num_layers=5, n_head=4, dropout=0.25):
    model = Meshnet(num_joint, embed_dim, num_layers, n_head, dropout)
    return model

