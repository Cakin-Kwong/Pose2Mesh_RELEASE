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
from timm.models.layers import DropPath



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

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()

        self.w_1 = nn.Linear(d_model, d_ff)

        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Block(nn.Module):
    def __init__(self, in_channel=17, hid_dim=128, n_head=4, dropout=0.5, drop_path=0.5, mlp_ratio=4.):
        super(Block, self).__init__()
        self.in_channel = in_channel
        self.DropPath = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mask = torch.ones(1, 1, in_channel, dtype=torch.bool).cuda()
        attn = MultiHeadedAttention(n_head, hid_dim)
        gcn = GraphNet(in_features=hid_dim, out_features=hid_dim, n_pts=in_channel)
        mlp_hidden_dim = int(hid_dim * mlp_ratio)
        self.GraAttenLayer = GraAttenLayer(hid_dim, attn, gcn, dropout)
        self.norm = nn.LayerNorm(hid_dim)
        self.mlp = PositionwiseFeedForward(d_model=hid_dim, d_ff=mlp_hidden_dim, dropout=dropout)

    def forward(self, x):
        out = x + self.DropPath(self.GraAttenLayer(x, mask=self.mask))
        out = out + self.DropPath(self.mlp(self.norm(out)))
        return out

class G_Block(nn.Module):

    def __init__(self, in_channel=17, hid_dim=128, n_head=4, dropout=0.2, drop_path=0.2, mlp_ratio=4.):
        super(G_Block, self).__init__()
        self.Block1 = Block(in_channel=in_channel, hid_dim=hid_dim, n_head=n_head, dropout=dropout, drop_path=drop_path, mlp_ratio=mlp_ratio)
        self.Block2 = Block(in_channel=in_channel, hid_dim=hid_dim, n_head=n_head, dropout=dropout, drop_path=drop_path, mlp_ratio=mlp_ratio)
        self.Block3 = Block(in_channel=in_channel, hid_dim=hid_dim, n_head=n_head, dropout=dropout, drop_path=drop_path, mlp_ratio=mlp_ratio)

        self.conv1 = nn.Conv2d(3, 1, 1)
        self.norm = nn.LayerNorm(hid_dim)
        self.gelu = nn.GELU()

    def forward(self, x):
        # B = x.shape[0]
        x1 = self.Block1(x).unsqueeze(1)
        x2 = self.Block2(x).unsqueeze(1)
        x3 = self.Block3(x).unsqueeze(1)

        x = torch.cat((x1,x2,x3), dim = 1)
        x = self.conv1(x).squeeze(1)
        x = self.norm(x)
        x = self.gelu(x)

        return x

class Meshnet(nn.Module):
    def __init__(self, num_joints=17, hid_dim=128, num_layers=3,
                 n_head=4, dropout=0.2, drop_path=0.2, mlp_ratio=4.):
        super(Meshnet, self).__init__()
        self.num_joints = num_joints
        self.hid_dim = hid_dim
        self.n_layers = num_layers

        init_vertices = torch.from_numpy(np.load(SMPL_MEAN_vertices)).unsqueeze(0)
        self.register_buffer('init_vertices', init_vertices)

        self.up_feature = nn.Linear(hid_dim, hid_dim)
        self.up_linear = nn.Linear(689 * 2, hid_dim)


        _GBlock = []

        for i in range(num_layers):
            _GBlock.append(G_Block(in_channel=self.num_joints + 15, hid_dim=hid_dim, n_head=n_head, dropout=dropout, drop_path=drop_path, mlp_ratio=mlp_ratio))

        self.GBlock = nn.ModuleList(_GBlock)

        self.norm = nn.LayerNorm(hid_dim)
        self.gelu = nn.GELU()


        self.d_conv = nn.Conv1d(self.num_joints + 15, 3, kernel_size=3, padding=1)
        self.d_linear = nn.Linear(self.hid_dim, 6890)


    def forward(self, feature):
        B = feature.shape[0]
        feature = feature.view(B, self.num_joints, -1)
        feature = self.up_feature(feature)

        init_vertices = self.init_vertices.expand(B, -1, -1)
        mean_smpl = self.up_linear(init_vertices.transpose(1,2).reshape(B, 15, -1))

        x = torch.cat((feature, mean_smpl), dim=1)

        for i in range(self.n_layers):
            x = x + self.GBlock[i](x)

        x = self.norm(x)

        ####### after mesh regression module, x_out [B, J+T, emb_dim]
        x_out = x
        x_out = self.d_linear(x_out)
        x_out = self.gelu(x_out)
        x_out = self.d_conv(x_out).transpose(1, 2)
        x_out = x_out + init_vertices

        return x_out

def get_model(num_joints=17, hid_dim=128, num_layers=3, n_head=4, dropout=0.22, drop_path=0.2, mlp_ratio=4.):
    model = Meshnet(num_joints=num_joints, hid_dim=hid_dim, num_layers=num_layers, n_head=n_head, dropout=dropout, drop_path=drop_path, mlp_ratio=mlp_ratio)
    return model

