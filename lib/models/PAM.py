import numpy as np
import torch.nn as nn
import torch
import math
import graph_utils
from core.config import cfg as cfg
from funcs_utils import load_checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal(m.weight)

class Conv(nn.Module):
    def __init__(self, in_channel, drop=0.):
        super().__init__()

        self.channel = in_channel
        self.conv1 = nn.Conv1d(in_channel, 32, 1)
        self.act = nn.GELU()
        self.conv2 = nn.Conv1d(in_channel, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(32, in_channel, kernel_size=3, padding=1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x1 =self.conv1(x)
        x1 = self.act(x1)
        x1 = self.drop(x1)
        x2 = self.conv2(x)
        x2 = self.act(x2)
        x2 = self.drop(x2)
        x = x1 + x2
        x = self.conv3(x)
        return x

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


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MGCN(nn.Module):
    """
    Semantic graph convolution layer
    """

    def __init__(self, in_features, out_features, in_channel, adj, bias=True):
        super(MGCN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        if adj == None:
            adj = nn.Parameter(torch.eye(in_channel, in_channel, dtype=torch.float, requires_grad = True))

        self.adj = adj

        self.W = nn.Parameter(torch.zeros(size=(2, in_features, out_features), dtype=torch.float))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.M = nn.Parameter(torch.zeros(size=(adj.size(0), out_features), dtype=torch.float))
        nn.init.xavier_uniform_(self.M.data, gain=1.414)

        self.adj2 = nn.Parameter(torch.ones_like(adj))
        nn.init.constant_(self.adj2, 1e-6)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float))
            stdv = 1. / math.sqrt(self.W.size(2))
            self.bias.data.uniform_(-stdv, stdv)
        else:
            self.register_parameter('bias', None)

    def forward(self, input):
        h0 = torch.matmul(input, self.W[0])
        h1 = torch.matmul(input, self.W[1])

        adj = self.adj.to(input.device) + self.adj2.to(input.device)
        adj = (adj.T + adj) / 2
        E = torch.eye(adj.size(0), dtype=torch.float).to(input.device)

        output = torch.matmul(adj * E, self.M * h0) + torch.matmul(adj * (1 - E), self.M * h1)
        if self.bias is not None:
            return output + self.bias.view(1, 1, -1)
        else:
            return output


class Block(nn.Module):

    def __init__(self, dim, in_channel, graph_adj=None, num_heads=8, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0.5, attn_drop=0.,
                 drop_path=0.5, fix = False, norm_layer=nn.LayerNorm, act_layer=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)
        self.mgcn = MGCN(dim, dim, in_channel, graph_adj,)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.conv = Conv(in_channel=in_channel, drop=drop)
        self.mlpse = MLP_SE(in_features = dim, in_channel=in_channel,hidden_features=mlp_hidden_dim)

    def forward(self, x):
        x = self.mgcn(x)
        x = x + self.drop_path(self.attn(self.norm1(x)) + self.conv(self.norm2(x)))
        # x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlpse(self.norm3(x)))
        return x

class G_Block(nn.Module):

    def __init__(self, dim=64, in_channel=17, graph_adj = None, num_heads=8, mlp_ratio=4., qkv_bias=True,
                 qk_scale=None, drop_rate=0., attn_drop_rate=0.,drop_path=0.2, norm_layer=nn.LayerNorm, act_layer=None,):
        super().__init__()
        self.Block1 = Block(
            dim=dim, in_channel=in_channel, graph_adj = graph_adj, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
            qk_scale=qk_scale, fix = True,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path, norm_layer=norm_layer, act_layer=act_layer)

        self.Block2 = Block(
            dim=dim, in_channel=in_channel, graph_adj = graph_adj, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path, norm_layer=norm_layer, act_layer=act_layer)

        self.Block3 = Block(
            dim=dim, in_channel=in_channel, graph_adj = None, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path, norm_layer=norm_layer, act_layer=act_layer)

        self.Block4 = Block(
            dim=dim, in_channel=in_channel, graph_adj = None, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path, norm_layer=norm_layer, act_layer=act_layer)

        self.Block5 = Block(
            dim=dim, in_channel=in_channel, graph_adj = None, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path, norm_layer=norm_layer, act_layer=act_layer)

        self.Block6 = Block(
            dim=dim, in_channel=in_channel, graph_adj = None, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path, norm_layer=norm_layer, act_layer=act_layer)

        self.conv1 = nn.Conv2d(6, 1, 1)


    def forward(self, x):
        # B = x.shape[0]
        x1 = self.Block1(x).unsqueeze(1)
        x2 = self.Block2(x).unsqueeze(1)
        x3 = self.Block3(x).unsqueeze(1)
        x4 = self.Block4(x).unsqueeze(1)
        x5 = self.Block5(x).unsqueeze(1)
        x6 = self.Block6(x).unsqueeze(1)

        x = torch.cat((x1,x2,x3,x4,x5,x6), dim = 1)
        x = self.conv1(x).squeeze(1)

        return x

class LiftingModel(nn.Module):
    def __init__(self,
                 num_joint,
                 pretrained=False, embed_dim=128, graph_adj = None, num_heads=8, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0.2, attn_drop_rate = 0.2, drop_path_rate=0.2,norm_layer=nn.LayerNorm, act_layer=None,):
        super(LiftingModel, self).__init__()

        self.patch_number = num_joint
        self.embed_dim = embed_dim
        #graph_adj = graph_utils.sparse_python_to_torch(graph_adj[-1]).to_dense()
        self.register_buffer('graph_adj', graph_adj)


        # process input to linear size
        self.input_up = MGCN(2, embed_dim, self.patch_number, self.graph_adj)

        self.norm0 = nn.LayerNorm(embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.gelu0 = nn.GELU()
        self.gelu1 = nn.GELU()
        self.gelu2 = nn.GELU()
        self.drop = nn.Dropout(drop_rate)

        # post processing
        self.linear_down = nn.Linear(embed_dim, 3)

        self.pos_drop = nn.Dropout(p=drop_rate)

        self.G_block1 = G_Block(
            dim=embed_dim, in_channel=self.patch_number, graph_adj = self.graph_adj, num_heads=num_heads, mlp_ratio=mlp_ratio,
            drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path=drop_path_rate, norm_layer=norm_layer, act_layer=act_layer)

        self.G_block2 = G_Block(
            dim=embed_dim, in_channel=self.patch_number, graph_adj = self.graph_adj, num_heads=num_heads, mlp_ratio=mlp_ratio,
            drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path=drop_path_rate, norm_layer=norm_layer, act_layer=act_layer)


        if pretrained:
            self._load_pretrained_model()

    def forward(self, x):
        ####  input [B, J*2]  J is the number of joints
        B = x.shape[0]
        x = x.view(B,self.patch_number, -1)
        x = self.input_up(x)
        x = self.norm0(x)
        x = self.gelu0(x)
        x0 = x

        x1 = self.G_block1(x)
        x1 = self.norm1(x1)
        x1 = self.gelu1(x1)
        x1 = x1 + x0

        x2 = self.G_block2(x1)
        x2 = self.norm2(x2)
        x2 = self.gelu2(x2)

        y = x2 + x1 + x0

        y_feature = y
        y_3d = self.linear_down(y_feature)
        y_3d = y_3d.view(B,-1)
        ##### 3D pose output: y_3d [B, J*3]
        ##### pose feature: y_feature [B, J, embed_dim]
        #return y_3d, y_feature
        return y_3d

    def _load_pretrained_model(self):
        print("Loading pretrained posenet...")
        checkpoint = load_checkpoint(load_dir=cfg.MODEL.posenet_path, pick_best=True)
        self.load_state_dict(checkpoint['model_state_dict'])


# def get_model(num_joint, pretrained=False, embed_dim=128, graph_adj = None):
#     model = LiftingModel(num_joint, pretrained, embed_dim, graph_adj)
#
#     return model
def build_adj(joint_num, skeleton, flip_pairs):
    adj_matrix = torch.zeros((joint_num, joint_num))
    for line in skeleton:
        adj_matrix[line] = 1
        adj_matrix[line[1], line[0]] = 1
    for lr in flip_pairs:
        adj_matrix[lr] = 1
        adj_matrix[lr[1], lr[0]] = 1

    return adj_matrix + torch.eye(joint_num)

def get_model(joint_num, skeleton, flip_pairs, hid_dim=128, pretrained=False):
    joint_adj = build_adj(joint_num, skeleton, flip_pairs)
    model = LiftingModel(joint_num, pretrained, hid_dim, joint_adj)
    return model



#########################################################
# import torch
# import time
# model_deit = LiftingModel(num_joint = 17)
# input = torch.ones(32,34).cuda()
# model_pos = model_deit.cuda()
# start = time.time()
# x_3d, x = model_pos(input)
# end = time.time() - start
# print(end)
# print(x.shape)
# print(x_3d.shape)