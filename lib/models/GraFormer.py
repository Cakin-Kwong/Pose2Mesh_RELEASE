from __future__ import absolute_import

import torch.nn as nn
import torch
import numpy as np
import scipy.sparse as sp
import copy, math
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from funcs_utils import load_checkpoint
from core.config import cfg as cfg
#from network.ChebConv import ChebConv, _ResChebGC
from models.ChebConv import ChebConv, _ResChebGC
from timm.models.layers import DropPath
#from ChebConv import ChebConv, _ResChebGC


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def adj_mx_from_edges(num_pts, edges, sparse=True):
    edges = np.array(edges, dtype=np.int32)
    data, i, j = np.ones(edges.shape[0]), edges[:, 0], edges[:, 1]
    adj_mx = sp.coo_matrix((data, (i, j)), shape=(num_pts, num_pts), dtype=np.float32)

    # build symmetric adjacency matrix
    adj_mx = adj_mx + adj_mx.T.multiply(adj_mx.T > adj_mx) - adj_mx.multiply(adj_mx.T > adj_mx)
    adj_mx = normalize(adj_mx + sp.eye(adj_mx.shape[0]))
    if sparse:
        adj_mx = sparse_mx_to_torch_sparse_tensor(adj_mx)
    else:
        adj_mx = torch.tensor(adj_mx.todense(), dtype=torch.float)
    return adj_mx


gan_edges = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 4],
                          [0, 5], [5, 6], [6, 7], [7, 8],
                          [0, 9], [9, 10], [10, 11], [11, 12],
                          [0, 13], [13, 14], [14, 15], [15, 16]
                          ], dtype=torch.long)


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class LayerNorm(nn.Module):

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        # features=layer.size=512
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):

    def __init__(self, size, droppath):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        #self.dropout = nn.Dropout(dropout)
        self.droppath = DropPath(droppath)

    def forward(self, x, sublayer):
        return x + self.droppath(sublayer(self.norm(x)))


class GraAttenLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(GraAttenLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


def attention(Q, K, V, mask=None, dropout=None):
    # Query=Key=Value: [batch_size, 8, max_len, 64]
    d_k = Q.size(-1)
    # Q * K.T = [batch_size, 8, max_len, 64] * [batch_size, 8, 64, max_len]
    # scores: [batch_size, 8, max_len, max_len]
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    # padding mask
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, V), p_attn


class MultiHeadedAttention(nn.Module):

    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        Q, K, V = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                   for l, x in zip(self.linears, (query, key, value))]

        x, self.attn = attention(Q, K, V, mask=mask, dropout=self.dropout)

        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)

        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()

        self.w_1 = nn.Linear(d_model, d_ff)

        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


src_mask = torch.tensor([[[True, True, True, True, True, True, True, True, True, True, True,
                           True, True, True, True, True, True]]])


class LAM_Gconv(nn.Module):

    def __init__(self, in_features, out_features, activation=nn.ReLU(inplace=True)):
        super(LAM_Gconv, self).__init__()
        self.fc = nn.Linear(in_features=in_features, out_features=out_features)
        self.activation = activation

    def laplacian(self, A_hat):
        D_hat = (torch.sum(A_hat, 0) + 1e-5) ** (-0.5)
        L = D_hat * A_hat * D_hat
        return L

    def laplacian_batch(self, A_hat):
        batch, N = A_hat.shape[:2]
        D_hat = (torch.sum(A_hat, 1) + 1e-5) ** (-0.5)
        L = D_hat.view(batch, N, 1) * A_hat * D_hat.view(batch, 1, N)
        return L

    def forward(self, X, A):
        batch = X.size(0)
        A_hat = A.unsqueeze(0).repeat(batch, 1, 1)
        X = self.fc(torch.bmm(self.laplacian_batch(A_hat), X))
        if self.activation is not None:
            X = self.activation(X)
        return X


class GraphNet(nn.Module):

    def __init__(self, in_features=2, out_features=2, n_pts=17):
        super(GraphNet, self).__init__()

        self.A_hat = Parameter(torch.eye(n_pts).float(), requires_grad=True)
        self.gconv1 = LAM_Gconv(in_features, in_features * 2)
        self.gconv2 = LAM_Gconv(in_features * 2, out_features, activation=None)

    def forward(self, X):
        X_0 = self.gconv1(X, self.A_hat)
        X_1 = self.gconv2(X_0, self.A_hat)
        return X_1

class Block(nn.Module):
    def __init__(self, adj=None, hid_dim=128, n_head=4, dropout=0.5, drop_path=0.5):
        super(Block, self).__init__()
        self.DropPath = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mask = torch.ones(1, 1, adj.shape[0], dtype=torch.bool).cuda()
        attn = MultiHeadedAttention(n_head, hid_dim)
        gcn = GraphNet(in_features=hid_dim, out_features=hid_dim, n_pts=adj.shape[0])
        self.GraAttenLayer = GraAttenLayer(hid_dim, attn, gcn, dropout)
        self.gconv = _ResChebGC(adj=adj, input_dim=hid_dim, output_dim= hid_dim, hid_dim=hid_dim, p_dropout=dropout)

    def forward(self, x):
        out = x + self.DropPath(self.GraAttenLayer(x, mask=self.mask))
        out = out + self.DropPath(self.gconv(out))
        return out

class G_Block(nn.Module):

    def __init__(self, adj, hid_dim=128, n_head=4, dropout=0.2, drop_path=0.2):
        super(G_Block, self).__init__()
        self.Block1 = Block(adj=adj, hid_dim=hid_dim, n_head=n_head, dropout=dropout, drop_path=drop_path)
        self.Block2 = Block(adj=adj, hid_dim=hid_dim, n_head=n_head, dropout=dropout, drop_path=drop_path)
        self.Block3 = Block(adj=adj, hid_dim=hid_dim, n_head=n_head, dropout=dropout, drop_path=drop_path)
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


class GraFormer(nn.Module):
    def __init__(self, adj, hid_dim=128, coords_dim=(2, 3), num_layers=3,
                 n_head=4, dropout=0.2, drop_path=0.2, pretrained=False):
        super(GraFormer, self).__init__()
        self.n_layers = num_layers
        self.adj = adj

        self.gconv_input = ChebConv(in_c=coords_dim[0], out_c=hid_dim, K=2)
        self.norm = nn.LayerNorm(hid_dim)
        self.gelu = nn.GELU()
        _GBlock = []

        for i in range(num_layers):
            _GBlock.append(G_Block(adj=adj, hid_dim=hid_dim, n_head=n_head, dropout=dropout, drop_path=drop_path))

        self.GBlock = nn.ModuleList(_GBlock)
        self.gconv_output = ChebConv(in_c=hid_dim, out_c=3, K=2)

        if pretrained:
            self._load_pretrained_model()

    def forward(self, x):
        out = self.gconv_input(x, self.adj)
        out = self.norm(out)
        out = self.gelu(out)
        for i in range(self.n_layers):
            out = out + self.GBlock[i](out)

        pose_feature = out
        out = self.gconv_output(out, self.adj)
        return out, pose_feature

    def _load_pretrained_model(self):
        print("Loading pretrained posenet...")
        checkpoint = load_checkpoint(load_dir=cfg.MODEL.posenet_path, pick_best=True)
        self.load_state_dict(checkpoint['model_state_dict'])



def build_adj(joint_num, skeleton):
    adj_matrix = np.zeros((joint_num, joint_num))
    for line in skeleton:
        adj_matrix[line] = 1
        adj_matrix[line[1], line[0]] = 1

    return (torch.from_numpy(normalize(adj_matrix + np.eye(joint_num)))).float()


def get_model(joint_num, skeleton, hid_dim=128, pretrained=False):
    joint_adj = build_adj(joint_num, skeleton)
    model = GraFormer(adj=joint_adj.cuda(), hid_dim=hid_dim, pretrained=pretrained).cuda()
    return model

if __name__ == '__main__':
    #adj = adj_mx_from_edges(num_pts=17, edges=gan_edges, sparse=False)
    human36_skeleton = (
        (0, 7), (7, 8), (8, 9), (9, 10), (8, 11), (11, 12), (12, 13), (8, 14), (14, 15), (15, 16), (0, 1), (1, 2),
        (2, 3), (0, 4), (4, 5), (5, 6))
    adj = build_adj(17, human36_skeleton)
    model = GraFormer(adj=adj, hid_dim=128)
    x = torch.zeros((1, 17, 2))
    print(model(x, src_mask).shape)
    print(src_mask.shape)

