#!/usr/bin/env python 
# -*- coding:utf-8 -*-

import math
import numpy as np

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torchvision.models
from torch.nn.modules.container import ModuleList
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from utils import *


# 0 - Input Module
# This model is used to generate primary input embedding of one modality
class InputEmb(nn.Module):
    def __init__(self, mod, dropout):
        super(InputEmb, self).__init__()

        self.u = 65     # 输入特征矩阵 (S, E)
        self.d = 32     # 规整后的矩阵 (u, d)
        if mod == 'a':
            self.seq_len = 220
            self.feat_dim = 73
            self.conv1 = nn.Conv1d(in_channels=self.feat_dim, out_channels=self.d, kernel_size=28, stride=3)
        elif mod == 'v':
            self.seq_len = 350
            self.feat_dim = 512
            self.conv1 = nn.Conv1d(in_channels=self.feat_dim, out_channels=self.d, kernel_size=30, stride=5)
        else:
            self.seq_len = 610
            self.feat_dim = 200
            self.conv1 = nn.Conv1d(in_channels=self.feat_dim, out_channels=self.d, kernel_size=34, stride=9)

        # self.dropout = nn.Dropout(dropout)
        self.bn = nn.BatchNorm1d(self.d)
        self.relu = F.relu
        self.ln = nn.LayerNorm(self.d)

    def forward(self, src):
        # Input:    src(N, S, E)
        # Output:   out(N, u, d)
        x = src.permute(0, 2, 1)    # pytorch是对最后一维做卷积的, 因此需要把S换到最后
        x = self.conv1(x)           # (N, S, E) -> (N, d, u)
        # x = self.dropout(x)               # 1. dropout
        # x = self.relu(self.bn(x))         # 2. bn+relu
        # out = self.ln(x.permute(0, 2, 1)) # 3. ln
        out = x.permute(0, 2, 1)    # (N, u, d)
        return out


# 1 - Transformer Encoder
# positional encoder + transformer encoder
class TransformerEmb(nn.Module):
    def __init__(self, ninp, nhead, nhid, nlayers=1, dropout=0.1):
        super(TransformerEmb, self).__init__()
        # self.model_type = 'Transformer'
        self.ninp = ninp
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        # self.pos_encoder = LearnedPositionalEncoding(ninp, dropout)

        # 1-layer transformer encoder
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

    def forward(self, src):
        # Input:    src(S, N, E:16)
        # Output:   out(T, N, E:16)
        src *= math.sqrt(self.ninp)
        # positional encoder
        src = self.pos_encoder(src)
        # transformer encoder
        output = self.transformer_encoder(src)
        return output


# positional encoding
# 无可学习参数的PositionEncoding层
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


# The model generate the compact latent embedding of one modality
class LatentModel(nn.Module):
    def __init__(self, mod, dropout=0.4):
        super(LatentModel, self).__init__()

        # question: it means?
        # answer:
        nfeat = 32
        nhead = 4   # number of head attention
        nhid = 16   # number of hidden layers
        nlayers = 1     # transformer encoder layers

        self.feat_exa = InputEmb(mod, dropout)
        self.transformer_emb = TransformerEmb(nfeat, nhead, nhid, nlayers, dropout)

    def forward(self, src):
        # Input:    src(N, u, d),
        #           where N: batch size, u: sequence length, d: dimension of feature
        # Output:   out(N, u, d)
        feats = self.feat_exa(src)
        seq = self.transformer_emb(feats.transpose(0, 1))  # seq: (u, N, d)
        out = F.relu(seq)
        # max_pool: (u, N, d) -> (N, d)
        # out = torch.max(seq, dim=0)[0]    # (N, d)
        return out.transpose(0, 1)

# 2 - Cross Transformer
class CrossAttnModel(nn.Module):
    def __init__(self, dropout=0.4):
        super(CrossAttnModel, self).__init__()
        self.d = 32
        self.ln_alpha = nn.LayerNorm(self.d)
        self.ln_beta = nn.LayerNorm(self.d)
        self.weightQ = nn.Linear(self.d, self.d, bias=False)
        self.weightK = nn.Linear(self.d, self.d, bias=False)
        self.weightV = nn.Linear(self.d, self.d, bias=False)
        self.ln = nn.LayerNorm(self.d)
        # position-wise feed forward
        self.fc1 = nn.Linear(self.d, 4*self.d)
        self.fc2 = nn.Linear(4*self.d, self.d)
        self.dropout = nn.Dropout(dropout)

    # Input:    alpha , beta (N, u, d)
    # Output:   CCA-Bi(N, u, d)
    def forward(self, z_alpha, z_beta):
        z_alpha, z_beta = self.ln_alpha(z_alpha), self.ln_beta(z_beta)
        # self-attn
        residual = z_alpha
        Q = self.weightQ(z_alpha)
        K = self.weightK(z_beta)
        V = self.weightV(z_beta)
        attn = torch.bmm(Q, K.transpose(1, 2)) #/ torch.sqrt(torch.tensor(self.d))
        attn = F.softmax(attn, dim=-1)
        #attn = F.dropout(attn, p=self.attn_dropout) # dropout
        z_attn = torch.bmm(attn, V)
        z_attn = self.ln(residual + z_attn)
        # position-wise ffn
        residual = z_attn
        z = self.fc2(F.relu(self.fc1(z_attn)))
        z = self.dropout(z)
        z += residual
        return z



# 2 - Bi-Attention
class BiAttnModel(nn.Module):
    def __init__(self, dropout=0.4):
        super(BiAttnModel, self).__init__()
        self.d = 32
        self.trans_a_with_v = CrossAttnModel(dropout)
        self.trans_a_with_l = CrossAttnModel(dropout)
        self.trans_v_with_a = CrossAttnModel(dropout)
        self.trans_v_with_l = CrossAttnModel(dropout)
        self.trans_l_with_a = CrossAttnModel(dropout)
        self.trans_l_with_v = CrossAttnModel(dropout)

        self.fc1 = nn.Linear(2*self.d, 2*self.d)
        self.tanh = torch.tanh
        self.fc2 = nn.Linear(2*self.d, 1, bias=False)
        self.softmax = F.softmax

    # Input:    latent_emb_mod 3 x (N, u, d)
    # func:     首先由单模态特征矩阵F生成三个双模态融合信息
    #           但如果直接将这三个 Attn(u, 2d) 的矩阵拼接，特征维度将大大上升
    #           因此采用一种全局注意力机制对这部分融合信息进行进一步的筛选，压缩到(u, 2d)
    # Output:   CCA-Bi(N, u, 2d)
    def forward(self, latent_emb_mod):
        a_emb, v_emb, l_emb = latent_emb_mod['a'], latent_emb_mod['v'], latent_emb_mod['l']
        #z_v2a, z_l2a = self.trans_a_with_v(a_emb, v_emb), self.trans_a_with_l(a_emb, l_emb)
        #attnA = torch.cat([z_v2a, z_l2a], dim=2)
        #z_a2v, z_l2v = self.trans_v_with_a(v_emb, a_emb), self.trans_v_with_l(v_emb, l_emb)
        #attnV = torch.cat([z_a2v, z_l2v], dim=2)
        #z_a2l, z_v2l = self.trans_l_with_a(l_emb, a_emb), self.trans_l_with_v(l_emb, v_emb)
        #attnL = torch.cat([z_a2l, z_v2l], dim=2)    # u x 2d
        #bi_attn = torch.cat([attnA, attnV, attnL], dim=1)   # 3u x 2d
        #return bi_attn

        attnAV, attnAL, attnVL = self.BiAttn(a_emb, v_emb), self.BiAttn(a_emb, l_emb), self.BiAttn(v_emb, l_emb)
        u = a_emb.size()[1]
        CCA = []
        for i in range(u):
             Bi = torch.cat([attnAV[:, 0:1, :], attnAL[:, 0:1, :], attnVL[:, 0:1, :]], dim=1)
             Ci = self.fc2(self.tanh(self.fc1(Bi)))
             alpha = self.softmax(Ci, dim=0)
             CCA_i = torch.matmul(alpha.transpose(1, 2), Bi)
             CCA.append(CCA_i)
        CCA = torch.cat(CCA, dim=1) # u x 2d
        #CCA = torch.cat([attnAV, attnAL, attnVL], dim=1)   # 3u x 2d
        return CCA

    def BiAttn(self, feat1, feat2):
        # Input:    feat1, feat2 (N, u, d)
        # Output:   Attn12(N, u, 2d)
        # 矩阵相乘，生成跨模态信息矩阵, (u, u)
        B12 = torch.matmul(feat1, feat2.transpose(1, 2))
        B21 = torch.matmul(feat2, feat1.transpose(1, 2))
        # softmax, 得到相关矩阵注意力分布, (u, u)
        N1 = F.softmax(B12, dim=1)
        N2 = F.softmax(B21, dim=1)
        # 矩阵乘法，生成注意力表征矩阵, (u, d)
        O1 = torch.matmul(N1, feat2)
        O2 = torch.matmul(N2, feat1)
        # 逐元素相乘，得到互注意力信息矩阵, (u, d)
        A1 = torch.mul(O1, feat1)
        A2 = torch.mul(O2, feat2)
        # concat, 融合信息表征, (u, 2d)
        Attn12 = torch.cat([A1, A2], dim=2)
        return Attn12


# 3 - Tri-Attention
class TriAttnModel(nn.Module):
    def __init__(self):
        super(TriAttnModel, self).__init__()
        self.d = 32
        self.fc1 = nn.Linear(2 * self.d, self.d)
        self.fc2 = nn.Linear(2 * self.d, self.d)
        self.fc3 = nn.Linear(2 * self.d, self.d)
        self.relu = torch.relu
        self.softmax = F.softmax

    # Input:    latent_emb_mod 3 x (N, u, d)
    # Output:   CCA-Tri(N, u, 3d)
    def forward(self, latent_emb_mod):
        # 单模态特征矩阵
        F_a, F_v, F_l = latent_emb_mod['a'], latent_emb_mod['v'], latent_emb_mod['l']
        # 浅层融合的双模态信息矩阵
        F_av, F_al, F_vl = torch.cat([F_a, F_v], dim=2), torch.cat([F_a, F_l], dim=2), torch.cat([F_v, F_l], dim=2)
        F_av = self.relu(self.fc1(F_av))
        F_al = self.relu(self.fc2(F_al))
        F_vl = self.relu(self.fc3(F_vl))
        # 三模态融合特征
        attn_lav = self.TriAttn(F_l, F_av)
        attn_val = self.TriAttn(F_v, F_al)
        attn_avl = self.TriAttn(F_a, F_vl)
        CCA = torch.cat([attn_lav, attn_val, attn_avl], dim=2)
        return CCA

    def TriAttn(self, F1, F23):
        # Input:    F1 (u, d), F23 (u, d)
        # func:     输入单模态特征矩阵与双模态浅层融合特征，通过计算其注意力分布得到最终的三模态融合信息
        # Output:   Attn123 (u, d)
        # 跨模态信息矩阵, (u, u)
        C1 = torch.matmul(F1, F23.transpose(1, 2))
        # 注意力分布, (u, u)
        P1 = self.softmax(C1, dim=-1)
        # 交互注意力表征信息, (u, d)
        T1 = torch.matmul(P1, F1)
        # 三模态融合信息, (u, d)
        Attn123 = torch.mul(T1, F23)
        return Attn123


# 4 - Persuasiveness Module
class PersModel(nn.Module):
    def __init__(self, nmod=3, nfeat=32, dropout=0.1):
        super(PersModel, self).__init__()

        # input: latent_emb emb (nmod * nfeat), bi_attn_emb (2 * nfeat), tri_attn_emb (3 * nfeat), debate meta-data (1)
        ninp = (2 + 3) * nfeat + 2
        nout = 1
        self.fc1 = nn.Linear(ninp, 2 * ninp)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(2 * ninp, nout)
        self.sigm = nn.Sigmoid()

    def forward(self, latent_emb_mod, bi_attn_emb, tri_attn_emb, meta_emb):
        latent_emb = torch.cat([torch.mean(emb, dim=1) for emb in latent_emb_mod.values()], dim=1)
        bi_attn_emb = torch.max(bi_attn_emb, dim=1)[0]
        tri_attn_emb = torch.max(tri_attn_emb, dim=1)[0]
        x = torch.cat([bi_attn_emb, tri_attn_emb, meta_emb], dim=1)
        # x = torch.cat([latent_emb, bi_attn_emb, tri_attn_emb, meta_emb], dim=1)
        x = self.fc1(x)
        x = F.relu(self.dropout(x))
        x = self.fc2(x)
        return self.sigm(x)

