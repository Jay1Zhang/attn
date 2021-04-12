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
class LatentModel(nn.Module):
    def __init__(self, mod, dropout):
        super(LatentModel, self).__init__()

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
        self.bn = nn.BatchNorm1d(self.d)
        self.relu = F.relu

    def forward(self, src):
        # Input:    src(N, S, E)
        # Output:   out(N, u, d)
        x = src.permute(0, 2, 1)    # pytorch是对最后一维做卷积的, 因此需要把S换到最后
        x = self.conv1(x)
        x = self.relu(self.bn(x))
        out = x.permute(0, 2, 1)
        return out

# 1 - Bi-Attention
def BiAttn(feat1, feat2):
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

class AttnModel(nn.Module):
    def __init__(self):
        super(AttnModel, self).__init__()
        self.d = 32
        self.fc1 = nn.Linear(2*self.d, 2*self.d)
        self.tanh = torch.tanh
        self.fc2 = nn.Linear(2*self.d, 1, bias=False)
        self.softmax = F.softmax

    # Input:    latent_emb_mod 3 x (N, u, d)
    # func:     首先由单模态特征矩阵F生成三个双模态融合信息
    #           但如果直接将这三个 Attn(u, 2d) 的矩阵拼接，特征维度将大大上升
    #           因此采用一种全局注意力机制对这部分融合信息进行进一步的筛选，压缩到(u, 2d)
    # Output:   CCA(N, u, 2d)
    def forward(self, latent_emb_mod):
        a_emb, v_emb, l_emb = latent_emb_mod['a'], latent_emb_mod['v'], latent_emb_mod['l']
        attnAV, attnAL, attnVL = BiAttn(a_emb, v_emb), BiAttn(a_emb, l_emb), BiAttn(v_emb, l_emb)
        u = a_emb.size()[1]
        CCA = []
        for i in range(u):
            Bi = torch.cat([attnAV[:, 0:1, :], attnAL[:, 0:1, :], attnVL[:, 0:1, :]], dim=1)
            Ci = self.fc2(self.tanh(self.fc1(Bi)))
            alpha = self.softmax(Ci, dim=0)
            CCA_i = torch.matmul(alpha.transpose(1, 2), Bi)
            CCA.append(CCA_i)
        CCA = torch.cat(CCA, dim=1)
        return CCA


# 3 - Persuasiveness Module
class PersModel(nn.Module):
    def __init__(self, nmod=3, nfeat=32, dropout=0.1):
        super(PersModel, self).__init__()

        # input: latent_emb emb (nmod * nfeat), bi_attn_emb (2 * nfeat), debate meta-data (1)
        ninp = (nmod + 2) * nfeat + 2
        nout = 1
        self.fc1 = nn.Linear(ninp, 2 * ninp)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(2 * ninp, nout)
        self.sigm = nn.Sigmoid()

    def forward(self, latent_emb_mod, bi_attn_emb, meta_emb):
        latent_emb = torch.cat([torch.mean(emb, dim=1) for emb in latent_emb_mod.values()], dim=1)
        bi_attn_emb = torch.mean(bi_attn_emb, dim=1)
        x = torch.cat([latent_emb, bi_attn_emb, meta_emb], dim=1)
        x = self.fc1(x)
        x = F.relu(self.dropout(x))
        x = self.fc2(x)
        return self.sigm(x)

