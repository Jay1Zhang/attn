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

        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # Input:    src(N, S, E)
        # Output:   out(N, u, d)
        x = src.permute(0, 2, 1)    # pytorch是对最后一维做卷积的, 因此需要把S换到最后
        x = self.conv1(x)           # (N, S, E) -> (N, d, u)
        x = self.dropout(x)
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
        # Input:    src(N, S, E),  seq_msk(N, S)
        #           where N: batch size, S: sequence length, E: {a:73, v:512, l:200}
        # Output:   out(, N, 16)
        # N, S = src.size()[0], src.size()[1]
        # (N, S, E) -> (N, u, d)
        # feats = torch.stack([self.feat_exa(src[i]) for i in range(N)], dim=0).transpose(0, 1)
        feats = self.feat_exa(src)
        seq = self.transformer_emb(feats.transpose(0, 1))  # seq: (u, N, d)
        out = F.relu(seq)
        # max_pool: (u, N, d) -> (N, d)
        # out = torch.max(seq, dim=0)[0]    # (N, d)
        return out.transpose(0, 1)


# 3 - Persuasiveness Module
class PersModel(nn.Module):
    def __init__(self, nmod=3, nfeat=32, dropout=0.1):
        super(PersModel, self).__init__()

        # input: heterogeneity emb (nmod * nfeat), debate meta-data (1)
        ninp = nmod * nfeat + 2
        nout = 1
        self.fc1 = nn.Linear(ninp, 2 * ninp)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(2 * ninp, nout)
        self.sigm = nn.Sigmoid()

    def forward(self, latent_emb_mod, meta_emb):
        latent_emb = torch.cat([torch.mean(emb, dim=1) for emb in latent_emb_mod.values()], dim=1)
        x = torch.cat([latent_emb, meta_emb], dim=1)
        x = self.fc1(x)
        x = F.relu(self.dropout(x))
        x = self.fc2(x)
        return self.sigm(x)

