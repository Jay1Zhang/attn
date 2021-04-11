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
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # Input:    src(N, S, E)
        # Output:   out(N, u, d)
        x = src.permute(0, 2, 1)    # pytorch是对最后一维做卷积的, 因此需要把S换到最后
        x = self.conv1(x)           # (N, S, E) -> (N, d, u)
        x = self.bn(x)
        x = F.relu(self.dropout(x))
        out = x.permute(0, 2, 1)    # (N, u, d)
        return out


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
        emb = torch.cat([torch.mean(emb, dim=1) for emb in latent_emb_mod.values()], dim=1)
        x = torch.cat([emb, meta_emb], dim=1)
        x = self.fc1(x)
        x = F.relu(self.dropout(x))
        x = self.fc2(x)
        return self.sigm(x)

