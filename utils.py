#!/usr/bin/env python 
# -*- coding:utf-8 -*-

import os
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import r2_score

# file path
FOLDS_DIR = './folds_split/'    # folds directory
META_FILE = './qps_index.csv'   # meta-data file

# model params


# employed hyper-parameters & constants
BATCH = 32
N_WORKERS = 4   # thread number of dataloader
N_FEATS = 32
MAX_DUR = 482   # we divided all speaking length by this max length to normalize

GAMMA = 0.2     # loss_final = L_pers + GAMMA * L_align
ALPHA = 0.5     # update rate for modality weights
BETA = 50       # weight in the softmax function for modality weights

N_EPOCHS = 40   # master training procedure (alg 1 in paper)

# optimizer
LR = 1e-3
W_DECAY = 1e-5      # L2正则系数
STEP_SIZE = 10
SCHE_GAMMA = 0.2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def config_device():
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    torch.manual_seed(3)    # 设置seed能够保证代码在同一设备上的可复现性
    torch.cuda.manual_seed_all(3)
    torch.backends.cudnn.deterministic = True
    # torch.autograd.set_detect_anomaly(True)


def get_hyper_params(model_dict):
    all_params = []
    for model in model_dict.values():
        params = [p for p in model.parameters() if p.requires_grad]
        all_params += params

    return all_params


def count_hyper_params(params):
    return sum(p.numel() for p in params if p.requires_grad)


def setModelMode(model_dict, is_train_mode=True):
    if is_train_mode:
        for model in model_dict.values():
            model.train()
    else:
        for model in model_dict.values():
            model.eval()


def saveModel(FOLD, model_dict):
    dirs = f'./new_trained_models/fold{FOLD}/'
    if not os.path.exists(dirs):
        os.makedirs(dirs)
    for mod, model in model_dict.items():
        torch.save(model.state_dict(), f'{dirs}/{mod}')


def loadModel(FOLD, model_dict):
    dirs = f'./pre_trained_models/fold{FOLD}'
    for mod in model_dict.keys():
        filename = f'{dirs}/{mod}'
        if os.path.isfile(filename):
            model_dict[mod].load_state_dict(torch.load(filename))

    return model_dict


def calcAccuracy(y_pred, y_true):
    n = y_true.shape[0]
    THRESHOLD = 0.10
    threshold = torch.tensor([THRESHOLD for i in range(n)]).to(device)
    diff = torch.abs(y_pred.squeeze() - y_true)
    acc = torch.ge(threshold, diff).sum() / n
    return acc


def calcR2Score(y_pred, y_true):
    if y_pred.size()[0] == 1:
        return 0.2
    y_pred = y_pred[:, 0]
    criterion = nn.MSELoss()
    mse = criterion(y_pred, y_true)
    var = y_true.var()
    r2 = 1 - mse / var
    return r2


def calcMAE(y_pred, y_true):
    mae = nn.L1Loss()
    return mae(y_pred[:, 0], y_true)


def calcPersLoss(pred, target):
    criterion = nn.MSELoss()    # input: (N), (N)
    return criterion(pred[:, 0], target)


def calc_epoch_time(st_time, ed_time):
    elapsed_time = ed_time - st_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs
