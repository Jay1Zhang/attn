#!/usr/bin/env python 
# -*- coding:utf-8 -*-

import torch

from utils import *


def gen_align_emb(s_emb_mod):
    # generate H^align with H^s_m
    align_cat = torch.cat([emb.unsqueeze(dim=0) for emb in s_emb_mod.values()], dim=0)
    align_emb = torch.mean(align_cat, dim=0)  # H^align
    return align_emb


def gen_het_emb(latent_emb_mod, weight_mod, MODS):
    # generate H^het with H^latent_m, w_m
    het_emb_mod = {}
    for mod in MODS:
        het_emb_mod[mod] = weight_mod[mod] * latent_emb_mod[mod]
    # het_emb_mod = [torch.tensor(weight_mod[mod]) * latent_emb_mod[mod] for mod in MODS]
    het_emb = torch.cat([v for v in het_emb_mod.values()], dim=1)
    return het_emb, het_emb_mod


def gen_meta_emb(sample):
    st_vote = (sample['ed_vote'] - sample['change']).float().unsqueeze(1).to(device)
    dur_time = (sample['dur'] / MAX_DUR).float().unsqueeze(1).to(device)
    meta_emb = torch.cat([st_vote, dur_time], dim=1)
    return meta_emb


def fit_m2p2(m2p2_models, MODS, sample_batched):
    latent_emb_mod = {}
    for mod in MODS:
        latent_emb_mod[mod] = m2p2_models[mod](sample_batched[f'{mod}_data'].to(device))

    bi_attn_emb = m2p2_models['bi_attn'](latent_emb_mod)
    tri_attn_emb = m2p2_models['tri_attn'](latent_emb_mod)
    #tri_attn_emb = None
    meta_emb = gen_meta_emb(sample_batched)

    y_pred = m2p2_models['pers'](latent_emb_mod, bi_attn_emb, tri_attn_emb, meta_emb)   # (N, 1)
    y_true = sample_batched['ed_vote'].float().to(device)   # (N)

    # calc loss
    loss_pers = calcPersLoss(y_pred, y_true)
    #acc = calcAccuracy(y_pred, y_true)
    acc = calcMAE(y_pred, y_true)

    return loss_pers, acc


def train_m2p2(m2p2_models, MODS, iterator, optimizer, scheduler):
    setModelMode(m2p2_models, is_train_mode=True)
    total_loss = 0
    total_acc = 0

    for i_batch, sample_batched in enumerate(iterator):
        optimizer.zero_grad()
        # forward
        loss, acc = fit_m2p2(m2p2_models, MODS, sample_batched)
        total_loss += loss.item()
        total_acc += acc.item()

        # backward
        loss.backward()
        optimizer.step()

    scheduler.step()
    return total_loss / (i_batch+1), total_acc / (i_batch + 1)    # mean


def eval_m2p2(m2p2_models, MODS, iterator):
    setModelMode(m2p2_models, is_train_mode=False)
    total_loss = 0
    total_acc = 0

    for i_batch, sample_batched in enumerate(iterator):
        # forward
        with torch.no_grad():
            loss, acc = fit_m2p2(m2p2_models, MODS, sample_batched)
            total_loss += loss.item()
            total_acc += acc.item()

    return total_loss / (i_batch+1), total_acc / (i_batch + 1)    # mean

