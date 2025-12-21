# from inferno.utils.other import get_path_to_externals
from pathlib import Path
import sys, os
import torch
import json
user_name = os.getcwd().split('/')[2]
sys.path.append(f'../')
#print current directory
from DEE.utils.pcme import sample_gaussian_tensors
# video emotion loss
# from inferno.models.temporal.Renderers import cut_mouth_vectorized
'''
E2E should be implemented from same version that LipReading used
'''
# from externals.spectre.external.Visual_Speech_Recognition_for_Multiple_Languages.espnet.nets.pytorch_backend.e2e_asr_transformer import E2E
## new lip reading loss
import torchvision.transforms as t
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import face_alignment

import argparse
import math
import numpy as np
import torch.nn as nn
import pickle
import omegaconf
## for video emotion loss
from omegaconf import OmegaConf
sys.path.append('DEEPTalk/models')
# from video_emotion import SequenceClassificationEncoder, ClassificationHead, TransformerSequenceClassifier, MultiheadLinearClassificationHead, EmoCnnModule
import pytorch_lightning as pl 
from typing import Any, Optional, Dict, List
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
import torchvision
from scipy.signal import find_peaks

def check_nan(sample: Dict): 
    ok = True
    nans = []
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            if torch.isnan(value).any():
                print(f"NaN found in '{key}'")
                nans.append(key)
                ok = False
                # raise ValueError("Nan found in sample")
    if len(nans) > 0:
        raise ValueError(f"NaN found in {nans}")
    return ok

# Following EMOTE paper,
# λrec is set to 1000000 and λKL to 0.001, which makes the
# converged KL divergence term less than one order of magnitude
# lower than the reconstruction terms
def calc_vae_loss(pred,target,mu, logvar, recon_weight=1_000_000, kl_weight=0.001):                            
    """ function that computes the various components of the VAE loss """
    reconstruction_loss = nn.MSELoss()(pred, target)
    KLD = torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1))
    return recon_weight * reconstruction_loss + kl_weight * KLD, recon_weight *reconstruction_loss,kl_weight * KLD


def calc_vq_loss(pred, target, quant_loss, quant_loss_wight,alpha=1.0):
    """ function that computes the various components of the VQ loss """

    exp_loss = nn.L1Loss()(pred[:,:,:50], target[:,:,:50])
    rot_loss = nn.L1Loss()(pred[:,:,50:53], target[:,:,50:53])
    jaw_loss = alpha * nn.L1Loss()(pred[:,:,53:], target[:,:,53:])
    ## loss is VQ reconstruction + weighted pre-computed quantization loss
    return quant_loss.mean()*quant_loss_wight + \
            (exp_loss + rot_loss + jaw_loss)

def calc_vq_flame_L1_loss(pred, target, quant_loss,quant_loss_wight=1.0, alpha=1.0):
    exp_loss = nn.L1Loss()(pred[:,:,:50], target[:,:,:50])
    jaw_loss = alpha * nn.L1Loss()(pred[:,:,50:53], target[:,:,50:53])
    ## loss is VQ reconstruction + weighted pre-computed quantization loss
    return quant_loss.mean()*quant_loss_wight + \
            (exp_loss + jaw_loss), (exp_loss + jaw_loss)
            
def calc_vq_flame_L2_loss(pred, target, quant_loss, quant_loss_wight=1.0,alpha=1.0):
    exp_loss = nn.MSELoss()(pred[:,:,:50], target[:,:,:50])
    jaw_loss = alpha * nn.MSELoss()(pred[:,:,50:53], target[:,:,50:53])
    ## loss is VQ reconstruction + weighted pre-computed quantization loss
    return quant_loss.mean()*quant_loss_wights + \
            (exp_loss + jaw_loss), (exp_loss + jaw_loss)
            
def calc_vq_vertice_L2_loss(pred, target, quant_loss, quant_loss_wight=1.0, recon_weight=1000000):
    reconstruction_loss = nn.MSELoss()(pred, target)
    ## loss is VQ reconstruction + weighted pre-computed quantization loss
    return quant_loss.mean()*quant_loss_wight + reconstruction_loss*recon_weight  , reconstruction_loss*recon_weight

def calculate_vertice_loss(pred, target):
     reconstruction_loss = nn.MSELoss()(pred, target)
     return reconstruction_loss

def calculate_vertex_velocity_loss(pred, target):
    """
    pred, target torch tensor of shape (BS, T, V*3)
    """
    pred_diff = pred[:, 1:,:] - pred[:, :-1,:]
    target_diff = target[:, 1:,:] - target[:, :-1,:]
    velocity_loss = nn.MSELoss()(pred_diff, target_diff)
    return velocity_loss


def calculate_consistency_loss(model, audio, pred_exp, 
                               point_DEE, normalize_exp=False,
                               num_samples = None, affectnet_feature_extractor=None,
                               prob_method='csd') :
    if normalize_exp :
        pred_exp = (pred_exp - torch.mean(pred_exp, dim=1, keepdim=True)) / torch.std(pred_exp, dim=1, keepdim=True)
    if point_DEE :

        DEE_audio_embedding = model.encode_audio(audio)
        if affectnet_feature_extractor is not None:
            pred_exp = affectnet_feature_extractor.extract_feature_from_layer(pred_exp, layer_num = -2)
        DEE_exp_embedding = model.encode_parameter(pred_exp)
        # Normalize before computing cosine similarity
        DEE_audio_embedding /= DEE_audio_embedding.norm(dim=1, keepdim=True)
        DEE_exp_embedding /= DEE_exp_embedding.norm(dim=1, keepdim=True)
        emotion_consistency_loss = 1-F.cosine_similarity(DEE_audio_embedding, DEE_exp_embedding, dim=1).mean()
        return emotion_consistency_loss
    else :
        _, audio_mean, audio_logvar = model.audio_encoder(audio)
        if affectnet_feature_extractor is not None:
            pred_exp = affectnet_feature_extractor.extract_feature_from_layer(pred_exp, layer_num = -2)
        _, exp_mean, exp_logvar = model.exp_encoder(pred_exp)
        # sampling
        if prob_method == 'csd' :
            mu_pdist = ((audio_mean - exp_mean) ** 2).sum(-1)
            sigma_pdist = ((torch.exp(audio_logvar) + torch.exp(exp_logvar))).sum(-1)
            logits = mu_pdist + sigma_pdist
            logits = -model.criterion.negative_scale * logits + model.criterion.shift # (Bs, BS)
            labels = torch.ones(logits.shape[0], dtype=logits.dtype, device=logits.device)
            loss = model.criterion.bceloss(logits, labels)
            return loss
        elif prob_method == 'mean':
            mu_pdist = ((audio_mean - exp_mean) ** 2).sum(-1)
            return mu_pdist.mean()
        
        elif prob_method == 'sample' :
            raise NotImplementedError
            DEE_audio_embedding = sample_gaussian_tensors(audio_mean, audio_logvar, num_samples, normalize=True)
            DEE_exp_embedding = sample_gaussian_tensors(exp_mean, exp_logvar, num_samples, normalize=True)
            if num_samples == 1 :
                DEE_audio_embedding = DEE_audio_embedding.squeeze(1)
                DEE_exp_embedding = DEE_exp_embedding.squeeze(1)
    raise ValueError(f"Invalid prob_method '{prob_method}'")

def calculate_VV_emo_loss(model, target_exp, pred_exp, 
                               point_DEE, normalize_exp=True,
                               num_samples = None, affectnet_feature_extractor=None,
                               prob_method='csd'):
    if normalize_exp :
        pred_exp = (pred_exp - torch.mean(pred_exp, dim=1, keepdim=True)) / torch.std(pred_exp, dim=1, keepdim=True)
        target_exp = (target_exp - torch.mean(target_exp, dim=1, keepdim=True)) / torch.std(target_exp, dim=1, keepdim=True)
    if point_DEE :
        if affectnet_feature_extractor is not None:
            pred_exp = affectnet_feature_extractor.extract_feature_from_layer(pred_exp, layer_num = -2)
            target_exp = affectnet_feature_extractor.extract_feature_from_layer(target_exp, layer_num = -2)
        pred_embedding = model.encode_parameter(pred_exp)
        target_embedding = model.encode_parameter(target_exp)
        # Normalize before computing cosine similarity
        pred_embedding /= pred_embedding.norm(dim=1, keepdim=True)
        target_embedding /= target_embedding.norm(dim=1, keepdim=True)
        emotion_consistency_loss = 1-F.cosine_similarity(pred_embedding, target_embedding, dim=1).mean()
        return emotion_consistency_loss
    else :
        if affectnet_feature_extractor is not None:
            pred_exp = affectnet_feature_extractor.extract_feature_from_layer(pred_exp, layer_num = -2)
            target_exp = affectnet_feature_extractor.extract_feature_from_layer(target_exp, layer_num = -2)
        target_exp_mean, target_exp_logvar = model.exp_encoder(target_exp)
        exp_mean, exp_logvar = model.exp_encoder(pred_exp)
        # sampling
        if prob_method == 'csd' :
            mu_pdist = ((target_exp_mean - exp_mean) ** 2).sum(-1)
            sigma_pdist = ((torch.exp(target_exp_logvar) + torch.exp(exp_logvar))).sum(-1)
            logits = mu_pdist + sigma_pdist
            logits = -model.criterion.negative_scale * logits + model.criterion.shift # (Bs, BS)
            labels = torch.ones(logits.shape[0], dtype=logits.dtype, device=logits.device)
            loss = model.criterion.bceloss(logits, labels)
            return loss
        elif prob_method == 'mean':
            mu_pdist = ((target_exp_mean - exp_mean) ** 2).sum(-1)
            return mu_pdist.mean()
        
        elif prob_method == 'sample' :
            raise NotImplementedError
            DEE_target_exp_embedding = sample_gaussian_tensors(target_exp_mean, target_exp_logvar, num_samples, normalize=True)
            DEE_exp_embedding = sample_gaussian_tensors(exp_mean, exp_logvar, num_samples, normalize=True)
            if num_samples == 1 :
                DEE_audio_embedding = DEE_target_exp_embedding.squeeze(1)
                DEE_exp_embedding = DEE_exp_embedding.squeeze(1)
    raise ValueError(f"Invalid prob_method '{prob_method}'")
