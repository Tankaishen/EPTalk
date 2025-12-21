import torch
import torch.nn as nn
import torch.nn.functional as F
# import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append("/data/code/EPTalk/model/My_transformer")
from attention import SemanticAlignment, SemanticAttention, EmotionAttention, DynamicSemanticModule
# from Models import Encoder_emo
# from Models import Decoder_emo_nomul
    
class Gate(nn.Module):
    def __init__(self, config, feature_size, hidden_size, output_size):
        super(Gate, self).__init__()
        self.sem_align_hidden_size = hidden_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.feature_size = feature_size
        self.is_ScaledDotProduct = 1

        # self.semantic_alignment = SemanticAlignment(
        #     query_size=self.feature_size,
        #     feat_size=self.feature_size,
        #     bottleneck_size=self.sem_align_hidden_size)
        if(self.is_ScaledDotProduct):
            self.intensity_alignment = nn.MultiheadAttention(self.feature_size,num_heads=4)
        else:
            self.intensity_alignment = SemanticAlignment(
                query_size=self.feature_size,
                feat_size=self.feature_size,
                bottleneck_size=self.hidden_size)

        self.w_1 = nn.Linear(feature_size,feature_size)
        self.w_2 = nn.Linear(feature_size,feature_size)
        self.gate = nn.Linear(feature_size,1)
        self.out = nn.Linear(self.feature_size*2, self.feature_size)

    def sp(self,em_embededing, audio_embedding):
        B = em_embededing.shape[0]
        if(self.is_ScaledDotProduct):
            intensity_feat, attn_weights = self.intensity_alignment(em_embededing, audio_embedding, audio_embedding)
        else:
            intensity_feat,_,_ = self.intensity_alignment(
                phr_feats=em_embededing,
                vis_feats=audio_embedding)
        intensity_feat = self.l2norm(self.w_1(intensity_feat))
        audio_embedding = self.l2norm(self.w_2(audio_embedding))
        scaled_feat = intensity_feat * audio_embedding # elementwise-scale
        emo_inten = self.gate(scaled_feat)
        emo_inten = torch.sigmoid(emo_inten)
        # emo_w_intensity = emo_inten * em_embededing
        # reemo_audio_emb = torch.cat((
        #     emo_inten*em_embededing,
        #     audio_embedding),dim=2)
        # feat_evw_c = self.out(feat_evw_c)
        return emo_inten, em_embededing
    
    def l2norm(self, X, dim=-1, eps=1e-8):
        """L2-normalize columns of X
        """
        norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
        X = torch.div(X, norm)
        return X

    def forward(self,reconstruct_emo, audio_embedding):
        output = self.sp(reconstruct_emo,audio_embedding)
        return output

class Gate2(nn.Module):
    def __init__(self, config, feature_size, hidden_size, output_size):
        super(Gate2, self).__init__()
        self.sem_align_hidden_size = hidden_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.feature_size = feature_size
        self.is_ScaledDotProduct = 1

        # self.semantic_alignment = SemanticAlignment(
        #     query_size=self.feature_size,
        #     feat_size=self.feature_size,
        #     bottleneck_size=self.sem_align_hidden_size)
        if(self.is_ScaledDotProduct):
            self.intensity_alignment = nn.MultiheadAttention(self.feature_size,num_heads=4)
        else:
            self.intensity_alignment = SemanticAlignment(
                query_size=self.feature_size,
                feat_size=self.feature_size,
                bottleneck_size=self.hidden_size)

        # self.w_1 = nn.Linear(feature_size,feature_size)
        # self.w_2 = nn.Linear(feature_size,feature_size)
        self.gate = nn.Linear(feature_size,1)
        self.out = nn.Linear(self.feature_size*2, self.feature_size)

    def sp(self,em_embededing, audio_embedding):
        B = em_embededing.shape[0]
        if(self.is_ScaledDotProduct):
            intensity_feat, _ = self.intensity_alignment(em_embededing, audio_embedding, audio_embedding)
        else:
            intensity_feat,_,_ = self.intensity_alignment(
                phr_feats=em_embededing,
                vis_feats=audio_embedding)
        # intensity_feat = self.l2norm(self.w_1(intensity_feat))
        # audio_embedding = self.l2norm(self.w_2(audio_embedding.detach()))
        # similarity = intensity_feat * audio_embedding
        emo_inten = self.gate(intensity_feat)
        emo_inten = torch.sigmoid(emo_inten)
        # emo_w_intensity = emo_inten * em_embededing
        # reemo_audio_emb = torch.cat((
        #     emo_inten*em_embededing,
        #     audio_embedding),dim=2)
        # feat_evw_c = self.out(feat_evw_c)
        return emo_inten, em_embededing
    
    def l2norm(self, X, dim=-1, eps=1e-8):
        """L2-normalize columns of X
        """
        norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
        X = torch.div(X, norm)
        return X

    def forward(self,reconstruct_emo, audio_embedding):
        output = self.sp(reconstruct_emo,audio_embedding)
        return output

