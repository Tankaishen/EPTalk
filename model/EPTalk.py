import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
from funasr import AutoModel
sys.path.append("/path/to/EPTalk/model")
from MLP import MLP
import fairseq
from dataclasses import dataclass
from wav2vec import Wav2Vec2Model, Wav2Vec2ForCTC, linear_interpolation #, Wav2Vec2Encoder
from My_transformer.attention import *
from My_transformer.Emo_Gate import *
from omegaconf import OmegaConf

class MultiLevelPeriodicPositionalEncoding(nn.Module):
    def __init__(self, d_model, PPE_scale, dropout=0.1, period=25, max_seq_len=600,scale_factor=1):
        # self.is_768, self.PPE_scale, period=25, scale_factor=scale_factor
        super(MultiLevelPeriodicPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe_list = []
        for scale in PPE_scale:
            repeat_num = (max_seq_len//(period*scale))
            position = torch.arange(0, period*scale, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe = torch.zeros(period*scale, d_model)
            pe[:, 0::2] += torch.sin(position * div_term)
            pe[:, 1::2] += torch.cos(position * div_term)
            pe = pe.unsqueeze(0) # (1, period, d_model)
            pe = pe.repeat(1, repeat_num, 1)
            pe_list.append(pe)
        pe_list = torch.sum(torch.cat(pe_list,dim=0),dim=0)/scale_factor
        pe_list = pe_list.unsqueeze(0)
        # self.register_buffer('pe', pe)
        self.register_buffer('pe_list', pe_list)

    def forward(self, x):
        x = x + self.pe_list[:, :x.size(1), :]
        return self.dropout(x)

def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X

@dataclass
class UserDirModule:
    user_dir: str

def get_slopes(n):
    def get_slopes_power_of_2(n):
        start = (2**(-2**-(math.log2(n)-3)))
        ratio = start
        return [start*ratio**i for i in range(n)]
    if math.log2(n).is_integer():
        return get_slopes_power_of_2(n)                   
    else:                                                 
        closest_power_of_2 = 2**math.floor(math.log2(n)) 
        return get_slopes_power_of_2(closest_power_of_2) + get_slopes(2*closest_power_of_2)[0::2][:n-closest_power_of_2]

def init_alibi_biased_mask_future(num_heads, max_seq_len):
    """
    Returns a mask for the self-attention layer. The mask is initialized according to the ALiBi paper but 
    with not with the future masked out.
    The diagonal is filled with 0 and the lower triangle is filled with a linear function of the position. 
    The upper triangle is filled symmetrically with the lower triangle.
    That lowers the attention to the past and the future (the number gets lower the further away from the diagonal it is).
    """
    period = 1
    slopes = torch.Tensor(get_slopes(num_heads))
    bias = torch.arange(start=0, end=max_seq_len).unsqueeze(1).repeat(1,period).view(-1)//(period)
    bias = - torch.flip(bias,dims=[0])
    alibi = torch.zeros(max_seq_len, max_seq_len)
    for i in range(max_seq_len):
        alibi[i, :i+1] = bias[-(i+1):]
    alibi = slopes.unsqueeze(1).unsqueeze(1) * alibi.unsqueeze(0)
    # mask = alibi - torch.flip(alibi, [1, 2])
    mask = alibi + torch.flip(alibi, [1, 2])
    return mask

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len=512):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=0.1)
        
        # Compute the positional encodings in advance
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pos_enc = torch.zeros((max_seq_len, d_model))
        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)
        pos_enc = pos_enc.unsqueeze(0)  # Add batch dimension
        
        self.register_buffer('pos_enc', pos_enc)

    def forward(self, x):
        x = x + self.pos_enc[:, :x.size(1)].clone().detach()
        return self.dropout(x)
    
# Temporal Bias, brrowed from https://github.com/EvelynFan/FaceFormer/blob/main/faceformer.py
def init_biased_mask(n_head, max_seq_len, period):
    def get_slopes(n):
        def get_slopes_power_of_2(n):
            start = (2 ** (-2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio ** i for i in range(n)]

        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)
        else:
            closest_power_of_2 = 2 ** math.floor(math.log2(n))
            return get_slopes_power_of_2(closest_power_of_2) + get_slopes(2 * closest_power_of_2)[0::2][
                                                               :n - closest_power_of_2]

    slopes = torch.Tensor(get_slopes(n_head))
    bias = torch.arange(start=0, end=max_seq_len, step=period).unsqueeze(1).repeat(1, period).view(-1) // (period)
    bias = - torch.flip(bias, dims=[0])
    alibi = torch.zeros(max_seq_len, max_seq_len)
    for i in range(max_seq_len):
        alibi[i, :i + 1] = bias[-(i + 1):]
    alibi = slopes.unsqueeze(1).unsqueeze(1) * alibi.unsqueeze(0)
    mask = (torch.triu(torch.ones(max_seq_len, max_seq_len)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    mask = mask.unsqueeze(0) + alibi
    return mask


# Input Representation Adjustment, brrowed from https://github.com/galib360/FaceXHuBERT
def inputRepresentationAdjustment(audio_embedding_matrix, vertex_matrix, ifps, ofps):
    if ifps % ofps == 0:
        factor = -1 * (-ifps // ofps)
        if audio_embedding_matrix.shape[1] % 2 != 0:
            audio_embedding_matrix = audio_embedding_matrix[:, :audio_embedding_matrix.shape[1] - 1]

        if audio_embedding_matrix.shape[1] > vertex_matrix.shape[1] * 2:
            audio_embedding_matrix = audio_embedding_matrix[:, :vertex_matrix.shape[1] * 2]

        elif audio_embedding_matrix.shape[1] < vertex_matrix.shape[1] * 2:
            vertex_matrix = vertex_matrix[:, :audio_embedding_matrix.shape[1] // 2]
    else:
        factor = -1 * (-ifps // ofps)
        audio_embedding_seq_len = vertex_matrix.shape[1] * factor
        audio_embedding_matrix = audio_embedding_matrix.transpose(1, 2)
        audio_embedding_matrix = F.interpolate(audio_embedding_matrix, size=audio_embedding_seq_len, align_corners=True,
                                               mode='linear')
        audio_embedding_matrix = audio_embedding_matrix.transpose(1, 2)

    frame_num = vertex_matrix.shape[1]
    audio_embedding_matrix = torch.reshape(audio_embedding_matrix, (
    audio_embedding_matrix.shape[0], audio_embedding_matrix.shape[1] // factor, audio_embedding_matrix.shape[2] * factor))
    return audio_embedding_matrix, vertex_matrix, frame_num

def inputRepresentationAdjustment_audio_only(audio_embedding_matrix, ifps, ofps):
    if ifps % ofps == 0:
        factor = -1 * (-ifps // ofps)
        if audio_embedding_matrix.shape[1] % 2 != 0:
            audio_embedding_matrix = audio_embedding_matrix[:, :audio_embedding_matrix.shape[1] - 1]

    audio_embedding_matrix = torch.reshape(audio_embedding_matrix, (
    audio_embedding_matrix.shape[0], audio_embedding_matrix.shape[1] // factor, audio_embedding_matrix.shape[2] * factor))
    return audio_embedding_matrix

class EPTalk(nn.Module):
    def __init__(self, args, FER):
        super(EPTalk, self).__init__()
        self.args = args
        self.Fer = FER
        for param in self.Fer.parameters():
            param.requires_grad = False
        self.device = args.device
        self.dataset = args.dataset
        self.args = args
        self.audio_encoder = Wav2Vec2Model.from_pretrained(args.wav2vec_model)
        self.text_encoder = Wav2Vec2ForCTC.from_pretrained(args.wav2vec_model)
        self.audio_encoder.feature_extractor._freeze_parameters()
        for param in self.text_encoder.parameters():
            param.requires_grad = False

        decoder_layer = nn.TransformerDecoderLayer(d_model=args.feature_dim, nhead=args.num_heads,
                                                   dim_feedforward=2 * args.feature_dim, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=4)
        self.audio_feature_map = nn.Linear(2*args.wav2vec_dim, args.feature_dim)
        self.audio_feature_map2 = nn.Linear(768*2, args.feature_dim)
        self.vertice_map_r = nn.Linear(args.feature_dim, args.vertice_dim)

        self.transformer = nn.Transformer(d_model=args.feature_dim, batch_first=True)
        self.dropout = nn.Dropout(p=0.0, inplace=False)
        self.lm_head = nn.Linear(args.wav2vec_dim, 33)

        self.emo_relu = nn.LeakyReLU()
        self.emo_dropout = nn.Dropout(0.1)

        nn.init.constant_(self.vertice_map_r.weight, 0)
        nn.init.constant_(self.vertice_map_r.bias, 0)

        self.emo_encoder = ProbAudioEncoder(args)
        self.emo_proj_2 = nn.Linear(2*args.feature_dim, args.feature_dim)

        self.edp = EmotionDynamicPerceive(args)
        if(args.cosine_similarity):
            self.cosine_similarity = nn.CosineSimilarity(dim=-1, eps=1e-6)
        else:
            self.mse_loss = nn.MSELoss()

    def forward(self, audio, template, vertice=None, labels=None):
        if(vertice is not None):
            vertice = vertice.reshape(vertice.shape[0],vertice.shape[1],-1)
        template = template.unsqueeze(1)
        # frame_num = vertice.shape[1]
        hidden_states = self.audio_encoder(audio, self.dataset).last_hidden_state  # Wav2Vec2Model
        audio_emotion = audio
        # audio_emotion = torch.nn.functional.layer_norm(audio,(audio[0].shape[-1],))
        emo_hidden_states = self.emo_encoder(audio_emotion)
        # hidden_states = self.audio_encoder(audio)

        if self.dataset != "vocaset":
            if vertice==None:
                vertice = template.repeat(1,hidden_states.shape[1]//2,1)
            hidden_states, vertice, frame_num = inputRepresentationAdjustment(hidden_states, vertice, 50, 25)
            emo_hidden_states, _, frame_num = inputRepresentationAdjustment(emo_hidden_states, vertice, 50, 25)
            assert hidden_states.shape[1]==emo_hidden_states.shape[1]
            hidden_states = hidden_states[:, :frame_num]
            motion_gt = (vertice - template).detach()
        elif self.dataset == "vocaset":
            pass
        hidden_states_content = self.audio_feature_map(hidden_states) # self.audio_feature_map = nn.Linear

        hidden_states_emotion = self.audio_feature_map2(emo_hidden_states) # self.audio_feature_map = nn.Linear
        hidden_states_emotion = self.emo_relu(hidden_states_emotion)
        hidden_states_emotion = self.emo_dropout(hidden_states_emotion)
        hidden_states_emotion = l2norm(hidden_states_emotion,dim=-1)
        
        weights, q = self.edp(hidden_states_emotion, hidden_states_content)
        # if(self.args.analysis):
        #     return weights
        vertice_input = torch.cat((q, hidden_states_content),dim=-1)
        vertice_input = self.emo_proj_2(vertice_input) # self.emo_proj_2 = nn.Linear

        cross_feat2 = self.transformer_decoder(vertice_input, hidden_states_content)
        vertice_out = self.vertice_map_r(cross_feat2) #


        audio_model = self.text_encoder(audio)
        text_hidden_states = audio_model.hidden_states
        text_logits = audio_model.logits
        frame_num = text_hidden_states.shape[1]
        lip_out = vertice_out.reshape(vertice_out.shape[0], vertice_out.shape[1], -1, 3)[:, :, self.lip_mask,
                  :].reshape(vertice_out.shape[0], vertice_out.shape[1], -1)
        # lip_gt = vertice.reshape(vertice.shape[0], vertice.shape[1], -1, 3)[:, :, self.lip_mask, :].reshape(
        #     vertice.shape[0], vertice.shape[1], -1)
        lip_offset = self.lip_map(lip_out)

        if self.dataset == "vocaset":
            lip_offset = linear_interpolation(lip_offset, 30, 50, output_len=frame_num)
        elif self.dataset != "vocaset":
            text_hidden_states = text_hidden_states[:, :vertice_out.shape[1] * 2]
            text_logits = text_logits[:, :vertice_out.shape[1] * 2]
            frame_num = text_hidden_states.shape[1]
            lip_offset = linear_interpolation(lip_offset, 25, 50, output_len=frame_num)
        lip_features = self.transformer(lip_offset, lip_offset)
        logits = self.lm_head(self.dropout(lip_features))
        motion = vertice_out
        vertice_out = vertice_out + template

        # return vertice_out, vertice, lip_features, text_hidden_states, logits, text_logits, motion, motion_gt, weights
        return vertice_out, vertice, lip_features, text_hidden_states, logits, text_logits, motion, motion_gt
    
    def predict(self,audio,template):
        hidden_states = self.audio_encoder(audio, self.dataset).last_hidden_state
        audio_emotion = audio
        emo_hidden_states = self.emo_encoder(audio_emotion)
        hidden_states = inputRepresentationAdjustment_audio_only(hidden_states, 50, 25)
        emo_hidden_states = inputRepresentationAdjustment_audio_only(emo_hidden_states, 50, 25)
        assert hidden_states.shape[1]==emo_hidden_states.shape[1]

        hidden_states_content = self.audio_feature_map(hidden_states) # self.audio_feature_map = nn.Linear

        hidden_states_emotion = self.audio_feature_map2(emo_hidden_states) # self.audio_feature_map = nn.Linear
        hidden_states_emotion = self.emo_relu(hidden_states_emotion)
        hidden_states_emotion = self.emo_dropout(hidden_states_emotion)
        hidden_states_emotion = l2norm(hidden_states_emotion,dim=-1)
        
        weights, q = self.edp(hidden_states_emotion, hidden_states_content)
        # if(self.args.analysis):
        #     return weights
        vertice_input = torch.cat((q, hidden_states_content),dim=-1)
        vertice_input = self.emo_proj_2(vertice_input) # self.emo_proj_2 = nn.Linear

        cross_feat2 = self.transformer_decoder(vertice_input, hidden_states_content)
        vertice_out = self.vertice_map_r(cross_feat2)
        vertice_out = vertice_out + template

        return vertice_out

    def emo_loss(self, motion, motion_gt):
        feat = self.Fer.extract_feature_from_layer(motion, layer_num = -2)
        feat_gt = self.Fer.extract_feature_from_layer(motion_gt, layer_num = -2)
        if(self.args.cosine_similarity):
            cosine_loss = 1 - self.cosine_similarity(feat, feat_gt)
            cosine_loss = torch.sum(cosine_loss)/cosine_loss.shape[0]
            return feat, feat_gt, cosine_loss
        else:
            mse_loss = self.mse_loss(feat, feat_gt)
            return feat, feat_gt, mse_loss

class DynamicPoolingLayer(nn.Module):
    def __init__(self, pool_type):
        super(DynamicPoolingLayer, self).__init__()
        self.pool_type = pool_type
    def forward(self, x):
        if self.pool_type == 'avg':
            return torch.mean(x, dim=1)
        elif self.pool_type == 'max':
            return torch.max(x, dim=1).values
        else:
            raise NotImplementedError('pool type {} is not implemented'.format(self.pool_type))
        
class EmotionDynamicPerceive(nn.Module):
    def __init__(self,config):
        super(EmotionDynamicPerceive,self).__init__()
        self.feature_dim = 1024
        self.args = config
        # Multi Granularity PPE
        self.scale_factor = 3
        self.PPE_scale = [1,2,4]
        self.multi_granularity_PPE = MultiLevelPeriodicPositionalEncoding(self.feature_dim, self.PPE_scale, period=25, scale_factor=self.scale_factor)
        emo_encoder_layer = torch.nn.TransformerEncoderLayer(
                    d_model=config.feature_dim, 
                    nhead=4, dim_feedforward=2*config.feature_dim, 
                    activation='gelu',
                    dropout=0.1, batch_first=True)
        self.emo_decoder = torch.nn.TransformerEncoder(emo_encoder_layer, num_layers=1)
        self.emo_proj1 = nn.Linear(self.feature_dim, self.feature_dim)
        # self.emo_activate1 = nn.PReLU(decoder_config['feature_dim'])

        # Multi Granularity Pooling
        self.multi_granularity_pooling = []
        for scale in self.PPE_scale[1:]:
            self.multi_granularity_pooling.append(nn.AvgPool1d(kernel_size=scale,stride=scale))
        self.adp_pooling = nn.AdaptiveAvgPool1d
        # self.agg_conv = PositionwiseFeedForward(decoder_config['feature_dim'] * 3,decoder_config['feature_dim'] * 3)
        self.emo_proj2 = nn.Linear(self.feature_dim * self.scale_factor, self.feature_dim)
        self.emo_activate = nn.PReLU(self.feature_dim)
        self.emo_proj3 = nn.Linear(self.feature_dim , self.feature_dim)
        # self.emo_dropout = nn.Dropout(p=0.1)
        # Subspace-wise Route
        self.dynamic_module = Subspace_routing(self.feature_dim,self.feature_dim,
                                                    64,
                                                    [2,4,8,16])
        # self.gate = Gate2(config, self.feature_dim, self.feature_dim*2 , self.feature_dim)
        self.gate = Gate(config, self.feature_dim, self.feature_dim*2 , self.feature_dim)

    def element_wise_route(self,em_embedding):
        # Emotion embedding attention
        T = em_embedding.shape[1]
        emo_feat = self.multi_granularity_PPE(em_embedding)
        emo_feat = em_embedding
        emo_feat = self.emo_decoder(emo_feat) # transformerencoderblock
        emo_feat_route = self.emo_proj1(emo_feat) # nn.Linear
        # Multi granularity pooling
        emo_feat_pooled = []
        emo_feat_route = emo_feat_route.permute(0,2,1)
        for pool in self.multi_granularity_pooling:
            pooled_feat = pool(emo_feat_route)
            pooled_feat_upsampling = self.adp_pooling(T)(pooled_feat).permute(0,2,1)
            emo_feat_pooled.append(pooled_feat_upsampling)
        emo_feat_route = emo_feat_route.permute(0,2,1)
        emo_feat_pooled.append(emo_feat_route)
        # Post project
        multi_pooled_emo_ori = torch.cat(emo_feat_pooled,dim=-1)
        multi_pooled_emo = self.emo_proj2(multi_pooled_emo_ori) # nn.Linear
        multi_pooled_emo = multi_pooled_emo.permute(0,2,1)
        multi_pooled_emo = self.emo_activate(multi_pooled_emo) # Prelu

        multi_pooled_emo = multi_pooled_emo.permute(0,2,1)
        multi_pooled_emo = self.emo_proj3(multi_pooled_emo) # nn.Linear
        multi_pooled_emo_res = emo_feat_route + multi_pooled_emo
        multi_pooled_emo_res = l2norm(multi_pooled_emo_res,dim=-1)
        return multi_pooled_emo_res # reconstructed emo & attentioned emo

    def forward(self,em_embedding, audio_embedding_new):
        multi_pooled_emo = self.element_wise_route(em_embedding)
        if self.args.subspace:
            reconstruct_emo = self.dynamic_module(em_embedding, multi_pooled_emo, audio_embedding_new)
        else:
            reconstruct_emo = multi_pooled_emo
        emo_w_intensity, em_embedding_new  = self.gate(reconstruct_emo,audio_embedding_new)
        q = emo_w_intensity * reconstruct_emo
        # q = audio_embedding_new * (1-emo_w_intensity) + emo_w_intensity * em_embedding_new
        return emo_w_intensity, q
    
class ProbAudioEncoder(nn.Module): 
    def __init__(self, args) :
        super(ProbAudioEncoder, self).__init__()
        self.args=args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_dir = 'DEE/models/emo2vec'
        model_path = UserDirModule(model_dir)
        emo2vec_checkpoint = 'DEE/models/emo2vec/checkpoint/emotion2vec_base.pt'
        fairseq.utils.import_user_module(model_path)
        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([emo2vec_checkpoint])
        self.embedding_model = model[0]
        finetuned_model = AutoModel(model='iic/emotion2vec_base_finetuned',
                                            disable_update=True,device=args.device)
        finetuned_model_state_dict = finetuned_model.model.state_dict()
        pt_model_state_dict = self.embedding_model.state_dict()
        new_checkpoint = {k: v for k, v in finetuned_model_state_dict.items() if k in list(pt_model_state_dict.keys())}
        self.embedding_model.load_state_dict(new_checkpoint)

        for param in self.embedding_model.parameters():
            param.requires_grad = False
        self.embedding_model.eval()
        
        self.unfreeze_emo2vec = 0

        self.audio_encoder_layer = nn.TransformerEncoderLayer(d_model=768, nhead=8, batch_first=True)
        self.audio_encoder = nn.TransformerEncoder(self.audio_encoder_layer, num_layers=4)
        
    def forward_base(self, audio) :
        '''
        audio : (BS, seq_length) : (BS, 6400)
        '''
        if not self.unfreeze_emo2vec:
            self.embedding_model.eval() # always set to eval mode
        embeddings = self.embedding_model.extract_features(audio, padding_mask=None) 
        embeddings = embeddings['x'] # (BS, 50*sec, 768)
        feats = embeddings
        return feats
    
    def forward(self, audio):
        feats = self.forward_base(audio)
        feats = self.audio_encoder(feats)
        return feats