import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import math
from modules import *
from typing import Optional, Any, Union, Callable
import copy

# Temporal Bias,some are borrowed from https://github.com/EvelynFan/FaceFormer/blob/main/faceformer.py
# Some are borrowed from https://github.com/huifu99/Mimic
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

class FeatureProjection(nn.Module):
    def __init__(self, feature_dim, hidden_size, feat_proj_dropout=0.0):
        super().__init__()
        self.layer_norm = nn.LayerNorm(feature_dim)
        self.projection = nn.Linear(feature_dim, hidden_size)
        self.dropout = nn.Dropout(feat_proj_dropout)

    def forward(self, hidden_states):
        # non-projected hidden states are needed for quantization
        norm_hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.projection(norm_hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states, norm_hidden_states

class FeatureProjectionIN(nn.Module):
    def __init__(self, feature_dim, hidden_size, feat_proj_dropout=0.0):
        super().__init__()
        self.instance_norm = nn.InstanceNorm1d(feature_dim)
        self.projection = nn.Linear(feature_dim, hidden_size)
        self.dropout = nn.Dropout(feat_proj_dropout)

    def forward(self, hidden_states):  # hidden_states (B,L,C)
        # non-projected hidden states are needed for quantization
        hidden_states = hidden_states.transpose(-2,-1)  # (B,C,L)
        norm_hidden_states = self.instance_norm(hidden_states)
        norm_hidden_states = norm_hidden_states.transpose(-2,-1)
        hidden_states = self.projection(norm_hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states, norm_hidden_states

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
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

class PeriodicPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, period=25, max_seq_len=600):
        super(PeriodicPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(period, d_model)
        position = torch.arange(0, period, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # (1, period, d_model)
        repeat_num = (max_seq_len//period) + 1
        pe = pe.repeat(1, repeat_num, 1)
        self.register_buffer('pe', pe)
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class Conv1DLN(nn.Module):
    def __init__(self, in_conv_dim, out_conv_dim, kernel, stride, padding, bias=False):
        super().__init__()

        self.conv = nn.Conv1d(
            in_conv_dim,
            out_conv_dim,
            kernel_size=kernel,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        self.layer_norm = nn.LayerNorm(out_conv_dim, elementwise_affine=True)
        self.activation = F.relu

    def forward(self, hidden_states):  # hidden_states: (B,C,L)
        hidden_states = self.conv(hidden_states)
        hidden_states = hidden_states.transpose(-2, -1)  # (B,L,C)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = hidden_states.transpose(-2, -1)  # (B,C,L)

        hidden_states = self.activation(hidden_states, inplace=True)
        return hidden_states
    
class Conv1DIN(nn.Module):
    def __init__(self, in_conv_dim, out_conv_dim, kernel, stride, padding, bias=False):
        super().__init__()

        self.conv = nn.Conv1d(
            in_conv_dim,
            out_conv_dim,
            kernel_size=kernel,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        self.IN = nn.InstanceNorm1d(out_conv_dim, affine=False)
        self.activation = F.relu

    def forward(self, hidden_states):  # hidden_states: (B,C,L)
        hidden_states = self.conv(hidden_states)

        # hidden_states = hidden_states.transpose(-2, -1)
        hidden_states = self.IN(hidden_states)
        # hidden_states = hidden_states.transpose(-2, -1)

        hidden_states = self.activation(hidden_states)
        return hidden_states

class AdaIN(nn.Module):
    def __init__(self, c_cond: int, c_h: int):
        super(AdaIN, self).__init__()
        self.c_h = c_h
        self.norm_layer = nn.InstanceNorm1d(c_h, affine=False)
        self.linear_layer = nn.Linear(c_cond, c_h * 2)

    def forward(self, x: Tensor, x_cond: Tensor) -> Tensor:
        x_cond = self.linear_layer(x_cond)
        mean, std = x_cond[:, : self.c_h], x_cond[:, self.c_h :]
        mean, std = mean.unsqueeze(-1), std.unsqueeze(-1)
        x = x.transpose(1,2)  # (N,C,L)
        x = self.norm_layer(x)
        x = x * std + mean
        x = x.transpose(1,2)  # (N,L,C)
        return x
    
class TransformerDecoderLayer(nn.TransformerDecoderLayer):
  def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1, activation: Union[str, Callable[[Tensor], Tensor]] = F.relu, layer_norm_eps: float = 0.00001, batch_first: bool = False, norm_first: bool = False, device=None, dtype=None) -> None:
     super().__init__(d_model, nhead, dim_feedforward, dropout, activation, layer_norm_eps, batch_first, norm_first, device, dtype)
     self.norm1 = StyleAdaptiveLayerNorm(d_model, d_model)
     self.norm2 = StyleAdaptiveLayerNorm(d_model, d_model)
     self.norm3 = StyleAdaptiveLayerNorm(d_model, d_model)
  
  def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None,
              tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None,
              style_code=None) -> Tensor:
      r"""Pass the inputs (and mask) through the decoder layer.

      Args:
          tgt: the sequence to the decoder layer (required).
          memory: the sequence from the last layer of the encoder (required).
          tgt_mask: the mask for the tgt sequence (optional).
          memory_mask: the mask for the memory sequence (optional).
          tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
          memory_key_padding_mask: the mask for the memory keys per batch (optional).

      Shape:
          see the docs in Transformer class.
      """
      # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf

      x = tgt
      if self.norm_first:
          x = x + self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask)
          x = x + self._mha_block(self.norm2(x), memory, memory_mask, memory_key_padding_mask)
          x = x + self._ff_block(self.norm3(x))
      else:
          x = x + self._sa_block(x, tgt_mask, tgt_key_padding_mask)
          x = self.norm1(x, style_code)
          x = x + self._mha_block(x, memory, memory_mask, memory_key_padding_mask)
          x = self.norm2(x, style_code)
          x = x + self._ff_block(x)
          x = self.norm3(x, style_code)
      return x
  

from torch.nn.modules.container import ModuleList
def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])
class TransformerDecoder(nn.Transformer):
    r"""TransformerDecoder is a stack of N decoder layers

    Args:
        decoder_layer: an instance of the TransformerDecoderLayer() class (required).
        num_layers: the number of sub-decoder-layers in the decoder (required).
        norm: the layer normalization component (optional).

    Examples::
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        >>> transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        >>> memory = torch.rand(10, 32, 512)
        >>> tgt = torch.rand(20, 32, 512)
        >>> out = transformer_decoder(tgt, memory)
    """
    __constants__ = ['norm']

    def __init__(self, decoder_layer, num_layers, norm=None):
        super(TransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                style_code=None) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer in turn.

        Args:
            tgt: the sequence to the decoder (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = tgt

        for mod in self.layers:
            output = mod(output, memory, tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask,
                         style_code=style_code)

        if self.norm is not None:
            output = self.norm(output)

        return output