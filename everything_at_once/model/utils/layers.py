import torch
import torch as th
from torch import nn as nn
from torch.nn import functional as F
from timm.models.vision_transformer import DropPath, Mlp, Attention


class GatedEmbeddingUnit(nn.Module):
    def __init__(self, input_dimension, output_dimension):
        super().__init__()
        self.fc = nn.Linear(input_dimension, output_dimension)
        self.cg = ContextGating(output_dimension)

    def forward(self, x):
        x = self.fc(x)
        x = self.cg(x)
        return x


class FusedGatedUnit(nn.Module):
    def __init__(self, input_dimension, output_dimension):
        super(FusedGatedUnit, self).__init__()
        self.fc_audio = nn.Linear(input_dimension, output_dimension)
        self.fc_text = nn.Linear(input_dimension, output_dimension)
        self.cg = ContextGating(output_dimension)

    def forward(self, audio, text):
        audio = self.fc_audio(audio)
        text = self.fc_text(text)
        x = audio + text
        x = self.cg(x)
        return x


class ContextGating(nn.Module):
    def __init__(self, dimension):
        super(ContextGating, self).__init__()
        self.fc = nn.Linear(dimension, dimension)

    def forward(self, x):
        x1 = self.fc(x)
        x = th.cat((x, x1), 1)
        return F.glu(x, 1)


class SentenceMaxpool(nn.Module):
    def __init__(self, word_dimension, output_dim):
        super(SentenceMaxpool, self).__init__()
        self.fc = nn.Linear(word_dimension, output_dim)

    def forward(self, x):
        x = self.fc(x)
        x = F.relu(x)
        return torch.max(x, dim=1)[0]


class FusionBlock(nn.Module):
    """
        Adopted from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        Copyright 2020, Ross Wightman
    """
    def __init__(
            self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., init_values=None,
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = FusionAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, attention_mask=None):
        x = x + self.drop_path(self.attn(self.norm1(x), attention_mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class FusionAttention(Attention):
    """
    Adopted from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    Copyright 2020, Ross Wightman
    """
    def forward(self, x, attention_mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if attention_mask is not None:
            zero_attention_mask = (attention_mask == 0).view(B, 1, 1, N).expand_as(attn)  # (bs, n_heads, q_length, k_length)
            attn.masked_fill_(zero_attention_mask, -float("inf"))  # (bs, n_heads, q_length, k_length)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def get_projection(input_dim, output_dim, projection_type):
    if projection_type == 'minimal':
        return nn.Linear(input_dim, output_dim)
    if projection_type == 'gated':
        return GatedEmbeddingUnit(input_dim, output_dim)
    elif projection_type == '':
        return nn.Identity()
    else:
        raise NotImplementedError