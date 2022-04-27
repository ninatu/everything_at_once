import collections

from timm.models.vision_transformer import _init_vit_weights, trunc_normal_
import torch.nn as nn
from functools import partial
import torch
from everything_at_once.model.utils.layers import FusionBlock


class FusionTransformer(nn.Module):
    def __init__(self, embed_dim=768, depth=1, num_heads=12, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=None,
                 act_layer=None,
                 use_cls_token=False,
                 ):
        super().__init__()

        self.embed_dim = embed_dim

        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        else:
            self.cls_token = None

        self.masking_token = nn.Parameter(torch.zeros(embed_dim))

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            FusionBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer,
            )
            for i in range(depth)])

        self.norm = norm_layer(embed_dim) # TODO: not needed, remove?
        self.init_weights()

    def init_weights(self):
        trunc_normal_(self.masking_token, std=.02)
        if self.cls_token is not None:
            trunc_normal_(self.cls_token, std=.02)
        self.apply(_init_vit_weights)

    def forward(self, text=None, video=None, audio=None):
        # concatenate tokens
        data = [text, video, audio]
        tokens = [x['all_tokens'] for x in data if x is not None]
        tokens = torch.cat(tokens, dim=1)

        # concatenate attention masks
        tokens_mask = [x['attention_mask'] for x in data if x is not None]
        tokens_mask = torch.cat(tokens_mask, dim=1)

        # concatenate cls token
        if self.cls_token is None:
            offset = 0
        else:
            cls_token = self.cls_token.expand(tokens.shape[0], -1, -1)
            tokens = torch.cat((cls_token, tokens), dim=1)
            cls_token_mask = torch.ones((1, 1)).to(tokens_mask.device).expand(tokens_mask.shape[0], -1)
            tokens_mask = torch.cat((cls_token_mask, tokens_mask), dim=1)
            offset = 1

        for block in self.blocks:
            tokens = block(tokens, attention_mask=tokens_mask)

        output = collections.OrderedDict()

        def _get_average(tokens, attention_mask):
            attention_mask = attention_mask.unsqueeze(2).expand_as(tokens)
            return (tokens * attention_mask).sum(1) / attention_mask.sum(1)

        if text is not None:
            n_tokens = text['all_tokens'].size(1)
            attention_mask = text['attention_mask']
            all_tokens = tokens[:, offset:offset + n_tokens]

            offset += n_tokens
            output['text'] = {
                "all_tokens": all_tokens,
                "attention_mask": attention_mask,
            }

        if video is not None:
            n_tokens = video['all_tokens'].size(1)
            attention_mask = video['attention_mask']
            all_tokens = tokens[:, offset:offset + n_tokens]

            offset += n_tokens
            output['video'] = {
                "all_tokens": all_tokens,
                "attention_mask": attention_mask,
            }

        if audio is not None:
            n_tokens = audio['all_tokens'].size(1)
            attention_mask = audio['attention_mask']
            all_tokens = tokens[:, offset: offset + n_tokens]

            offset += n_tokens
            output['audio'] = {
                "all_tokens": all_tokens,
                "attention_mask": attention_mask,
            }

        if self.cls_token is None:
            for key, value in output.items():
                output[key]['embed'] = _get_average(value["all_tokens"], value['attention_mask'])
        else:
            modalities = list(output.keys())
            modalities = '_'.join(modalities)
            if modalities not in output:
                output[modalities] = {}
            output[modalities]['embed'] = tokens[:, 0]

        return output
