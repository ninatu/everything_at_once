import numpy as np
import torch
from timm.models.layers import trunc_normal_
from torch import nn as nn

from everything_at_once.model.utils.utils import normalize_embeddings
from everything_at_once.model.utils.layers import get_projection
from everything_at_once.model.utils.fusion_transformer import FusionTransformer
from everything_at_once.model.utils.davenet import load_DAVEnet


class EverythingAtOnceModel(nn.Module):
    def __init__(self,
                 video_embed_dim,
                 text_embed_dim,
                 fusion_params,
                 video_max_tokens=None,
                 text_max_tokens=None,
                 audio_max_num_STFT_frames=None,
                 projection_dim=6144,
                 token_projection='gated',
                 projection='gated',
                 cross_modal=True,
                 strategy_audio_pooling='none',
                 davenet_v2=True,
                 individual_projections=True,
                 use_positional_emb=False,
                 ):
        super().__init__()

        self.fusion = FusionTransformer(**fusion_params)

        self.individual_projections = individual_projections
        self.use_positional_emb = use_positional_emb
        self.strategy_audio_pooling = strategy_audio_pooling
        self.cross_modal = cross_modal

        embed_dim = fusion_params['embed_dim']

        self.video_norm_layer = nn.LayerNorm(embed_dim, eps=1e-6)
        self.text_norm_layer = nn.LayerNorm(embed_dim, eps=1e-6)
        self.audio_norm_layer = nn.LayerNorm(embed_dim, eps=1e-6)

        # audio token preprocess
        self.davenet = load_DAVEnet(v2=davenet_v2)

        if audio_max_num_STFT_frames is not None:
            if davenet_v2:
                audio_max_tokens = int(audio_max_num_STFT_frames / 64)
            else:
                audio_max_tokens = int(audio_max_num_STFT_frames / 16)
            self.audio_max_tokens = audio_max_tokens
        else:
            self.audio_max_tokens = None

        if self.use_positional_emb:
            assert video_max_tokens is not None
            assert text_max_tokens is not None
            assert audio_max_num_STFT_frames is not None
            self.video_pos_embed = nn.Parameter(torch.zeros(1, video_max_tokens, embed_dim))
            self.text_pos_embed = nn.Parameter(torch.zeros(1, text_max_tokens, embed_dim))
            self.audio_pos_embed = nn.Parameter(torch.zeros(1, self.audio_max_tokens, embed_dim))
        else:
            self.video_pos_embed = None
            self.text_pos_embed = None
            self.audio_pos_embed = None

        audio_embed_dim = 4096 if davenet_v2 else 1024
        self.video_token_proj = get_projection(video_embed_dim, embed_dim, token_projection)
        self.text_token_proj = get_projection(text_embed_dim, embed_dim, token_projection)
        self.audio_token_proj = get_projection(audio_embed_dim, embed_dim, token_projection)
        
        if not self.individual_projections:
            self.proj = get_projection(embed_dim, projection_dim, projection)
        else:
            self.video_proj = get_projection(embed_dim, projection_dim, projection)
            self.text_proj = get_projection(embed_dim, projection_dim, projection)
            self.audio_proj = get_projection(embed_dim, projection_dim, projection)

        self.init_weights()

    def init_weights(self):
        for weights in [self.video_pos_embed, self.audio_pos_embed, self.text_pos_embed]:
            if weights is not None:
                trunc_normal_(weights, std=.02)

    def _check_and_fix_if_input_empty(self, x, attention_mask):
        nonempty_input_mask = attention_mask.sum(-1) != 0

        # if all tokens of modality is empty, add one masking token
        empty_input_mask = nonempty_input_mask == 0
        n_masking_tokens = 1
        x[empty_input_mask, :n_masking_tokens] = self.fusion.masking_token.type(x.dtype)
        attention_mask[empty_input_mask, :n_masking_tokens] = 1
        return x, attention_mask, nonempty_input_mask

    def extract_video_tokens(self, video, attention_mask):
        x = self.video_token_proj(video)
        x = self.video_norm_layer(x)

        x, attention_mask, nonempty_input_mask = self._check_and_fix_if_input_empty(x, attention_mask)
        special_token_mask = attention_mask == 0

        return {'all_tokens': x, 'attention_mask': attention_mask, 'special_token_mask': special_token_mask,
                'nonempty_input_mask': nonempty_input_mask}

    def extract_audio_tokens(self, audio, attention_mask, audio_STFT_nframes):
        audio = self.davenet(audio)
        audio = audio.permute(0, 2, 1)

        coef = int(np.ceil(attention_mask.shape[1] / audio.shape[1]))
        attention_mask = torch.nn.functional.max_pool1d(attention_mask.unsqueeze(0), kernel_size=coef).squeeze(0)
        audio_STFT_nframes = (audio_STFT_nframes / coef).int()

        if (self.audio_max_tokens is not None) and (audio.shape[1] > self.audio_max_tokens):
            new_audio, new_audio_mask = [], []
            for i in range(len(audio)):
                cur_audio, cur_audio_mask = create_audio_tokens(
                    audio[i], attention_mask[i], audio_STFT_nframes[i], self.audio_max_tokens, strategy=self.strategy_audio_pooling)
                new_audio.append(cur_audio)
                new_audio_mask.append(cur_audio_mask)
            audio = torch.stack(new_audio, dim=0)
            attention_mask = torch.stack(new_audio_mask, dim=0)

        audio = self.audio_token_proj(audio)
        audio = self.audio_norm_layer(audio)

        audio, attention_mask, nonempty_input_mask = self._check_and_fix_if_input_empty(audio, attention_mask)
        special_token_mask = attention_mask == 0
        return {'all_tokens': audio, 'attention_mask': attention_mask, 'special_token_mask': special_token_mask,
                'nonempty_input_mask': nonempty_input_mask}

    def extract_text_tokens(self, text, attention_mask):
        x = self.text_token_proj(text)
        x = self.text_norm_layer(x)

        x, attention_mask, nonempty_input_mask = self._check_and_fix_if_input_empty(x, attention_mask)
        special_token_mask = attention_mask == 0
        return {'all_tokens': x, 'attention_mask': attention_mask, 'special_token_mask': special_token_mask,
                'nonempty_input_mask': nonempty_input_mask}

    def forward(self, data, force_cross_modal=False):
        output = {}

        text_raw_embed = self.extract_text_tokens(data['text'], data['text_mask'])
        video_raw_embed = self.extract_video_tokens(data['video'], data['video_mask'])
        audio_raw_embed = self.extract_audio_tokens(data['audio'], data['audio_mask'], data['audio_STFT_nframes'])
        output['text_nonempty_input_mask'] = text_raw_embed['nonempty_input_mask']
        output['video_nonempty_input_mask'] = video_raw_embed['nonempty_input_mask']
        output['audio_nonempty_input_mask'] = audio_raw_embed['nonempty_input_mask']

        # add positional embedding after masking
        if self.use_positional_emb:
            text_raw_embed['all_tokens'] = text_raw_embed['all_tokens'] + self.text_pos_embed
            video_raw_embed['all_tokens'] = video_raw_embed['all_tokens'] + self.video_pos_embed
            audio_raw_embed['all_tokens'] = audio_raw_embed['all_tokens'] + self.audio_pos_embed

        text = self.fusion(text=text_raw_embed)['text']
        video = self.fusion(video=video_raw_embed)['video']
        audio = self.fusion(audio=audio_raw_embed)['audio']

        if self.individual_projections:
            text_proj, video_proj, audio_proj = self.text_proj, self.video_proj, self.audio_proj
        else:
            text_proj, video_proj, audio_proj = self.proj, self.proj, self.proj

        output["text_embed"] = text_proj(text['embed'])
        output["video_embed"] = video_proj(video['embed'])
        output["audio_embed"] = audio_proj(audio['embed'])

        if self.cross_modal or force_cross_modal:
            tv = self.fusion(text=text_raw_embed,
                             video=video_raw_embed)
            ta = self.fusion(text=text_raw_embed,
                             audio=audio_raw_embed)
            va = self.fusion(video=video_raw_embed,
                             audio=audio_raw_embed)

            if self.fusion.cls_token is not None:
                assert not self.individual_projections
                output["tv_embed"] = self.proj(tv['text_video']['embed'])
                output["ta_embed"] = self.proj(ta['text_audio']['embed'])
                output["va_embed"] = self.proj(va['video_audio']['embed'])
            else:
                output["tv_embed"] = (normalize_embeddings(text_proj(tv['text']['embed'])) +
                                      normalize_embeddings(video_proj(tv['video']['embed']))) / 2

                output["ta_embed"] = (normalize_embeddings(text_proj(ta['text']['embed'])) +
                                      normalize_embeddings(audio_proj(ta['audio']['embed']))) / 2

                output["va_embed"] = (normalize_embeddings(video_proj(va['video']['embed'])) +
                                      normalize_embeddings(audio_proj(va['audio']['embed']))) / 2

        if force_cross_modal:
            #  needed for ablation
            output["t+v_embed"] = (normalize_embeddings(output["text_embed"]) +
                                   normalize_embeddings(output["video_embed"])) / 2
            output["t+a_embed"] = (normalize_embeddings(output["text_embed"]) +
                                   normalize_embeddings(output["audio_embed"])) / 2
            output["v+a_embed"] = (normalize_embeddings(output["video_embed"]) +
                                   normalize_embeddings(output["audio_embed"])) / 2

        return output


class EverythingAtOnceModel_TV_Only(EverythingAtOnceModel):
    def forward(self, data, force_cross_modal=False):
        output = {}

        text_raw_embed = self.extract_text_tokens(data['text'], data['text_mask'])
        video_raw_embed = self.extract_video_tokens(data['video'], data['video_mask'])
        audio_raw_embed = self.extract_audio_tokens(data['audio'], data['audio_mask'], data['audio_STFT_nframes'])
        output['text_nonempty_input_mask'] = text_raw_embed['nonempty_input_mask']
        output['video_nonempty_input_mask'] = video_raw_embed['nonempty_input_mask']
        output['audio_nonempty_input_mask'] = audio_raw_embed['nonempty_input_mask']

        # add positional embedding after masking
        if self.use_positional_emb:
            text_raw_embed['all_tokens'] = text_raw_embed['all_tokens'] + self.text_pos_embed
            video_raw_embed['all_tokens'] = video_raw_embed['all_tokens'] + self.video_pos_embed

        text = self.fusion(text=text_raw_embed)['text']
        video = self.fusion(video=video_raw_embed)['video']

        if not self.individual_projections:
            output["text_embed"] = self.proj(text['embed'])
            output["video_embed"] = self.proj(video['embed'])
        else:
            output["text_embed"] = self.text_proj(text['embed'])
            output["video_embed"] = self.video_proj(video['embed'])
        return output


class TransformerPerModalityModel(EverythingAtOnceModel):
    def __init__(self,
                 video_embed_dim,
                 text_embed_dim,
                 fusion_params,
                 video_max_tokens=None,
                 text_max_tokens=None,
                 audio_max_num_STFT_frames=None,
                 projection_dim=6144,
                 token_projection='gated',
                 projection='gated',
                 strategy_audio_pooling='none',
                 davenet_v2=True,
                 use_positional_emb=False,
                 ):
        super().__init__(video_embed_dim,
                         text_embed_dim,
                         fusion_params,
                         video_max_tokens=video_max_tokens,
                         text_max_tokens=text_max_tokens,
                         audio_max_num_STFT_frames=audio_max_num_STFT_frames,
                         projection_dim=projection_dim,
                         token_projection=token_projection,
                         projection=projection,
                         cross_modal=False,
                         strategy_audio_pooling=strategy_audio_pooling,
                         davenet_v2=davenet_v2,
                         individual_projections=True,
                         use_positional_emb=use_positional_emb,
                         )

        self.fusion_text = self.fusion
        self.fusion_video = FusionTransformer(**fusion_params)
        self.fusion_audio = FusionTransformer(**fusion_params)

    def forward(self, data, force_cross_modal=False):
        output = {}

        text_raw_embed = self.extract_text_tokens(data['text'], data['text_mask'])
        video_raw_embed = self.extract_video_tokens(data['video'], data['video_mask'])
        audio_raw_embed = self.extract_audio_tokens(data['audio'], data['audio_mask'], data['audio_STFT_nframes'])

        output['text_nonempty_input_mask'] = text_raw_embed['nonempty_input_mask']
        output['video_nonempty_input_mask'] = video_raw_embed['nonempty_input_mask']
        output['audio_nonempty_input_mask'] = audio_raw_embed['nonempty_input_mask']

        text = self.fusion_text(text=text_raw_embed)['text']
        output["text_embed"] = self.text_proj(text['embed'])

        video = self.fusion_video(video=video_raw_embed)['video']
        output["video_embed"] = self.video_proj(video['embed'])

        audio = self.fusion_audio(audio=audio_raw_embed)['audio']
        output["audio_embed"] = self.audio_proj(audio['embed'])

        if force_cross_modal:
            #  needed for ablation
            output["t+v_embed"] = (normalize_embeddings(output["text_embed"]) +
                                   normalize_embeddings(output["video_embed"])) / 2
            output["t+a_embed"] = (normalize_embeddings(output["text_embed"]) +
                                   normalize_embeddings(output["audio_embed"])) / 2
            output["v+a_embed"] = (normalize_embeddings(output["video_embed"]) +
                                   normalize_embeddings(output["audio_embed"])) / 2

        return output


def create_audio_tokens(audio, audio_mask, audio_STFT_nframes, n_tokens, strategy='avg_pool'):
    if torch.is_tensor(audio_STFT_nframes):
        audio_STFT_nframes = int(audio_STFT_nframes.cpu().item())
    if strategy == 'clip':
        return audio[:n_tokens], audio_mask[:n_tokens]
    elif strategy == 'nearest':
        if audio_STFT_nframes <= n_tokens:
            return create_audio_tokens(audio, audio_mask, audio_STFT_nframes, n_tokens, strategy='clip')
        audio = audio[:audio_STFT_nframes]
        audio = torch.nn.functional.interpolate(
            audio.permute(1, 0).unsqueeze(0),
            size=n_tokens,
            mode='nearest').squeeze(0).permute(1, 0)
        return audio, audio_mask[:n_tokens]
    elif strategy == 'max_pool':
        if audio_STFT_nframes <= n_tokens:
            return create_audio_tokens(audio, audio_mask, audio_STFT_nframes, n_tokens, strategy='clip')
        audio = audio[:audio_STFT_nframes]
        audio = torch.nn.functional.adaptive_max_pool1d(
            audio.permute(1, 0).unsqueeze(0),
            output_size=n_tokens).squeeze(0).permute(1, 0)
        return audio, audio_mask[:n_tokens]
    elif strategy == 'avg_pool':
        if audio_STFT_nframes <= n_tokens:
            return create_audio_tokens(audio, audio_mask, audio_STFT_nframes, n_tokens, strategy='clip')
        audio = audio[:audio_STFT_nframes]
        audio = torch.nn.functional.adaptive_avg_pool1d(
            audio.permute(1, 0).unsqueeze(0),
            output_size=n_tokens).squeeze(0).permute(1, 0)
        return audio, audio_mask[:n_tokens]
    elif strategy == 'none':
        return audio, audio_mask
    else:
        raise NotImplementedError
