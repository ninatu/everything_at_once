import torch
from torch import nn as nn

from everything_at_once.model.utils.layers import GatedEmbeddingUnit, SentenceMaxpool


class AVLnetBaselineModel(nn.Module):
    def __init__(
            self,
            embd_dim=4096,
            video_dim=4096,
            we_dim=300,
            davenet_v2=False
    ):
        super().__init__()
        from everything_at_once.model.utils.davenet import load_DAVEnet

        self.DAVEnet = load_DAVEnet(v2=davenet_v2)
        self.DAVEnet_projection = nn.Linear(1024, embd_dim)
        self.GU_audio = GatedEmbeddingUnit(1024, 1024)
        self.GU_video = GatedEmbeddingUnit(video_dim, embd_dim)
        self.text_pooling_caption = SentenceMaxpool(we_dim, embd_dim)
        self.GU_text_captions = GatedEmbeddingUnit(embd_dim, embd_dim)

    def forward(self, data, force_cross_modal=False):
        output = {}

        output['video_nonempty_input_mask'] = data['video_mask'].sum(-1) != 0
        output['text_nonempty_input_mask'] = data['text_mask'].sum(-1) != 0
        output['audio_nonempty_input_mask'] = data['video_mask'].sum(-1) != 0

        output["text_embed"] = self.GU_text_captions(self.text_pooling_caption(data['text']))

        video = data['video']
        if len(video.shape) == 3:
            video = torch.nn.functional.normalize(torch.max(video, dim=1)[0], dim=1)
        output["video_embed"] = self.GU_video(video)

        if 'audio' in data:
            audio_input = data['audio']
            audio_STFT_nframes = data['audio_STFT_nframes']

            audio = self.DAVEnet(audio_input)
            if audio_STFT_nframes[0] != -1:
                # if not self.training:  # controlled by net.train() / net.eval() (use for downstream tasks)
                # Mean-pool audio embeddings and disregard embeddings from input 0 padding
                pooling_ratio = round(audio_input.size(-1) / audio.size(-1))
                audio_STFT_nframes = audio_STFT_nframes / pooling_ratio
                audioPoolfunc = torch.nn.AdaptiveAvgPool2d((1, 1))
                audio_outputs = audio.unsqueeze(2)
                pooled_audio_outputs_list = []
                for idx in range(audio.shape[0]):
                    nF = max(1, audio_STFT_nframes[idx].cpu().item())
                    pooled_audio_outputs_list.append(audioPoolfunc(audio_outputs[idx][:, :, 0:int(nF)]).unsqueeze(0))
                audio = torch.cat(pooled_audio_outputs_list).squeeze(3).squeeze(2)
            else:
                audio = audio.mean(dim=2)  # this averages features from 0 padding too
            # Gating in lower embedding dimension (1024 vs 4096) for stability with mixed-precision training
            audio = self.GU_audio(audio)
            output["audio_embed"] = self.DAVEnet_projection(audio)

        return output
