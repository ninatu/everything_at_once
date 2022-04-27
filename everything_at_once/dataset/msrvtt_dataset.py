from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import pickle
import random
import torch
from torch.utils.data.dataloader import default_collate
from torch.utils.data import Dataset

from everything_at_once.dataset.utils import _tokenize_text, create_audio_features, create_text_features, \
    create_video_features, cut_into_clips


class MSRVTT_Dataset(Dataset):
    """MSRVTT dataset"""

    def __init__(
            self,
            data_path,
            we,
            data_path_2D=None,
            data_path_3D=None,
            we_dim=300,
            max_words=20,
            training=True,
            n_video_tokens=12,
            num_audio_STFT_frames=768,
            video_sampling_strategy='clip',
            cut_clips=False,
            n_clips=1,
            use_2D=True,
            use_3D=True,
            key_2d='2d',
            key_3d='3d',
    ):
        assert use_2D or use_3D
        self.data = pickle.load(open(data_path, 'rb'))
        self.data_2D = self.data if data_path_2D is None else pickle.load(open(data_path_2D, 'rb'))
        self.data_3D = self.data if data_path_3D is None else pickle.load(open(data_path_3D, 'rb'))
        self.use_2D = use_2D
        self.use_3D = use_3D
        self.key_2d = key_2d
        self.key_3d = key_3d

        self.we = we
        self.we_dim = we_dim
        self.max_words = max_words
        self.training = training

        self.n_video_tokens = n_video_tokens
        self.num_audio_STFT_frames = num_audio_STFT_frames
        self.video_sampling_strategy = video_sampling_strategy
        self.n_clips = n_clips
        self.cut_clips = cut_clips

        if not self.cut_clips:
            assert n_clips == 1

        if self.cut_clips:
            assert video_sampling_strategy == 'clip'

        if not training:
            assert len(self.data) == 968

            # TODO: make it more clean:
            # we need to know the real test size for a fair comparison
            # we count the missing test clips as mistakes
            self.complete_dataset_size = 1000

    def __len__(self):
        return len(self.data)

    def custom_collate(self, batch):
        return default_collate(batch)

    def __getitem__(self, idx):
        # load 2d and 3d features
        id_ = self.data[idx]['id']
        feat_2d = None
        feat_3d = None
        if self.use_2D:
            feat_2d = torch.from_numpy(self.data_2D[idx][self.key_2d]).float()
        if self.use_3D:
            feat_3d = torch.from_numpy(self.data_3D[idx][self.key_3d]).float()
            
        target_nvideo_tokens = self.n_video_tokens * self.n_clips
        video, video_mask = create_video_features(feat_2d, feat_3d, target_nvideo_tokens, strategy=self.video_sampling_strategy)
        
        # load audio
        audio = self.data[idx]['audio']
        max_audio_STFT_nframes = self.num_audio_STFT_frames * self.n_clips
        audio, audio_mask, audio_STFT_nframes = create_audio_features(audio, max_audio_STFT_nframes)

        # choose a caption
        if self.training:
            caption = random.choice(self.data[idx]['caption'])
        else:
            caption = self.data[idx]['eval_caption']
        words = _tokenize_text(caption)
        text, text_mask, raw_text = create_text_features(words, self.max_words, self.we, self.we_dim)

        # meta data
        dataset = 'MSRVTT'

        if self.cut_clips:
            video, video_mask, audio, audio_mask, text, text_mask, raw_text, audio_STFT_nframes, id_, dataset = \
                cut_into_clips(video, video_mask, audio, audio_mask, text, text_mask, raw_text, audio_STFT_nframes, id_, dataset,
                               n_clips=self.n_clips)
            unroll_clips = 1
        else:
            unroll_clips = 0

        return {'video': video, 'audio': audio, 'text': text, 'audio_STFT_nframes': audio_STFT_nframes,
                'video_mask': video_mask, 'audio_mask': audio_mask, 'text_mask': text_mask,
                'raw_text': raw_text,
                'unroll_clips': unroll_clips,
                'meta': {'paths': id_, 'ids': id_, 'dataset': dataset}}
