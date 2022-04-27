from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import pickle
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate

from everything_at_once.dataset.utils import _tokenize_text, create_audio_features, create_text_features, \
    create_video_features, cut_into_clips


class Youcook_Dataset(Dataset):
    """Youcook dataset."""

    def __init__(
            self,
            we,
            data_path=None,
            data_path_2D=None,
            data_path_3D=None,
            we_dim=300,
            max_words=20,
            n_video_tokens=12,
            num_audio_STFT_frames=768,
            video_sampling_strategy='clip',
            cut_clips=False,
            n_clips=1,
            use_2D=True,
            use_3D=True,
            key_2d='2d_full',
            key_3d='3d_full',
    ):
        # Some Resnet152&Resnet101 test features were missing (only 3339 out of 3500 were available)
        # So when testing with S3D or CLIP features,
        # -- set data_path to the Resnet152&Resnet101 feature path -- used just for filtering (using only 3339 items)
        # -- set data_path_2D or data_path_3D to the S3D or CLIP feature path
        with open(data_path, 'rb') as fin:
            data = pickle.load(fin)
        if data_path_2D is None:
            data_2D = data
        else:
            data_2D = pickle.load(open(data_path_2D, 'rb'))
        if data_path_3D is None:
            data_3D = data
        else:
            data_3D = pickle.load(open(data_path_3D, 'rb'))

        self.data = []
        self.data_2D = []
        self.data_3D = []
        for i in range(len(data)):
            #  11 spatial features were missed in original datafiles
            #  we always count them as missing and as mistakes
            if '2d_full' in data[i]:
                self.data.append(data[i])
                self.data_2D.append(data_2D[i])
                self.data_3D.append(data_3D[i])

        self.use_2D = use_2D
        self.use_3D = use_3D
        self.key_2d = key_2d
        self.key_3d = key_3d

        self.we = we
        self.we_dim = we_dim
        self.max_words = max_words

        self.num_audio_STFT_frames = num_audio_STFT_frames
        self.n_video_tokens = n_video_tokens
        self.video_sampling_strategy = video_sampling_strategy
        self.cut_clips = cut_clips
        self.n_clips = n_clips

        if not self.cut_clips:
            assert n_clips == 1

        if self.cut_clips:
            assert video_sampling_strategy == 'clip'

        # TODO: make it more clean:
        # we need to know the real test size for a fair comparison
        # we count the missing test clips as mistakes
        self.complete_dataset_size = 3350

    def __len__(self):
        return len(self.data_2D)

    def custom_collate(self, batch):
        return default_collate(batch)

    def __getitem__(self, idx):
        # load 2d and 3d features
        feat_2d = None
        feat_3d = None
        if self.use_2D:
            feat_2d = torch.from_numpy(self.data_2D[idx][self.key_2d]).float()
        if self.use_3D:
            feat_3d = torch.from_numpy(self.data_3D[idx][self.key_3d]).float()
            
        target_nvideo_tokens = self.n_video_tokens * self.n_clips
        video, video_mask = create_video_features(feat_2d, feat_3d, target_nvideo_tokens,
                                                  strategy=self.video_sampling_strategy,
                                                  )
        # load audio
        audio = self.data[idx]['audio']
        max_audio_STFT_nframes = self.num_audio_STFT_frames * self.n_clips
        audio, audio_mask, audio_STFT_nframes = create_audio_features(audio, max_audio_STFT_nframes)

        # load text
        caption = self.data[idx]['caption']
        words = _tokenize_text(caption)
        text, text_mask, raw_text = create_text_features(words, self.max_words, self.we, self.we_dim)

        id_ = str(self.data[idx]['id'])
        dataset = 'YouCook2'

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
