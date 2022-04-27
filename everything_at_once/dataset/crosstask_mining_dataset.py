from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import torch as th
from torch.utils.data import Dataset
import pandas as pd
import os
import numpy as np
import json
import librosa

from everything_at_once.dataset.utils import _tokenize_text, create_audio_features, create_text_features, \
    _crop_audio_from_mel_spec, _get_video


class CrossTaskMiningYoutubeDataset(Dataset):
    def __init__(
            self,
            csv,
            features_path,
            annot_path,
            steps_path,
            audio_path,
            we,
            n_video_tokens=12,
            we_dim=300,
            max_words=20,
            feature_per_sec_2D=1.0,
            feature_per_sec_3D=24.0 / 16.0,
            num_audio_STFT_frames=768,
            mining_youtube=False,
            clip_radius=0,
            audio_padding=0,
            use_2D=True,
            use_3D=True
    ):
        self.use_2D = use_2D
        self.use_3D = use_3D

        self.csv = pd.read_csv(csv)
        self.annot_path = annot_path
        self.steps_path = steps_path
        self.audio_path = audio_path
        self.we = we
        self.we_dim = we_dim
        self.max_words = max_words
        self.num_audio_STFT_frames = num_audio_STFT_frames
        self.feature_per_sec_2D = feature_per_sec_2D
        self.feature_per_sec_3D = feature_per_sec_3D
        self.feature_path = features_path
        self.n_video_tokens = n_video_tokens
        self.mining_youtube = mining_youtube
        self.clip_radius = clip_radius
        self.audio_padding = audio_padding

        with open(steps_path, "r") as read_file:
            self.step_dict = json.load(read_file)

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, idx):
        video_id = self.csv['video_id'][idx]

        # load video
        features_2d = None
        features_3d = None
        if self.use_2D:
            features_2d = np.load(os.path.join(self.feature_path, self.csv['video_id'][idx] + '_2d.npy'))
            features_2d = th.from_numpy(features_2d).float()

        if self.use_3D:
            path = os.path.join(self.feature_path, self.csv['video_id'][idx] + '_3d.npy')
            if os.path.exists(path):
                features_3d = np.load(path)
            else:
                path = os.path.join(self.feature_path, self.csv['video_id'][idx] + '.npz')
                features_3d = np.load(path)['features']
            features_3d = th.from_numpy(features_3d).float()

        # load audio
        audio = np.load(os.path.join(self.audio_path, self.csv['video_id'][idx] + '.npz'))['arr_0']

        # load annotation
        if self.mining_youtube:
            task = ''
            y_true = th.from_numpy(np.load(os.path.join(self.annot_path, video_id + '.npy')))
            steps = self.step_dict[video_id]
        else:
            task = str(self.csv['task'][idx])
            y_true = th.from_numpy(np.load(os.path.join(self.annot_path, task + '_' + video_id + '.npy')))
            steps = self.step_dict[task]
        T = y_true.size()[0]  # number of frames = number  of segments

        # create audio, text, video
        audio, audio_mask, audio_STFT_nframes, starts, ends = _get_audio(T, audio, self.num_audio_STFT_frames,
                                                              radius=self.clip_radius,
                                                              padding=self.audio_padding)

        video, video_mask = _get_video(features_2d, features_3d,
                                       self.feature_per_sec_2D, self.feature_per_sec_3D,
                                       starts, ends,
                                       self.n_video_tokens,
                                       video_sampling_strategy='clip',
                                       accurate_borders=True)

        text, text_mask, raw_text = _get_text(steps, self.max_words, self.we, self.we_dim)
        paths = [video_id] * T
        ids = paths
        datasetname = 'MiningYoutube' if self.mining_youtube else 'CrossTask'

        return {'video': video, 'audio': audio, 'text': text,
                'video_mask': video_mask, 'audio_mask': audio_mask, 'text_mask': text_mask,
                'raw_text': raw_text,
                'unroll_clips': 1, 'audio_STFT_nframes': audio_STFT_nframes,
                'meta': {'paths': paths, 'ids': ids, 'dataset': [datasetname] * len(video)},
                'task': task, 'y_true': y_true,
                }


def _get_text(step_texts, max_words, we, we_dim):
    n_steps = len(step_texts)
    text = [0 for i in range(n_steps)]
    text_mask = [0 for i in range(n_steps)]
    raw_text = [0 for i in range(n_steps)]
    for i in range(n_steps):
        words = _tokenize_text(step_texts[i])
        text[i], text_mask[i], raw_text[i] = create_text_features(words, max_words, we, we_dim)
    text = th.stack(text, dim=0)
    text_mask = th.stack(text_mask, dim=0)
    return text, text_mask, raw_text


def _get_audio(T, mel_spec, num_audio_STFT_frames, radius=0, padding=0):
    audio = [0 for i in range(T)]
    audio_mask = [0 for i in range(T)]
    starts = [0 for i in range(T)]
    ends = [0 for i in range(T)]
    audio_STFT_nframes = np.zeros(T)

    def get_start_end(i, T):
        start, end = max(0, i - radius), min(i + radius + 1, T)
        padded_start, padded_end = max(0, start - padding), min(end + padding, T)
        relative_start, relative_end = start - padded_start, end - padded_start
        return start, end, padded_start, padded_end, relative_start, relative_end

    for i in range(T):
        starts[i], ends[i], padded_start, padded_end, relative_start, relative_end = get_start_end(i, T)
        cropped_mel_spec = _crop_audio_from_mel_spec(padded_start, padded_end, mel_spec)
        audio[i], audio_mask[i], audio_STFT_nframes[i] = create_audio_features(cropped_mel_spec, num_audio_STFT_frames)

        frames = librosa.core.time_to_frames([relative_start, relative_end], sr=16000, hop_length=160,
                                             n_fft=400)
        start_frame, end_frame = max(0, frames[0]), frames[1]
        audio_mask[i][:start_frame] = 0
        audio_mask[i][end_frame:] = 0

    audio = th.stack(audio, dim=0)
    audio_mask = th.stack(audio_mask, dim=0)
    return audio, audio_mask, audio_STFT_nframes, starts, ends
