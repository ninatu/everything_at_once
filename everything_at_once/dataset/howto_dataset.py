from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import torch as th
from torch.utils.data import Dataset
import pandas as pd
import os
import numpy as np
import librosa

from everything_at_once.dataset.utils import _tokenize_text, create_audio_features, \
    create_text_features, _crop_audio_from_mel_spec, _get_video


class HowTo_Dataset(Dataset):
    """HowTo100M dataset"""

    def __init__(
            self,
            csv,
            caption,
            we,
            features_path,
            features_path_audio,
            features_path_3D=None,
            min_time=8.0,
            n_video_tokens=None,
            feature_per_sec_2D=1.0,
            feature_per_sec_3D=24.0 / 16.0,
            we_dim=300,
            max_words=20,
            n_clips=1,
            num_audio_STFT_frames=768,
            video_sampling_strategy='clip',
            flag_text_audio_misaligned=False,
            use_2D=True,
            use_3D=True,
    ):
        """
        Args:
        """
        self.csv = pd.read_csv(csv)
        self.caption = caption
        self.we = we

        self.features_path = features_path
        self.features_path_audio = features_path_audio
        self.features_path_3D = features_path_3D if features_path_3D is not None else features_path

        self.video_sampling_strategy = video_sampling_strategy
        self.flag_text_audio_misaligned = flag_text_audio_misaligned

        self.min_time = min_time
        self.n_video_tokens = n_video_tokens if n_video_tokens is not None else int(1.5 * min_time)
        self.feature_per_sec_2D = feature_per_sec_2D
        self.feature_per_sec_3D = feature_per_sec_3D
        self.we_dim = we_dim
        self.max_words = max_words
        self.num_audio_STFT_frames = num_audio_STFT_frames
        self.n_clips = n_clips
        self.use_2D = use_2D
        self.use_3D = use_3D

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, idx):
        # load video
        vid_path = self.csv['path'].values[idx].replace("None/", "")
        video_id = vid_path.split("/")[-1]

        features_2d = None
        features_3d = None
        if self.use_2D:
            path = os.path.join(self.features_path, vid_path, video_id + "_2d.npz")
            if not os.path.exists(path):
                path = os.path.join(self.features_path, vid_path + "_2d.npz")

            features_2d = np.load(path)['features']
            features_2d = th.from_numpy(features_2d).float()
        if self.use_3D:
            path = os.path.join(self.features_path_3D, vid_path, video_id + "_3d.npz")
            if not os.path.exists(path):
                path = os.path.join(self.features_path_3D, vid_path, video_id + ".npz")
            features_3d = np.load(path)['features']
            features_3d = th.from_numpy(features_3d).float()

        # load audio
        audio_path = os.path.join(self.features_path_audio, vid_path, video_id + "_spec.npz")
        mel_spec = np.load(audio_path)['arr_0']

        # load text
        caption = self.caption[video_id]

        video, video_mask, audio, audio_mask, text, text_mask, raw_text, audio_STFT_nframes, paths, ids, starts, ends = \
            _sample_video_audio_text(caption, mel_spec, features_2d, features_3d, vid_path, self.n_clips,
                                     self.feature_per_sec_2D, self.feature_per_sec_3D,
                                     self.min_time, self.max_words, self.we, self.we_dim,
                                     self.num_audio_STFT_frames, self.n_video_tokens,
                                     video_sampling_strategy=self.video_sampling_strategy,
                                     flag_text_audio_misaligned=self.flag_text_audio_misaligned)

        return {'video': video, 'audio': audio, 'text': text,
                'video_mask': video_mask, 'audio_mask': audio_mask, 'text_mask': text_mask,
                'raw_text': raw_text,
                'unroll_clips': 1, 'audio_STFT_nframes': audio_STFT_nframes,
                'meta': {'paths': paths, 'ids': ids, 'dataset': ['HowTo100M'] * len(video)}}


def _sample_video_audio_text(caption, mel_spec, features_2d, features_3d, vid_path, n_clips, fps_2d, fps_3d,
                             min_time, max_words, we, we_dim, num_audio_STFT_frames, n_video_tokens,
                             video_sampling_strategy='clip',
                             flag_text_audio_misaligned=False,
                             ):
    audio, audio_mask, audio_STFT_nframes, starts, ends, text, text_mask, raw_text = \
        _sample_audio_and_text(caption, n_clips, mel_spec,
                               min_time, max_words, we, we_dim, num_audio_STFT_frames,
                               flag_text_audio_misaligned=flag_text_audio_misaligned)

    video, video_mask = _get_video(features_2d, features_3d, fps_2d, fps_3d, starts, ends, n_video_tokens, video_sampling_strategy)

    audio_STFT_nframes = -np.ones(len(audio_STFT_nframes))

    paths = [vid_path] * len(video)
    ids = [vid_path + str(start) for start in starts]

    return video, video_mask, audio, audio_mask, text, text_mask, raw_text, audio_STFT_nframes, paths, ids, starts, ends


def _sample_audio_and_text(caption, n_clips, mel_spec, min_time, max_words, we, we_dim, num_audio_STFT_frames,
                           flag_text_audio_misaligned=False):
    starts = np.zeros(n_clips)
    ends = np.zeros(n_clips)
    text = [0 for _ in range(n_clips)]
    text_mask = [0 for _ in range(n_clips)]
    raw_text = [0 for _ in range(n_clips)]
    audio = [0 for _ in range(n_clips)]
    audio_mask = [0 for _ in range(n_clips)]
    audio_STFT_nframes = np.zeros(n_clips)

    for i in range(n_clips):
        audio[i], audio_mask[i], audio_STFT_nframes[i], starts[i], ends[i], text[i], text_mask[i], raw_text[i] = \
                _get_single_random_audio_and_text(caption, mel_spec, min_time, max_words, we, we_dim, num_audio_STFT_frames,
                                                  flag_text_audio_misaligned=flag_text_audio_misaligned)
    audio = th.stack(audio, dim=0)
    audio_mask = th.stack(audio_mask, dim=0)
    text = th.stack(text, dim=0)
    text_mask = th.stack(text_mask, dim=0)
    return audio, audio_mask, audio_STFT_nframes, starts, ends, text, text_mask, raw_text


def _get_single_random_audio_and_text(caption, mel_spec, min_time,  max_words, we, we_dim, num_audio_STFT_frames,
                                      flag_text_audio_misaligned=False):
    video_duration_seconds = int(librosa.core.frames_to_time(mel_spec.shape[1], sr=16000, hop_length=160, n_fft=400))
    start = np.random.rand() * max(0, video_duration_seconds - min_time)
    end = start + min_time

    # Search if there is ASR narration in the sampled clip
    ind_start = max(0, np.searchsorted(caption['start'], start, side='left') - 1)
    ind_end = max(0, np.searchsorted(caption['start'], end, side='left') - 1)
    if caption['start'][ind_start] <= start <= caption['end'][ind_start]:
        return _get_text_and_audio_by_ind(caption, mel_spec, min_time, max_words, we, we_dim, num_audio_STFT_frames,
                                          ind=ind_start,
                                          flag_text_audio_misaligned=flag_text_audio_misaligned)
    elif caption['start'][ind_end] <= end <= caption['end'][ind_end]:
        return _get_text_and_audio_by_ind(caption, mel_spec, min_time, max_words, we, we_dim, num_audio_STFT_frames,
                                          ind=ind_end,
                                          flag_text_audio_misaligned=flag_text_audio_misaligned)
    else:
        words = [] # no ASR in this clip
        start_frame = max(0, librosa.core.time_to_frames(start, sr=16000, hop_length=160, n_fft=400))
        mel_spec = mel_spec[:, start_frame:start_frame + num_audio_STFT_frames]

        # padded audio features
        audio, audio_mask, audio_STFT_nframes = create_audio_features(mel_spec, num_audio_STFT_frames)

        # create padded text features
        text, text_mask, raw_text = create_text_features(words, max_words, we, we_dim)

        return audio, audio_mask, audio_STFT_nframes, start, end, text, text_mask, raw_text


def _get_text_and_audio_by_ind(caption, mel_spec, min_time, max_words, we, we_dim, num_audio_STFT_frames, ind=None,
                               flag_text_audio_misaligned=False):
    if ind is None:
        n_caption = len(caption['start'])
        ind = np.random.choice(range(n_caption))

    start, end = ind, ind
    words = _tokenize_text(caption['text'][ind])
    diff = caption['end'][end] - caption['start'][start]
    # Extend the video clip if shorter than the minimum desired clip duration
    while diff < min_time:
        if start > 0 and end < len(caption['end']) - 1:
            next_words = _tokenize_text(caption['text'][end + 1])
            prev_words = _tokenize_text(caption['text'][start - 1])
            d1 = caption['end'][end + 1] - caption['start'][start]
            d2 = caption['end'][end] - caption['start'][start - 1]
            # Use the closest neighboring video clip
            if d2 <= d1:
                start -= 1
                words.extend(prev_words)
            else:
                end += 1
                words.extend(next_words)
        # If no video clips after it, use the clip before it
        elif start > 0:
            words.extend(_tokenize_text(caption['text'][start - 1]))
            start -= 1
        # If no video clips before it, use the clip after it.
        elif end < len(caption['end']) - 1:
            words.extend(_tokenize_text(caption['text'][end + 1]))
            end += 1
        # If there's no clips before or after
        else:
            break
        diff = caption['end'][end] - caption['start'][start]

    start, end = caption['start'][start], caption['end'][end]
    if flag_text_audio_misaligned:
        if np.random.rand() > 0:
            start_audio = start + 0.5 * min_time
            end_audio = start_audio + min_time
        else:
            start_audio = max(0, start - 0.5 * min_time)
            end_audio = start_audio + min_time
    else:
        start_audio = start
        end_audio = end

    # create padded audio features
    mel_spec = _crop_audio_from_mel_spec(start_audio, end_audio, mel_spec)
    audio, audio_mask, audio_STFT_nframes = create_audio_features(mel_spec, num_audio_STFT_frames)

    # create padded text features
    text, text_mask, raw_text = create_text_features(words, max_words, we, we_dim)

    return audio, audio_mask, audio_STFT_nframes, start, end, text, text_mask, raw_text
