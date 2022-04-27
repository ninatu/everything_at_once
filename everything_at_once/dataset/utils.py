import re

import librosa
import numpy as np
import torch
from torch.nn import functional as F


def _tokenize_text(sentence):
    w = re.findall(r"[\w']+", str(sentence))
    return w


def create_audio_features(mel_spec, max_audio_STFT_nframes):
    audio = np.zeros((mel_spec.shape[0], max_audio_STFT_nframes), dtype=np.float32)
    audio_mask = np.zeros(max_audio_STFT_nframes, dtype=np.float32)

    audio_STFT_nframes = min(mel_spec.shape[1], max_audio_STFT_nframes)
    audio[:, :audio_STFT_nframes] = mel_spec[:, :audio_STFT_nframes]
    audio_mask[:audio_STFT_nframes] = 1

    audio = torch.from_numpy(audio).float()
    audio_mask = torch.from_numpy(audio_mask).float()
    return audio, audio_mask, audio_STFT_nframes


def create_text_features(words, max_words, we, we_dim):
    raw_text = ' '.join(words)
    words = [word for word in words if word in we.vocab]
    text = np.zeros((max_words, we_dim), dtype=np.float32)
    text_mask = np.zeros(max_words, dtype=np.float32)
    nwords = min(len(words), max_words)
    if nwords > 0:
        text[:nwords] = we[words][:nwords]
        text_mask[:nwords] = 1
    text = torch.from_numpy(text).float()
    text_mask = torch.from_numpy(text_mask).float()

    return text, text_mask, raw_text


def create_video_features(feat_2d, feat_3d, n_tokens, strategy='clip'):
    if n_tokens == 0:
        feat_2d = F.normalize(torch.max(feat_2d, dim=0)[0], dim=0) if len(feat_2d) else torch.zeros(feat_2d.shape[1])
        feat_3d = F.normalize(torch.max(feat_3d, dim=0)[0], dim=0) if len(feat_3d) else torch.zeros(feat_3d.shape[1])
        video = torch.cat((feat_2d, feat_3d))
        video_mask = torch.ones(1)  # TODO: not quite right, really 0
        return video, video_mask
    else:
        if strategy == 'clip':
            if feat_2d is None:
                video = torch.zeros(n_tokens, feat_3d.shape[-1])
                video_mask = torch.zeros(n_tokens)
                cur_n_tokens_3d, dim_3d = feat_3d.shape
                video[:cur_n_tokens_3d] = F.normalize(feat_3d[:n_tokens], dim=1)
                video_mask[:cur_n_tokens_3d] = 1
                return video, video_mask
            elif feat_3d is None:
                video = torch.zeros(n_tokens, feat_2d.shape[-1])
                video_mask = torch.zeros(n_tokens)
                cur_n_tokens_2d, dim_2d = feat_2d.shape
                video[:cur_n_tokens_2d] = F.normalize(feat_2d[:n_tokens], dim=1)
                video_mask[:cur_n_tokens_2d] = 1
                return video, video_mask
            else:
                video = torch.zeros(n_tokens, feat_2d.shape[-1] + feat_3d.shape[-1])
                video_mask = torch.zeros(n_tokens)
                cur_n_tokens_2d, dim_2d = feat_2d.shape
                cur_n_tokens_3d, dim_3d = feat_3d.shape

                if cur_n_tokens_2d != 0 and cur_n_tokens_3d != 0:
                    feat_2d = torch.nn.functional.interpolate(
                        feat_2d.permute(1, 0).unsqueeze(0),
                        size=cur_n_tokens_3d,
                        mode='nearest').squeeze(0).permute(1, 0)

                    video[:cur_n_tokens_3d, :dim_2d] = F.normalize(feat_2d[:n_tokens], dim=1)
                    video[:cur_n_tokens_3d, dim_2d:] = F.normalize(feat_3d[:n_tokens], dim=1)
                    video_mask[:cur_n_tokens_3d] = 1
                return video, video_mask
        elif strategy == 'nearest':
            if feat_2d is None:
                cur_n_tokens_3d, dim_3d = feat_3d.shape
                if cur_n_tokens_3d <= n_tokens:
                    return create_video_features(feat_2d, feat_3d, n_tokens, strategy='clip')
                feat_3d = torch.nn.functional.interpolate(
                    feat_3d.permute(1, 0).unsqueeze(0),
                    size=n_tokens,
                    mode='nearest').squeeze(0).permute(1, 0)
                video = F.normalize(feat_3d, dim=1)
                video_mask = torch.ones(n_tokens)
                return video, video_mask
            elif feat_3d is None:
                cur_n_tokens_2d, dim_2d = feat_2d.shape
                if cur_n_tokens_2d <= n_tokens:
                    return create_video_features(feat_2d, feat_2d, n_tokens, strategy='clip')
                feat_2d = torch.nn.functional.interpolate(
                    feat_2d.permute(1, 0).unsqueeze(0),
                    size=n_tokens,
                    mode='nearest').squeeze(0).permute(1, 0)
                video = F.normalize(feat_2d, dim=1)
                video_mask = torch.ones(n_tokens)
                return video, video_mask
            else:
                cur_n_tokens_2d, dim_2d = feat_2d.shape
                cur_n_tokens_3d, dim_3d = feat_3d.shape
                if cur_n_tokens_3d <= n_tokens or cur_n_tokens_2d == 0:
                    return create_video_features(feat_2d, feat_3d, n_tokens, strategy='clip')

                video = torch.zeros(n_tokens, feat_2d.shape[-1] + feat_3d.shape[-1])
                video_mask = torch.zeros(n_tokens)
                feat_2d = torch.nn.functional.interpolate(
                    feat_2d.permute(1, 0).unsqueeze(0),
                    size=n_tokens,
                    mode='nearest').squeeze(0).permute(1, 0)
                feat_3d = torch.nn.functional.interpolate(
                    feat_3d.permute(1, 0).unsqueeze(0),
                    size=n_tokens,
                    mode='nearest').squeeze(0).permute(1, 0)
                video[:, :dim_2d] = F.normalize(feat_2d, dim=1)
                video[:, dim_2d:] = F.normalize(feat_3d, dim=1)
                video_mask[:] = 1
                return video, video_mask

        elif strategy == 'max_pool':
            if feat_2d is None:
                cur_n_tokens_3d = feat_3d.shape[0]
                if cur_n_tokens_3d <= n_tokens:
                    return create_video_features(feat_2d, feat_3d, n_tokens, strategy='clip')
                kernel_size_3d = int(np.floor(cur_n_tokens_3d / n_tokens))
                if kernel_size_3d <= 1:  # we don't have what to max pool
                    return create_video_features(feat_2d, feat_3d, n_tokens, strategy='nearest')
                feat_3d = torch.nn.functional.max_pool1d(feat_3d.permute(1, 0), kernel_size=kernel_size_3d).permute(1,
                                                                                                                    0)
                return create_video_features(feat_2d, feat_3d, n_tokens, strategy='nearest')
            elif feat_3d is None:
                cur_n_tokens_2d = feat_2d.shape[0]
                if cur_n_tokens_2d <= n_tokens:
                    return create_video_features(feat_2d, feat_2d, n_tokens, strategy='clip')
                kernel_size_2d = int(np.floor(cur_n_tokens_2d / n_tokens))
                if kernel_size_2d <= 1:  # we don't have what to max pool
                    return create_video_features(feat_2d, feat_2d, n_tokens, strategy='nearest')
                feat_2d = torch.nn.functional.max_pool1d(feat_2d.permute(1, 0), kernel_size=kernel_size_2d).permute(1,
                                                                                                                    0)
                return create_video_features(feat_2d, feat_2d, n_tokens, strategy='nearest')
            else:
                cur_n_tokens_2d = feat_2d.shape[0]
                cur_n_tokens_3d = feat_3d.shape[0]
                if cur_n_tokens_3d <= n_tokens or cur_n_tokens_2d == 0:
                    return create_video_features(feat_2d, feat_3d, n_tokens, strategy='clip')

                kernel_size_3d = int(np.floor(cur_n_tokens_3d / n_tokens))
                kernel_size_2d = int(np.floor(cur_n_tokens_2d / n_tokens))

                if kernel_size_2d <= 1 or kernel_size_3d <= 1: # we don't have what to max pool
                    return create_video_features(feat_2d, feat_3d, n_tokens, strategy='nearest')

                feat_2d = torch.nn.functional.max_pool1d(feat_2d.permute(1, 0), kernel_size=kernel_size_2d).permute(1, 0)
                feat_3d = torch.nn.functional.max_pool1d(feat_3d.permute(1, 0), kernel_size=kernel_size_3d).permute(1, 0)
                return create_video_features(feat_2d, feat_3d, n_tokens, strategy='nearest')
        else:
            raise NotImplementedError


def _crop_audio_from_mel_spec(start, end, mel_spec):
    frames = librosa.core.time_to_frames([start, end], sr=16000, hop_length=160,
                                         n_fft=400)
    mel_spec = mel_spec[:, max(0, frames[0]): frames[1]]
    return mel_spec


def _get_video(features_2d, features_3d, fps_2d, fps_3d, starts, ends, n_video_tokens, video_sampling_strategy='clip',
               accurate_borders=False):
    def get_slice(features, fps, start, end):
        if accurate_borders:
            start = int(np.floor(start * fps))
            end = int(np.ceil(end * fps))
        else:
            # this was in baseline code
            start = int(start * fps)
            end = int(end * fps) + 1
        if features is not None:
            return features[start:end]
        else:
            return None

    all_videos = []
    all_video_masks = []
    for i in range(len(starts)):
        slice_2d = get_slice(features_2d, fps_2d, starts[i], ends[i])
        slice_3d = get_slice(features_3d, fps_3d, starts[i], ends[i])
        video, video_mask = create_video_features(slice_2d, slice_3d, n_video_tokens, strategy=video_sampling_strategy)
        all_videos.append(video)
        all_video_masks.append(video_mask)
    all_videos = torch.stack(all_videos, dim=0)
    all_video_masks = torch.stack(all_video_masks, dim=0)

    return all_videos, all_video_masks


def cut_into_clips(video, video_mask, audio, audio_mask, text, text_mask, raw_text, audio_STFT_nframes, id_, dataset,
                   n_clips):
    # create audio clips
    max_num_audio_STFT_frames = int(audio_mask.shape[0] // n_clips)
    audio = audio.permute(1, 0) \
        .view(n_clips, max_num_audio_STFT_frames, audio.size(0)) \
        .permute(0, 2, 1)
    audio_mask = audio_mask.view(n_clips, max_num_audio_STFT_frames)

    # create video clips
    n_video_tokens = int(video_mask.shape[0] // n_clips)
    video = video.view(n_clips, n_video_tokens, video.size(-1))
    video_mask = video_mask.view(n_clips, n_video_tokens)

    # copy text
    text = text.unsqueeze(0).expand(n_clips, -1, -1)
    text_mask = text_mask.unsqueeze(0).expand(n_clips, -1)

    # determine audio_STFT_nframes
    new_audio_STFT_nframes = []
    new_id = []
    for i in range(n_clips):
        left_frame = audio_STFT_nframes - i * max_num_audio_STFT_frames
        if (i == 0) or (left_frame > 0.7 * max_num_audio_STFT_frames):
            new_audio_STFT_nframes.append(min(max_num_audio_STFT_frames, left_frame))
            new_id.append(id_)
        else:
            new_audio_STFT_nframes.append(max_num_audio_STFT_frames)
            new_id.append('-1')
    audio_STFT_nframes = torch.tensor(new_audio_STFT_nframes)
    id_ = new_id
    dataset = [dataset] * n_clips
    raw_text = [raw_text] * n_clips
    return video, video_mask, audio, audio_mask, text, text_mask, raw_text, audio_STFT_nframes, id_, dataset