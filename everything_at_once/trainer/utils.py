import itertools

import numpy as np
import torch


def average_embeddings(ids_arr, embed_arr, verbose=False):
    # check if ids are unique, if not average embedings with the same ids
    ids_arr = np.array(ids_arr)
    unique_ids, counts = np.unique(ids_arr, return_counts=True)

    if len(ids_arr) != len(unique_ids):
        # group and average https://discuss.pytorch.org/t/groupby-aggregate-mean-in-pytorch/45335
        index = {id_: idx for idx, id_ in enumerate(unique_ids)}
        indexed_ids = [index[id_] for id_ in ids_arr]

        for name, embed in embed_arr.items():
            labels = torch.LongTensor(indexed_ids).to(embed.device)
            labels = labels.view(labels.size(0), 1).expand(-1, embed.size(1))
            unique_labels, labels_count = labels.unique(dim=0, return_counts=True)

            res = torch.zeros_like(unique_labels, dtype=embed.dtype).scatter_add_(0, labels, embed)
            embed_arr[name] = res / labels_count.float().unsqueeze(1)

            # if there were items that we wanted to skip (id=-1):
            if '-1' in index:
                bad_label = index['-1']
                idx_bad_label = (unique_labels[:, 0] == bad_label).nonzero(as_tuple=True)[0] #https://stackoverflow.com/questions/47863001/how-pytorch-tensor-get-the-index-of-specific-value
                if verbose:
                    print('Percent of skipped:', (labels == bad_label).sum() / (labels.shape[0] * labels.shape[1]) )
                embed = embed_arr[name]
                if idx_bad_label == 0:
                    embed = embed[1:]
                elif idx_bad_label == len(unique_labels) - 1:
                    embed = embed[:-1]
                else:
                    embed = torch.cat(
                        (embed[:idx_bad_label], embed[idx_bad_label + 1:]),
                        dim=0)
                embed_arr[name] = embed

    return embed_arr


def _move_to_device(data, device):
    if torch.is_tensor(data):
        return data.to(device)
    else:
        return {key: val.to(device) for key, val in data.items()}


def short_verbose(epoch, dl_nested_metrics, dataset_name):
    for metric_set_name in ['t2v_metrics', 't2v+a_metrics', 't2va_metrics']:
        if metric_set_name in dl_nested_metrics:
            metrics = dl_nested_metrics[metric_set_name]
            if all((metric_name in metrics) for metric_name in ['R1', 'R5', 'R10', 'R50', 'MedR', 'MeanR']):
                r1, r5, r10, r50 = metrics["R1"], metrics["R5"], metrics["R10"], metrics["R50"]

                msg = f"[{metric_set_name}]{dataset_name:s} epoch {epoch}         {r1:.1f} {r5:.1f} {r10:.1f} {metrics['MedR']:g}"
                msg += f"           {r50:.1f} {metrics['MedR']:g} {metrics['MeanR']:.1f}"
                print(msg)


def verbose(epoch, metrics, mode, name="TEST"):
    if all((metric_name in metrics) for metric_name in ['R1', 'R5', 'R10', 'R50', 'MedR', 'MeanR']):
        r1, r5, r10, r50 = metrics["R1"], metrics["R5"], metrics["R10"], metrics["R50"]
        msg = f"[{mode}]{name:s} epoch {epoch}, R@1: {r1:.1f}"
        msg += f", R@5: {r5:.1f}, R@10 {r10:.1f}, R@50 {r50:.1f}"
        msg += f"MedR: {metrics['MedR']:g}, MeanR: {metrics['MeanR']:.1f}"
        print(msg)


def format_nested_metrics_for_writer(metrics, mode, name="TEST"):
    res = {}
    for key, val in metrics.items():
        log_name = f"[{mode}]{name}_{key}"
        res[log_name] = val
    return res


def format_dataloader_output(data):
    if data.get('unroll_clips', [False])[0]:
        # TODO: make it more clean
        # We need to unroll clip axis if we samples several clips for one item
        for field in ['video', 'video_mask', 'audio', 'audio_mask', 'text', 'text_mask', 'audio_STFT_nframes', 'y_true']:
            if field in data:
                data[field] = data[field].view(-1, *data[field].shape[2:])

        data['raw_text'] = list(itertools.chain.from_iterable(zip(*data['raw_text'])))  # TODO: make it more clean
        data['meta'] = {
            'dataset': list(itertools.chain.from_iterable(zip(*data['meta']['dataset']))),  # TODO: make it more clean
            'paths': list(itertools.chain.from_iterable(zip(*data['meta']['paths']))),
            'ids': list(itertools.chain.from_iterable(zip(*data['meta']['ids'])))
        }
    return data
