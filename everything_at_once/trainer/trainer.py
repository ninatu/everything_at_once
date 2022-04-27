import collections

import numpy as np
import torch
import tqdm

from everything_at_once.model.utils.utils import sim_matrix
from everything_at_once.trainer.clip_utils import _apply_clip_text_model
from everything_at_once.trainer.utils import average_embeddings, _move_to_device, short_verbose, verbose, \
    format_nested_metrics_for_writer, format_dataloader_output
from everything_at_once.base import BaseTrainer
from torch.cuda.amp import autocast, GradScaler


class Trainer(BaseTrainer):
    def __init__(self, model, loss, metrics, optimizer, config, data_loader,
                 valid_data_loaders=[], lr_scheduler=None, writer=None):
        super().__init__(model, loss, metrics, optimizer, config, writer)
        self.config = config
        self.data_loader = data_loader

        self.valid_data_loader = valid_data_loaders
        self.lr_scheduler = lr_scheduler
        self.batch_size = self.data_loader.effective_batch_size
        self.log_step = int(np.sqrt(self.batch_size))
        self.scaler = GradScaler()

        cfg_trainer = config['trainer']
        self.max_samples_per_epoch = cfg_trainer.get('max_samples_per_epoch', None)
        self.mixed_precision = cfg_trainer.get("mixed_precision", False)
        self.use_eval_mode_always = cfg_trainer.get("use_eval_mode_always", False)
        self.save_latest = cfg_trainer.get('save_latest', True)

        self.use_clip_text_model = cfg_trainer.get("use_clip_text_model", False)
        if self.use_clip_text_model:
            import clip
            device, device_ids = self._prepare_device(config['n_gpu'])
            self.clip_text_model, _ = clip.load("ViT-B/32", device=device)
            self.clip_text_model.eval()
        else:
            self.clip_text_model = None

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.

        Note:
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log

            The metrics in log must have the key 'metrics'.
        """
        torch.cuda.empty_cache()

        if self.use_eval_mode_always:
            self.model.eval()
        else:
            self.model.train()

        total_loss = 0
        total_iter = 0
        for batch_idx, data in enumerate(self.data_loader):
            if self.max_samples_per_epoch is not None and \
                    (batch_idx + 1) * self.batch_size > self.max_samples_per_epoch:
                break

            data = format_dataloader_output(data)

            if self.clip_text_model is not None:
                data = _apply_clip_text_model(self.clip_text_model, data, self.device)

            for field in ['text', 'text_mask', 'video', 'video_mask', 'audio', 'audio_mask', 'audio_STFT_nframes']:
                if field in data:
                    data[field] = _move_to_device(data[field], self.device)

            if self.mixed_precision:
                self.optimizer.zero_grad()
                with autocast():
                    output = self.model(data)
                    loss, loss_info = self.loss(output)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.zero_grad()
                output = self.model(data)
                loss, loss_info = self.loss(output)
                loss.backward()
                self.optimizer.step()

            if self.writer is not None:
                for loss_name, loss_value in loss_info.items():
                    self.writer.log_scalar(f'loss_train_{loss_name}', loss_value, step=self.step)
                self.writer.log_scalar(f'loss_train', loss.detach().item(), step=self.step)

            total_loss += loss.detach().item()
            total_iter += 1

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.detach().item()))

            self.step += 1
            del data, output, loss

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        if self.save_latest:
            self._save_checkpoint(epoch, save_latest=True)

        log = {f'loss': total_loss / total_iter}
        val_log = self._valid_epoch(epoch)
        log.update(val_log)

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :return: A log that contains information about validation

        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        torch.cuda.empty_cache()
        self.model.eval()

        nested_metrics = {x: {} for x in range(len(self.valid_data_loader))}
        val_loss = [0] * len(self.valid_data_loader)

        with torch.no_grad():
            for dl_idx, dl in enumerate(self.valid_data_loader):
                dl_nested_metrics, dl_val_loss, dl_val_loss_detailed = eval(self.model, dl,
                                                                            self.device,
                                                                            self.metrics,
                                                                            self.loss,
                                                                            self.clip_text_model)

                val_loss[dl_idx] = dl_val_loss
                nested_metrics[dl_idx] = dl_nested_metrics

                if self.writer is not None:
                    self.writer.log_scalar(f'loss_val_{dl_idx}', dl_val_loss, step=epoch)
                    for loss_name, loss_value in dl_val_loss_detailed.items():
                        self.writer.log_scalar(f'loss_val_{loss_name}_{dl_idx}', loss_value, step=epoch)

                short_verbose(epoch=epoch, dl_nested_metrics=dl_nested_metrics, dataset_name=dl.dataset_name)
                for metric in self.metrics:
                    metric_name = metric.__name__
                    res = dl_nested_metrics[metric_name]
                    verbose(epoch=epoch, metrics=res, name=dl.dataset_name,
                            mode=metric_name)

                    if self.writer is not None:
                        to_write = format_nested_metrics_for_writer(res, mode=metric_name,
                                                                    name=self.valid_data_loader[dl_idx].dataset_name)
                        for key, val in to_write.items():
                            self.writer.log_scalar(key, val, step=epoch)

        res_dict = {f'val_loss_{dl_idx}': val_loss[dl_idx]
                    for dl_idx in range(len(self.valid_data_loader))}
        res_dict['nested_val_metrics'] = nested_metrics

        return res_dict

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        current = batch_idx * self.data_loader.effective_batch_size
        total = self.data_loader.n_samples
        return base.format(current, total, 100.0 * current / total)


def eval(model, dl, device, metrics, loss_func=None, clip_text_model=None):
    torch.cuda.empty_cache()

    total_val_loss = 0
    total_val_loss_detailed = collections.defaultdict(lambda: 0)
    meta_arr = []
    ids_arr = []
    embed_arr = collections.defaultdict(lambda: [])

    with torch.no_grad():
        for data in tqdm.tqdm(dl):
            data = format_dataloader_output(data)

            meta_arr.append(data['meta'])
            ids_arr.extend(data['meta']['ids'])

            if clip_text_model is not None:
                data = _apply_clip_text_model(clip_text_model, data, device)

            for field in ['text', 'text_mask', 'video', 'video_mask', 'audio', 'audio_mask', 'audio_STFT_nframes', 'caption', 'image']:
                if field in data:
                    data[field] = _move_to_device(data[field], device)
            embeds = model(data, force_cross_modal=True)
            for name, embed in embeds.items():
                if '_embed' in name:
                    embed_arr[name].append(embed.cpu())

            if loss_func is not None:
                loss, loss_info = loss_func(embeds)
                for loss_name, loss_value in loss_info.items():
                    total_val_loss_detailed[loss_name] += loss_value
                total_val_loss += loss.item()
                del loss
            del data, embeds

    val_loss = total_val_loss / len(dl)
    val_loss_detailed = {loss_name: loss_value / len(dl) for loss_name, loss_value in total_val_loss_detailed.items()}

    # compute scores
    nested_metrics = {}

    for name, embed in embed_arr.items():
        embed_arr[name] = torch.cat(embed, dim=0)
        print(embed_arr[name].shape)

    # needed for 'cut_clips: true' ablation
    embed_arr = average_embeddings(ids_arr, embed_arr, verbose=True)

    sims = {}
    for name1 in ['text_embed', 'video_embed', 'audio_embed']:
        if name1 not in embed_arr:
            continue
        embed1 = embed_arr[name1]
        for name2, embed2 in embed_arr.items():
            name1 = name1.replace('_embed', '').replace('text', 't').replace('audio', 'a').replace('video', 'v')
            name2 = name2.replace('_embed', '').replace('text', 't').replace('audio', 'a').replace('video', 'v')
            if name1 in name2 or f'{name2}2{name1}' in sims:
                continue
            sims[f'{name1}2{name2}'] = sim_matrix(embed1, embed2).detach().cpu().numpy()

    for metric in metrics:
        metric_name = metric.__name__
        # TODO: make it more clean:
        # we need to know the real test size for a fair comparison
        # we count the missing test clips as mistakes
        if hasattr(dl.dataset, 'complete_dataset_size'):
            complete_dataset_size = dl.dataset.complete_dataset_size
        else:
            complete_dataset_size = None

        res = metric(sims, complete_dataset_size=complete_dataset_size)
        nested_metrics[metric_name] = res

    return nested_metrics, val_loss, val_loss_detailed
