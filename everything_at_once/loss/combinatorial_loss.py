import torch.nn as nn

from everything_at_once.loss.contrastive_losses import NormSoftmaxLoss, MMS_Loss
from everything_at_once.model.utils.utils import sim_matrix


class CombinatorialLoss(nn.Module):
    def __init__(self, contrastive_loss='NormSoftmax', temperature=0.05,
                 tv_weight=0, ta_weight=0, va_weight=0,
                 t_va_weight=0, v_ta_weight=0, a_tv_weight=0):
        super().__init__()

        if contrastive_loss == 'NormSoftmax':
            self.contrastive_loss = NormSoftmaxLoss(temperature=temperature)
        elif contrastive_loss == 'MMS':
            self.contrastive_loss = MMS_Loss()
        else:
            raise NotImplementedError()

        self.tv_weight = tv_weight
        self.ta_weight = ta_weight
        self.va_weight = va_weight
        self.t_va_weight = t_va_weight
        self.v_ta_weight = v_ta_weight
        self.a_tv_weight = a_tv_weight

    def forward(self, input_data):

        nonempty = {}
        nonempty['tv'] = input_data['text_nonempty_input_mask'] & input_data['video_nonempty_input_mask']
        nonempty['ta'] = input_data['text_nonempty_input_mask'] & input_data['audio_nonempty_input_mask']
        nonempty['va'] = input_data['video_nonempty_input_mask'] & input_data['audio_nonempty_input_mask']

        nonempty['t_va'] = input_data['text_nonempty_input_mask'] & (
                    input_data['video_nonempty_input_mask'] & input_data['audio_nonempty_input_mask'])
        nonempty['v_ta'] = input_data['video_nonempty_input_mask'] & (
                    input_data['text_nonempty_input_mask'] & input_data['audio_nonempty_input_mask'])
        nonempty['a_tv'] = input_data['audio_nonempty_input_mask'] & (
                    input_data['text_nonempty_input_mask'] & input_data['video_nonempty_input_mask'])

        loss_sum = 0
        weight_sum = 0
        loss_info = {}

        for name, embed_name1, embed_name2, weight in [
            ('tv', 'text_embed', 'video_embed', self.tv_weight),
            ('ta', 'text_embed', 'audio_embed', self.ta_weight),
            ('va', 'video_embed', 'audio_embed', self.va_weight),
            ('t_va', 'text_embed', 'va_embed', self.t_va_weight),
            ('v_ta', 'video_embed', 'ta_embed', self.v_ta_weight),
            ('a_tv', 'audio_embed', 'tv_embed', self.a_tv_weight),
        ]:
            if (embed_name1 in input_data) and (embed_name2 in input_data) and (weight != 0):
                nonempty_mask = nonempty[name]
                embed1 = input_data[embed_name1][nonempty_mask]
                embed2 = input_data[embed_name2][nonempty_mask]

                loss = self.contrastive_loss(sim_matrix(embed1, embed2))
                loss_info[name] = loss.item()
                loss_sum += weight * loss
                weight_sum += weight

        final_loss = loss_sum / weight_sum
        loss_info['Retrieval'] = final_loss.item()
        return final_loss, loss_info
