import argparse
import torch
import collections
from sacred import Experiment

from everything_at_once import data_loader as module_data
from everything_at_once import model as module_arch

from everything_at_once.metric import RetrievalMetric
from everything_at_once.trainer import eval
from everything_at_once.trainer.utils import short_verbose, verbose
from everything_at_once.utils.util import state_dict_data_parallel_fix

from parse_config import ConfigParser


ex = Experiment('test')


@ex.main
def run():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # build model architecture
    if config['trainer'].get("use_clip_text_model", False):
        import clip
        clip_text_model, _ = clip.load("ViT-B/32", device=device)
        clip_text_model.eval()
    else:
        clip_text_model = None
    model = config.initialize('arch', module_arch)

    # setup data_loader instances
    data_loader = config.initialize('data_loader', module_data)

    metrics = [RetrievalMetric(met) for met in config['metrics']]

    checkpoint = torch.load(config.resume)
    epoch = checkpoint['epoch']
    state_dict = checkpoint['state_dict']
    new_state_dict = state_dict_data_parallel_fix(state_dict, model.state_dict())
    model.load_state_dict(new_state_dict, strict=True)

    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)

    # prepare model for testing
    model = model.to(device)
    model.eval()

    nested_metrics, val_loss, val_loss_detailed = eval(model, data_loader, device, metrics,
                                                       loss_func=None,
                                                       clip_text_model=clip_text_model)

    short_verbose(epoch=epoch, dl_nested_metrics=nested_metrics, dataset_name=data_loader.dataset_name)
    for metric in metrics:
        metric_name = metric.__name__
        res = nested_metrics[metric_name]
        verbose(epoch=epoch, metrics=res, name="", mode=metric_name)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')

    args.add_argument('-r', '--resume', required=True, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-c', '--config', default=True, type=str,
                      help='config file path (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

   # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--n_gpu'], type=int, target=('n_gpu',)),
    ]
    config = ConfigParser(args, options, test=True)
    args = args.parse_args()
    ex.add_config(config.config)

    ex.run()
