import argparse
import collections
from sacred import Experiment
from neptunecontrib.monitoring.sacred import NeptuneObserver

import torch.optim as module_optim
import torch.optim.lr_scheduler as module_lr_scheduler

from everything_at_once import data_loader as module_data
from everything_at_once import model as module_arch
from everything_at_once import loss as module_loss

from everything_at_once.trainer import Trainer
from everything_at_once.metric import RetrievalMetric

from parse_config import ConfigParser


ex = Experiment('train', save_git_info=False)


@ex.main
def run():
    logger = config.get_logger('train')
    logger.info(f"Config: {config['name']}")

    # build model architecture, then print to console
    model = config.initialize('arch', module_arch)
    logger.info(model)

    # setup data_loader instances
    data_loader = config.initialize('data_loader', module_data)
    valid_data_loaders = init_dataloaders(config, module_data, data_loader='val_data_loaders')
    print('Train dataset: ', data_loader.n_samples, ' samples')
    print('Val dataset: ', [x.n_samples for x in valid_data_loaders], ' samples')

    # get function handles of loss and metrics
    loss = config.initialize(name="loss", module=module_loss)
    metrics = [RetrievalMetric(met) for met in config['metrics']]

    # build optimizer, learning rate scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.initialize('optimizer', module_optim, params=trainable_params)
    lr_scheduler = config.initialize('lr_scheduler', module_lr_scheduler, optimizer=optimizer)

    writer = ex if config['trainer']['neptune'] else None

    trainer = Trainer(model, loss, metrics, optimizer,
                      config=config,
                      data_loader=data_loader,
                      valid_data_loaders=valid_data_loaders,
                      lr_scheduler=lr_scheduler,
                      writer=writer)
    trainer.train()


def init_dataloaders(config, module_data, data_loader):
    if "type" in config[data_loader] and "args" in config[data_loader]:
        return [config.initialize(data_loader, module_data)]
    elif isinstance(config[data_loader], list):
        return [config.initialize(data_loader, module_data, index=idx) for idx in
                range(len(config[data_loader]))]
    else:
        raise ValueError("Check data_loader config, not correct format.")


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-n', '--neptune', action='store_true',
                      help='Whether to observe (neptune)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target=('optimizer', 'args', 'lr')),
        CustomArgs(['--bs', '--batch_size'], type=int, target=('data_loader', 'args', 'batch_size')),
        CustomArgs(['--ep', '--epochs'], type=int, target=('trainer', 'epochs')),
        CustomArgs(['--sp', '--save_period'], type=int, target=('trainer', 'save_period')),
        CustomArgs(['--mixed_precision'], type=int, target=('trainer', 'mixed_precision')),
        CustomArgs(['--save_latest'], type=int, target=('trainer', 'save_latest')),
        CustomArgs(['--n_gpu'], type=int, target=('n_gpu',)),
    ]
    config = ConfigParser(args, options)
    ex.add_config(config.config)

    if config['trainer']['neptune']:
        # delete this error if you have added your own neptune credentials neptune.ai
        raise ValueError

        ex.observers.append(NeptuneObserver(
            api_token='',
            project_name='',
            source_extensions=['train.py', 'test.py', 'parse_config.py',
                               'everything_at_once/**/*.py',
                               'configs/**/*.yaml', 'configs/**/*.yml']))
        ex.run()
    else:
        run()
