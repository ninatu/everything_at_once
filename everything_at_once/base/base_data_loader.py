from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate


class BaseDataLoaderExplicitSplit(DataLoader):
    """
    Base class for all data loaders
    """
    def __init__(self, dataset, batch_size, shuffle, num_workers, collate_fn=default_collate, drop_last=False):
        self.shuffle = shuffle
        self.batch_idx = 0
        self.n_samples = len(dataset)
        self.effective_batch_size = batch_size

        self.init_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'collate_fn': collate_fn,
            'num_workers': num_workers,
            'pin_memory': True,
            'drop_last': drop_last
        }
        super().__init__(**self.init_kwargs)
