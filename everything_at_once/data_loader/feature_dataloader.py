import pickle
from gensim.models.keyedvectors import KeyedVectors

from everything_at_once.base import BaseDataLoaderExplicitSplit

from everything_at_once.dataset import HowTo_Dataset, MSRVTT_Dataset, Youcook_Dataset, \
    CrossTaskMiningYoutubeDataset


caption = None
we = None


class FeatureDataloader(BaseDataLoaderExplicitSplit):
    def __init__(self,
                 dataset_name,
                 dataset_kwargs,
                 drop_last=False,
                 batch_size=1,
                 num_workers=1,
                 shuffle=True):


        word2vec_path = dataset_kwargs.pop('word2vec_path')
        global we
        if we is None:
            we = KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
        else:
            print('Using loaded we')

        if 'HowTo100M' in dataset_name:
            caption_path = dataset_kwargs.pop('caption_path')

            global caption
            if caption is None:
                with open(caption_path, 'rb') as fin:
                    caption = pickle.load(fin)
            else:
                print('Using loaded caption')
            dataset = HowTo_Dataset(**dataset_kwargs, caption=caption, we=we)
        elif 'MSRVTT' in dataset_name:
            dataset = MSRVTT_Dataset(**dataset_kwargs, we=we)
        elif 'YouCook2' in dataset_name:
            dataset = Youcook_Dataset(**dataset_kwargs, we=we)
        elif 'CrossTask' in dataset_name:
            assert batch_size == 1
            dataset = CrossTaskMiningYoutubeDataset(**dataset_kwargs, we=we, mining_youtube=False)
        elif 'MiningYoutube' in dataset_name:
            assert batch_size == 1
            dataset = CrossTaskMiningYoutubeDataset(**dataset_kwargs, we=we, mining_youtube=True)
        else:
            raise NotImplementedError(f"Dataset: {dataset_name} not found.")

        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                         drop_last=drop_last)

        self.dataset_name = dataset_name