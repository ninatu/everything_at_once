import numpy as np
import scipy.stats
import itertools


class RetrievalMetric:
    def __init__(self, task='t2v', break_ties='averaging'):
        assert break_ties in ['averaging', 'optimistically']
        task = task.replace('_metrics', '')
        assert task in ['t2v', 'v2t', 't2a', 'a2t', 'v2a', 'a2v',
                        't2va', 'va2t', 'v2ta', 'ta2v', 'a2tv', 'tv2a',
                        't2v+a', 'v+a2t', 'v2t+a', 't+a2v', 'a2t+v', 't+v2a',
                        't2v_va', 'v_va2t', 't2a_va', 'a_va2t',
                        'v2t_ta', 't_ta2v', 'v2a_ta', 'a_ta2v',
                        'a2t_tv', 't_tv2a', 'a2v_tv', 'v_tv2a'
                        ]
        self._task = task

        mod1, mod2 = self._task.split('2')
        self._inv_task = f"{mod2}2{mod1}"
        self.__name__ = f"{self._task}_metrics"
        self.break_ties = break_ties

    def __call__(self, sims_dict, complete_dataset_size=None):
        if self._task in sims_dict:
            return retrieval_metrics(sims_dict[self._task], complete_dataset_size=complete_dataset_size)
        elif self._inv_task in sims_dict:
            return retrieval_metrics(sims_dict[self._inv_task].T, complete_dataset_size=complete_dataset_size)
        else:
            return {}

    def __repr__(self):
        return f"{self._task}_metrics"


def retrieval_metrics(sims, break_ties='averaging', complete_dataset_size=None):
    num_queries, num_vids = sims.shape
    if complete_dataset_size is not None:
        num_queries = complete_dataset_size

    sx = np.sort(-sims, axis=1)
    d = np.diag(-sims)
    d = d[:, np.newaxis]
    diff = sx - d
    if break_ties == 'optimistically':
        ind = np.argmax(diff == 0, axis=1)
    elif break_ties == 'averaging':
        locs = np.argwhere(diff == 0)
        grouped_locs = [list(values) for n_row, values in itertools.groupby(locs, key=lambda x: x[0])]
        ind = [np.mean(list(map(lambda x: x[1], locs))) for locs in grouped_locs]
        ind = np.array(ind)
    else:
        raise NotImplementedError
    return cols2metrics(ind, num_queries)


def cols2metrics(cols, num_queries):
    metrics = {}
    metrics["R1"] = 100 * float(np.sum(cols == 0)) / num_queries
    metrics["R5"] = 100 * float(np.sum(cols < 5)) / num_queries
    metrics["R10"] = 100 * float(np.sum(cols < 10)) / num_queries
    metrics["R50"] = 100 * float(np.sum(cols < 50)) / num_queries
    metrics["MedR"] = np.median(cols) + 1
    metrics["MeanR"] = np.mean(cols) + 1
    stats = [metrics[x] for x in ("R1", "R5", "R10")]
    metrics["geometric_mean_R1-R5-R10"] = scipy.stats.mstats.gmean(stats)
    return metrics
