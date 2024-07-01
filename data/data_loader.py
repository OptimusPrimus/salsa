import os
from sacred import Ingredient
from torch.utils.data import WeightedRandomSampler, DataLoader, Sampler
import numpy as np
import torch
data_loader = Ingredient('data_loader')


@data_loader.config
def config():
    batch_size = 24
    batch_size_eval = 24
    n_workers = 16

    # only used if targets are given
    queue_random_sampling = False


@data_loader.capture
def get_train_data_loader(data_set, batch_size, n_workers, queue_random_sampling, targets=None, collate_fun=None, shuffle=True):

    if targets is None:
        return DataLoader(data_set, batch_size=batch_size, num_workers=n_workers, shuffle=shuffle, collate_fn=collate_fun, drop_last=True)
    else:
        if queue_random_sampling:
            sampler = CustomQueueRandomSampler(targets=targets)
        else:
            sampler = CustomWeightedRandomSampler(targets=targets, replacement=False)
        return DataLoader(data_set, batch_size=batch_size, num_workers=n_workers, sampler=sampler)


@data_loader.capture
def get_eval_data_loader(data_set, batch_size_eval, n_workers, collate_fun=None, shuffle=False, distributed=False):
    if distributed:
        print("Using distributed sampler")
        num_replicas = int(os.environ['WORLD_SIZE'])
        rank = int(os.environ['NODE_RANK'])
        sampler = torch.utils.data.DistributedSampler(data_set, shuffle=shuffle, num_replicas=num_replicas, rank=rank)
        return DataLoader(data_set, batch_size=batch_size_eval, num_workers=n_workers, sampler=sampler
                          , collate_fn=collate_fun)
    return DataLoader(data_set, batch_size=batch_size_eval, num_workers=n_workers, shuffle=shuffle, collate_fn=collate_fun)


class CustomQueueRandomSampler(Sampler):

    def __init__(self, targets, random_seed=122):
        """Balanced sampler. Generate batch meta for training. Data are equally
        sampled from different sound classes.

        Args:

          batch_size: int
          black_list_csv: string
          random_seed: int
        """
        super(CustomQueueRandomSampler, self).__init__(targets)

        self.targets = targets
        self.classes_num = targets.shape[1]

        self.samples_num_per_class = np.sum(self.targets, axis=0)
        self.random_state = np.random.RandomState(random_seed)

        # Training indexes of all sound classes. E.g.:
        # [[0, 11, 12, ...], [3, 4, 15, 16, ...], [7, 8, ...], ...]
        self.indexes_per_class = []

        for k in range(targets.shape[1]):
            self.indexes_per_class.append(np.where(self.targets[:, k] == 1)[0])

        # Shuffle indexes
        for k in range(self.classes_num):
            self.random_state.shuffle(self.indexes_per_class[k])

        self.queue = []
        self.pointers_of_classes = [0] * self.classes_num

    def expand_queue(self, queue):
        classes_set = np.arange(self.classes_num).tolist()
        self.random_state.shuffle(classes_set)
        queue += classes_set
        return queue

    def __iter__(self):
        """Generate batch meta for training.

        Returns:
          batch_meta: e.g.: [
            {'hdf5_path': string, 'index_in_hdf5': int},
            ...]
        """

        while True:
            if len(self.queue) == 0:
                self.queue = self.expand_queue(self.queue)

            class_id = self.queue.pop(0)
            pointer = self.pointers_of_classes[class_id]
            self.pointers_of_classes[class_id] += 1
            index = self.indexes_per_class[class_id][pointer]

            # When finish one epoch of a sound class, then shuffle its indexes and reset pointer
            if self.pointers_of_classes[class_id] >= self.samples_num_per_class[class_id]:
                self.pointers_of_classes[class_id] = 0
                self.random_state.shuffle(self.indexes_per_class[class_id])

            yield index

    def state_dict(self):
        state = {
            'indexes_per_class': self.indexes_per_class,
            'queue': self.queue,
            'pointers_of_classes': self.pointers_of_classes}
        return state

    def load_state_dict(self, state):
        self.indexes_per_class = state['indexes_per_class']
        self.queue = state['queue']
        self.pointers_of_classes = state['pointers_of_classes']


class CustomWeightedRandomSampler(WeightedRandomSampler):

    def __init__(self, targets: np.ndarray, *args, **kwargs):
        # flatten weights
        frequencies = 1000. / (targets.sum(axis=0, keepdims=True) + 100)
        weights = (targets * frequencies).sum(axis=1)

        super().__init__(weights=weights, num_samples=len(weights), *args, **kwargs)
