import os
import numpy as np
import csv
import tqdm
from data.datasets.dataset_base_classes import audio_dataset, DatasetBaseClass

from sacred import Ingredient

from utils.directories import directories, get_dataset_dir

audioset = Ingredient('audioset', ingredients=[directories, audio_dataset])

SPLITS = ['train', 'evaluation', 'balanced']
N_CLASSES = 527

cached_audiosets = None

@audioset.config
def config():
    # dirs
    folder_name = 'audioset'
    # data set configuration
    update_parent_labels = False
    compress = True


@audioset.capture
def get_audioset(split, update_parent_labels):
    return AudioSetDataset(split=split, update_parent_labels=update_parent_labels)


@audioset.capture
def get_class_labels_ids(folder_name):
    root_dir = os.path.join(get_dataset_dir(), folder_name)
    assert os.path.exists(root_dir), f'Parameter \'root_dir\' is invalid. {root_dir} does not exist.'

    metadata_dir = os.path.join(root_dir, 'metadata')

    with open(os.path.join(metadata_dir, 'class_labels_indices.csv'), 'r') as f:
        reader = csv.reader(f, delimiter=',')
        lines = list(reader)

    labels = []
    ids = []  # Each label has a unique id such as "/m/068hy"
    for i in range(1, len(lines)):
        _, id, label = lines[i]
        ids.append(id)
        labels.append(label)

    return labels, ids


class AudioSetDataset(DatasetBaseClass):
    @audioset.capture
    def __init__(self, folder_name, split='train', compress=True, update_parent_labels=False):
        super().__init__()
        root_dir = os.path.join(get_dataset_dir(), folder_name)
        assert os.path.exists(root_dir), f'Parameter \'root_dir\' is invalid. {root_dir} does not exist.'
        assert split in SPLITS, f'Parameter \'split\' must be in {SPLITS}.'

        self.metadata_dir = os.path.join(root_dir, 'metadata')
        self.update_parent_labels = update_parent_labels
        self.split = split
        self.compress = compress

        ###
        # load the labels, create label index mapping
        ###
        self.labels, self.ids = get_class_labels_ids()
        self.lb_to_ix = {label: i for i, label in enumerate(self.labels)}
        self.id_to_ix = {id: i for i, id in enumerate(self.ids)}
        self.ix_to_lb = {i: label for i, label in enumerate(self.labels)}

        ###
        # load file list
        ###
        self.ytids, self.parts, self.targets, self.paths = [], [], [], []

        if split == 'train':
            self.audio_dir = os.path.join(root_dir, 'audios', 'unbalanced_train_segments')
            for i in tqdm.tqdm(range(41), desc='Loading meta data'):
                file_path = os.path.join(self.metadata_dir, 'unbalanced_partial_csvs',
                                         f'unbalanced_train_segments_part{i:02d}.csv')
                self.append_csv_to_lists(file_path, part=i)
        elif split == 'balanced':
            self.audio_dir = os.path.join(root_dir, 'audios', 'balanced_train_segments')
            file_path = os.path.join(self.metadata_dir, 'balanced_train_segments.csv')
            self.append_csv_to_lists(file_path)
        elif split == 'evaluation':
            self.audio_dir = os.path.join(root_dir, 'audios', 'eval_segments')
            file_path = os.path.join(self.metadata_dir, 'eval_segments.csv')
            self.append_csv_to_lists(file_path)



    def get_updated_csv_rows(self, file_path, part):
        cleaned_csv_file = file_path[:-4] + '_cleaned.csv'

        if not os.path.exists(cleaned_csv_file):
            with open(file_path, 'r') as f:
                lines = f.readlines()
                lines = list(lines)[3:]

            with open(cleaned_csv_file, 'w') as f:
                for line in lines:
                    line = line.split(', ')
                    if self.split == 'train':
                        path = os.path.join(self.audio_dir, f'unbalanced_train_segments_part{part:02d}',
                                            'Y' + line[0] + '.wav')
                    else:
                        path = os.path.join(self.audio_dir, 'Y' + line[0] + '.wav')
                    if os.path.exists(path) or os.path.exists(path[:-3] + 'mp3'):
                        f.write(', '.join(line))

        with open(cleaned_csv_file, 'r') as f:
            lines = f.readlines()
            lines = list(lines)

        return lines

    def append_csv_to_lists(self, file_path, part=None):
        lines = self.get_updated_csv_rows(file_path, part)
        for line in lines:
            line = line.split(', ')
            self.ytids.append('Y' + line[0])
            self.parts.append(part)
            label_ids = line[3].split('"')[1].split(',')
            target = np.zeros(len(self.lb_to_ix), dtype=np.bool_)
            for id in label_ids:
                ix = self.id_to_ix[id]
                target[ix] = 1
            self.targets.append(target)
            if self.split == 'train':
                path = os.path.join(
                    self.audio_dir,
                    f'unbalanced_train_segments_part{part:02d}',
                    'Y' + line[0] + '.wav'
                )
            else:
                path = os.path.join(self.audio_dir, 'Y' + line[0] + '.wav')
            self.paths.append(path)

    def __len__(self):
        return len(self.ytids)

    def __get_audio_paths__(self):
        return self.paths

    def __getitem__(self, item):
        audio = self.__get_audio__(item)
        return {
            'path': self.paths[item],
            'audio': audio['audio'],
            'audio_length': audio['audio_length'],
            'ytid': self.ytids[item],
            'target': self.targets[item].copy() # copy, just to be save ...
        }

    def __str__(self):
        return f'AudioSet_{self.split}'


if __name__ == '__main__':
    from sacred import Experiment

    ex = Experiment('test', ingredients=[audioset])

    @ex.main
    def main_():
        ds = get_audioset('train')

        ds[0]
        # ds.cache_audios()

        ds = ds.set_quick(quick=True)
        metadata = [s for s in ds]
        ds = ds.set_quick(quick=False)

        ds.set_fixed_length(10)
        ds[0]

        ds.set_fixed_length(30)
        ds[0]

        ds.set_fixed_length(None)
        ds[0]

        class_labels = get_class_labels_ids()
        print(ds)

    ex.run()