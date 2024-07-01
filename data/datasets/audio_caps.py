import os
import torch
import csv
import numpy as np
from utils.directories import directories, get_dataset_dir
from sacred import Ingredient
from data.datasets.audioset import audioset, get_audioset
from data.datasets.dataset_base_classes import audio_dataset, DatasetBaseClass

audiocaps = Ingredient('audiocaps', ingredients=[directories, audioset, audio_dataset])

SPLITS = ['train', 'val', 'test']

@audiocaps.config
def config():
    folder_name = 'audiocaps'
    compress = True


@audiocaps.capture
def get_audiocaps(split, folder_name, compress):
    path = os.path.join(get_dataset_dir(), folder_name)
    ds = AudioCapsDataset(path, split=split)
    ds.compress = compress
    return ds


class AudioCapsDataset(DatasetBaseClass):

    @audiocaps.capture
    def __init__(self, folder_name, compress, split='train', mp3=False):
        super().__init__()
        root_dir = os.path.join(get_dataset_dir(), folder_name)
        # check parameters
        assert os.path.exists(root_dir), f'Parameter \'audio_caps_root\' is invalid. {root_dir} does not exist.'
        assert split in SPLITS, f'Parameter \'split\' must be in {SPLITS}.'

        self.audio_caps_root = root_dir
        self.split = split
        self.compress = compress

        if split == 'validation':   # rename validation split
            split = 'val'

        # read ytids and captions from csv
        with open(os.path.join(self.audio_caps_root, 'dataset', f'{split}.csv'), 'r') as f:
            reader = csv.reader(f, delimiter=',')
            lines = list(reader)[1:]
        _, self.ytids, _, self.captions = list(map(list, zip(*lines)))
        # sort captions by ytid
        self.ytids, self.captions = list(zip(*sorted(zip(self.ytids, self.captions))))

        # get paths and prediction targets
        self.audioset = get_audioset('train').set_quick(True)
        idx = dict(zip(self.audioset.ytids, range(0, len(self.audioset.ytids))))

        self.paths, self.targets, self.keywords = [], [], []
        for ytid, caption in zip(self.ytids, self.captions):
            i = idx.get('Y' + ytid)
            if i is None:
                continue
            self.paths.append(self.audioset[i]['path'][:-3]+'mp3' if mp3 else self.audioset[i]['path'])
            self.keywords.append(";".join([
                self.audioset.ix_to_lb[i] for i in np.where(self.audioset[i]['target'])[0]
            ]))
            self.targets.append(caption)
            # self.targets.append(self.audioset[i]['target'])

        self.captions = self.targets

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):

        a = self.__get_audio__(index)

        a['keywords'] = self.keywords[index]
        a['caption'] = self.targets[index]
        a['idx'] = index + 1000000
        a['caption_hard'] = ''
        a['html'] = ''
        return a

    def __get_audio_paths__(self):
        return self.paths

    def __str__(self):
        return f'AudioCaps_{self.split}'


if __name__ == '__main__':
    from sacred import Experiment

    ex = Experiment('test', ingredients=[audiocaps])

    @ex.main
    def main_():
        ds = get_audiocaps('train')
        ds.cache_audios()
        print(ds[0])

        import scipy.io.wavfile
        scipy.io.wavfile.write("test.wav", 32000, ds[0]['audio'])

    ex.run()
