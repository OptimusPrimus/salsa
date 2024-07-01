import os
import torch
import csv
import json
from data.datasets.audioset import AudioSetDataset
from utils.directories import directories, get_dataset_dir
from sacred import Ingredient
from data.datasets.audioset import audioset, get_audioset
from data.datasets.dataset_base_classes import audio_dataset, DatasetBaseClass

wavcaps = Ingredient('wavcaps', ingredients=[directories, audioset, audio_dataset])


@wavcaps.config
def config():
    folder_name = 'wavcaps'
    compress = True
    exclude_clothov2 = True


@wavcaps.capture
def get_wavecaps(folder_name, exclude_clothov2):
    path = os.path.join(get_dataset_dir(), folder_name)
    wc = WaveCaps(path)

    if exclude_clothov2:
        with open(os.path.join(path, 'dcase2024_task6_excluded_freesound_ids.csv'), 'r') as f:
            ids = set([r[0] for r in csv.reader(f)][1:])
            print("WavCaps before filtering ClothoV2:", len(wc))
            wc = wc.get_subset(lambda s: not(s['path'].split(os.sep)[-2] == 'FreeSound' and s['path'].split(os.sep)[-1].split('.')[0] in ids))
            print("WavCaps after filtering ClothoV2:", len(wc))
    return wc


def get_audioset_subset(wave_caps_root):

    with open(os.path.join(wave_caps_root, 'json_files', 'AudioSet_SL', 'as_final.json'), 'r') as f:
        files = json.load(f)['data']


    missing = {'Y06-g5jz-OGc.wav', 'Y3sSblRfEG2o.wav', 'YFli8wjBFV2M.wav', 'YVcu0pVF1npM.wav', 'YWudGD6ZHRoY.wav',
               'YmW3S0u8bj58.wav'}

    return [
        {
            'path': os.path.join(wave_caps_root, 'audio', 'AudioSet_SL', f['id'][:-4] + '.flac'),
            'caption': f['caption'],
            'keywords': ""
            #'description': "",
            #'url': f['id']
        } for f in sorted(files, key=lambda x: x['id']) if f['id'] not in missing
    ]


def get_soundbible_subset(wave_caps_root):

    with open(os.path.join(wave_caps_root, 'json_files', 'SoundBible', 'sb_final.json'), 'r') as f:
        files = json.load(f)['data']

    return [
        {
            'path': os.path.join(wave_caps_root, 'audio', 'SoundBible', f['id'] + '.flac'),
            'caption': f['caption'],
            'keywords': f['title']
            # 'description': f['description'],
            # 'url': f['id']
        } for f in sorted(files, key=lambda x: x['id'])
    ]

def get_bbc_subset(wave_caps_root, filter=False):

    with open(os.path.join(wave_caps_root, 'json_files', 'BBC_Sound_Effects', 'bbc_final.json'), 'r') as f:
        files = json.load(f)['data']

    return [
        {
            'path': os.path.join(wave_caps_root, 'audio', 'BBC_Sound_Effects', f['id'] + '.flac'),
            'caption': f['caption'],
            'keywords': ";".join([p.replace('[', '').replace(']', '').replace("'", '') for p in f['category'].split(",")])
            # 'description': f['description'],
            # 'url': f['id']
        } for f in sorted(files, key=lambda x: x['id'])
    ]

def get_freesound_subset(wave_caps_root):

    with open(os.path.join(wave_caps_root, 'json_files', 'FreeSound', 'fsd_final_2s.json'), 'r') as f:
        files = json.load(f)['data']

    return [
        {
            'path': os.path.join(wave_caps_root, 'audio', 'FreeSound', f['id'] + '.flac'),
            'caption': f['caption'],
            'keywords': ";".join(f['tags'])
            # 'description': f['description'],
            # 'url': f['href']
        } for f in sorted(files, key=lambda x: x['id'])
    ]


class WaveCaps(DatasetBaseClass):

    @wavcaps.capture
    def __init__(self, folder_name, compress):
        super().__init__()
        root_dir = os.path.join(get_dataset_dir(), folder_name)
        # check parameters
        assert os.path.exists(root_dir), f'Parameter \'audio_caps_root\' is invalid. {root_dir} does not exist.'

        self.wave_caps_root = root_dir

        # with open(os.path.join(root_dir, 'missing_files.json'), 'r') as f:
        #    self.missing = json.load(f)
        #    self.missing = set(["/".join(m.split('/')[-2:]) for m in self.missing])

        samples_as = get_audioset_subset(self.wave_caps_root)
        samples_soundbible = get_soundbible_subset(self.wave_caps_root)
        samples_fsd = get_freesound_subset(self.wave_caps_root)
        # samples_fsd = get_freesound_2_subset(self.wave_caps_root)
        samples_bbc = get_bbc_subset(self.wave_caps_root)

        # filter

        print("Files per data set:")
        print("AudioSet: ", len(samples_as))
        print("FreeSound: ", len(samples_fsd))
        print("SoundBible: ", len(samples_soundbible))
        print("BBC: ", len(samples_bbc))
        # print("missing_files: ", len(self.missing))

        self.samples = samples_bbc + samples_soundbible + samples_fsd + samples_as

        # self.samples = [s for s in self.samples if "/".join(s['path'].split('/')[-2:]) not in self.missing]

        self.captions = [s["caption"] for s in self.samples]
        self.paths = [s["path"] for s in self.samples]
        self.compress = compress

    def __get_audio_paths__(self):
        return self.paths

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        audio = self.__get_audio__(index)
        for k in self.samples[index]:
            audio[k] = self.samples[index][k]
        # audio["keywords"] = ''
        audio["idx"] = index
        audio["caption_hard"] = ''
        audio["html"] = ''
        return audio

    def __str__(self):
        return f'WavCaps'


if __name__ == '__main__':
    from sacred import Experiment
    from data.datasets.clotho_v2 import clotho_v2, get_clotho_v2
    from data.datasets.audio_caps import audiocaps, get_audiocaps
    from data.datasets.audioset import audioset, get_audioset
    from data.audio_loader import LoadMP3CompressedIntoRam
    ex = Experiment('test', ingredients=[wavcaps, audiocaps, audioset, clotho_v2])

    @ex.main
    def main_():
        wc = get_wavecaps().set_quick()
        # wc = wc.cache_audios()
        ss = wc.get_subset(lambda x: 'FreeSound' in x['path'])
        # wc.set_fixed_length(30)

        #from torch.utils.data import Subset
        #import numpy as np
        #dl = torch.utils.data.DataLoader(Subset(wc, np.arange(258180, len(wc))), 10, num_workers=16)

        #from tqdm import tqdm
        #for b in tqdm(dl):
        #    continue

        # a = get_audioset('evaluation')
        train = ['/' + "/".join(s['sound_link'].split('/')[3:]) + '/' for s in get_clotho_v2('train').set_quick() if type(s['sound_link']) is str]
        val = ['/' + "/".join(s['sound_link'].split('/')[3:]) + '/' for s in get_clotho_v2('val').set_quick()  if type(s['sound_link']) is str]
        test = ['/' + "/".join(s['sound_link'].split('/')[3:]) + '/' for s in get_clotho_v2('test').set_quick() if type(s['sound_link']) is str]

        urls = set([w['url'] for w in wc])


        len(urls.intersection(set(train))) / len(train)
        len(urls.intersection(set(val))) / len(val)
        len(urls.intersection(set(test))) / len(test)


        #missing = []
        #for s in wc:
        #    if not os.path.exists(s['path']):
        #        print(s['path'])
        #        missing.append(s['path'])

        #import json
        #print("missing", len(missing))
        #with open('missing_files.json', 'w') as f:
        #    json.dump(missing, f)

        # wc = LoadMP3CompressedIntoRam(wc, sample_rate=32000, processes=16, compress=True, shared=False)
        # print("Finished creating wavecaps....")

    ex.run()
