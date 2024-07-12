import torch
import pandas as pd
import os
from sacred import Ingredient
from utils.directories import directories, get_dataset_dir
from data.datasets.dataset_base_classes import audio_dataset, DatasetBaseClass

SPLITS = ['development', 'validation', 'evaluation']

clotho_v2 = Ingredient('clotho_v2', ingredients=[directories, audio_dataset])


@clotho_v2.config
def config():
    folder_name = 'clotho_v2'
    compress = False
    add_hard_negatives = False
    ablate_while = False
    add_hard_negatives_gpt = False


@clotho_v2.capture
def get_clotho_v2(split, folder_name, compress):

    splits = {'train': 'development', 'val': 'validation', 'test': 'evaluation'}
    assert split in list(splits.keys())
    ds = Clotho_v2Dataset(splits[split])
    return ds


class Clotho_v2Dataset(DatasetBaseClass):

    @clotho_v2.capture
    def __init__(self, split, folder_name, compress, add_hard_negatives, add_hard_negatives_gpt, ablate_while=False):
        super().__init__()
        self.compress = compress
        self.add_hard_negatives = add_hard_negatives
        self.add_hard_negatives_gpt = add_hard_negatives_gpt
        self.ablate_while=ablate_while
        root_dir = os.path.join(get_dataset_dir(), folder_name)

        assert os.path.exists(root_dir), f'Parameter \'root_dir\' is invalid. {root_dir} does not exist.'
        assert split in SPLITS, f'Parameter \'split\' must be in {SPLITS}.'
        self.split = split

        self.root_dir = root_dir
        self.files_dir = os.path.join(root_dir, split)
        captions_csv = f'clotho_captions_{split}.csv'
        metadata_csv = f'clotho_metadata_{split}.csv'

        kwargs = {'sep': ';'} if split in ['analysis'] else {}

        if split in ['analysis']:
            files = [file for file in os.listdir(self.files_dir) if file.endswith(".wav")]
            metadata = pd.DataFrame({'file_name': files}, index=files)
            captions = pd.DataFrame({'file_name': files}, index=files)
        else:
            metadata = pd.read_csv(os.path.join(root_dir, metadata_csv), encoding="ISO-8859-1", **kwargs)
            metadata = metadata.set_index('file_name')
            captions = pd.read_csv(os.path.join(root_dir, captions_csv))
            captions = captions.set_index('file_name')

        self.metadata = pd.concat([metadata, captions], axis=1)
        self.metadata.reset_index(inplace=True)
        self.num_captions = 1 if split in ['analysis'] else 5

        self.paths, self.attributes = [], []

        for i in range(len(self.metadata) * self.num_captions):
            attributes = dict(self.metadata.iloc[i // self.num_captions].items())

            # append to paths
            path = os.path.join(self.files_dir, attributes['file_name'])
            self.paths.append(path)

            # append to attributes
            caption_idx = i % self.num_captions
            if f'caption_{caption_idx + 1}' in attributes:
                attributes['caption'] = attributes[f'caption_{caption_idx + 1}']
                if 'caption_2' in attributes:
                    del attributes['caption_1'], attributes['caption_2'], attributes['caption_3'], attributes['caption_4'], attributes[
                        'caption_5']
                else:
                    del attributes['caption_1']
            else:
                attributes['caption'] = ''
            attributes['html'] = f'<iframe frameborder="0" scrolling="no" src="https://freesound.org/embed/sound/iframe/{attributes["sound_id"]}/simple/small/" width="375" height="30"></iframe>'
            if 'sound_id' in attributes:
                del attributes['sound_id'], attributes['sound_link']
            if 'start_end_samples' in attributes:
                del attributes['start_end_samples']
            if 'manufacturer' in attributes:
                del attributes['manufacturer']
            if 'license' in attributes:
                del attributes['license']
            if 'file_name' in attributes:
                del attributes['file_name']
            self.attributes.append(attributes)

        hard_captions_csv = f'hard_negative_captions_{split}.csv'
        self.hard_negatives = {}
        if os.path.exists(os.path.join(root_dir, hard_captions_csv)) and add_hard_negatives_gpt:
            import csv
            with open(os.path.join(root_dir, hard_captions_csv)) as csvfile:
                spamreader = csv.reader(csvfile)
                for row in spamreader:
                    self.hard_negatives[int(row[0])] = row[3:]

    def __get_audio_paths__(self):
        return self.paths

    def __getitem__(self, item):
        # get audio
        audio = self.__get_audio__(item)
        # get additional attributes
        attributes = self.attributes[item]
        # add attributes to dict
        for k in attributes:
            audio[k] = attributes[k]
        audio['idx'] = item
        audio['caption_hard'] = '' # get_hard_negative(audio['caption'], ablate_while=self.ablate_while) if self.add_hard_negatives else ''

        if audio['caption_hard'] != '' and self.hard_negatives.get(item):
            hard_index = torch.randint(len(self.hard_negatives.get(item)), (1,)).item()
            audio['caption_hard'] = self.hard_negatives.get(item)[hard_index]

        i = (item + 1) % 5
        j = item // 5

        # audio['caption_other'] = self.attributes[j*5 + i]['caption']

        return audio

    def __len__(self):
        return len(self.metadata) * self.num_captions

    def __str__(self):
        return f'ClothoV2_{self.split}'



if __name__ == '__main__':
    import torch
    from sacred import Experiment

    ex = Experiment(ingredients=[clotho_v2])
    @ex.automain
    def main(_config):
        train = get_clotho_v2('train')
        print(train[7])

    ex.run(config_updates={'directories': {'data_dir': '~/shared'}})


