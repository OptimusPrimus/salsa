from sacred import Ingredient
import os
import torch
import numpy as np
import multiprocessing
import librosa
import h5py
import tqdm
import shutil
import psutil
import minimp3py

from typing import List, NoReturn, Dict, Union

from utils.directories import directories, get_persistent_cache_dir

audio_dataset = Ingredient('audio_dataset', ingredients=[directories])

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

@audio_dataset.config
def config():
    sample_rate = 32000
    processes = 32
    shared = False

    run_cached_features_test = 1


class LoadAudios:

    @audio_dataset.capture
    def __init__(self, sample_rate=32000, processes=1, shared=False):
        self.sample_rate = sample_rate
        self.processes = processes
        self.compress = False
        self.shared = shared
        self.__cached__ = False
        self.__quick__ = False

        assert type(sample_rate) == int
        assert sample_rate > 0

    def __get_audio_paths__(self) -> List[str]:
        """
        Returns the list of audio file paths that will be loaded by 'cache_audios'
        """
        raise NotImplementedError

    def __str__(self) -> str:
        """
        Returns the name of the data set. Required for caching.
        """
        raise NotImplementedError

    def set_quick(self, quick=True):
        """
        Prohibits returning of audios; speeds up iteration over the DS.
        """
        self.__quick__ = quick
        return self

    def cache_audios(self):
        """
        Reads all audio files specified in paths into the main memory.
        """
        if self.__cached__:
            return

        paths = self.__get_audio_paths__()
        ds_name = str(self)

        filename = f'{ds_name}_{self.sample_rate}.hdf' if self.compress else f'{ds_name}_{self.sample_rate}_wav.hdf'
        file_path = os.path.join(get_persistent_cache_dir(), filename)

        unique_paths = sorted(list(set([p for p in paths])))
        print(f"trying to load {len(unique_paths)} files from {file_path}")
        if not os.path.exists(file_path):
            # compress and load files
            with multiprocessing.Pool(processes=self.processes) as pool:
                self.__mp3s__ = list(
                    tqdm.tqdm(
                        pool.imap(encode if self.compress else load_wavs, path_iter(unique_paths, self.sample_rate)),
                        total=len(unique_paths),
                        desc='Compressing and loading files'
                    )
                )

            # save files to hdf file
            with h5py.File(file_path, 'w') as hdf5_file:
                dt = h5py.vlen_dtype(np.dtype('uint8') if self.compress else np.float32)
                mp3s = hdf5_file.create_dataset('mp3', shape=(len(unique_paths),), dtype=dt)
                for i, s in enumerate(self.__mp3s__):
                    mp3s[i] = self.__mp3s__[i]

        # copy hd5py file to ram
        self.__dataset_file__ = file_path
        self.__path_file_map__ = {p: i for i, p in enumerate(unique_paths)}

        self.__hdf5_file__ = None

        self.__cached__ = True

        self.run_test()

        return self

    def __get_audio__(self, item: int) -> Union[Dict]:
        """
        Loads audio file specified via path
        """
        path = self.__get_audio_paths__()[item]
        # quick mode, skip loading
        if self.__quick__:
            return {'path': path, 'audio': [], 'audio_length': 0}

        # if not cached, load file from path
        if not self.__cached__:
            audio = load_wavs((0, path, self.sample_rate))
            return {'path': path, 'audio': audio, 'audio_length': 1.0}

        # load from RAM, decode if necessary
        if self.__hdf5_file__ is None:
            self.__open_hdf5__()
        data = self.__hdf5_file__["mp3"][self.__path_file_map__[path]]

        audio = decode(data, path) if self.compress else data

        return {'path': path, 'audio': audio, 'audio_length': 1.0}

    @audio_dataset.capture
    def run_test(self, run_cached_features_test: int):
        if run_cached_features_test:
            idxes = np.random.randint(len(self.__get_audio_paths__()), size=(run_cached_features_test,))
            for i in idxes:
                self.__check_audio__(i)
        return self

    def __check_audio__(self, index: int):
        q, c = self.__quick__, self.__cached__

        self.__quick__ = False
        self.__cached__ = False
        a1 = self.__get_audio__(index)['audio']
        self.__cached__ = True
        a2 = self.__get_audio__(index)['audio']

        if a1 is None:
            print("Audio files not available. Skipping cached features test.")
        else:
            if a1.shape != a2.shape:
                print("file shape: ", a1.shape)
                print("cached shape: ", a2.shape)

            if any(a1 != a2):
                print("Mean difference: ", np.abs(a1 - a2).mean())

        self.__quick__, self.__cached__ = q, c

    def __open_hdf5__(self):
        self.__hdf5_file__ = h5py.File(self.__dataset_file__, 'r')

    def __del__(self):
        if hasattr(self, '__hdf5_file__') and self.__hdf5_file__ and type(self.__hdf5_file__) is h5py.File:
            self.__hdf5_file__.close()
            self.__hdf5_file__ = None

    def __len__(self) -> int:
        return len(self.__get_audio_paths__())


class FixedLengthAudio(LoadAudios):

    @audio_dataset.capture
    def __init__(self):
        super().__init__()
        self.__fixed_length__ = False
        self.__offset__ = False

    def set_fixed_length(self, length: int):
        if not length:
            self.__fixed_length__ = False
        else:
            self.__fixed_length__ = length * self.sample_rate
        return self

    def set_offset(self, offset: int):
        if not offset:
            self.__offset__ = False
        else:
            self.__offset__ = offset * self.sample_rate
        return self

    def __get_audio__(self, index: int) -> Union[Dict, None]:
        s = super().__get_audio__(index)
        s['audio'], s['audio_length'] = self.__get_fixed_length_audio__(s['audio'])
        return s

    def __get_fixed_length_audio__(self, x):
        # load audio from super class
        if x is None or not self.__fixed_length__ or len(x) == 0:
            return x, 1

        audio_length = min(len(x), self.__fixed_length__) / self.__fixed_length__

        if x.shape[-1] < self.__fixed_length__:
            x = self.__pad__(x, self.__fixed_length__)
        elif x.shape[-1] > self.__fixed_length__:
            if self.__offset__:
                offset = self.__offset__
            else:
                offset = torch.randint(x.shape[-1] - self.__fixed_length__ + 1, size=(1,)).item()
            x = x[offset:offset+self.__fixed_length__]

        return x, audio_length

    def __pad__(self, x, length):
        assert len(x) <= length, 'audio sample is longer than the max length'
        y = np.zeros((self.__fixed_length__,)).astype(np.float32)
        y[:len(x)] = x
        return y


class DatasetBaseClass(FixedLengthAudio, torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()

    def get_subset(self, filter):

        q = self.__quick__

        self.set_quick(True)
        idx = [i for i in range(len(self)) if filter(self[i])]
        self.set_quick(q)

        subset = Subset(self, idx)
        subset.set_quick(q)
        return subset

class ConcatDataset(DatasetBaseClass):

    def __init__(self, datasets: List[DatasetBaseClass]):
        super().__init__()
        self.datasets = datasets
        self.lengths = [len(ds) for ds in datasets]
        self.lower = np.cumsum([0] + self.lengths[:-1])
        self.upper = np.cumsum(self.lengths)
        self.__quick__ = False
        self.__cached__ = False

    def set_quick(self, quick=True):
        self.__quick__ = quick
        for ds in self.datasets:
            ds.set_quick(quick=quick)
        return self

    def set_offset(self, offset: int):
        if not offset:
            self.__offset__ = False
        else:
            self.__offset__ = offset * self.sample_rate
        for ds in self.datasets:
            ds.set_offset(offset)
        return self

    def set_fixed_length(self, length: int):
        if not length:
            self.__fixed_length__ = False
        else:
            self.__fixed_length__ = length * self.sample_rate
        for ds in self.datasets:
            ds.set_fixed_length(length)
        return self

    def cache_audios(self):
        for ds in self.datasets:
            ds.cache_audios()
        self.__cached__ = True
        return self

    def __getitem__(self, item: int):
        i = 0
        assert item < len(self)

        while not (self.lower[i] <= item < self.upper[i]):
            i = i + 1  # next data set

        item = item - self.lower[i]  # remove offset
        return self.datasets[i][item]

    def __get_audio_paths__(self):
        paths = []
        for ds in self.datasets:
            paths = paths + ds.__get_audio_paths__()
        return paths

    def __len__(self) -> int:
        return sum(self.lengths)


class Subset(DatasetBaseClass):

    def __init__(self, dataset: DatasetBaseClass, indices: List[int]):
        super().__init__()
        self.dataset = dataset
        self.indices = indices
        self.__quick__ = False
        self.__cached__ = False

    def set_quick(self, quick=True):
        self.__quick__ = quick
        self.dataset.set_quick(quick=quick)
        return self

    def set_offset(self, offset: int):
        if not offset:
            self.__offset__ = False
        else:
            self.__offset__ = offset * self.sample_rate

        self.dataset.set_offset(offset)
        return self

    def set_fixed_length(self, length: int):
        if not length:
            self.__fixed_length__ = False
        else:
            self.__fixed_length__ = length * self.sample_rate
        self.dataset.set_fixed_length(length)
        return self

    def cache_audios(self):
        self.dataset.cache_audios()
        self.__cached__ = True
        return self

    def __get_audio__(self, index: int) -> Union[Dict, None]:
        item = self.indices[index]
        return self.dataset.__get_audio__(item)

    def __getitem__(self, item: int) -> Dict:
        item = self.indices[item]
        return self.dataset[item]

    def __len__(self) -> int:
        return len(self.indices)

def decode(array, path, max_length=32):
    """
    decodes an array if uint8 representing an mp3 file
    :rtype: np.array
    """
    # taken from https://github.com/kkoutini/PaSST/blob/main/audioset/prepare_scripts
    try:
        data = array.tobytes()
        duration, ch, sr =  minimp3py.probe(data)
        # if ch != 1:
        #     print(f"Unexpected number of channels {ch} {path}")
        assert sr == 32000, f"Unexpected sample rate {sr}   {path}"

        max_length = max_length * sr
        offset=0
        if duration > max_length:
            max_offset = max(int(duration - max_length), 0) + 1
            offset = torch.randint(max_offset, (1,)).item()

        waveform, _ = minimp3py.read(data, start=offset, length=max_length)
        waveform = waveform[:,0]

        if waveform.dtype != 'float32':
            raise RuntimeError("Unexpected wave type")

    except Exception as e:
        print(path)
        raise e
        # print(e)
        # print("Error decompressing: ", path, "Returning empty arrray instead...")
        waveform = np.zeros((10 * 32000)).astype(np.float32)

    return waveform



def encode(params, codec='mp3'):
    i, file, sr = params
    # taken from https://github.com/kkoutini/PaSST/blob/main/audioset/prepare_scripts
    target_file = os.path.join(get_persistent_cache_dir(), f'{i}.{codec}')
    print(file)
    print(target_file)
    if not os.path.exists(file[:-3] + codec):
        os.system(f"ffmpeg  -hide_banner -nostats -loglevel error -n -i '{file}' -codec:a {codec} -ar {sr} -ac 1 '{target_file}'")
        array = np.fromfile(target_file, dtype='uint8')
        os.remove(target_file)
    else:
        array = np.fromfile(file[:-3] + codec, dtype='uint8')

    return array


def load_wavs(params):
    i, file, sr = params
    try:
        audio = librosa.load(path=file, sr=sr, mono=True)[0]
    except:
        print("File not found:", file)
        audio = None

    return audio


def path_iter(dataset, sr):
    for i, s in enumerate(dataset):
        yield i, s, sr


if __name__ == '__main__':
    import torch
    from sacred import Experiment
    from data.datasets.clotho_v2 import clotho_v2, get_clotho_v2

    ex = Experiment(ingredients=[clotho_v2])
    @ex.automain
    def main(_config):
        ds = get_clotho_v2('test')
        ds.set_quick(True)
        ds[0]
        ds.set_quick(False)
        ds[0]
        ds_ = ConcatDataset([ds, ds])
        ds_[0]
        ds__ = Subset(ds, [0, 5, 10])
        ds__[0]
        ds.cache_audios()
    ex.run()