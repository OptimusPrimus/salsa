from sacred import Ingredient
import os
import torch
import numpy as np
import multiprocessing
import av
import io
import librosa
import h5py
import tqdm
import shutil
import psutil
import minimp3py

from data.datasets.wrapper import FixedLengthAudio
from utils.directories import directories, get_persistent_cache_dir, get_ram_cache_dir

audio_loader = Ingredient('audio_loader', ingredients=[directories])

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

@audio_loader.config
def config():
    sample_rate = 32000
    max_audio_length = 10


@audio_loader.capture
def load_cached_audio_set(audio_set, sample_rate, max_audio_length, compress=True, shared=False):

    audio_set = LoadMP3CompressedIntoRam(audio_set, sample_rate=sample_rate, compress=compress, shared=shared)

    assert type(max_audio_length) == int or max_audio_length in [None]

    if type(max_audio_length) == int:
        return FixedLengthAudio(audio_set, fixed_length=max_audio_length, sampling_rate=sample_rate)

    return audio_set


def encode(params, codec='mp3'):
    i, file, sr = params
    # taken from https://github.com/kkoutini/PaSST/blob/main/audioset/prepare_scripts
    target_file = os.path.join(get_persistent_cache_dir(), f'{i}.{codec}')
    if not os.path.exists(file[:-3] + codec):
        os.system(f"ffmpeg  -hide_banner -nostats -loglevel error -n -i '{file}' -codec:a {codec} -ar {sr} -ac 1 '{target_file}'")
        array = np.fromfile(target_file, dtype='uint8')
        os.remove(target_file)
    else:
        array = np.fromfile(file[:-3] + codec, dtype='uint8')

    return array


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



def path_iter(dataset, sr):
    for i, s in enumerate(dataset):
        yield i, s, sr


class LoadMP3CompressedIntoRam(torch.utils.data.Dataset):

    def __init__(self, dataset, sample_rate=32000, processes=32, compress=True, shared=False):
        self.dataset = dataset
        self.sample_rate = sample_rate
        self.compress = compress
        assert type(sample_rate) == int
        assert sample_rate > 0

        filename = f'{str(self.dataset)}_{sample_rate}.hdf' if compress else f'{str(self.dataset)}_{sample_rate}_wav.hdf'
        file_path = os.path.join(get_persistent_cache_dir(), filename)

        self.unique_paths = sorted(list(set([d['path'] for d in self.dataset])))
        print(f"trying to load {len(self.unique_paths)} files from {file_path}")
        if not os.path.exists(file_path):
            # compress and load files
            with multiprocessing.Pool(processes=processes) as pool:
                self.mp3s = list(
                    tqdm.tqdm(
                        pool.imap(encode if compress else load_wavs, path_iter(self.unique_paths, sample_rate)),
                        total=len(self.unique_paths),
                        desc='Compressing and loading files'
                    )
                )

            # save files to hdf file
            with h5py.File(file_path, 'w') as hdf5_file:
                dt = h5py.vlen_dtype(np.dtype('uint8') if compress else np.float32)
                mp3s = hdf5_file.create_dataset('mp3', shape=(len(self.unique_paths),), dtype=dt)
                for i, s in enumerate(self.mp3s):
                    mp3s[i] = self.mp3s[i]

        # copy hd5py file to ram
        if shared:
            ram_file = os.path.join(get_ram_cache_dir(), filename)
            print(f"copying {file_path} to {ram_file}\n\n*** Don't forget to delete this file when you're done! ***\n\n")
            if not os.path.exists(ram_file):
                available = getattr(psutil.virtual_memory(), 'available')
                file_size = os.path.getsize(file_path)
                assert file_size < available, "File is too large to fit into RAM."
                shutil.copyfile(file_path, ram_file)

            # self.hdf5_file = h5py.File(ram_file, 'r')
            # self.mp3s = self.hdf5_file['mp3']
            self.dataset_file = ram_file
            self.path_file_map = {p: i for i, p in enumerate(self.unique_paths)}

        else:
            self.dataset_file = file_path # self.hdf5_file = h5py.File(, 'r') # ['mp3'][:]
            self.path_file_map = {p: i for i, p in enumerate(self.unique_paths)}

        self.hdf5_file = None

    def open_hdf5(self):
        self.hdf5_file = h5py.File(self.dataset_file, 'r')

    def __del__(self):
        if self.hdf5_file and type(self.hdf5_file) is h5py.File:
            self.hdf5_file.close()
            self.hdf5_file = None

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):

        if self.hdf5_file is None:
            self.open_hdf5()

        # get sample and replace default path
        s = self.dataset[index].copy()
        data = self.hdf5_file["mp3"][self.path_file_map[s['path']]]
        x = decode(data, s['path']) if self.compress else data
        s['audio'] = x
        return s


def load_wavs(params):
    i, file, sr = params
    try:
        audio = librosa.load(path=file, sr=sr, mono=True)[0]
    except:
        print(file)
        audio = None

    return audio


class LoadIntoRam(torch.utils.data.Dataset):

    def __init__(self, dataset, sample_rate=32000, max_len=10, processes=4):
        self.dataset = dataset
        self.sample_rate = sample_rate
        self.max_len = max_len
        assert type(sample_rate) == int
        assert sample_rate > 0
        def path_iter(dataset, sr):
            for i, s in enumerate(dataset):
                yield i, s['path'], sr

        with multiprocessing.Pool(processes=processes) as pool:
            self.wavs = list(
                tqdm.tqdm(
                    pool.imap(load_wavs, path_iter(self.dataset, sample_rate)),
                    total=len(self.dataset),
                    desc='Compressing and loading files'
                )
            )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        s = self.dataset[index]
        x = self.wavs[index]
        if self.max_len != None:
            s['audio'] = x[:self.max_len*self.sample_rate]
        else:
            s['audio'] = x
        return s