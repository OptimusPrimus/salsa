import torch
import numpy as np
import json
import nltk
#nltk.download('stopwords')
#nltk.download('wordnet')
from textaugment import EDA as EDA_
import string
from wordcloud import WordCloud
import collections

def remove_stop_words(counts):
    stopwords = WordCloud(collocations = False, background_color = 'white').stopwords

    for w in stopwords:
        if w in counts:
            del counts[w]

    return counts


class BalancedDataSet(torch.utils.data.Dataset):

    def __init__(self, data_set, source_captions, target_captions, size):
        import numpy as np
        self.size = size
        self.data_set = data_set

        target_counts = remove_stop_words(collections.Counter(" ".join(target_captions).split(" ")))
        target_counts_total = sum(target_counts.values())

        # build mapping word -> caption
        index = {}
        for i, s in enumerate(source_captions):
            for w in s.split(" "):
                if w in target_counts:
                    word_list = index.get(w, [])
                    word_list.append(i)
                    index[w] = word_list

        print("finished building index")
        print("Built index of size:", len(index))
        self.p_word = np.array([v for k, v in target_counts.items()]) / target_counts_total
        self.words = [k for k, v in target_counts.items()]
        self.index = index

    def __getitem__(self, item):

        word = None
        while word not in self.index:
            word = np.random.choice(self.words, size=1, replace=True, p=self.p_word)[0]
        id = int(np.random.choice(self.index[word], size=1)[0].item())

        return self.data_set[id]

    def __len__(self):
        return self.size



class Tokenize(torch.utils.data.Dataset):

    def __init__(self, dataset, encode_fun, cache=False):
        self.dataset = dataset
        self.encode_fun = encode_fun
        if cache:
            self.encoded = [self.encode_fun(d['caption']) for d in self.dataset]
        else:
            self.encoded = None

    def __getitem__(self, item):
        sample = self.dataset[item].copy()
        if self.encoded:
            sample['input_ids'], sample['attention_mask'] = self.encoded[item]
        else:
            sample['input_ids'], sample['attention_mask'] = self.encode_fun(sample['caption'])
        return sample

    def __len__(self):
        return len(self.dataset)


class TranslateAugment(torch.utils.data.Dataset):

    def __init__(self, dataset, cache_dir, p=0.5, lng=['de'], method=['google'], stochastic=True):
        self.dataset = dataset
        self.backtranslated = [list() for s in dataset]
        self.p = p
        self.stochastic = stochastic
        self.n_languages = len(lng)

        for l in lng:
            for m in method:
                if m == 'google':
                    from deep_translator import GoogleTranslator
                    translate = GoogleTranslator
                else:
                    raise ValueError('translation method unknown')
                fn = os.path.join(cache_dir, f'{str(self.dataset.data_set.dataset)}_{l}_{m}.npy')
                if os.path.exists(fn):
                    # load translated caption
                    augmented = list(np.load(fn))
                else:
                    # backtranslate captions
                    def back_translate(caption):
                        translated = translate(source='en', target=l).translate(caption)
                        backtranslated = translate(source=l, target='en').translate(translated)
                        return backtranslated

                    augmented = [back_translate(s['caption']) for s in tqdm(dataset)]
                    np.save(fn, augmented)

                # append sentences
                for i, li in enumerate(self.backtranslated):
                    li.append(augmented[i])

    def __getitem__(self, item):
        if self.stochastic:
            if self.p == 0 or torch.rand((1,)).item() > self.p:
                return self.dataset[item]
            else:
                sample = self.dataset[item].copy()
                i = np.random.choice(len(self.backtranslated[item]))
                sample['caption'] = str(self.backtranslated[item][i])
                return sample
        else:
            i = item//(self.n_languages + 1)
            l = item % (self.n_languages + 1)
            sample = self.dataset[i].copy()
            sample['idx'] = item

            if l == 0:
                return sample
            sample['caption'] = str(self.backtranslated[i][l-1])

            return sample

    def __len__(self):
        return len(self.dataset) if self.stochastic else ((len(self.dataset) * self.n_languages) + len(self.dataset))

if __name__ == '__main__':
    ds = [{'caption': 'Good Morning!'}]
    ta = TranslateAugment(ds)

    print(ta)


class EDA(torch.utils.data.Dataset):

    def __init__(self, dataset, p=0, p_syn=0, p_swap=0, p_ins=0, p_del=0):
        self.p = p
        self.p_swap = p_swap
        self.p_ins = p_ins
        self.p_del = p_del
        self.p_syn = p_syn

        self.dataset = dataset

        self.augmenter = EDA_()

    def __getitem__(self, item):
        if self.p == 0 or torch.rand((1,)).item() > self.p:
            sample = self.dataset[item]
        else:
            sample = self.dataset[item].copy()
            caption = sample['caption'].lower()
            caption = caption.translate(str.maketrans('', '', string.punctuation))
            length = len(caption.split())

            aug = torch.randint(4, size=(1,)).item()
            if aug == 0:
                n = torch.distributions.Binomial(length, torch.tensor([self.p_syn])).sample().item()
                if n > 0:
                    caption = self.augmenter.synonym_replacement(caption, n=int(n))
            elif aug == 1:
                caption = self.augmenter.random_deletion(caption, p=self.p_del)
            elif aug == 2:
                n = torch.distributions.Binomial(length, torch.tensor([self.p_swap])).sample().item()
                caption = self.augmenter.random_swap(caption, n=int(n))
            elif aug == 3:
                n = torch.distributions.Binomial(length, torch.tensor([self.p_ins])).sample().item()
                caption = self.augmenter.random_insertion(caption, n=int(n))

            sample['caption'] = caption

        return sample

    def __len__(self):
        return len(self.dataset)


class FixedLengthAudio(torch.utils.data.Dataset):

    def __init__(self, data_set, fixed_length=10, sampling_rate=32000):
        self.data_set = data_set
        self.fixed_length = fixed_length*sampling_rate
        self.sampling_rate = sampling_rate

    def __getitem__(self, item):
        sample = self.data_set[item].copy()
        x = sample.get('audio')
        sample['audio_length'] = min(len(x), self.fixed_length) / self.fixed_length

        if x is None:
            return sample
        if x.shape[-1] < self.fixed_length:
            x = self.__pad__(x, self.fixed_length)
        elif x.shape[-1] > self.fixed_length:
            offset = torch.randint(x.shape[-1] - self.fixed_length + 1, size=(1,)).item()
            x = x[offset:offset+self.fixed_length]
        sample['audio'] = x
        return sample

    def __pad__(self, x, length):
        assert len(x) <= length, 'audio sample is longer than the max length'
        y = np.zeros((self.fixed_length,)).astype(np.float32)
        y[:len(x)] = x
        return y

    def __len__(self):
        return len(self.data_set)

import pickle
import os
from tqdm import tqdm


class AddVGGishFeatures(torch.utils.data.Dataset):

    def __init__(self, data_set, data_set_id):

        self.data_set = data_set

        path = os.path.join(os.path.expanduser('~'), 'shared', 'paul', 'embeddings', 'vggish_' + data_set_id + '.pkl')

        if os.path.exists(path):
            with open(path, 'rb') as f:
                self.vggish_features = pickle.load(f)
        else:
            vggish_model = torch.hub.load('harritaylor/torchvggish', 'vggish')
            vggish_model.eval()
            vggish_model.postprocess = False
            vggish_model.embeddings[5] = torch.nn.Sequential()  # Remove last ReLU

            self.vggish_features = []
            for i, s in enumerate(tqdm(data_set)):
                vggish_embeddings = vggish_model.forward(s['path']).detach().cpu().numpy()
                self.vggish_features.append(vggish_embeddings)

            with open(path, 'wb') as f:
                pickle.dump(self.vggish_features, f)

        self.fixed_length = max([s.shape[0] for s in self.vggish_features])

    def __getitem__(self, item):

        s = self.data_set[item].copy()
        s['audio_features'] = np.zeros((self.fixed_length, 128)).astype(np.float32)
        features_length = len(self.vggish_features[item])
        s['audio_features'][:features_length] = self.vggish_features[item]

        s['audio_mask'] = np.zeros((self.fixed_length,)).astype(np.int32)
        s['audio_mask'][:features_length] = 1

        return s

    def __len__(self):
        return len(self.data_set)


class CacheAudioFeatures(torch.utils.data.Dataset):

    def __init__(self, data_set, embed_function, dataset_id, model_id):

        self.data_set = data_set

        path = os.path.join(
            os.path.expanduser('~'),
            'shared',
            'paul',
            'embeddings',
            dataset_id + '_' + model_id + '.pkl'
        )

        if os.path.exists(path):
            with open(path, 'rb') as f:
                self.features, self.masks = pickle.load(f)
        else:
            self.features, self.masks = [], []
            from torch.utils.data import DataLoader
            dl = DataLoader(data_set, batch_size=32, num_workers=8)
            for b in tqdm(dl):
                embeddings, masks = embed_function(b['audio'], b['audio_length'])
                self.features.append(embeddings)
                self.masks.append(masks)

            self.features = np.concatenate(self.features)
            self.masks = np.concatenate(self.masks)

            with open(path, 'wb') as f:
                pickle.dump((self.features, self.masks), f)

    def __getitem__(self, item):

        s = self.data_set[item].copy()
        del s['audio']
        s['audio_features'] = self.features[item]
        s['audio_mask'] = self.masks[item]
        return s

    def __len__(self):
        return len(self.data_set)
