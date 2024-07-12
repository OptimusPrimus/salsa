import csv
import os
from abc import ABC
import math
import sys
import pytorch_lightning as pl
import wandb
import copy
import itertools

from data.datasets.dataset_base_classes import ConcatDataset
from data.datasets.clotho_v2 import clotho_v2, get_clotho_v2
from data.datasets.audio_caps import audiocaps, get_audiocaps
from data.datasets.wavcaps import wavcaps, get_wavecaps
from data.datasets.audioset import audioset, get_audioset
from data.data_loader import data_loader, get_train_data_loader, get_eval_data_loader
from sacred import Experiment

from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import torch
from glob import glob
from pytorch_lightning.loggers import WandbLogger
from utils.directories import directories, get_model_dir, get_dataset_dir
import numpy as np
import torch.distributed as dist

import string

from nltk.corpus import stopwords

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# nltk.download('stopwords')
stopwords = set(stopwords.words('english'))

wandb.login()
audio_retrieval = Experiment('audio_retrieval', ingredients=[
    directories,
    clotho_v2,
    audiocaps,
    wavcaps,
    data_loader,
    audioset
])

@audio_retrieval.config
def default_config():

    log_db_name = 'dcase24_workshop'

    # model loading and saving
    load_parameters = None
    load_last = 'best'
    resume_training = None

    audio_features = {
        'model': 'passt',
        'model_config': {
            'freq_scale': 1.0,
            's_patchout_t': 15,
            's_patchout_f': 2,
            'freqm': 0,
            'timem': 0,
            'return_sequence': False
        },
        'use_local_model': False,
        'frozen': False,
        'adopt_n_layers': 0,
        'adopt_layer_size': 2048,
        'segment_length': 10,
        'hop_length': 10,
        'aggregate': 'mean',
        'sequence_model': {
            'num_layers': 0,
            'dim': 768, # 3840
            'nhead': 12,
            'dim_feedforward': 2048,
            'use_first_token': True,
            'norm_first': True,
            'dropout': 0.1,
            'posencode_init': 'sinosoidal_fixed',
            'posencode_project': False,
            'posencode_length': 64*3,
            'posencode_weight': 1.0,
            'ablate_posencode': False,
            'normalize_input': False,
            'audio_token_dropout': 0.0,
            'reduce_sequence': False
        }
    }

    sentence_features = {
        'model': 'bert-base-uncased',
        'frozen': False,
        'adopt_n_layers': 0,
        'adopt_layer_size': 2048,
        'use_first_token': True,
        'aggregate': 'mean',
        'max_sentence_tokens': 32,
        'remove_punctuation': True
    }

    # loss function
    initial_tau = 0.05
    freeze_tau = True
    shared_representation_size = 1024
    normalize = True

    # data set & augmentations
    train_on = 'clothov2'

    # optimizer
    max_epochs = 25
    max_samples_per_epoch = None
    adamw = True
    beta1 = 0.9
    beta2 = 0.999
    weight_decay = 0.0
    amsgrad = False
    lr_audio_encoder = 2e-5
    lr_audio_project = 2e-5
    lr_sentence_encoder = 2e-5
    lr_sentence_project = 2e-5
    min_lr = 1e-7
    hard_steps = False
    warmup_length = 1
    rampdown_start = 1
    rampdown_stop = 20
    rampdown_type = 'cosine'

    eps = 1e-8
    accumulate_grad_batches = 1
    gradient_clip_val = None

    # technical stuff
    gpus = 1
    accelerator = 'cuda' if torch.cuda.is_available() else 'cpu'
    half_precision = True
    fast_dev_run = False
    strategy = 'auto'
    monitor = 'mAP@10'
    enable_checkpointing = True

    num_nodes=1

    run_ba_benchmark = False

    loss_weight = 1.0

    projection_dropout = 0.0

    timing_loss_weight = 0.0
    timing_loss_margin = None
    initial_timing_tau = None
    ranking_loss_margin = False
    use_same_tau = False

    # audio_reconstruction_weight = 0
    # audio_reconstruction_layers = 0

    use_keywords = False
    add_filename_to_keywords = False
    use_captions = False
    ablate_audio_embedding = False
    normalize_before_combine = False
    init_audio_token_with_mean = True
    use_raw_sentence_features = False
    use_separate_metadata_embedding_model = False

    run_cmd=None
    average_models=[]

    loss_weight=1.0
    distill_weight=0.0
    distill_from=[]

@audio_retrieval.capture
def run_one(log_db_name, train_on, resume_training, _config=None):

    run = wandb.init(reinit=True, project='dev' if _config['fast_dev_run'] else log_db_name)

    if not torch.cuda.is_available():
        return 0

    # make un-mutable dict mutable
    _config = dict(_config)

    print('Initialize model...')
    model = get_model()

    # create the data module
    print(f'Loading {train_on} data sets...')
    train = get_data_set(train_on, 'train')
    val = get_data_set(train_on, 'val')
    test = get_data_set(train_on, 'test')

    print(f'Training set size: {len(train)}')
    print(f'Val set size: {len(val)}')
    print(f'Test set size: {len(test)}')

    # training
    print('Start training...')

    distributed = _config['num_nodes'] > 1
    if distributed and int(os.environ['NODE_RANK']) > 0:
        print(f"Slave node {int(os.environ['NODE_RANK'])}...")
        wandb_logger = None
    else:
        print(f"Master node...")
        print('Logging to ', log_db_name)
        wandb_logger = WandbLogger(log_model=False, project='dev' if _config['fast_dev_run'] else log_db_name)

    print('Logger created')
    t = get_trainer(wandb_logger)

    t.fit(
        model,
        train_dataloaders=get_train_data_loader(train, targets=None),
        val_dataloaders=[get_eval_data_loader(val, shuffle=True, distributed=distributed)],
        ckpt_path=str(os.path.join(get_model_dir(), resume_training, 'last.ckpt')) if resume_training else None
    )
    t.test(model, get_eval_data_loader(test, shuffle=True, distributed=distributed))
    return 0


@audio_retrieval.capture
def get_data_set(data_set_id, mode, _config):
    if data_set_id == 'clothov2' or (data_set_id in ['wavcaps', 'all'] and mode != 'train'):
        assert mode in ['train', 'val', 'test', 'analysis']
        ds = get_clotho_v2(mode)
        ds.set_fixed_length(30)
    elif data_set_id == 'audiocaps':
        assert mode in ['train', 'val', 'test']
        ds = get_audiocaps(mode)
        ds.set_fixed_length(10)
    elif data_set_id == 'wavcaps':
        ds = get_wavecaps()
        ds.compress = True
        ds.set_fixed_length(30)
    elif data_set_id == 'all':
        ds = ConcatDataset(
            [
                get_clotho_v2('train'),
                get_audiocaps('train'),
                get_wavecaps()
            ]
        )
        ds.set_fixed_length(30)
    else:
        raise NotImplementedError(f'Data set {data_set_id} unknown.')

    ds.cache_audios()
    return ds


@audio_retrieval.capture
def get_model(load_parameters, _config):
    ac = AudioRetrievalModel(**_config)

    # init parameters from pre-trained model
    if load_parameters:
        print(f'Loading model {load_parameters} ...')
        save_dir = os.path.join(get_model_dir(), load_parameters)
        assert os.path.exists(save_dir)
        if _config['load_last'] == 'last':
            print('Loading last checkpoint.')
            model_path = list(glob(os.path.join(save_dir, 'last.ckpt')))[-1]
        elif _config['load_last'] == 'best':
            print('Loading best checkpoint.')
            paths = glob(os.path.join(save_dir, 'epoch_*.ckpt'))
            paths.sort(key=lambda x: float(os.path.basename(x).split('-')[-1].split('.')[1]))
            model_path = paths[-1]
        else:
            raise AttributeError(_config['load_last'])
        print(model_path)
        ac_ = AudioRetrievalModel.load_from_checkpoint(model_path)
        print('Loading state dict')
        missing_keys = ac.load_state_dict(ac_.state_dict())
        print(missing_keys)

    return ac


class AudioRetrievalModel(pl.LightningModule, ABC):

    def __init__(
            self,
            **kwargs
    ):

        super().__init__()

        self.save_hyperparameters(kwargs)

        self.kwargs = kwargs
        self.distributed_mode = kwargs.get('num_nodes', 1) > 1

        from models.audio.base import get_audio_embedding_model
        self.audio_embedding_model, audio_output_size = get_audio_embedding_model(
            self.kwargs['audio_features']['model'],
            segment_length=self.kwargs['audio_features']['segment_length'],
            hop_size=self.kwargs['audio_features']['hop_length'],
            model_config=self.kwargs['audio_features']['model_config'],
            multi_window=self.kwargs['audio_features']['use_local_model']
        )

        from models.text.sentence_embedding_models import get_sentence_embedding_model
        self.sentence_embedding_model, self.tokenizer, text_output_size = get_sentence_embedding_model(self.kwargs['sentence_features']['model'])

        layer_sizes = [self.kwargs['audio_features']['sequence_model']['dim'] if self.kwargs['audio_features']['sequence_model']['num_layers'] > 0 else audio_output_size]
        layer_sizes += [self.kwargs['audio_features']['adopt_layer_size']] * self.kwargs['audio_features']['adopt_n_layers']
        layer_sizes += [self.kwargs['shared_representation_size']]
        audio_layers = []
        for i, o in zip(layer_sizes[:-1], layer_sizes[1:]):
            audio_layers.append(torch.nn.Linear(i, o))
            audio_layers.append(torch.nn.ReLU())

        audio_layers.pop()
        self.project_audio = torch.nn.Sequential(*audio_layers)

        layer_sizes = [text_output_size]
        layer_sizes += [self.kwargs['sentence_features']['adopt_layer_size']] * self.kwargs['sentence_features']['adopt_n_layers']
        layer_sizes += [self.kwargs['shared_representation_size']]
        sentence_layers = []
        for i, o in zip(layer_sizes[:-1], layer_sizes[1:]):
            sentence_layers.append(torch.nn.Linear(i, o))
            sentence_layers.append(torch.nn.ReLU())

        sentence_layers.pop()
        self.project_sentence = torch.nn.Sequential(*sentence_layers)


        initial_tau = torch.zeros((1,)) + self.kwargs['initial_tau']
        self.tau = torch.nn.Parameter(initial_tau, requires_grad=not self.kwargs['freeze_tau'])

        # assert audio_output_size == text_output_size
        if self.kwargs['audio_features']['sequence_model'].get('reduce_sequence', 0) > 0:
            n = self.kwargs['audio_features']['sequence_model']['reduce_sequence']
            emb_dim = self.kwargs['audio_features']['sequence_model']['dim']
            self.audio_output_projection = torch.nn.Linear(audio_output_size*n, emb_dim)
        else:
            self.audio_output_projection = torch.nn.Identity() if audio_output_size == self.kwargs['audio_features']['sequence_model']['dim'] else torch.nn.Linear(audio_output_size, self.kwargs['audio_features']['sequence_model']['dim'])
        self.posencode = PositionalEncoding(
            self.kwargs['audio_features']['sequence_model']['dim'],
            max_len=self.kwargs['audio_features']['sequence_model']['posencode_length'],
            posencode_init=self.kwargs['audio_features']['sequence_model']['posencode_init'],
            project=self.kwargs['audio_features']['sequence_model']['posencode_project'],
            weight=self.kwargs['audio_features']['sequence_model']['posencode_weight']
        )

        num_layers = self.kwargs['audio_features']['sequence_model']['num_layers']
        ce_layers = []
        from torch.nn import TransformerEncoderLayer
        for i in range(num_layers):
            ce_layers.append(
                TransformerEncoderLayer(
                    self.kwargs['audio_features']['sequence_model']['dim'],
                    self.kwargs['audio_features']['sequence_model']['nhead'],
                    dim_feedforward=self.kwargs['audio_features']['sequence_model']['dim_feedforward'],
                    dropout=self.kwargs['audio_features']['sequence_model']['dropout'],
                    batch_first=True,
                    norm_first=self.kwargs['audio_features']['sequence_model']['norm_first'],
                    activation=torch.nn.functional.gelu
                )
            )

        self.audio_sequence = torch.nn.ModuleList(ce_layers)

        self.audio_token = torch.nn.Parameter(torch.randn((1, 1, self.kwargs['audio_features']['sequence_model']['dim'])) * 0.04, requires_grad=True)
        self.first = True

        initial_tau = torch.zeros((1,)) + (self.kwargs['initial_timing_tau'] if self.kwargs.get('initial_timing_tau') else self.kwargs['initial_tau'])
        self.timing_tau = torch.nn.Parameter(initial_tau, requires_grad=not self.kwargs['freeze_tau'])

        self.validation_outputs = []

        self.audio_embeddings = {}
        self.sentence_embeddings = {}

        if len(self.kwargs.get('distill_from', "")):
            for pt_model in self.kwargs['distill_from'].split(";"):
                out_path = os.path.join(get_model_dir(), pt_model)
                if not os.path.exists(out_path):
                    continue
                if not os.path.exists(os.path.join(out_path, self.kwargs['train_on'] + '_sentence_embeddings_train.pt')):
                    print("embeddings do not exist")
                    # cmd_generate_embeddings(model=None, load_parameters=pt_model)
                se = torch.load(os.path.join(out_path, self.kwargs['train_on'] + '_sentence_embeddings_train.pt'))
                ae = torch.load(os.path.join(out_path, self.kwargs['train_on'] + '_audio_embeddings_train.pt'))
                for k in ae:
                    e = self.audio_embeddings.get(k, [])
                    e.append(ae[k])
                    self.audio_embeddings[k] = e
                for k in se:
                    e = self.sentence_embeddings.get(k, [])
                    e.append(se[k])
                    self.sentence_embeddings[k] = e

        self.experiment_name = 'none'
        self.store_predictions = True

    def forward_audio(self, batch, y=None, y_mask=None):

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        batch['audio'] = batch['audio'].to(device)

        # embed audios
        with torch.set_grad_enabled(not self.kwargs['audio_features']['frozen']):
            if self.kwargs['audio_features']['frozen']:
                self.audio_embedding_model.eval()
            batch['audio_features'] = self.audio_embedding_model(batch['audio'], audio_length=batch['audio_length'])
            batch['audio_features'] = batch['audio_features'].mean(1) # average over frequency dimension

        audio_mask = []
        for i, f in enumerate(batch['audio_features']):
            m = torch.zeros(f.shape[0])
            m[:math.ceil((batch['audio_length'][i].item() * f.shape[0]))] = 1
            audio_mask.append(m)

        batch['audio_features_mask'] = torch.stack(audio_mask).to(device)

        audio_features = batch['audio_features']
        audio_features_mask = batch['audio_features_mask']

        # combine embeddings of the individual snippets
        if self.kwargs['audio_features']['sequence_model']['num_layers'] > 0:
            if self.kwargs['audio_features']['sequence_model'].get('reduce_sequence', 0) > 0:
                B, L, D = audio_features.shape
                N = self.kwargs['audio_features']['sequence_model']['reduce_sequence']
                audio_features = audio_features.reshape(B, L//N, N, D).flatten(2)
                audio_features_mask = audio_features_mask.reshape(B, L//N, N).max(2).values

            audio_features = self.audio_output_projection(audio_features)
            if self.kwargs['audio_features']['sequence_model']['normalize_input']:
                audio_features = torch.nn.functional.normalize(audio_features, p=2, dim=-1)

            audio_token = self.audio_token.expand(len(audio_features), -1, -1)
            audio_token_mask = torch.ones(len(audio_features_mask), 1).to(audio_features_mask.device)

            if self.kwargs['init_audio_token_with_mean']:
                audio_token = audio_token + audio_features.mean(1)[:, None, :]

            # add positional encoding
            if not self.kwargs['audio_features']['sequence_model']['ablate_posencode']:
                audio_features = self.posencode(audio_features)
                if self.posencode.posencode_init == 'sinosoidal_fixed':
                    self.log('train/posencode_weight', self.posencode.w.data.item())

            # i = torch.randperm(self.posencode.pe.shape[1])[:audio_features.shape[1]].sort().values
            # batch['audio_posencode_raw'] = self.posencode.project(self.posencode.pe[:, i, :]).detach()
            # batch['audio_features_raw'] = audio_features

            if self.kwargs['audio_features']['sequence_model']['audio_token_dropout'] > 0 and self.training:
                i = torch.randperm(audio_features.shape[1])[:int(audio_features.shape[1] * (1-self.kwargs['audio_features']['sequence_model']['audio_token_dropout']))].sort().values
                audio_features = audio_features[:, i, :]
                audio_features_mask = audio_features_mask[:, i]



            if y is not None:
                y = torch.concatenate([y, audio_token], dim=1)
                y_mask = torch.concatenate([y_mask, audio_token_mask], dim=1)
            else:
                y = audio_token
                y_mask = audio_token_mask

            # add special token to the embeddings
            # if self.kwargs['audio_features']['sequence_model']['use_first_token']:
            audio_features = torch.concatenate([y, audio_features], dim=1)
            audio_features_mask = torch.concatenate([y_mask, audio_features_mask], dim=1)

            # forward audio embeddings
            for block_a in self.audio_sequence:
                audio_features = block_a(audio_features, src_key_padding_mask=audio_features_mask == 0)

            # use first token or average of the rest
            if self.kwargs['audio_features']['sequence_model']['use_first_token']:
                audio_features = audio_features[:, :1, :]
                audio_features_mask = audio_features_mask[:, :1]
            else:
                audio_features = audio_features[:, :, :]
                audio_features_mask = audio_features_mask[:, :]

        if self.kwargs['audio_features']['aggregate'] == 'mean':
            # average, get the embeddings
            num_unmasked = audio_features_mask[:, :, None].sum(1)
            audio_features = (audio_features * audio_features_mask[:, :, None]).sum(1) / num_unmasked

            batch['audio_features'] = audio_features[:, None, :]
            batch['audio_features_mask'] = audio_features_mask[:, :1]
        else:
            batch['audio_features'] = audio_features
            batch['audio_features_mask'] = audio_features_mask

        batch['audio_features'] = torch.nn.functional.dropout(batch['audio_features'], p=self.kwargs['projection_dropout'])
        batch['audio_features'] = self.project_audio(batch['audio_features'])

        return batch

    def forward_sentence(self, batch):

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        captions = []
        for i, b in enumerate(batch['caption']):
            if not (type(b) == str):
                print(b)
                b = b[0]
            if self.kwargs['sentence_features']['remove_punctuation']:
                captions.append(b.lower().translate(str.maketrans('', '', string.punctuation)))

        tokenized = self.tokenizer(
            captions,
            add_special_tokens=True,
            padding='max_length',
            return_tensors='pt',
            max_length=self.kwargs['sentence_features']['max_sentence_tokens'],
            truncation=True
        )

        batch['input_ids'] = tokenized['input_ids'].to(device)
        batch['attention_mask'] = tokenized['attention_mask'].to(device)

        with torch.set_grad_enabled(not self.kwargs['sentence_features']['frozen']):
            if self.kwargs['sentence_features']['frozen']:
                self.sentence_embedding_model.eval()
            token_embeddings = self.sentence_embedding_model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])[0]

        batch['sentence_features'] = token_embeddings
        batch['sentence_features_mask'] = batch['attention_mask']

        ### forward audio features
        sentence_features = batch['sentence_features']
        sentence_features_mask = batch['sentence_features_mask']
        # first token or all tokens
        if self.kwargs['sentence_features']['use_first_token']:
            sentence_features = sentence_features[:, :1, :]
            sentence_features_mask = sentence_features_mask[:, :1]
        else:
            sentence_features = sentence_features
            sentence_features_mask = sentence_features_mask

        if self.kwargs['audio_features']['aggregate'] == 'mean':
            num_unmasked = sentence_features_mask[:, :, None].sum(1)
            sentence_features = (sentence_features * sentence_features_mask[:, :, None]).sum(1) / num_unmasked
            batch['sentence_features'] = sentence_features[:, None, :]
            batch['sentence_features_mask'] = sentence_features_mask[:, :1]
        else:
            batch['sentence_features'] = sentence_features
            batch['sentence_features_mask'] = sentence_features_mask

        batch['raw_sentence_features'] = torch.nn.functional.dropout(batch['sentence_features'], p=self.kwargs['projection_dropout'])
        batch['sentence_features'] = torch.nn.functional.dropout(batch['sentence_features'], p=self.kwargs['projection_dropout'])

        batch['sentence_features'] = self.project_sentence(batch['sentence_features'])

        return batch

    def forward(self, batch):

        ### forward audio
        batch = self.forward_audio(batch)

        ### forward audio features
        batch = self.forward_sentence(batch)

        return batch['audio_features'], batch['sentence_features'], batch['audio_features_mask'], batch['sentence_features_mask']

    def rank_sequences(self, audio_features, audio_mask, sentence_features, sentence_mask, batch_idx=0, batch=None):

        if self.kwargs['normalize']:
            sentence_features = torch.nn.functional.normalize(sentence_features, p=2, dim=-1)
            audio_features = torch.nn.functional.normalize(audio_features, p=2, dim=-1)

        C = (audio_features[:, None, 0] * sentence_features[None, :, 0]).sum(-1)
        return C

    def training_step(self, batch, batch_idx):
        self.update_scalars(batch_idx)

        audio_features, sentence_features, audio_mask, sentence_mask = self(batch)
        paths = np.array([hash(p) for p in batch['path']])

        if self.distributed_mode:
            paths_all = self.all_gather(paths).reshape(-1)
        else:
            paths_all = torch.tensor(paths)

        I = (paths_all.unsqueeze(0) == paths_all.unsqueeze(1))

        if self.distributed_mode:
            audio_features = self.all_gather(audio_features, sync_grads=True).reshape(-1, audio_features.shape[1], audio_features.shape[-1])
            sentence_features = self.all_gather(sentence_features, sync_grads=True).reshape(-1, sentence_features[1], sentence_features.shape[-1])

        assert len(audio_features) == len(sentence_features), f"Captions: {len(batch['caption'])}, Audios: {len(batch['audio'])}, Audio Features Shape: {audio_features.shape} Sentence Features Shape: {sentence_features.shape}"

        if self.first:
            print("Audio Features Shape:", audio_features.shape)
            print("Sentence Features Shape:", sentence_features.shape)
            self.first = False

        C = self.rank_sequences(audio_features, audio_mask, sentence_features, sentence_mask, batch=batch)
        C = C / torch.abs(self.tau)

        C_audio = torch.log_softmax(C, dim=0)
        C_text = torch.log_softmax(C, dim=1)

        assert C_audio.shape[0] == C_audio.shape[1], f'Audio Features Shape: {C_audio.shape} Sentence Features Shape: {C_text.shape}'
        assert C_text.shape[0] == C_text.shape[1]

        loss = -0.5 * (C_audio[torch.where(I)].mean() + C_text[torch.where(I)].mean())

        self.log("train/loss", loss, batch_size=len(audio_features), sync_dist=True)
        self.log('train/tau', torch.abs(self.tau), sync_dist=True)

        loss = self.kwargs['loss_weight'] * loss

        if self.kwargs.get('distill_weight', 0) > 0:
            l = len(get_dataset_dir().split(os.path.sep))
            ae = torch.stack([torch.stack(self.audio_embeddings[os.path.sep.join(p.split(os.path.sep)[l:])]) for p in batch['path']]) # A, EM, D
            se = torch.stack([torch.stack(self.sentence_embeddings[c]) for c in batch['caption']]) # S, EM, D

            # C_distilled = self.rank_sequences(ae, None, se, None).to(C.device)
            se = torch.nn.functional.normalize(se, p=2, dim=-1)
            ae = torch.nn.functional.normalize(ae, p=2, dim=-1)
            C_distilled = (ae[:, None] * se[None, :]).sum(-1).mean(-1).to(C.device)
            C_distilled = C_distilled / torch.abs(self.tau)

            distill_loss = 0.5 * (
                    torch.nn.functional.cross_entropy(C, torch.softmax(C_distilled, dim=1)) +
                    torch.nn.functional.cross_entropy(C.T, torch.softmax(C_distilled.T, dim=1))
            )

            self.log('train/distill_C', distill_loss, sync_dist=True)

            loss = self.kwargs['loss_weight'] * loss + self.kwargs['distill_weight'] * distill_loss

        return loss


    def validation_step(self, batch, batch_idx, dl_index=0, mode='val'):

        if dl_index == 1 and mode == 'val':
            mode = 'test'
        with torch.no_grad():
            audio_features, sentence_features, audio_mask, sentence_mask = self(batch)

        args = {
            'audio_features': copy.deepcopy(audio_features[:, :2].detach()),
            'audio_mask':  copy.deepcopy(audio_mask[:, :2].detach()),
            'sentence_features': copy.deepcopy(sentence_features[:, :1].detach()),
            'sentence_mask': copy.deepcopy(sentence_mask[:, :1].detach()),
            'keywords': batch['keywords'],
            'caption': batch['caption'],
            'path': batch['path'],
            'idx': batch['idx'],
            'mode': mode,
            'html': batch['html']
        }

        if type(args) is not tuple:
            args = [args]
        import itertools
        import numpy as np
        mode = args[0]['mode']
        paths = list(itertools.chain(*[batch['path'] for batch in args]))
        captions = list(itertools.chain(*[batch['caption'] for batch in args]))
        keywords = list(itertools.chain(*[batch['keywords'] for batch in args]))
        html = list(itertools.chain(*[batch['html'] for batch in args]))

        I = torch.from_numpy((np.array(paths)[:, None] == np.array(paths)[None, :])).type(torch.bool)
        args = {k: torch.cat([batch[k] for batch in args], dim=0) for k in args[0] if
                k in ['audio_features', 'sentence_features', 'audio_mask', 'sentence_mask', 'idx']}

        audio_features = args['audio_features']
        sentence_features = args['sentence_features']
        audio_mask = args['audio_mask']
        sentence_mask = args['sentence_mask']

        with torch.cuda.amp.autocast(enabled=True):
            C = self.rank_sequences(audio_features, audio_mask, sentence_features, sentence_mask)

        C_ = C / torch.abs(self.tau)

        C_audio = torch.log_softmax(C, dim=0)
        C_text = torch.log_softmax(C, dim=1)

        C_audio_ = torch.log_softmax(C_, dim=0)
        C_text_ = torch.log_softmax(C_, dim=1)

        assert C_audio.shape[0] == C_audio.shape[
            1], f'Audio Features Shape: {audio_features.shape} Sentence Features Shape: {sentence_features.shape}'
        assert C_text.shape[0] == C_text.shape[1]

        # mode = kwargs.get('mode', 'val')
        loss = -0.5 * (C_audio[torch.where(I)].mean() + C_text[torch.where(I)].mean())
        loss_ = -0.5 * (C_audio_[torch.where(I)].mean() + C_text_[torch.where(I)].mean())
        self.log(f"{mode}/loss", loss.item(), batch_size=len(audio_features), add_dataloader_idx=False, sync_dist=True)
        self.log(f"{mode}/loss_tau", loss_.item(), batch_size=len(audio_features), add_dataloader_idx=False, sync_dist=True)
        args['path'] = paths
        args['caption'] = captions
        args['keywords'] = keywords
        args['html'] = html

        self.validation_outputs.append(args)

    def on_validation_epoch_end(self, mode='val'):
        outputs = self.validation_outputs
        if len(outputs) == 0:
            return
        if type(outputs[0]) == list:
            # multiple data sets, run validation for every one
            for o in outputs:
                self.validation_epoch_end(o, mode='val')
            return

        import numpy as np
        paths = [p for b in outputs for p in b['path']]
        captions = [p for b in outputs for p in b['caption']]
        idxs = [i.item() for b in outputs for i in b['idx']]
        keywords = [i for b in outputs for i in b['keywords']]
        html = [i for b in outputs for i in b['html']]
        print('concatenating outputs')

        audio_features = torch.cat([o['audio_features'] for o in outputs])
        sentence_features = torch.cat([o['sentence_features'] for o in outputs])
        audio_mask = torch.cat([o['audio_mask'] for o in outputs])
        sentence_mask = torch.cat([o['sentence_mask'] for o in outputs])

        if self.distributed_mode:
            #print("Gathering validation results from all nodes")
            #print(f"local paths: {len(paths)}")
            lp = len(paths)
            #print(f"local paths: {paths[0]}")
            all_paths= [None for _ in range(word_size)]
            dist.all_gather_object(all_paths, paths)
            paths = list(itertools.chain(*all_paths))
            #print(f"all paths concat: {len(paths)}")

            all_paths= [None for _ in range(word_size)]
            dist.all_gather_object(all_paths, captions)
            captions = list(itertools.chain(*all_paths))

            all_paths= [None for _ in range(word_size)]
            dist.all_gather_object(all_paths, idxs)
            idxs = list(itertools.chain(*all_paths))


            #print("audio_features.shape=", audio_features.shape)
            all_audio_features  = self.all_gather(audio_features)
            #print("all_audio_features.shape=", all_audio_features.shape)
            audio_features = all_audio_features.reshape(-1, audio_features.shape[-1])
            #print("audio_features.shape=", audio_features.shape)

            #print("sentence_features.shape=", sentence_features.shape)
            all_sentence_features  = self.all_gather(sentence_features)
            #print("all_sentence_features.shape=", all_sentence_features.shape)
            sentence_features = all_sentence_features.reshape(-1, sentence_features.shape[-1])
            #print("sentence_features.shape=", sentence_features.shape)
        print('sorting outputs')
        _, sorted = np.unique(idxs, return_index=True)

        audio_features = audio_features[sorted]
        sentence_features = sentence_features[sorted]
        audio_mask = audio_mask[sorted]
        sentence_mask = sentence_mask[sorted]
        paths = np.array(paths)[sorted]
        captions = np.array(captions)[sorted]

        from collections import Counter
        n_captions = Counter(paths)
        assert [v == n_captions[paths[0]] for k, v in n_captions.items()]
        n_captions = n_captions[paths[0]] if not self.kwargs['use_captions'] else 1
        print('compute global ranking')
        C = torch.empty((len(sentence_features), len(audio_features) // n_captions))
        all_audio_features = torch.concatenate([audio_features[::n_captions]])
        all_audio_masks = torch.concatenate([audio_mask[::n_captions]])
        for i in range(len(all_audio_features)):
            with torch.cuda.amp.autocast(enabled=True):
                C_ = self.rank_sequences(
                    all_audio_features[i:i+1],
                    all_audio_masks[i:i+1],
                    sentence_features[:],
                    sentence_mask[:]
                )
            C[:, i:i+1] = C_.T

        if self.trainer.is_global_zero and self.store_predictions:
            print(self.logger)
            experiment_name = self.experiment_name if not self.logger else self.logger.experiment.name
            path = os.path.join(get_model_dir(), experiment_name)
            print("\nSaving predictions to ", path)
            os.makedirs(path, exist_ok=True)
            torch.save(C.cpu(), os.path.join(path, f"predictions_{mode}_{self.current_epoch}.pt"))
            torch.save(sentence_features.cpu(), os.path.join(path, f"sentence_embeddings_{mode}.pt"))
            torch.save(audio_features.cpu(), os.path.join(path, f"audio_embeddings_{mode}.pt"))
            np.save(os.path.join(path, f"paths_{mode}"), paths)
            np.save(os.path.join(path, f"captions_{mode}"), captions)
            print("\nSaving done!\n to ", path)

        if self.kwargs['use_captions']:
            rows = []
            for i, c in enumerate(C):
                j = (i +1) % 5
                rows.append(c[j::5])
            C = torch.stack(rows)
            n_captions = 5

        top_ten = C.topk(10, dim=1)[1]

        target = torch.arange(len(audio_features)//n_captions)
        target = torch.repeat_interleave(target, n_captions)

        r_1 = (top_ten[:, :1] == target[:, None]).float().sum(axis=1).mean().item()
        r_5 = (top_ten[:, :5] == target[:, None]).float().sum(axis=1).mean().item()
        r_10 = (top_ten == target[:, None]).float().sum(axis=1).mean().item()

        AP = 1 / ((top_ten == target[:, None]).float().argmax(dim=1) + 1)
        AP[~(top_ten == target[:, None]).any(dim=1)] = 0
        mAP = AP.mean().item()

        self.log(f'{mode}/R@1', r_1, add_dataloader_idx=False, sync_dist=True)
        self.log(f'{mode}/R@5', r_5, add_dataloader_idx=False, sync_dist=True)
        self.log(f'{mode}/R@10', r_10, add_dataloader_idx=False, sync_dist=True)
        self.log(f'{mode}/mAP@10', mAP, add_dataloader_idx=False, sync_dist=True)

        self.validation_outputs.clear()

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx, mode='test')

    def on_test_epoch_end(self):
        return self.on_validation_epoch_end(mode='test')

    def configure_optimizers(self):
        audio_encoder = []
        text_encoder = []

        audio_seq = []
        text_seq = []

        for k, p in self.named_parameters():
            if 'audio_embedding' in k:
                audio_encoder.append(p)
            elif 'sentence_embedding' in k or 'metadata_embedding_model' in k:
                text_encoder.append(p)
            elif 'project_sentence' in k or 'project_metadata' in k:
                text_seq.append(p)
            elif 'project_audio' in k or 'audio_output_projection' in k or 'posencode' in k or 'audio_sequence' in k or 'audio_token' in k:
                audio_seq.append(p)
            elif 'tau' in k or 'timing_tau' in k:
                audio_seq.append(p)
            else:
                raise ValueError

        param_groups = [
            {"params": audio_encoder, "lr": self.kwargs['lr_audio_encoder']},
            {"params": text_encoder, "lr": self.kwargs['lr_sentence_encoder']},
            {"params": audio_seq, "lr": self.kwargs['lr_audio_project']},
            {"params": text_seq, "lr": self.kwargs['lr_sentence_project']},
        ]

        optimizer = get_optimizer(param_groups)
        return {
            "optimizer": optimizer
        }

    @audio_retrieval.capture
    def update_scalars(self, batch_idx, hard_steps=True):
        epoch = self.current_epoch + (batch_idx / self.trainer.num_training_batches)
        if hard_steps:
            epoch = epoch // 1

        # weight decay - keep constant
        self.log('trainer/weight_decay', self.trainer.optimizers[0].param_groups[0]['weight_decay'])

        # learning rate
        update_lr(self.optimizers(use_pl_optimizer=False), epoch)
        for i, pm in enumerate(self.trainer.optimizers[0].param_groups):
            self.log(f'trainer/lr_{i}', pm['lr'])


@audio_retrieval.capture
def update_lr(optimizer, epoch, lr_audio_encoder, lr_sentence_encoder, lr_audio_project, lr_sentence_project, min_lr, warmup_length, rampdown_start, rampdown_stop, max_epochs, warmup_type='linear', rampdown_type='cosine'):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if rampdown_stop <= 0:
        rampdown_stop = max_epochs

    lrs = [lr_audio_encoder, lr_sentence_encoder, lr_audio_project, lr_sentence_project]
    for i, lr in enumerate(lrs):
        if epoch < warmup_length:
            if warmup_type == 'linear':
                lr = lr * epoch / warmup_length
            elif warmup_type == 'exp':
                epoch = np.clip(epoch, 0.5, warmup_length)
                phase = 1.0 - epoch / warmup_length
                lr = lr * float(np.exp(-5.0 * phase * phase))
            else:
                raise NotImplementedError
        elif epoch < rampdown_start:
            lr = lr
        elif epoch < rampdown_stop:

            if rampdown_type == 'cosine':
                offset = rampdown_start
                lr = min_lr + (lr - min_lr) * 0.5 * \
                     (1. + math.cos(math.pi * (epoch - offset) / (rampdown_stop - offset)))
            elif rampdown_type.startswith('step'):
                distance, factor = rampdown_type.split('_')[1:]
                distance, factor = int(distance), float(factor)
                steps = epoch // distance
                lr = lr*(factor**steps)
                lr = max(lr, min_lr)
            elif rampdown_type == 'linear':
                e = epoch - rampdown_start
                m = rampdown_stop - rampdown_start
                lr -= (lr - min_lr) * (e / m)
                lr = max(lr, min_lr)
            else:
                raise NotImplementedError
        else:
            lr = min_lr

        optimizer.param_groups[i]["lr"] = lr
    return lr


@audio_retrieval.capture
def get_optimizer(parameters, beta1, beta2, eps, weight_decay, amsgrad, adamw):
    if adamw:
        return torch.optim.AdamW(parameters, betas=(beta1, beta2), eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
    else:
        return torch.optim.Adam(parameters, betas=(beta1, beta2), eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)


@audio_retrieval.capture
def get_trainer(wandb_logger, max_epochs, max_samples_per_epoch, gpus, half_precision, accelerator, enable_checkpointing, fast_dev_run, accumulate_grad_batches,
                gradient_clip_val, strategy, num_nodes, _config):
    if max_samples_per_epoch is None:
        max_steps_per_epoch = 0
    else:
        max_steps_per_epoch = max_samples_per_epoch // _config['data_loader']['batch_size']
    kwargs = {}
    if fast_dev_run:
       if max_steps_per_epoch == 0:
           print('Using fast_dev_run with max_epochs=5')
           max_steps_per_epoch = 5
       else:
           raise ValueError('Cannot use fast_dev_run with max_samples_per_epoch')
    return pl.Trainer(
        devices=gpus,
        num_nodes=num_nodes,
        accelerator=accelerator,
        val_check_interval=1.0,
        enable_checkpointing=enable_checkpointing > 0,
        logger=wandb_logger,
        max_epochs=max_epochs,
        callbacks=get_callbacks(wandb_logger),
        precision=16 if half_precision else 32,
        limit_train_batches=1.0 if max_steps_per_epoch <= 0 else max_steps_per_epoch // gpus,
        limit_val_batches=1.0,
        reload_dataloaders_every_n_epochs=1 if max_steps_per_epoch > 0 else 0,
        num_sanity_val_steps=0,
        accumulate_grad_batches=accumulate_grad_batches,
        gradient_clip_val=gradient_clip_val,
        fast_dev_run=False,
        strategy=strategy,
        **kwargs
    )

@audio_retrieval.capture
def get_callbacks(wandb_logger, monitor, enable_checkpointing):
    callbacks = []

    if wandb_logger == False or wandb_logger is None:
         print('No logger; skipping checkpoints')
    else:
        callbacks.append(LearningRateMonitor(logging_interval='epoch'))

    if enable_checkpointing:
        monitor = f'val/{monitor}'
        experiment_name = 'none' if wandb_logger is None or callable(wandb_logger.experiment.name) else wandb_logger.experiment.name
        save_dir = os.path.join(get_model_dir(), experiment_name)
        os.makedirs(save_dir, exist_ok=True)

        callbacks.append(
            ModelCheckpoint(
                dirpath=save_dir,
                monitor=monitor,
                mode='max',
                save_top_k=1,
                every_n_epochs=enable_checkpointing,
                save_last=True,
                auto_insert_metric_name=False,
                filename='epoch_{epoch}-{' + f'{monitor}' + '}'
            )
        )

    return callbacks


@audio_retrieval.command
def print_lr(max_epochs):
    import matplotlib.pyplot as plt
    import numpy as np

    lrs = [update_lr(torch.optim.Adam(torch.nn.Linear(1,1).parameters()), x) for x in np.linspace(0, max_epochs, 10000)]

    plt.plot(np.linspace(0, max_epochs, 10000), lrs)
    plt.show()


@audio_retrieval.command
def cmd_generate_embeddings(model=None, load_parameters=None, train_on=None):
    if model is None:
        model = get_model(load_parameters)
    model = model.cuda()
    model.eval()

    # create the data module
    print(f'Loading {train_on} data sets...')
    train = get_data_set(train_on, 'train')

    print(f'Training set size: {len(train)}')

    dl = get_eval_data_loader(train, shuffle=False)

    audio_embeddings = {}
    sentence_embeddings = {}
    l = len(get_dataset_dir().split(os.path.sep))
    from tqdm import tqdm
    with torch.no_grad():
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
            for b in tqdm(dl, total=len(dl)):
                a, s, am, sm = model(b)
                for i, p in enumerate(b['path']):
                    p = os.path.sep.join(p.split(os.path.sep)[l:]) # remove dataset dir
                    audio_embeddings[p] = copy.deepcopy(a[i, 0].detach().cpu())
                    sentence_embeddings[b['caption'][i]] = copy.deepcopy(s[i, 0].detach().cpu().clone())

    out_path = os.path.join(get_model_dir(), load_parameters)

    torch.save(sentence_embeddings, os.path.join(out_path, train_on + '_sentence_embeddings_train.pt'))
    torch.save(audio_embeddings, os.path.join(out_path, train_on + '_audio_embeddings_train.pt'))


@audio_retrieval.command
def cmd_test_on_clothov2(load_model, _config):
    print('Initialize model...')
    print(load_model)
    model = AudioRetrievalModel.load_from_checkpoint(load_model, strict=False)
    model.store_predictions = False
    model = model.cuda()
    model.eval()
    t = get_trainer(None)

    predict = get_data_set('clothov2', 'test')
    result = t.test(model, get_eval_data_loader(predict, shuffle=True, distributed=False))

    print(result)
    return result[0]


def multiprocessing_run(rank, word_size, pernode=None):
    import socket
    print("rank ", rank, os.getpid(), "hash=", hash("kk test"), " on node ", socket.gethostname())
    print("word_size ", word_size)
    if pernode is None:
        pernode = word_size
    print("Tasks per node = ", pernode)

    os.environ['NODE_RANK'] = str(rank)
    os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['CUDA_VISIBLE_DEVICES'].split(",")[
        rank%pernode]
    print("Sat os.environ['CUDA_VISIBLE_DEVICES']=", os.environ['CUDA_VISIBLE_DEVICES'])
    # torch.cuda.set_device(int(os.environ['CUDA_VISIBLE_DEVICES'].split(",")[
    #     rank]))
    argv = sys.argv
    if rank != 0:
        print(f"Unobserved {os.getpid()} with rank {rank}")
        argv = argv + ["-u"]  # only rank 0 is observed
    if "with" not in argv:
        argv = argv + ["with"]

    argv = argv + \
        [f"num_nodes={word_size}", f"strategy=ddp"]
    print(argv)

    @audio_retrieval.main
    def main():
        return run_one()

    audio_retrieval.run_commandline(argv)

class PositionalEncoding(torch.nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 500, posencode_init=0.04, project=False, project_bias=True, pe_requires_grad=True, weight=1.0):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout)
        self.max_len = max_len
        self.posencode_init = posencode_init
        reweight_only=False

        if posencode_init == 'sinosoidal':

            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len).unsqueeze(1)
            div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                                  -(math.log(10000.0) / d_model)))
            pe[:, 0::2] = torch.sin(position.float() * div_term)
            pe[:, 1::2] = torch.cos(position.float() * div_term)
        elif posencode_init == 'passt':
            from models.audio.passt import get_passt
            import torch.nn.functional as F
            passt = get_passt('passt')
            pe = passt[0].model.time_new_pos_embed[0,:,0,:].permute(1,0)

            posemb_grid = pe.permute(1, 0)[None, :, :, None]
            posemb_grid = F.interpolate(posemb_grid, size=(max_len, 1), mode='bicubic', align_corners=False)
            pe = posemb_grid[0, :, :, 0].permute(1, 0)
        elif posencode_init == 'sinosoidal_fixed':
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len).unsqueeze(1)
            div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                                  -(math.log(10000.0) / d_model)))
            pe[:, 0::2] = torch.sin(position.float() * div_term)
            pe[:, 1::2] = torch.cos(position.float() * div_term)
            pe_requires_grad=False
            reweight_only=True
        else:
            pe = torch.randn((max_len, d_model)) * posencode_init

        if reweight_only:
            w = torch.ones((1,)) * weight
            self.w = torch.nn.Parameter(w, requires_grad=True)
            self.project = lambda x: x * self.w
        else:
            self.project = torch.nn.Linear(pe.shape[-1], d_model, bias=project_bias) if project else torch.nn.Identity()

        self.pe = torch.nn.Parameter(
            pe[None, :, :],
            requires_grad=pe_requires_grad
        )


    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """

        # i = torch.randperm(self.pe.shape[1])[:x.shape[1]].sort().values
        x = x + self.project(self.pe[:, :x.shape[1], :])

        return self.dropout(x)


if __name__ == '__main__':
    # set DDP=2 forks two processes to run on two GPUs
    # the environment variable "DDP" define the number of processes to fork
    # With two 2x 2080ti you can train the full model to .47 in around 24 hours
    # you may need to set NCCL_P2P_DISABLE=1
    global word_size
    word_size = os.environ.get("DDP", None)
    DDP_SLURM = os.environ.get("DDP_SLURM", None)
    if DDP_SLURM:
        print("\n***SLLURM DDP MODE***\n\n")
        if "SLURM_NTASKS" in os.environ:
            del os.environ["SLURM_NTASKS"]
        if "SLURM_JOB_NAME" in os.environ:
            del os.environ["SLURM_JOB_NAME"]
        word_size = int(os.environ.get("WORLD_SIZE", None))
        print("word_size = ", word_size)
        pernode = int(os.environ.get("SLURM_NTASKS_PER_NODE", None))
        print("pernode = ", pernode)
        rank = int(os.environ.get("SLURM_PROCID", None))
        print("rank = ", rank)
        os.environ['PL_IN_DDP_SUBPROCESS'] = '1'
        print("I'm runing  with, pid=", os.getpid())
        multiprocessing_run(rank, word_size, pernode)
        exit(0)

    if word_size:
        import random
        if "SLURM_NTASKS" in os.environ:
            del os.environ["SLURM_NTASKS"]
        if "SLURM_JOB_NAME" in os.environ:
            del os.environ["SLURM_JOB_NAME"]
        word_size = int(word_size)
        print(f"\n\nDDP TRAINING WITH WORD_SIZE={word_size}\n\n")
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        # plz no collisions
        os.environ['MASTER_PORT'] = f"{9999 + random.randint(0, 9999)}"
        os.environ['PL_IN_DDP_SUBPROCESS'] = '1'
        os.environ['WORLD_SIZE'] = str(word_size)
        for rank in range(word_size):
            pid = os.fork()
            if pid == 0:
                print("Child Forked, pid=", os.getpid())
                multiprocessing_run(rank, word_size)
                exit(0)

        pid, exit_code = os.wait()
        print(pid, exit_code)
        exit(0)

print("__main__ is running pid", os.getpid(), "in module main: ", __name__)




@audio_retrieval.automain
def main():
    return run_one()


