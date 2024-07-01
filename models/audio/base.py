import torch
from torchaudio.functional import resample
import torch.nn.functional


def get_audio_embedding_model(name, segment_length=10, hop_size=10, model_config=dict(), multi_window=False):
    from models.audio.passt import get_passt
    from models.audio.atst import get_atst
    from models.audio.efficientat import get_efficientat

    if name.startswith('passt'):
        model, emb_dim = get_passt(name, **model_config)
    elif name.startswith('mn'):
        model, emb_dim = get_efficientat(name, **model_config)
    elif name.startswith('atst'):
        model = get_atst('atstframe_base_as2M.ckpt', **model_config)
        emb_dim = 768
    else:
        raise AttributeError("")

    if multi_window:
        assert False

    return WindowedPrediction(model, segment_length=segment_length, hop_size=hop_size), emb_dim

class WindowedPrediction(torch.nn.Module):

    def __init__(self, model, segment_length, hop_size, sr=32000):
        super().__init__()
        self.model = model
        self.segment_length = segment_length
        self.hop_size = hop_size
        self.sr = sr

    def forward(self, x, y=None, **kwargs):
        B = len(x)
        x = split_audio(x, self.sr, self.segment_length, hop_size=self.hop_size)
        if y is not None:
            y = y[:, None].expand((-1,len(x)//B,-1,-1)).reshape(len(x), y.shape[1], y.shape[2])
        tokens = self.model(x, y=y, **kwargs) # (B, f, t, d)
        tokens = merge_embeddings(tokens, B)
        return tokens

def split_audio(x, sampling_rate, segment_length, hop_size=10):
    segment_length = int(segment_length * sampling_rate)
    hop_size = int(hop_size * sampling_rate)
    if x.shape[1] < segment_length:
        return x
    x = x.unfold(dimension=1, size=segment_length, step=hop_size).reshape(-1, segment_length)
    return x

def merge_embeddings(x, B):
    embedding_sequence = x.reshape(B, len(x) // B, x.shape[1], x.shape[2], x.shape[3]) # B, s, f, t, d
    embedding_sequence = torch.split(embedding_sequence, 1, dim=1) # s, B, f, t, d
    embedding_sequence = torch.concatenate(embedding_sequence, dim=3)[:, 0] # B, f, s*t, d
    return embedding_sequence


if __name__ == '__main__':
    from sacred import Experiment
    ex = Experiment(ingredients=[])

    @ex.main
    def automain():

        x = torch.zeros(10, 1, 1, 2)
        o = merge_embeddings(x, 2)


        from models.audio.passt import get_passt

        x = torch.zeros(1, 10*32000)

    ex.run()