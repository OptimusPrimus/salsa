import torch
from models.architecture.MobileNetV3 import get_model as get_mobile_net
from models.architecture.MobileNetV3 import AugmentMelSTFT




def get_efficientat(model_name='mn40_as_ext', freqm=48, timem=192, return_sequence=False, **kwargs):

    # get the PaSST model wrapper, includes Melspectrogram and the default pre-trained transformer
    if "mn40_as_ext" == model_name:
        model = get_mobile_net(width_mult=4.0, pretrained_name=model_name)
    else:
        raise ValueError

    # print(model.mel)  # Extracts mel spectrogram from raw waveforms.

    mel = AugmentMelSTFT(n_mels=128, sr=32000, win_length=800, hopsize=320, n_fft=1024, freqm=freqm,
                               timem=timem)


    class Wrapper(torch.nn.Module):

        def __init__(self, mel, model):
            super().__init__()
            self.mel = mel
            self.model = model

        def forward(self, x, **kwargs):
            with torch.no_grad():
                mel = self.mel(x)
            if return_sequence:
                out = self.model(mel[:, None])[1].permute(0, 2, 3, 1)
            else:
                out = self.model(mel[:, None])[0][:, None, None, :]
            return out

    wrapper = Wrapper(mel, model)

    return wrapper, 3840

if __name__ == '__main__':


    from data.datasets.audioset import audioset, get_audioset
    from sacred import Experiment

    ex = Experiment('test_as', ingredients=[audioset])

    @ex.main
    def run_test():

        model = get_efficientat('mn40_as_ext', return_sequence=True)[0]

        aus = get_audioset("evaluation")
        aus.set_fixed_length(10).cache_audios()

        predicted = []
        true = []
        model.eval()
        for a in aus:
            with torch.no_grad():
                predicted.append(model(torch.from_numpy(a['audio'])[None,:]).detach().numpy())
                true.append(a['target'])

        import numpy as np
        #predicted = np.stack(predicted)
        #true = np.stack(true)
        from sklearn import metrics
        metrics.average_precision_score(np.stack(true), np.stack(predicted)[:,0,:], average=None)
        print(predicted)

    ex.run()