import torch
from hear21passt.base import get_model_passt
from hear21passt.base import AugmentMelSTFT


def get_passt(model_name, s_patchout_t=0, s_patchout_f=0, freqm=48, timem=192, return_sequence=False, **kwargs):

    # get the PaSST model wrapper, includes Melspectrogram and the default pre-trained transformer
    if "passt_s" == model_name:
        print("#### Using PaSST-S ap486 model with overlap ####\n")
        model = get_model_passt("passt_s_kd_p16_128_ap486", input_tdim=998, fstride=10, tstride=10,
                                s_patchout_t=s_patchout_t, s_patchout_f=s_patchout_f)
    elif "passt_20" == model_name:
        print("#### Using PaSST-S  train on `20` seconds ####\n")
        model = get_model_passt(arch="passt_20sec", input_tdim=2000, fstride=10, tstride=10, s_patchout_t=s_patchout_t,
                                s_patchout_f=s_patchout_f)
    elif "passt_l" == model_name:
        print("#### Using PaSST-L  ####\n")
        model = get_model_passt(arch="passt_l_kd_p16_128_ap47", input_tdim=998, fstride=10, tstride=10,
                                s_patchout_t=s_patchout_t, s_patchout_f=s_patchout_f)
    else:
        print("#### Using PaSST model with no overlap ####")
        model = get_model_passt("passt_s_p16_s16_128_ap468", input_tdim=1000, fstride=16, tstride=16,
                                s_patchout_t=s_patchout_t, s_patchout_f=s_patchout_f)
    # print(model.mel)  # Extracts mel spectrogram from raw waveforms.

    model.mel = AugmentMelSTFT(n_mels=128, sr=32000, win_length=800, hopsize=320, n_fft=1024, freqm=freqm,
                               timem=timem, htk=False, fmin=0.0, fmax=None, norm=1, fmin_aug_range=10,
                               fmax_aug_range=2000)

    audio_embedding_model = model

    class Wrapper(torch.nn.Module):

        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x, y=None, **kwargs):
            with torch.no_grad():
                mel = self.model.mel(x)
            if return_sequence:
                tokens = self.model(mel[:, None], y=y, **kwargs)[-1]
            else:
                tokens = self.model(mel[:, None], y=y, **kwargs)[-2][:, None, None, :]
            return tokens

    return Wrapper(audio_embedding_model), 768