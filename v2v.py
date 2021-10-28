import yaml
from munch import Munch
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchaudio
import time
from JDC.model import JDCNet
from stmodels import Generator, MappingNetwork, StyleEncoder


to_mel = torchaudio.transforms.MelSpectrogram(
    n_mels=80, n_fft=2048, win_length=1200, hop_length=300)
mean, std = -4, 4

def preprocess(wave):
    wave_tensor = torch.from_numpy(wave).float()
    mel_tensor = to_mel(wave_tensor)
    mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - mean) / std
    return mel_tensor

def compute_style(starganv2):
    label = torch.LongTensor([0])
    latent_dim = starganv2.mapping_network.shared[0].in_features
    # ref = starganv2.mapping_network(torch.randn(1, latent_dim).to('cuda'), label)
    ref = starganv2.mapping_network(torch.randn(1, latent_dim), label)

    return ref

def load_F0_model():
    F0_model = JDCNet(num_class=1, seq_len=192)
    params = torch.load("JDC/bst.t7", map_location=torch.device('cpu'))['net']
    F0_model.load_state_dict(params)
    _ = F0_model.eval()
    # F0_model = F0_model.to('cuda')
    return F0_model

def build_model(model_params={}):
    args = Munch(model_params)
    generator = Generator(args.dim_in, args.style_dim, args.max_conv_dim, w_hpf=args.w_hpf, F0_channel=args.F0_channel)
    mapping_network = MappingNetwork(args.latent_dim, args.style_dim, args.num_domains, hidden_dim=args.max_conv_dim)
    style_encoder = StyleEncoder(args.dim_in, args.style_dim, args.num_domains, args.max_conv_dim)
    
    nets_ema = Munch(generator=generator,
                     mapping_network=mapping_network,
                     style_encoder=style_encoder)

    return nets_ema

def load_stargan_v2():
    model_path = 'Models/epoch_00150.pth'

    with open('Models/config.yml') as f:
        starganv2_config = yaml.safe_load(f)
    starganv2 = build_model(model_params=starganv2_config["model_params"])
    params = torch.load(model_path, map_location='cpu')
    params = params['model_ema']
    _ = [starganv2[key].load_state_dict(params[key]) for key in starganv2]
    _ = [starganv2[key].eval() for key in starganv2]
    # starganv2.style_encoder = starganv2.style_encoder.to('cuda')
    # starganv2.mapping_network = starganv2.mapping_network.to('cuda')
    # starganv2.generator = starganv2.generator.to('cuda')
    starganv2.style_encoder = starganv2.style_encoder.to('cpu')
    starganv2.mapping_network = starganv2.mapping_network.to('cpu')
    starganv2.generator = starganv2.generator.to('cpu')

    return starganv2

def conversion(audio, F0_model, starganv2, ref):
    # conversion 
    start = time.time()
        
    # source = preprocess(audio).to('cuda:0')
    source = preprocess(audio)

    with torch.no_grad():
        f0_feat = F0_model.get_feature_GAN(source.unsqueeze(1))
        out = starganv2.generator(source.unsqueeze(1), ref, F0=f0_feat)
        
        # c = out.transpose(-1, -2).squeeze().to('cuda')
        # y_out = vocoder.inference(c)
        # y_out = y_out.view(-1).cpu()

        # if key not in speaker_dicts or speaker_dicts[key][0] == "":
        #     recon = None
        # else:
        #     wave, sr = librosa.load(speaker_dicts[key][0], sr=24000)
        #     mel = preprocess(wave)
        #     c = mel.transpose(-1, -2).squeeze().to('cuda')
        #     recon = vocoder.inference(c)
        #     recon = recon.view(-1).cpu().numpy()

    end = time.time()
    print('total processing time: %.3f sec' % (end - start) )

    return out