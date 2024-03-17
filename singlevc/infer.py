
import torch
from torch.backends import cudnn
from torch.utils.data import DataLoader
import torchaudio
import numpy as np
import yaml
import time
from singlevc.any2one import Generator

from singlevc.any2any import MagicModel


def mel_denormalize(S, clip_val=1e-5):
    clip = torch.log(torch.Tensor([clip_val]).to(S.device))
    S = S*(0-clip) + clip
    return S


def mel_normalize(S, clip_val=1e-5):
    clip = torch.log(torch.Tensor([clip_val]).to(S.device))
    S = (S - clip) * 1.0/(0-clip)
    return S


def count_parameters_in_M(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6


class Solver():
    def __init__(self, config):
        # super(Solver, self).__init__()
        self.config = config
        self.local_rank = self.config['local_rank']
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.Generator = Generator().to(self.device)
        self.init_epoch = 0
        if self.config['resume']:
            self.resume_model(self.config['resume_model_path'])
        print('config = %s', self.config)
        print('param Generator size = %fM ' %
              (count_parameters_in_M(self.Generator)))

    def resume_model(self, resume_model_path):
        checkpoint_file = resume_model_path
        print('loading the model from %s' % (checkpoint_file))
        checkpoint = torch.load(checkpoint_file, map_location='cpu')
        self.init_epoch = checkpoint['epoch']
        self.Generator.load_state_dict(checkpoint['Generator'])
        self.Generator.eval()
        self.Generator.remove_weight_norm()

    def infer(self, input_mel):
        # infer  prepare
        with torch.no_grad():
            mel = input_mel.squeeze(0).transpose(0, 1)
            mel = mel_normalize(mel)
            mel = mel.unsqueeze(0)
            fake_mel = self.Generator(mel, None)
            fake_mel = torch.clamp(fake_mel, min=0, max=1)
            fake_mel = mel_denormalize(fake_mel)
            fake_mel = fake_mel.transpose(1, 2)
            fake_mel = fake_mel.detach().cpu()
        return fake_mel


class SolverA():
    def __init__(self, config):
        # super(Solver, self).__init__()
        self.config = config
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.Generator = MagicModel().to(self.device)
        if self.config['pre_train_singlevc']:
            singlevc_checkpoint = torch.load(
                self.config['singlevc_model_path'], map_location='cpu')
            self.Generator.any2one.load_state_dict(
                singlevc_checkpoint['Generator'])
            self.Generator.any2one.eval()
            self.Generator.any2one.remove_weight_norm()
        self.resume_model(self.config['resume_path'])
        print('config = %s', self.config)
        print('param Generator size = %fM ' %
              (count_parameters_in_M(self.Generator)))

        self.wav2mel_model = torch.jit.load(
            self.config['wav2mel_model_path']).eval()
        self.dvector_model = torch.jit.load(
            self.config['dvector_model_path']).eval()

    def resume_model(self, resume_path):
        print("*********  [load]   ***********")
        checkpoint_file = resume_path
        print('loading the model from %s' % (checkpoint_file))
        checkpoint = torch.load(checkpoint_file, map_location='cpu')
        self.Generator.load_state_dict(checkpoint['Generator'])
        self.Generator.eval()
        self.Generator.cont_encoder.remove_weight_norm()
        self.Generator.generator.remove_weight_norm()

    def get_spkemb_mels(self, wave_path):
        wav_tensor, sample_rate = torchaudio.load(wave_path)
        # shape: (frames, mel_dim)
        mel_tensor = self.wav2mel_model(wav_tensor, sample_rate)
        spk_emb = self.dvector_model.embed_utterance(
            mel_tensor)  # shape: (emb_dim)
        spk_emb = spk_emb.unsqueeze(0)
        return spk_emb

    def infer(self, input_mel, spk_emb):
        # infer  prepare
        with torch.no_grad():
            input_mel = input_mel.transpose(1, 2)
            input_mel = mel_normalize(input_mel)
            # input_mel = input_mel.unsqueeze(0)
            fake_mel = self.Generator(spk_emb, input_mel, None)
            # fake_mel = input_mel
            fake_mel = torch.clamp(fake_mel, min=0, max=1)
            fake_mel = mel_denormalize(fake_mel)
            fake_mel = fake_mel.transpose(1, 2)
            # fake_mel = fake_mel.detach().cpu()

            return fake_mel


# if __name__ == '__main__':
# 	print("【Solver】" )
# 	cudnn.benchmark = True
# 	config_path = r"infer_config_any.yaml"
# 	with open(config_path) as f:
# 		config = yaml.load(f, Loader=yaml.Loader)
# 	solver = Solver(config)
# 	solver.infer()


# if __name__ == '__main__':
#     cudnn.benchmark = True
#     config_path = r"infer_config.yaml"
#     with open(config_path) as f:
#         config = yaml.load(f, Loader=yaml.Loader)
#     solver = Solver(config)
#     x = torch.randn([5, 259, 80])
#     print(solver.infer(x))
