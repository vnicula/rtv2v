"""
PyAudio Example: Make a wire between input and output (i.e., record a
few samples and play them back immediately).

This is the callback (non-blocking) version.
"""

import json
import yaml
import librosa
import librosa.display
import matplotlib
import matplotlib.pyplot as plt
# from numba import jit
import numpy as np
import pyaudio
import time
from librosa.filters import mel as librosa_mel_fn
import torch
import torch.utils.data
# import v2v
from singlevc.infer import Solver
# import tensorflow as tf

from models import Generator

attr_d = {
    "segment_size": 8192,
    "num_mels": 80,
    "num_freq": 1025,
    "n_fft": 1024,
    "hop_size": 256,
    "win_size": 1024,

    "sampling_rate": 22050,

    "fmin": 0,
    "fmax": 8000,
    "fmax_for_loss": 0,
}

CHANNELS = 1
RATE = 44100
CHUNK_SIZE = RATE // 4

frames = []

plt.figure(figsize=(10, 4))
do_melspec = librosa.feature.melspectrogram
pwr_to_db = librosa.core.power_to_db


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)

def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C

def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C

def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output

def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output

# mel_spec ?
mel_basis = {}
hann_window = {}
# @jit
def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    if torch.min(y) < -1.:
        print('min value is ', torch.min(y))
    if torch.max(y) > 1.:
        print('max value is ', torch.max(y))

    global mel_basis, hann_window
    if fmax not in mel_basis:
        mel = librosa_mel_fn(sampling_rate, n_fft, num_mels, fmin, fmax)
        mel_basis[str(fmax)+'_'+str(y.device)] = torch.from_numpy(mel).float().to(y.device)
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(1)
    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(y.device)],
                      center=center, pad_mode='reflect', normalized=False, onesided=True)
    spec = torch.sqrt(spec.pow(2).sum(-1)+(1e-9))
    spec = torch.matmul(mel_basis[str(fmax)+'_'+str(y.device)], spec)
    spec = spectral_normalize_torch(spec)

    return spec


def mel_spectrogram_singlevc(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    if torch.min(y) < -1.:
        print('min value is ', torch.min(y))
    if torch.max(y) > 1.:
        print('max value is ', torch.max(y))
    
    global mel_basis, hann_window
    if fmax not in mel_basis:
        mel = librosa_mel_fn(sampling_rate, n_fft, num_mels, fmin, fmax)
        mel_basis[str(fmax)+'_'+str(y.device)] = torch.from_numpy(mel).float().to(y.device)
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)
    
    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(1)
    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(y.device)],
                      center=center, pad_mode='reflect', normalized=False, onesided=True,return_complex=False)
    spec = torch.sqrt(spec.pow(2).sum(-1)+(1e-9))
    spec = torch.matmul(mel_basis[str(fmax)+'_'+str(y.device)], spec)
    spec = spectral_normalize_torch(spec)
    
    return spec


"""
# TF inference
model_name = f'lite-model_hifi-gan_dr_1.tflite'
interpreter = tf.lite.Interpreter(model_path=model_name)
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
resized_input = False

def tflite_inference(input, quantization='dr'):
    global resized_input
    if not resized_input:
        interpreter.resize_tensor_input(input_details[0]['index'],  [1, input.shape[1], input.shape[2]], strict=True)
        interpreter.allocate_tensors()
        resized_input=True
    interpreter.set_tensor(input_details[0]['index'], input)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    return output
"""

p = pyaudio.PyAudio()
for i in range(p.get_device_count()):
    print(p.get_device_info_by_index(i))

config = 'singlevc/pretrained/HiFi-GAN/UNIVERSAL_V1/config.json'
device = 'cpu'
with open(config) as f:
    data = f.read()

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

json_config = json.loads(data)
h = AttrDict(json_config)

# Load HiFi GAN
torch_checkpoints = torch.load("singlevc/pretrained/HiFi-GAN/UNIVERSAL_V1/g_02500000", map_location=torch.device('cpu'))
torch_generator_weights = torch_checkpoints["generator"]
torch_model = Generator(h)
torch_model.load_state_dict(torch_checkpoints["generator"])
torch_model.eval()
torch_model.remove_weight_norm()

# # Conversion model stargan
# f0model = v2v.load_F0_model()
# stgv2 = v2v.load_stargan_v2()
# myref = v2v.compute_style(stgv2)
# voco = v2v.load_vocoder()


# Conversion model singlevc
config_path = r"singlevc/infer_config.yaml"
with open(config_path) as f:
    config = yaml.load(f, Loader=yaml.Loader)
SVCGen = Solver(config)


"""
def callback(in_data, frame_count, time_info, status):
    data = np.frombuffer(in_data, dtype=np.float32)
    audio = torch.FloatTensor(data)
    audio = audio.unsqueeze(0)

    spec = mel_spectrogram(audio, attr_d["n_fft"], attr_d["num_mels"], attr_d["sampling_rate"], 
        attr_d["hop_size"], attr_d["win_size"], attr_d["fmin"], attr_d["fmax"])

    # matplotlib.image.imsave('myfig_' + str(frame_count) + '.png', spec[0])

    # melspec = do_melspec(y=data, sr=RATE, n_mels=128, fmax=4000)
    # norm_melspec = pwr_to_db(melspec, ref=np.max)

    # frames.append(norm_melspec)
    
    # if len(frames) == 20:
    #     stack = np.hstack(frames)
    #     librosa.display.specshow(stack, y_axis='mel', fmax=4000, x_axis='time')
    #     plt.colorbar(format='%+2.0f dB')
    #     plt.title('Mel spectrogram')
    #     plt.plot()
    #     plt.savefig('myfig_' + str(frame_count) + '.png')

    #     #break
    #     frames.pop(0)

    # spec_in = spec.detach().numpy()
    # output = tflite_inference(spec_in).squeeze()
    with torch.no_grad():
        hifigan_output = torch_model(spec)
    output = hifigan_output.squeeze().detach().numpy()
    # print(spec[:10])
    # print(spec.shape)

    return (output, pyaudio.paContinue)
"""

def callback(in_data, frame_count, time_info, status):
    data = np.frombuffer(in_data, dtype=np.float32)
    # wave = torch.from_numpy(data).float()
    # wave = torch.FloatTensor(data)

    spec = v2v.conversion(data, f0model, stgv2, myref, voco) #.squeeze(1)
    print("spec shpe:", spec.shape)

    # with torch.no_grad():
    #     hifigan_output = torch_model(spec)
    # output = hifigan_output.squeeze().detach().numpy()
    # print(output[:10], output.shape)

    return (spec[:24000], pyaudio.paContinue)


def callback_singlevc(in_data, frame_count, time_info, status):
    data = np.frombuffer(in_data, dtype=np.float32)
    audio = torch.FloatTensor(data)
    audio = audio.unsqueeze(0)

    spec = mel_spectrogram_singlevc(audio, attr_d["n_fft"], attr_d["num_mels"], attr_d["sampling_rate"], 
        attr_d["hop_size"], attr_d["win_size"], attr_d["fmin"], attr_d["fmax"])
    print(spec[:10])
    print(spec.shape)

    with torch.no_grad():
        spec = SVCGen.infer(spec.transpose(1,2))
        hifigan_output = torch_model(spec)
    
    output = hifigan_output.squeeze().detach().numpy()

    return (output, pyaudio.paContinue)


stream = p.open(format=pyaudio.paFloat32,
                channels=CHANNELS,
                rate=attr_d["sampling_rate"],
                # rate=24000,
                input=True,
                output=True,
                frames_per_buffer=attr_d["segment_size"],
                # frames_per_buffer=24000,
                stream_callback=callback_singlevc)

print("Starting to listen.")
stream.start_stream()

while stream.is_active():
    time.sleep(0.1)

stream.stop_stream()
stream.close()

p.terminate()

