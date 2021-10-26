"""
PyAudio Example: Make a wire between input and output (i.e., record a
few samples and play them back immediately).

This is the callback (non-blocking) version.
"""

import librosa
import librosa.display
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pyaudio
import time
from librosa.filters import mel as librosa_mel_fn
import torch
import torch.utils.data
import tensorflow as tf

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

mel_basis = {}
hann_window = {}

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


model_name = f'lite-model_hifi-gan_dr_1.tflite'
interpreter = tf.lite.Interpreter(model_path=model_name)
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


def tflite_inference(input, quantization='dr'):
    interpreter.resize_tensor_input(input_details[0]['index'],  [1, input.shape[1], input.shape[2]], strict=True)
    interpreter.allocate_tensors()
    interpreter.set_tensor(input_details[0]['index'], input)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    return output

p = pyaudio.PyAudio()
for i in range(p.get_device_count()):
    print(p.get_device_info_by_index(i))

def callback(in_data, frame_count, time_info, status):
    data = np.fromstring(in_data, dtype=np.float32)
    audio = torch.FloatTensor(data)
    audio = audio.unsqueeze(0)

    spec = mel_spectrogram(audio, attr_d["n_fft"], attr_d["num_mels"], attr_d["sampling_rate"], 
        attr_d["hop_size"], attr_d["win_size"], attr_d["fmin"], attr_d["fmax"])

    matplotlib.image.imsave('myfig_' + str(frame_count) + '.png', spec[0])

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

    spec_in = spec.detach().numpy()
    output = tflite_inference(spec_in).squeeze()

    print(spec[:10])
    print(spec.shape)

    return (output, pyaudio.paContinue)

stream = p.open(format=pyaudio.paFloat32,
                channels=CHANNELS,
                rate=attr_d["sampling_rate"],
                input=True,
                output=True,
                frames_per_buffer=attr_d["segment_size"],
                stream_callback=callback)

print("Starting to listen.")
stream.start_stream()

while stream.is_active():
    time.sleep(0.1)

stream.stop_stream()
stream.close()

p.terminate()

