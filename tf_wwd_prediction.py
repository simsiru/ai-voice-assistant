import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf

import librosa

import sounddevice as sd
from scipy.io.wavfile import write
from tensorflow.keras.models import load_model

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)])
    except RuntimeError as e:
        print(e)


def get_spectrogram(waveform):
    # Zero-padding for an audio waveform with less than 16,000 samples.
    input_len = 16000
    waveform = waveform[:input_len]
    zero_padding = tf.zeros(
        [16000] - tf.shape(waveform),
        dtype=tf.float32)
    # Cast the waveform tensors' dtype to float32.
    waveform = tf.cast(waveform, dtype=tf.float32)
    # Concatenate the waveform with `zero_padding`, which ensures all audio
    # clips are of the same length.
    equal_length = tf.concat([waveform, zero_padding], 0)
    # Convert the waveform to a spectrogram via a STFT.
    spectrogram = tf.signal.stft(
        equal_length, frame_length=255, frame_step=128)
    # Obtain the magnitude of the STFT.
    spectrogram = tf.abs(spectrogram)
    # Add a `channels` dimension, so that the spectrogram can be used
    # as image-like input data with convolution layers (which expect
    # shape (`batch_size`, `height`, `width`, `channels`).
    spectrogram = spectrogram[..., tf.newaxis]
    return spectrogram



#fs = 44100
fs = 16000
seconds = 1
filename = "prediction.wav"
class_names = ['down', 'go', 'left', 'no', 'right', 'stop', 'up', 'yes']

#model = load_model("wake_word_detection_model/wwd_model.h5")
model = load_model("wwd_model/wwd_model.h5")

print("Prediction Started: ")
i = 0
while True:
    print("Say Now: ")
    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
    sd.wait()
    write(filename, fs, myrecording)

    audio, sample_rate = librosa.load(filename)
    spectrogram = get_spectrogram(audio)
    print(spectrogram.shape)

    prediction = model.predict(np.expand_dims(spectrogram, axis=0))

    clas_idx = np.argmax(prediction[0])

    print("Prediction: ", class_names[clas_idx])
    print("Confidence: ", prediction[0][clas_idx])
    """ if prediction[:, 1] > 0.99:
        print(f"Wake Word AMOGUS Detected for ({i})")
        print("Confidence:", prediction[:, 1])
        i += 1
    
    else:
        print(f"Wake Word NOT Detected")
        print("Confidence:", prediction[:, 0]) """