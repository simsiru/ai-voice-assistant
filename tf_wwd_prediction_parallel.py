###### IMPORTS ###################
import threading
import time
import sounddevice as sd
import librosa
import numpy as np
import tensorflow as tf
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


class_names = ['down', 'go', 'left', 'no', 'right', 'stop', 'up', 'yes']

fs = 16000
seconds = 1

model = load_model("wwd_model/wwd_model.h5")


def listener():
    while True:
        myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
        sd.wait()

        spectrogram = get_spectrogram(myrecording.ravel())
        print(spectrogram.shape)

        prediction_thread(spectrogram)
        time.sleep(0.001)


def prediction(y):
    prediction = model.predict(np.expand_dims(y, axis=0))

    clas_idx = np.argmax(prediction[0])
    print("Prediction: ", class_names[clas_idx])
    print("Confidence: ", prediction[0][clas_idx])

    time.sleep(0.1)

def prediction_thread(y):
    pred_thread = threading.Thread(target=prediction, name="PredictFunction", args=(y,))
    pred_thread.start()


listen_thread = threading.Thread(target=listener, name="ListeningFunction")
listen_thread.start()