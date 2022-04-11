###### IMPORTS ################
import os
import librosa
import librosa.display
from matplotlib.axis import XAxis
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import tensorflow as tf

#### LOADING THE VOICE DATA FOR VISUALIZATION ###
sample = "wake_word_dataset/background_sound/402.wav"
data, sample_rate = librosa.load(sample)

##### VISUALIZING WAVE FORM ##
""" plt.title("Wave Form")
librosa.display.waveshow(data, sr=sample_rate)
plt.show()

##### VISUALIZING MFCC #######
mfccs = librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=40)
print("Shape of mfcc:", mfccs.shape)

plt.title("MFCC")
librosa.display.specshow(mfccs, sr=sample_rate, x_axis='time', y_axis='mel')
#plt.colorbar()
plt.show() """



def get_spectrogram(waveform):
    # Zero-padding for an audio waveform with less than 16,000 samples.
    """ input_len = 16000
    waveform = waveform[:input_len]
    zero_padding = tf.zeros(
        [input_len] - tf.shape(waveform),
        dtype=tf.float32) """
    # Cast the waveform tensors' dtype to float32.
    waveform = tf.cast(waveform, dtype=tf.float32)
    # Concatenate the waveform with `zero_padding`, which ensures all audio
    # clips are of the same length.
    #equal_length = tf.concat([waveform, zero_padding], 0)
    # Convert the waveform to a spectrogram via a STFT.
    #spectrogram = tf.signal.stft(equal_length, frame_length=255, frame_step=128)
    spectrogram = tf.signal.stft(waveform, frame_length=255, frame_step=128)
    # Obtain the magnitude of the STFT.
    spectrogram = tf.abs(spectrogram)
    # Add a `channels` dimension, so that the spectrogram can be used
    # as image-like input data with convolution layers (which expect
    # shape (`batch_size`, `height`, `width`, `channels`).
    spectrogram = spectrogram[..., tf.newaxis]
    return spectrogram

def plot_spectrogram(spectrogram, ax):
    if len(spectrogram.shape) > 2:
        assert len(spectrogram.shape) == 3
        spectrogram = np.squeeze(spectrogram, axis=-1)
    # Convert the frequencies to log scale and transpose, so that the time is
    # represented on the x-axis (columns).
    # Add an epsilon to avoid taking a log of zero.
    log_spec = np.log(spectrogram.T + np.finfo(float).eps)
    height = log_spec.shape[0]
    width = log_spec.shape[1]
    X = np.linspace(0, np.size(spectrogram), num=width, dtype=int)
    Y = range(height)
    ax.pcolormesh(X, Y, log_spec)

spectrogram = get_spectrogram(data)
print(spectrogram.shape)
fig, axes = plt.subplots(2, figsize=(12, 8))
timescale = np.arange(data.shape[0])
axes[0].plot(timescale, data)
axes[0].set_title('Waveform')
#axes[0].set_xlim([0, 16000])

plot_spectrogram(spectrogram.numpy(), axes[1])
axes[1].set_title('Spectrogram')
plt.show()



#ig, ax = plt.subplots()
#mfcc_data = np.swapaxes(mfccs, 0 ,1)
#cax = ax.imshow(mfcc_data, interpolation='nearest', cmap='viridis', origin='lower', aspect='auto')
#ax.set_title('MFCC')
#plt.show()



##### Doing this for every sample ##
def process_audio_data():
    all_data = []

    data_path_dict = {
        0: ["wake_word_dataset/background_sound/" + file_path for file_path in os.listdir("wake_word_dataset/background_sound/")],
        1: ["wake_word_dataset/amogus_audio_data/" + file_path for file_path in os.listdir("wake_word_dataset/amogus_audio_data/")]
    }

    for class_label, list_of_files in data_path_dict.items():
        for single_file in list_of_files:
            audio, sample_rate = librosa.load(single_file) ## Loading file
            mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40) ## Apllying mfcc
            mfcc_processed = np.mean(mfcc.T, axis=0) ## some pre-processing
            #print("mfcc: {}, mean_mfcc: {}".format(mfcc.shape, mfcc_processed.shape))
            all_data.append([mfcc_processed, class_label])
        print(f"Info: Succesfully Preprocessed Class Label {class_label}")

    df = pd.DataFrame(all_data, columns=["feature", "class_label"])

    ###### SAVING FOR FUTURE USE ###
    df.to_pickle("wake_word_dataset/amogus_audio_data.csv")

#process_audio_data()