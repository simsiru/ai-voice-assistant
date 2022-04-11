import tensorflow as tf
from tflite_support import metadata
import json
import librosa
import numpy as np
import os
import sounddevice as sd


def get_labels(model):
    """Returns a list of labels, extracted from the model metadata."""
    displayer = metadata.MetadataDisplayer.with_model_file(model)
    labels_file = displayer.get_packed_associated_file_list()[0]
    labels = displayer.get_associated_file_buffer(labels_file).decode()
    return [line for line in labels.split('\n')]

def get_input_sample_rate(model):
    """Returns the model's expected sample rate, from the model metadata."""
    displayer = metadata.MetadataDisplayer.with_model_file(model)
    metadata_json = json.loads(displayer.get_metadata_json())
    input_tensor_metadata = metadata_json['subgraph_metadata'][0][
            'input_tensor_metadata'][0]
    input_content_props = input_tensor_metadata['content']['content_properties']
    return input_content_props['sample_rate']


""" # Get a WAV file for inference and list of labels from the model
tflite_file = "wwd_model/browserfft_wwd_amogus.tflite"
labels = get_labels(tflite_file)
#random_audio = "wake_word_dataset/test_amogus_2.wav"
random_audio = "wake_word_dataset/test_backround_2.wav"

# Ensure the audio sample fits the model input
interpreter = tf.lite.Interpreter(tflite_file)
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_size = input_details[0]['shape'][1]
sample_rate = get_input_sample_rate(tflite_file)

audio_data, _ = librosa.load(random_audio, sr=sample_rate)
if len(audio_data) < input_size:
  audio_data.resize(input_size)
audio_data = np.expand_dims(audio_data[:input_size], axis=0)

# Run inference
interpreter.allocate_tensors()
interpreter.set_tensor(input_details[0]['index'], audio_data)
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])

# Display prediction and ground truth
top_index = np.argmax(output_data[0])
label = labels[top_index]
score = output_data[0][top_index]
print('---prediction---')
print(f'Class: {label}\nScore: {score}')
#print('----truth----')
#show_sample(random_audio) """


# Get a WAV file for inference and list of labels from the model
tflite_file = "wwd_model/browserfft_wwd_amogus.tflite"
labels = get_labels(tflite_file)

# Ensure the audio sample fits the model input
interpreter = tf.lite.Interpreter(tflite_file)
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_size = input_details[0]['shape'][1]
sample_rate = get_input_sample_rate(tflite_file)

# Run inference
interpreter.allocate_tensors()

rec_seconds = 1

while True:
    try:
        audio_data = sd.rec(int(rec_seconds * sample_rate),
                            samplerate=sample_rate, channels=1)
        sd.wait()
        audio_data = audio_data[:, 0]
        #print(np.max(audio_data), np.min(audio_data), audio_data.shape)

        if len(audio_data) < input_size:
            audio_data.resize(input_size)
        audio_data = np.expand_dims(audio_data[:input_size], axis=0)
        interpreter.set_tensor(input_details[0]['index'], audio_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])

        top_index = np.argmax(output_data[0])
        label = labels[top_index]
        score = output_data[0][top_index]
        
        if top_index == 0 and score*100 > 99:
            print('---HOTWORD DETECTED---')
            #print(f'Class: {label}\nScore: {score}')
            print(f'Score: {score}\nClass:')
            print("\n" +  
                  " ______   __    __   ______   ______   _    _   ______         ______\n" + 
                  "|  __  | |  \  /  | |  __  | |  ____| | |  | | |  ____|       |___   |_\n" + 
                  "| |__| | |   \/   | | |  | | | |  __  | |  | | | |____       |____|  | |\n" + 
                  "|  __  | | |\  /| | | |  | | | | |_ | | |  | | |____  |       |      |_|\n" + 
                  "| |  | | | | \/ | | | |__| | | |__| | | |__| |  ____| |       \______| \n" + 
                  "|_|  |_| |_|    |_| |______| |______| |______| |______|        |_| |_| \n")
        else:
            print('---NO HOTWORD---')

    except KeyboardInterrupt:
        break











#Jetson nano version

""" import tensorflow as tf
import json
import numpy as np
import os
import sounddevice as sd

# Get a WAV file for inference and list of labels from the model
tflite_file = "browserfft_wwd_amogus.tflite"
labels = ['amogus', 'background']

# Ensure the audio sample fits the model input
interpreter = tf.lite.Interpreter(tflite_file)
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_size = input_details[0]['shape'][1]
sample_rate = 44100

# Run inference
interpreter.allocate_tensors()

rec_seconds = 1

while True:
    try:
        audio_data = sd.rec(rec_seconds * sample_rate,
                            samplerate=sample_rate, channels=1)
        sd.wait()
        audio_data = audio_data[:, 0]
        #print(np.max(audio_data), np.min(audio_data), audio_data.shape)

        if len(audio_data) < input_size:
            audio_data.resize(input_size)
        audio_data = np.expand_dims(audio_data[:input_size], axis=0)
        interpreter.set_tensor(input_details[0]['index'], audio_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])

        top_index = np.argmax(output_data[0])
        label = labels[top_index]
        score = output_data[0][top_index]
        
        if top_index == 0 and score*100 > 99:
            print('---prediction---')
            print(f'Class: {label}\nScore: {score}')
        else:
            print('---NO HOTWORD---')

    except KeyboardInterrupt:
        break """