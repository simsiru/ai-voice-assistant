#import tensorflow as tf
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import speech_recognition as sr
import io
from pydub import AudioSegment
import torch
import numpy as np

#import librosa

r = sr.Recognizer()

""" MODEL_ID = "facebook/wav2vec2-base-960h"
tokenizer = Wav2Vec2Processor.from_pretrained(MODEL_ID)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)

with sr.Microphone(sample_rate=16000) as source:
    print('Listening...')
    while True:
        audio = r.listen(source)
        data = io.BytesIO(audio.get_wav_data())
        clip = AudioSegment.from_wav(data)
        arr = np.array(clip.get_array_of_samples())
        x = torch.FloatTensor(arr)
        #print(arr, x)

        inputs = tokenizer(x, sampling_rate=16000, return_tensors='pt', padding='longest').input_values
        logits = model(inputs).logits
        tokens = torch.argmax(logits, dim=-1)
        text = tokenizer.batch_decode(tokens)

        print('You said: ', str(text).lower()) """






MODEL_ID = "jonatasgrosman/wav2vec2-large-xlsr-53-english"

processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)

with sr.Microphone(sample_rate=16000) as source:
    print('Listening...')
    while True:
        audio = r.listen(source)
        data = io.BytesIO(audio.get_wav_data())
        clip = AudioSegment.from_wav(data)
        arr = np.array(clip.get_array_of_samples())
        x = torch.FloatTensor(arr)

        inputs = processor(x, sampling_rate=16_000, return_tensors="pt", padding=True)

        with torch.no_grad():
            logits = model(inputs.input_values, attention_mask=inputs.attention_mask).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        predicted_sentences = processor.batch_decode(predicted_ids)

        print('You said: ', str(predicted_sentences).lower())