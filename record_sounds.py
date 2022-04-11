#### IMPORTS ####################
import sounddevice as sd
from scipy.io.wavfile import write
from pydub import AudioSegment
import librosa
import soundfile
import glob

def record_audio_and_save(save_path, n_times=50):
    """
    This function will run `n_times` and everytime you press Enter you have to speak the wake word

    Parameters
    ----------
    n_times: int, default=50
        The function will run n_times default is set to 50.

    save_path: str
        Where to save the wav file which is generated in every iteration.
    """

    input("To start recording Wake Word press Enter: ")
    for i in range(0, n_times):
        fs = 44100
        seconds = 1

        myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
        sd.wait()
        write(save_path + str(i) + ".wav", fs, myrecording)
        input(f"Press to record next or two stop press ctrl + C ({i + 1}/{n_times}): ")

def record_background_sound(save_path, n_times=50):
    """
    This function will run automatically `n_times` and record your background sounds so you can make some
    keybaord typing sound and saying something gibberish.
    Note: Keep in mind that you DON'T have to say the wake word this time.

    Parameters
    ----------
    n_times: int, default=50
        The function will run n_times default is set to 50.

    save_path: str
        Where to save the wav file which is generated in every iteration.
        Note: DON'T set it to the same directory where you have saved the wake word or it will overwrite the files.
    """

    input("To start recording your background sounds press Enter: ")
    for i in range(0, n_times):
        fs = 44100
        seconds = 1 

        myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
        sd.wait()
        write(save_path + str(i) + ".wav", fs, myrecording)
        print(f"Currently on {i+1}/{n_times}")

def split_audio_file_into_multiple(original_file_path, new_file_folder_path, duration=2000, index_offset=0):
    data = AudioSegment.from_wav(original_file_path)
    print(len(data))
    idx = 0
    for i in range(int(len(data) / duration)):
        if duration >= len(data):
            new_data = data[idx:]
        else:
            new_data = data[idx:idx + duration]

        idx += duration

        new_data.export(new_file_folder_path + str(i + index_offset) + '.wav', format="wav") #Exports to a wav file in the current path.

def convert_encoding(new_dir = "wake_word_dataset/"):
    for i, file in enumerate(glob.glob("wake_word_dataset/*.wav")):
        data, samplerate = soundfile.read(file)
        soundfile.write(new_dir + str(i) + "_.wav", data, samplerate, subtype='PCM_16')
        #print(i, file)

# Step 1: Record yourself saying the Wake Word
#print("Recording the Wake Word:\n")
#record_audio_and_save("wake_word_dataset/amogus_audio_data/", n_times=200)
#record_audio_and_save("wake_word_dataset/", n_times=200) 

# Step 2: Record your background sounds (Just let it run, it will automatically record)
#print("Recording the Background sounds:\n")
#record_background_sound("wake_word_dataset/background_sound/", n_times=400)


#split_audio_file_into_multiple("google_speech_commands_dataset/_background_noise_/white_noise.wav", "wake_word_dataset/background_sound/", duration=2000, index_offset=327)

""" sample = "wake_word_dataset/amogus_audio_data/amogus/140.wav"
data, sample_rate = librosa.load(sample)
print(data.shape)
print(sample_rate) """


convert_encoding()