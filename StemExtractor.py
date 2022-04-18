import time
import librosa
import soundfile as sf
from spleeter.separator import Separator
from spleeter.audio.adapter import AudioAdapter
import glob
import numpy as np
import os


def _check_rms(data, path="Unkown"):
    rms = librosa.feature.rms(y=data, frame_length=data.shape[0])
    if rms[0][0] > 0.00000001:
        return 0
    else:
        return 1  # The RMS Level is very low -> probably the song didn't have vocals


if __name__ == '__main__':
    separator = Separator('spleeter:4stems')
    path = os.path.join('/Volumes/LaptopData/FynnFrei/ML_Data/YoutubeSongs2', "*")
    list_of_files = glob.glob(path)
    start_time = time.time()

    audio_loader = AudioAdapter.default()
    sample_rate = 44100

    # Perform the Separation
    for i,file in enumerate(list_of_files):
        waveform, _ = audio_loader.load(file, sample_rate=sample_rate)
        prediction = separator.separate(waveform)
        vocals = prediction["vocals"]
        bass = prediction["bass"]
        drums = prediction["drums"]

        empty = _check_rms(vocals, file)
        save_path = os.path.join("demo_data/stems2/vocals", "vocals" + str(i) +".wav")
        if empty<1:
            sf.write(file=save_path, data=vocals, samplerate=44100)

        empty = _check_rms(bass, file)
        save_path = os.path.join("demo_data/stems2/bass", "bass" + str(i) +".wav")
        if empty<1:
            sf.write(file=save_path, data=bass, samplerate=44100)

        empty = _check_rms(drums, file)
        save_path = os.path.join("demo_data/stems2/drums", "drums" + str(i) + ".wav")
        if empty < 1:
            sf.write(file=save_path, data=drums, samplerate=44100)



        print(str(i+1) + " of " + str(len(list_of_files)) +" Songs done")


    end_time = time.time()

    print("took me " + str(end_time-start_time) + " seconds to finish " + str(len(list_of_files)) + " songs")



