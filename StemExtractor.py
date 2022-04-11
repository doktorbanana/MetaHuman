import time
import librosa
import soundfile as sf
from spleeter.separator import Separator
from spleeter.audio.adapter import AudioAdapter
import glob
import numpy as np


def _check_rms(data, path="Unkown"):
    rms = librosa.feature.rms(y=data, frame_length=data.shape[0])
    if rms[0][0] > 0.00000001:
        print("RMS of song " + str(path) + " is: " + str(rms[0][0]))
        return 0
    else:
        print("I don't think there were vocals in the song: " + str(path))
        return 1  # The RMS Level is very low -> probably the song didn't have vocals


if __name__ == '__main__':
    separator = Separator('spleeter:2stems')

    list_of_files = glob.glob('demo_data\\YouTubeSongs\\*')
   # print(list_of_files)
    start_time = time.time()

    audio_loader = AudioAdapter.default()
    sample_rate = 44100

    # Perform the Separation
    for i,file in enumerate(list_of_files):
        waveform, _ = audio_loader.load(file, sample_rate=sample_rate)
        prediction = separator.separate(waveform)
        vocals = prediction["vocals"]
        empty = _check_rms(vocals, file)
        save_path = "demo_data\\stems\\voclas" + str(i) +".wav"
        if empty<1:
            sf.write(file=save_path, data=vocals, samplerate=44100)
        print(str(i+1) + " of " + str(len(list_of_files)) +" Songs done")
    end_time = time.time()

    print("took me " + str(end_time-start_time) + " seconds to finish " + str(len(list_of_files)) + " songs")



