import time
import librosa
import soundfile as sf
from spleeter.separator import Separator
from spleeter.audio.adapter import AudioAdapter
import glob
import numpy as np
import os
import tensorflow as tf




def _check_rms(data, path="Unkown"):
    rms = librosa.feature.rms(y=data, frame_length=data.shape[0])
    if rms[0][0] > 0.00000001:
        return 0
    else:
        return 1  # The RMS Level is very low -> probably the song didn't have vocals


if __name__ == '__main__':
    #print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    num = 50
    step_size = 450

    separator = Separator('spleeter:4stems')
    path = os.path.join('E:/FynnFrei/ML_Data/YoutubeSongs', "*")
    list_of_files = glob.glob(path)
    start_time = time.time()

    list_of_files = list_of_files[num:num+step_size]

    for i,file in enumerate(list_of_files):
        separator.separate_to_file(file,
                                   "E:/FynnFrei/ML_Data/stems",
                                   filename_format="{instrument}/{instrument}" + str(i+num) + ".{codec}",
                                   synchronous=False)
    separator.join()

    # Perform the Separation
    #for i,file in enumerate(list_of_files):
        # waveform, _ = audio_loader.load(file, sample_rate=sample_rate)
        # prediction = separator.separate()
        # vocals = prediction["vocals"]
        # bass = prediction["bass"]
        # drums = prediction["drums"]

        # empty = _check_rms(vocals, file)
        # save_path = os.path.join("E:/FynnFrei/ML_Data/stems/vocals", "vocals" + str(i + (num)) +".wav")
        # if empty<1:
        #     sf.write(file=save_path, data=vocals, samplerate=44100)
        #
        # empty = _check_rms(bass, file)
        # save_path = os.path.join("E:/FynnFrei/ML_Data/stems/bass", "bass" + str(i + num) +".wav")
        # if empty<1:
        #     sf.write(file=save_path, data=bass, samplerate=44100)
        #
        # empty = _check_rms(drums, file)
        # save_path = os.path.join("E:/FynnFrei/ML_Data/stems/drums", "drums" + str(i + num) + ".wav")
        # if empty < 1:
        #     sf.write(file=save_path, data=drums, samplerate=44100)



        #print(str(i+1) + " of " + str(len(list_of_files)) +" Songs done")


    #end_time = time.time()

    #print("took me " + str(end_time-start_time) + " seconds to finish " + str(len(list_of_files)) + " songs")



