import time
import librosa
import soundfile as sf
from spleeter.separator import Separator
from spleeter.audio.adapter import AudioAdapter
import glob
import numpy as np
import os
import tensorflow as tf


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



