import librosa
import numpy as np
import soundfile
import math


def delete_silence(pcm, min_size):
    new_pcm = []
    for i in range(0, len(pcm), min_size):
        frame = pcm[i:i+min_size]
        #print(frame.shape[0])
        rms = math.sqrt(sum(frame**2)) / len(frame)
        #print(rms)
        if rms > 0.00001:
            new_pcm.append(frame)
    return np.array(new_pcm).flatten()

test_file, sr = librosa.load("test_file.wav")

new_test = delete_silence(test_file, 44100*2)
print(new_test.shape)

soundfile.write("new_test_file.wav", new_test, sr)