import time
import librosa
import soundfile as sf
from spleeter.separator import Separator
import glob


if __name__ == '__main__':

    separator = Separator('spleeter:2stems')
    path = 'H:\Musik\Peter\The Beatles\**\*.mp3'

    list_of_files = glob.glob(path, recursive=True)
    print(len(list_of_files))

    for i, file in enumerate(list_of_files):
        print(file)
        separator.separate_to_file(file,
                                   "D:\Daten\Studium\Semester_8\MetaHuman\Stems",
                                   filename_format="{instrument}/{instrument}" + str(i) + ".{codec}", synchronous=False)
    separator.join()



