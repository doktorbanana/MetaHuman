from spleeter.separator import Separator
import glob, os


if __name__ == "main":
    separator = Separator('spleeter:2stems')

    list_of_files = glob.glob('demo_data\*')
    print(list_of_files)

    for file in list_of_files:
        separator.separate_to_file(file, 'demo_data\stems')
