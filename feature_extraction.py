import librosa
import csv
import os

folder_path = './wav'

def local_extraction():
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    for file in files:
        print(file)


