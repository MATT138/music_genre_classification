import librosa
import csv
import os
import pandas as pd
import boto3
folder_path = './wav'
num_segment=1
num_mfcc=20
sample_rate=22050
n_fft=2048
hop_length=512

def divide_audio(y, sr):
    segment_duration = 3
    segments = []
    total_duration = librosa.get_duration(y=y, sr=sr)
    # Calculate the number of complete segments
    num_segments = int(total_duration // segment_duration)

    # Iterate over the segments and save each segment
    for i in range(num_segments):
        # Calculate start and end time for each segment
        start_time = i * segment_duration
        end_time = (i + 1) * segment_duration
        segment = y[int(start_time * sr):int(end_time * sr)]
        segments.append(segment)
    return segments


def local_extraction():
    my_csv={"filename":[], "chroma_stft_mean": [], "chroma_stft_var": [], "rms_mean": [], "rms_var": [], "spectral_centroid_mean": [],
        "spectral_centroid_var": [], "spectral_bandwidth_mean": [], "spectral_bandwidth_var": [], "rolloff_mean": [], "rolloff_var": [],
        "zero_crossing_rate_mean": [], "zero_crossing_rate_var": [], "harmony_mean": [], "harmony_var": [], "mfcc1_mean": [],
        "mfcc1_var" : [], "mfcc2_mean" : [], "mfcc2_var" : [], "mfcc3_mean" : [], "mfcc3_var" : [], "mfcc4_mean" : [], "mfcc4_var" : [],
        "mfcc5_mean" : [], "mfcc5_var" : [], "mfcc6_mean" : [], "mfcc6_var" : [], "mfcc7_mean" : [], "mfcc7_var" : [],
        "mfcc8_mean" : [], "mfcc8_var" : [], "mfcc9_mean" : [], "mfcc9_var" : [], "mfcc10_mean" : [], 
        "mfcc10_var" : [], "mfcc11_mean" : [], "mfcc11_var" : [], "mfcc12_mean" : [], "mfcc12_var" : [], 
        "mfcc13_mean" : [], "mfcc13_var" : [], "mfcc14_mean" : [], "mfcc14_var" : [], "mfcc15_mean" : [], 
        "mfcc15_var" : [], "mfcc16_mean" : [], "mfcc16_var" : [], "mfcc17_mean" : [], "mfcc17_var" : [], 
        "mfcc18_mean" : [], "mfcc18_var" : [], "mfcc19_mean" : [], "mfcc19_var" : [], "mfcc20_mean" : [], 
        "mfcc20_var":[], "label":[]}
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    for file in files:
        
        try:
            y, sr = librosa.load(rf"{folder_path}\{file}", sr=sample_rate)
            
            fname = file.split('.')[0]+file.split('.')[1]
            print(fname)
            label = file.split('.')[0]
            
            segments = divide_audio(y=y, sr=sr)
            for segment in segments:
                #Chromagram
                chromagram = librosa.feature.chroma_stft(y=segment, sr=sample_rate, hop_length=hop_length)    
                my_csv["chroma_stft_mean"].append(chromagram.mean())
                my_csv["chroma_stft_var"].append(chromagram.var())
                
                #Root Mean Square Energy
                RMSEn= librosa.feature.rms(y=segment)
                my_csv["rms_mean"].append(RMSEn.mean())
                my_csv["rms_var"].append(RMSEn.var())
                
                #Spectral Centroid
                spec_cent=librosa.feature.spectral_centroid(y=segment)
                my_csv["spectral_centroid_mean"].append(spec_cent.mean())
                my_csv["spectral_centroid_var"].append(spec_cent.var())
                
                #Spectral Bandwith
                spec_band=librosa.feature.spectral_bandwidth(y=segment,sr=sample_rate)
                my_csv["spectral_bandwidth_mean"].append(spec_band.mean())
                my_csv["spectral_bandwidth_var"].append(spec_band.var())

                #Rolloff
                spec_roll=librosa.feature.spectral_rolloff(y=segment,sr=sample_rate)
                my_csv["rolloff_mean"].append(spec_roll.mean())
                my_csv["rolloff_var"].append(spec_roll.var())
                
                #Zero Crossing Rate
                zero_crossing=librosa.feature.zero_crossing_rate(y=segment)
                my_csv["zero_crossing_rate_mean"].append(zero_crossing.mean())
                my_csv["zero_crossing_rate_var"].append(zero_crossing.var())
                
                #Harmonics and Perceptrual 
                harmony, _ = librosa.effects.hpss(y=segment)
                my_csv["harmony_mean"].append(harmony.mean())
                my_csv["harmony_var"].append(harmony.var())
                
                my_csv["filename"].append(fname)
                my_csv["label"].append(label)

                mfcc=librosa.feature.mfcc(y=segment,sr=sample_rate, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
                mfcc=mfcc.T
                for x in range(20):
                    feat1 = "mfcc" + str(x+1) + "_mean"
                    feat2 = "mfcc" + str(x+1) + "_var"
                    my_csv[feat1].append(mfcc[:,x].mean())
                    my_csv[feat2].append(mfcc[:,x].var())
        except Exception as e:
            print(e)
            # In case of corrupted audio file
            continue

    # for key, value in my_csv.items():
    #     print(f"Length of {key}: {len(value)}")
    df = pd.DataFrame(my_csv)
    df = df.dropna()
    df.to_csv('myfeatures.csv', index=False)
    
def extract_to_cloud():
    local_extraction()
    s3 = boto3.client('s3')
    file_path = "myfeatures.csv"
    bucket_name = "music-classification-project"

    try:
        s3.upload_file(file_path, bucket_name, "myfeatures.csv")
        print("uploaded")
    except Exception as e:
        print(f"Error uploading file to S3: {e}")



