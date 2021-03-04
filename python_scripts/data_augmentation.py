# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 10:17:35 2021

@author: eyaraz
"""


import numpy as np
import glob
import os
import librosa
import soundfile
import random

def adding_background_noise(wav,noise):
    if len(wav)>len(noise):
        print("Length Error: the noise length maust be bigger than signal length.")
        wav_plus_noise = wav
        wav_plus_noise[:len(noise)] += noise
    else:
        delta_L = len(noise) - len(wav)
        index = random.sample(range(0,delta_L),0)[0]        
        wav_plus_noise = wav + noise[index:index + len(wav)]
    return wav_plus_noise

def adding_gaussian_noise(wav,snr):
    energy = (np.linalg.norm(wav))**2
    sigma = energy / snr
    noise = np.random.randn(wav.shape[0],wav.shape[1],0)
    wav_plus_noise = wav + noise
    return wav_plus_noise


"""
L_record = 2_sec

method 1: adding background_noise     VVV
for noise in background_noise:
    L_noise = len(noise)
    for record in records:
        interval_noise = random(L_noise, L_record)
        temp = record + noise[interval_noise]
        write(temp, path_record)

method 2: shift right/left randomlly
 
method 3 : gaussian white noise

"""



DATA_DIR  = 'cnn_1d/data_wav/'
noise_background_path = 'Data/audio_data/audio/_background_noise_/'


dirs = [x[0] for x in os.walk(DATA_DIR)]
path_to_write = 'cnn_1d/data_augmentation/'

for noise_file in glob.glob(noise_background_path + "*.wav"):
    noise, sample_rate_noise =  librosa.load(noise_file, sr = None)
    for path in dirs:
        filename = path.split("/")[-1]
        path = path+"/"
        j=0
        for file in glob.glob(path + "*.wav"):
            wav, sample_rate = librosa.load(file, sr = None)
            wav_plus_noise = adding_background_noise(wav,noise)
            soundfile.write(path_to_write+filename+'/'+str(j)+'.wav', wav, sample_rate)
            j +=1
            soundfile.write(path_to_write+filename+'/'+str(j)+'.wav', wav_plus_noise, sample_rate)
            j += 1
            
            
            
            
            
