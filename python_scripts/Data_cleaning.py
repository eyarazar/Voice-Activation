# -*- coding: utf-8 -*-
"""
Created on Thu Dec 31 16:45:09 2020

@author: eyaraz
"""
import soundfile
import librosa
import numpy as np
# import scipy
import os
import glob
import matplotlib.pyplot as plt
import pandas as pd 

def convert_to_32k(wav, filename, signal_len = int(32e+3), amp_thr = 0.03):
    
    flag = False
    if len(wav)<signal_len:
        z = np.zeros((signal_len-len(wav),1))
        wav = np.reshape(wav, (len(wav),1))
        wav = np.concatenate((wav, z))
    
    else: # len(wav) > 32K
        thrs = [amp_thr, amp_thr +0.01, amp_thr +0.02, amp_thr +0.03, amp_thr +0.04, amp_thr +0.05]
        for thr in thrs:
            delta = len(wav) - signal_len
            index_start  = np.argwhere(abs(wav)>thr)[10][0]
            if index_start>100:
                index_start-=100
            index_end  = np.argwhere(abs(wav)>thr)[-10][0]
            if len(wav)-index_end>100:
                index_start+=100
            if index_start > delta: 
                wav = wav[delta:]
                flag = False
                break
            elif (index_start + len(wav)-index_end) > delta:
                wav = wav[index_start:]
                delta = len(wav) - signal_len
                wav = wav[:-delta]
                flag = False
                break
            else:
                wav = wav[index_start:index_end]
                flag = True
    if flag:
        print("The file {} still longer than 32K".format(filename))    
    return wav, flag

mode  = 5
DATA_DIR  = 'train/audio/'
dirs = [x[0] for x in os.walk(DATA_DIR)]

if mode == 1:  #Plotting
    path_to_write = 'clean_plots/'
    for path in dirs:
        filename = path.split("/")[-1]
        if filename == '_background_noise_':
            continue
        if filename == 'bed':
            continue
        if filename == 'bird':
            continue
        path = path+"/"
        j=0
        fig = plt.figure()
        for file in glob.glob(path + "*.wav"):
            wav, sample_rate = librosa.load(file, sr = None)
            wav, star_flag = convert_to_32k(wav, filename = filename)
            x = np.arange(0, len(wav)/sample_rate, 1/sample_rate)
            
            if len(x)!=len(wav):
                if len(x)==len(wav)+1:
                    x = x[:-1]
                else:
                    continue
            
            plt.plot(x, wav)
            fig.tight_layout()
            if star_flag:
                fig.savefig(path_to_write+filename+'/'+str(j)+'EEE.png')
            else:
                fig.savefig(path_to_write+filename+'/'+str(j)+'.png')
            fig.clear()
            # pylab.close(fig)
            
            j += 1
        
if mode == 2:  # Writing as .wav
    path_to_write = 'train/audio_2sec/'
    for path in dirs:
        filename = path.split("/")[-1]
        if filename == '_background_noise_':
            continue
        path = path+"/"
        j=0
        for file in glob.glob(path + "*.wav"):
            wav, sample_rate = librosa.load(file, sr = None)
            wav, star_flag = convert_to_32k(wav, filename = filename)
                        
            if star_flag:
                soundfile.write(path_to_write+filename+'/'+str(j)+'EEE.wav', wav, sample_rate)
            else:
                soundfile.write(path_to_write+filename+'/'+str(j)+'.wav', wav, sample_rate)
                    
            j += 1
        
if mode == 3:  # Writing as .csv
    path_to_write = 'data_csv/'
    for path in dirs:
        filename = path.split("/")[-1]
        if filename == '_background_noise_':
            continue
        path = path+"/"
        j=0
        for file in glob.glob(path + "*.wav"):
            wav, sample_rate = librosa.load(file, sr = None)
            wav, star_flag = convert_to_32k(wav, filename = filename)
            df = pd.DataFrame(wav)          
            if star_flag:
                df.to_csv(path_to_write+filename+'/'+str(j)+"EEE.csv")
                
            else:
                df.to_csv(path_to_write+filename+'/'+str(j)+".csv")
                    
            j += 1
            

if mode == 4: # create spectrogram images
    
    from scipy.signal import spectrogram    
    DATA_DIR  = 'audio_data/audio_2sec/'
    dirs = [x[0] for x in os.walk(DATA_DIR)]
    path_to_write = 'data_spec_images/'
    
    j=0
    for path in dirs:
        filename = path.split("/")[-1]
        if filename == 'call_911':
            label = 'call_911'
        else:
            label = 'others'
        path = path+"/"
        fig = plt.figure()
        for file in glob.glob(path + "*.wav"):
            wav, sample_rate = librosa.load(file, sr = None)
            x = np.arange(0, len(wav)/sample_rate, 1/sample_rate)
            
            if len(x)!=len(wav):
                print("The length of the record {} {} is not 2 sec ".format(filename, j))
                continue
            
            # f, t, spec = spectrogram(wav, sample_rate)
            noise_power = 0.000001
            noise = np.random.normal(scale=np.sqrt(noise_power), size=x.shape)
            
            
            plt.specgram(wav+noise, Fs = sample_rate)
            plt.axis('off')
            fig.tight_layout()
            plt.savefig(path_to_write+label+'/'+str(j)+".png", bbox_inches='tight',pad_inches = 0)
            
            fig.clear()
            j+=1
            
if mode == 5: # create train val test sets of spectrogram images 
    # import soundfile
    # import librosa
    # import pandas as pd 
    import os
    import glob
    import random
    import cv2
    
    training_set = 4000
    
    mode =2 
    i=0
    
  
    DATA_DIR  = 'Data/spec_images/others/'
    dirs = [x[0] for x in os.walk(DATA_DIR)]
    path_to_write = 'spectrogram_cnn/others/'

    for path in dirs:
        indcies = random.sample(range(0, 64000), 4000)
        filename = path.split("/")[-1]
        if filename == 'call_911':
            continue
        path = path+"/"
        j = 0
        for file in glob.glob(path + "*.png"):
            if j in indcies:
                spec = cv2.imread(file)
                cv2.imwrite(path_to_write+str(i)+".png", spec)
                # df = pd.read_csv(file)
                # df.to_csv(path_to_write+str(i)+".csv")
                i += 1                 
            j += 1
         
    
    
    # import os, os.path, shutil
    # from sklearn.model_selection import train_test_split
    
    # indcies = np.arange(3990)
    # train_idx, test_idx  = train_test_split(indcies, test_size=0.2, random_state=777,shuffle=True)
    # train_idx, val_idx  = train_test_split(train_idx, test_size=0.25, random_state=777,shuffle=True)
    
    # folder_path = 'spectrogram_cnn/call_911/'
    # images = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    # i=0
    # for image in images:
        
    #     if i in train_idx:
    #         folder_name = "train/call_911/"
    #     elif i in val_idx:
    #         folder_name = "val/call_911/"
    #     else:
    #         folder_name = "test/call_911/"
        
    #     new_path = os.path.join('spectrogram_cnn/', folder_name)
    #     # if not os.path.exists(new_path):
    #     #     os.makedirs(new_path)
    
    #     old_image_path = os.path.join(folder_path, image)
    #     new_image_path = os.path.join(new_path, image)
    #     shutil.move(old_image_path, new_image_path)
    #     i+=1
    
    
    # else: #wav    
    #     DATA_DIR  = 'audio_data/audio_2sec/'
    #     dirs = [x[0] for x in os.walk(DATA_DIR)]
    #     path_to_write = 'data_wav/others/'
        
    #     for path in dirs:
    #         if path.split("/")[-1] == 'call_911':
    #             continue
    #         indcies = random.sample(range(0, 1700), 133)
    #         filename = path.split("/")[-1]
    #         if filename == 'call_911':
    #             continue
    #         path = path+"/"
    #         j = 0
    #         for file in glob.glob(path + "*.wav"):
    #             if j in indcies:
    #                 wav, sample_rate = librosa.load(file, sr = None)
    #                 soundfile.write(path_to_write+str(i)+'.wav', wav, sample_rate)
    #                 i += 1            
    #             j += 1
                











