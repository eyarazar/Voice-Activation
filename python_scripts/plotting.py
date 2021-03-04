# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 10:34:54 2020

@author: eyaraz
"""

import librosa
import numpy as np
import scipy
import os
import glob
from pylab import *
import matplotlib.pyplot as plt



# def save_plots_Kaggle(mesure_array, filename, path_to_write, columns, fft_mode=False):
#     samples = mesure_array[0]
#     fig = plt.figure()
#     N = len(samples)
#     for i in range(1,len(mesure_array)):
#         if fft_mode:
#             mesure_array[i] -= mesure_array[i].mean()
#             mesure_array[i] = 1/N * np.abs(scipy.fft(mesure_array[i]))
#         plt.subplot(3, 3, i)
#         plt.plot(samples, mesure_array[i], 'b-', label=columns[i][-1]+'-axis')
#         if fft_mode:
#             plt.ylabel("fft"+columns[i]) 
#         else:
#             plt.ylabel(columns[i]) 
#     fig.tight_layout()
#     fig.savefig(path_to_write+filename+'.png')
#     plt.close(fig)
    

DATA_DIR  = 'train/audio/'
path_to_write = 'plots/'
dirs = [x[0] for x in os.walk(DATA_DIR)]

for path in dirs:
    filename = path.split("/")[-2]
    if filename == '_background_noise_':
        continue
    path = path+"/"
    j=0
    fig = plt.figure()
    for file in glob.glob(path + "*.wav"):
        wav, sample_rate = librosa.load(file, sr = None)
        x = np.arange(0, len(wav)/sample_rate, 1/sample_rate)
        
        if len(x)!=len(wav):
            if len(x)==len(wav)+1:
                x = x[:-1]
            else:
                continue
        
        plt.plot(x, wav)
        fig.tight_layout()
        fig.savefig(path_to_write+filename+'/'+str(j)+'.png')
        fig.clear()
        # pylab.close(fig)
        
        j += 1



  
  
            