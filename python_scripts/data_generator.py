# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 12:08:29 2021

@author: eyaraz
"""
import librosa
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split 

class data_generator:

    def __init__(self, data_path, labels, d_s_factor = 0.5):
        self.data_path = data_path
        self.labels = labels
        self.d_s_factor = d_s_factor
        
        
    def data_generator(self):
        all_wave = []
        all_label =[]        
        for label in self.labels:
            waves = [f for f in os.listdir(self.data_path + '/'+ label) if f.endswith('.wav')]
            for wav in waves:
                samples, sample_rate = librosa.load(self.data_path + '/' + label + '/' + wav, sr = 16000)
                # print(len(samples))
                samples = librosa.resample(samples, sample_rate, int(sample_rate * self.d_s_factor))
                if(len(samples)== sample_rate * self.d_s_factor * 2) : 
                    all_wave.append(samples)
                    all_label.append(label)

        len_samples = int(sample_rate * self.d_s_factor * 2)
        label_enconder = LabelEncoder()
        if len(self.labels)==2:
            y = label_enconder.fit_transform(all_label)
        else: # more than 2 labels
            classes = list(label_enconder.classes_)
            y = to_categorical(y, num_classes=len(self.labels))           

        X = np.array(all_wave).reshape(-1,len_samples,1)

        ######## Spliting the data ########
        X_train, X_test, y_train, y_test = train_test_split(X, np.array(y),stratify=y, test_size=0.2, random_state=777,shuffle=True)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1) # 0.25 x 0.8 = 0.2

        return X_train, y_train, X_val, y_val, X_test, y_test


class data_augmentation:
    
    def __init__(self, X_train, y_train):
        # add_noise_flag, ww_flag, shift_flag
        self.X_train = X_train
        self.y_train = y_train
        # self.ww_flag = ww_flag  
        # self.shift_flag = shift_flag
        self.num_feature = X_train.shape[2]
    
    
    def adding_gaussian_noise(self, mu, sigma):    
        X_new = self.X_train
        noise =  np.random.normal(mu, sigma, self.X_train.shape)
        X_new = np.vstack((X_new, self.X_train + noise))
        y_new = np.concatenate((self.y_train,self.y_train))
        self.X_train = X_new 
        self.y_train = y_new
        return X_new, y_new
    
        
    def shift_side(self, shift_factors = [50, 100, 150]):
        ## nee
        X_new = self.X_train
        y_new = self.y_train.copy()
        N,h,c = self.X_train.shape
        for shift_factor in shift_factors:
            first_row, last_row = self.X_train[:,0,:], self.X_train[:,-1,:]
            first_row = np.repeat(first_row[:, :, np.newaxis], step, axis=2).transpose(0,2,1)
            last_row = np.repeat(last_row[:, :, np.newaxis], step, axis=2).transpose(0,2,1)
            X_shift_right = shift(self.X_train,(0, step, 0), cval = 0)
            X_shift_right[: , :step, :] = first_row
            X_shift_left = shift(self.X_train,(0, -step, 0), cval = 0)
            X_shift_left[: , -step:, :] = last_row
            X_new = np.vstack((X_new, X_shift_left, X_shift_right))
            y_new += self.y_train
            y_new += self.y_train
        self.X_train = X_new 
        self.y_train = y_new        
        return X_new, y_new    
    
    
    
    
    
    
    
    
    
    
    
    
    
