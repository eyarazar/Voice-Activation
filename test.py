# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 10:56:17 2021

@author: eyaraz
"""


import librosa
import numpy as np
import os

def load_data(data_path):
    ####
    # need to edit
    ####
    X_test =[]
    y_test =[]
    return X_test, y_test


# method = "cnn_1d"   #  "cnn_1d", "spec"
model_path = "speech2text_model.hdf5" # speech2text_model.hdf5, spec_clf.hdf5
test_set_path =  "test_1d"   # "" , "spectrogram_cnn/test/"

X_test, y_test = load_data(test_set_path)

if model_path.split(".")[-1] == "hdf5": # Keras model
    from tensorflow.keras.models import load_model
    model = load_model('speech2text_model.hdf5')
    prob = model.predict(X_test)
    prob[prob>0.5] = 1
    prob[prob<1] = 0
    y_pred = np.reshape(prob,len(prob))
    error = abs(y_pred-y_test)
    acc = 1- error.sum()/len(error)
    print("Test accuracy: {}".format(acc))

elif model_path.split(".")[-1] == "tflite": # Tensorflow lite model
    from tensorflow.lite import Interpreter
    interpreter = Interpreter(model_path="converted_model.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    input_index = interpreter.get_input_details()[0]["index"]
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]['shape']
    acc = 0
    for i in range(len(X_test)):
        x = X_test[i].reshape(input_shape)
        interpreter.set_tensor(input_details[0]['index'], x)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        print(output_data)
        print(y_test[i])
        if(abs(output_data-y_test[i])<0.5):
            acc+=1
    acc = acc/len(X_test)
    print("Test accuracy: {}".format(acc))
