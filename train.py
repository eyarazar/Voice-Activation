# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 16:19:33 2020

@author: eyaraz
"""

import librosa
import numpy as np
import os

##### main ####
method = "cnn_1d"   #  "cnn_1d", "spec"
labels = ["others", "call_911"]
augmentaion = True
quantization = False
snr = 3.16 # 5db
######## Reading the data ########
if method == "cnn_1d":
    from cnn_1d.data_generator import data_generator
    data_path = "cnn_1d/data_wav/"
    dg = data_generator(data_path, labels, d_s_factor=0.5)
    X_train, y_train, X_val, y_val, X_test, y_test = dg.data_generator()
    Energy = ((np.linalg.norm(X_train,axis=1)**2)/len(X_train)).mean()
    if augmentaion:
        from cnn_1d.data_generator import data_augmentation
        da = data_augmentation(X_train,y_train)
        X_train, y_train = da.adding_gaussian_noise(mu=0, sigma = np.sqrt(Energy/snr))
        da = data_augmentation(X_val,y_val)
        X_val, y_val = da.adding_gaussian_noise(mu=0, sigma = np.sqrt(Energy/snr))
elif method == "spec":
    
    data_path = "spectrogram_cnn/"
    h,w = 128, 128

    # test_datagen = ImageDataGenerator(rescale = 1./255)    
    
######## Training ########
if method == "cnn_1d":
    from models.cnn_1d_classifier import cnn_1d_classifier
    clf = cnn_1d_classifier(X_train, y_train, X_val, y_val)
    classifier, early_stop, checkpoint = clf.build(filters=[2,2], kernels=[3,3], dense_units=[8])
    clf.fit(classifier,early_stop, checkpoint, epochs=10,batch_size=256)

elif method == "spec":
    from models.spec import spec_classifier
    h,w = 128, 128
    spec_clf = spec_classifier(data_path+'train', data_path+'val',h,w)
    classifier, early_stop, checkpoint = spec_clf.build()
    spec_clf.fit(classifier, early_stop, checkpoint, epochs = 10)    

######## Test ########

# prob = classifier.predict(X_test)
E = ((X_test**2).mean(axis=1)).mean()
snrs = [3.16, 10, 100,1000]
for snr in snrs:
    sigma = np.sqrt(E/snr)
    n = sigma * np.random.randn(X_test.shape[0], X_test.shape[1], 1)
    X_test_new = X_test + n
    prob = classifier.predict(X_test_new)
    prob[prob>0.5] = 1
    prob[prob<1] = 0
    y_pred = np.reshape(prob,len(prob))
    error = abs(y_pred-y_test)
    acc = 1- error.sum()/len(error)
    print("Test accuracy: {}".format(acc))

# from tensorflow.keras.models import load_model
# model = load_model('speech2text_model.hdf5')

DATA_DIR  = 'audio_data/audio_2sec/'
dirs = [x[0] for x in os.walk(DATA_DIR)]
for path in dirs:
        filename = path.split("/")[-1]
        if filename == 'call_911':
            continue
        path = path+"/"
        j=0
        for file in glob.glob(path + "*.wav"):
            wav, sample_rate = librosa.load(file, sr = None)
            
                    
            j += 1


#############  quantization  ################
if quantization:
    import tensorflow as tf
    Model_Path = "speech2text_model.hdf5"
    converter = tf.lite.TFLiteConverter.from_keras_model_file(Model_Path)
    converter.allow_custom_ops = False
    converter.post_training_quantize = True
    tflite_model = converter.convert()
    open("converted_model.tflite", "wb").write(tflite_model)
    
    interpreter = tf.lite.Interpreter(model_path="converted_model.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    input_index = interpreter.get_input_details()[0]["index"]
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]['shape']
    
    # import cv2
    # img = cv2.imread("Data/spec_images/call_911/3444.png")
    # img = cv2.imread("Data/spec_images/others/0.png")
    # img = cv2.resize(img, (128,128))
    # img = np.reshape(img,(1,128,128,3))
    # img = img.astype(np.float32)
    # interpreter.set_tensor(input_details[0]['index'], img)
    # output_data = interpreter.get_tensor(output_details[0]['index'])
    
    
    acc = 0
    for i in range(len(X_test)):
        # x = np.reshape(X_test[i],(1,len(X_test[i]),1))
        x = X_test[i].reshape(input_shape)
        interpreter.set_tensor(input_details[0]['index'], x)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        print(output_data)
        print(y_test[i])
        if(abs(output_data-y_test[i])<0.5):
            acc+=1
    acc = acc/len(X_test)
    print(acc*100)





    