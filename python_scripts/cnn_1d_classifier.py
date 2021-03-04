# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 21:26:18 2021

@author: eyaraz
"""


from tensorflow.keras.layers import Bidirectional, BatchNormalization, GRU, TimeDistributed
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv1D, Input, MaxPooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import optimizers
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt

class cnn_1d_classifier:
    
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
                
        
    def build(self, filters=[8,16], kernels=[11,9], dense_units=[256], dropout = 0.3):    
        
        signal_len = self.X_train.shape[1]
        inputs = Input(shape=(signal_len,1))
        # len_labels = self.y_train.shape[1]
        cnn_layers, dense_layers=len(filters), len(dense_units)        
        K.clear_session()
        inputs = Input(shape=(signal_len,1))
        
        x = BatchNormalization(axis=-1, momentum=0.99, epsilon=1e-3, center=True, scale=True)(inputs)
        
        for i in range(cnn_layers):
            if i == 0:
                #First Conv1D layer
                x = Conv1D(filters[i],kernels[i], padding='valid', activation='relu', strides=1)(x)
                x = MaxPooling1D(3)(x)
                x = Dropout(dropout)(x)
            else:
                #Second Conv1D layer
                x = Conv1D(filters[i],kernels[i], padding='valid', activation='relu', strides=1)(x)
                x = MaxPooling1D(3)(x)
                x = Dropout(dropout)(x)
        
                # #Third Conv1D layer
                # x = Conv1D(16, 7, padding='valid', activation='relu', strides=1)(x)
                # x = MaxPooling1D(3)(x)
                # x = Dropout(0.3)(x)
        
        # x = BatchNormalization(axis=-1, momentum=0.99, epsilon=1e-3, center=True, scale=True)(x)

        #Flatten layer
        x = Flatten()(x)
        #Dense Layer 1
        x = Dense(dense_units[0], activation='relu')(x)
        outputs = Dense(1, activation="sigmoid")(x)
        model = Model(inputs, outputs)
        model.summary()
        
        optimizer = [optimizers.Adam(lr=0.001, decay =5*1e-5), optimizers.RMSprop(lr=0.001, rho = 0.99, decay = 1e-4)]        
        # model.compile(loss='categorical_crossentropy',optimizer='nadam',metrics=['accuracy'])
        # model.compile(loss='binary_crossentropy',optimizer='nadam',metrics=['accuracy'])
        model.compile(loss='binary_crossentropy',optimizer='nadam', metrics=['accuracy'])
        
        early_stop = EarlyStopping(monitor='val_loss', mode='min', 
                                   verbose=1, patience=10, min_delta=0.0001)
        
        checkpoint = ModelCheckpoint('speech2text_model.hdf5', monitor='val_acc', 
                                     verbose=1, save_best_only=True, mode='max')

        return model, early_stop, checkpoint
    
    
    def fit(self, model, early_stop, checkpoint, epochs=15, batch_size=64):
        
        history = model.fit(
        x=self.X_train, 
        y=self.y_train,
        epochs=epochs, 
        callbacks=[early_stop, checkpoint], 
        batch_size=batch_size, 
        validation_data=(self.X_test, self.y_test)
        )
        
        _, accuracy = model.evaluate(self.X_test, self.y_test, batch_size=batch_size, verbose=0)
        print('1_D CNN Classifier accuracy:{0}'.format(accuracy))
        
        # Plot training & validation accuracy values
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        # plt.savefig('models/1_d_cnn/accuracy.png')
        plt.show()
        # Plot training & validation loss values
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        # plt.savefig('models/1_d_cnn/loss.png')
        plt.show()
        


