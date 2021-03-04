# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 16:18:37 2021

@author: eyaraz
"""


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class spec_classifier:
    
    def __init__(self, training_set_path, val_set_path, h, w):
        self.training_set_path = training_set_path
        self.val_set_path = val_set_path
        self.h = h
        self.w = w
        
    def build(self):
        
        model = Sequential()

        # Step 1 - Convolution
        model.add(Conv2D(16, (3, 3), input_shape = (self.h, self.w, 3), activation = 'relu'))
        model.add(MaxPooling2D(pool_size = (2, 2)))
        
        # Adding a second convolutional layer for Improving the results
        model.add(Conv2D(16, (3, 3), activation = 'relu'))
        model.add(MaxPooling2D(pool_size = (2, 2)))
        
        # # Adding a third convolutional layer for Improving the results
        # model.add(Conv2D(64, (3, 3), activation = 'relu'))
        # model.add(MaxPooling2D(pool_size = (2, 2)))
        
        # # Adding a forth convolutional layer for Improving the results
        # model.add(Conv2D(64, (3, 3), activation = 'relu'))
        # model.add(MaxPooling2D(pool_size = (2, 2)))
        
        # Step 3 - Flattening
        model.add(Flatten())
        
        # Step 4 - Full connection
        model.add(Dense(units = 32, activation = 'relu'))
        model.add(Dropout(0.5))
        # model.add(Dense(units = 64, activation = 'relu'))
        # model.add(Dropout(0.45))
        
        model.add(Dense(units = 1, activation = 'sigmoid'))
        
        #Optimizer
        optimizer = optimizers.Adam(lr=0.001, decay = 1e-5)
        # Compiling the CNN
        model.compile(optimizer = optimizer , loss = 'binary_crossentropy', metrics = ['accuracy'])
        
        early_stop = EarlyStopping(monitor='val_loss', mode='min', 
                               verbose=1, patience=10, min_delta=0.0001)
        
        checkpoint = ModelCheckpoint('soec_clf.hdf5', monitor='val_acc', 
                                 verbose=1, save_best_only=True, mode='max')
        return model, early_stop, checkpoint
    
    
    def fit(self, classifier, early_stop, checkpoint, epochs=20, batch_size=64):
        
        train_datagen = ImageDataGenerator(rescale = 1./255,
                                       width_shift_range=0.2, 
                                       zoom_range = 0.2,
                                       horizontal_flip = False)

        val_datagen = ImageDataGenerator(rescale = 1./255,
                                         width_shift_range=0.2)
        
        training_set = train_datagen.flow_from_directory(self.training_set_path,
                                                 target_size = (self.h, self.w),
                                                 batch_size = batch_size,
                                                 class_mode = 'binary')

        val_set = val_datagen.flow_from_directory(self.val_set_path,
                                            target_size = (self.h, self.w),
                                            batch_size = batch_size,
                                            class_mode = 'binary')

        history = classifier.fit_generator(training_set,
                         callbacks=[early_stop, checkpoint],
                         epochs = epochs,
                         validation_data = val_set,
                         )
        
        import matplotlib.pyplot as plt

        # Plot training & validation accuracy values
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()
        
        # Plot training & validation loss values
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()
                
        
        
        
        
