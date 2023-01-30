# -*- coding: utf-8 -*-
"""
Created on Sat Oct 15 09:05:29 2022

@author: ahmed
This code is a python script that uses various libraries like Numpy, Tensorflow, OpenCV, Keras etc. 
to create, train and test a Convolutional Neural Network (CNN) for image classification. 
The code reads and loads the input spectrum images from a specified folder path and pre-processes the images by resizing and normalizing.
It then uses the Keras library to define, compile and train the CNN model. 
The code also implements early stopping, checkpointing and 10-fold cross-validation to prevent overfitting and improve the accuracy of the model
"""
import numpy as np
import os
import tensorflow as tf
import cv2
from  matplotlib import pyplot as plt
from keras.models import Sequential, Model, load_model
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPool2D, MaxPooling2D, Input, InputLayer,Reshape, Conv2DTranspose
from keras.utils import to_categorical 
from keras import backend 
from keras.optimizers import SGD
from keras.constraints import MaxNorm
from sklearn.model_selection import  StratifiedKFold
from numpy import std
from numpy import mean
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from os.path import exists
from tensorflow.keras.utils import plot_model
import sys
from os import makedirs

# from numpy import expand_dims
from numpy import zeros
from numpy import ones
from keras.optimizers import Adam, RMSprop
from keras.layers import LeakyReLU, BatchNormalization, ReLU
# from keras.utils.vis_utils import plot_model
from numpy.random import randn, randint
from matplotlib import pyplot
import matplotlib.image as mpimg
from PIL import Image

# ============================= Loading Spectrum Images ========================
# defining the input images size    
IMG_WIDTH= 32
IMG_HEIGHT= 32
n_epochs = 300
n_batch = 20
cnn_batch_size = 9 
cnn_epochs = 100
Ad_times = 1
subject = "sub_a" #  "sub_a","sub_b", "sub_c","sub_d", "sub_e", "sub_f", "sub_g"
total_cnn_acc=list()

# for subject in subjects:
# for nfolds in range(0,10):
# for nfolds in R_nfolds:
    # fix random seed for reproducibility
seed = 7
tf.random.set_seed(seed)
np.random.seed(seed)


img_folder =r'--PATH--'.format(subject)   

def create_dataset(img_folder):       
    img_data_array=[]
    class_name=[]       
    for dir1 in os.listdir(img_folder):
        for file in os.listdir(os.path.join(img_folder, dir1)):
            image_path= os.path.join(img_folder, dir1,  file)
            image=mpimg.imread(image_path)
            # image= cv2.imread( image_path, cv2.COLOR_BGR2RGB)
            image=cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH),interpolation = cv2.INTER_AREA)
            image=np.array(image)
            image = image.astype('float32')
            image = (image - 127.5) / 127.5
            # image /= 255 
            img_data_array.append(image)
            class_name.append(dir1)
    # extract the image array and class name
    (img_data, class_name) = (img_data_array,class_name)
    # Create a dictionary for all unique values for the classes
    target_dict={s: v for v, s in enumerate(np.unique(class_name))}
    target_dict
    # Convert the class_names to their respective numeric value based on the dictionary
    target_val=  [target_dict[class_name[i]] for i in range(len(class_name))]
    x=tf.cast(np.array(img_data), tf.float64).numpy()
    y=tf.cast(list(map(int,target_val)),tf.int32).numpy()
    return x, y

xx,yy = create_dataset(img_folder)
          
def fold_split(xdata,ydata,folds=10):         
    trainX = np.empty((int(folds),int (len(xdata)-(len(xdata)/folds)) , IMG_HEIGHT, IMG_WIDTH))
    trainY = np.empty((int(folds),int (len(xdata)-(len(xdata)/folds)) ))
    testX = np.empty((int(folds),int (len(xdata)/folds) , IMG_HEIGHT, IMG_WIDTH))
    testY = np.empty((int(folds),int (len(xdata)/folds) ))
    sub_fold = StratifiedKFold(folds, shuffle=True, random_state=2) 
    i=0
    # ## enumerate splits
    for train, cv in sub_fold.split(xdata,ydata):
        # select data for train and test
        trainX[i,:,:,:], trainY[i,:], testX[i,:,:,:], testY[i,:] = xdata[train], ydata[train], xdata[cv], ydata[cv]
        i+=1
    return trainX, trainY, testX, testY

train_X, train_Y, testX, testY = fold_split(xx,yy)   

print ('Raw data = ',train_X.shape, train_Y.shape)
print ('Test data = ',testX.shape, testY.shape)
 
  


Drp1= 2
Drp2= 2
Drp3= 4
Drp4= 4
  
def create_cnn(): #dropout_rate1=0.0, dropout_rate2=0.0 , momentum=0, weight_constraint=0, 
    model= Sequential()
    model.add(Dropout(Drp1/10, input_shape = (IMG_HEIGHT,IMG_WIDTH,1)) )     # dropout 1
    model.add(Conv2D(8, kernel_size=(3,3), padding='same', activation= 'relu' 
                      , kernel_initializer='he_uniform' , kernel_constraint=MaxNorm(3)  ))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(Drp2/10))                                             # dropout 2
    model.add(Conv2D(8, (3,3), padding='same' ,activation= 'relu' 
                      , kernel_initializer='he_uniform', kernel_constraint=MaxNorm(3)   )) # kernel_regularizer=l2(0.001),
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(Drp3/10))                                             # dropout 3
    model.add(Flatten())
    # model.add(Dropout(0.2)) 
    model.add(Dense(100, activation= 'relu' , kernel_initializer='he_uniform', kernel_constraint=MaxNorm(3) ) )  
    model.add(Dropout(Drp4/10))                                             # dropout 4
    model.add(Dense(2, activation= 'softmax' , kernel_initializer='he_uniform' )) 
    opt = SGD(learning_rate=0.0001, momentum=0.99)
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy']) 
    return model
# model = create_cnn()
# model.summary()

def create_cnn2(): #dropout_rate1=0.0, dropout_rate2=0.0 , momentum=0, weight_constraint=0, 
    model= Sequential()
    model.add(Dropout(Drp1/10, input_shape = (IMG_HEIGHT,IMG_WIDTH,1)) )     # dropout 1
    model.add(Conv2D(16, kernel_size=(3,3), padding='same', activation= 'relu' 
                      , kernel_initializer='he_uniform' , kernel_constraint=MaxNorm(3)  ))
    model.add(Conv2D(16, kernel_size=(3,3), padding='same', activation= 'relu' 
                      , kernel_initializer='he_uniform' , kernel_constraint=MaxNorm(3)  ))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(Drp2/10))                                             # dropout 2
    model.add(Conv2D(32, (3,3), padding='same' ,activation= 'relu' 
                      , kernel_initializer='he_uniform', kernel_constraint=MaxNorm(3)   )) # kernel_regularizer=l2(0.001),
    model.add(Conv2D(32, (3,3), padding='same' ,activation= 'relu' 
                      , kernel_initializer='he_uniform', kernel_constraint=MaxNorm(3)   ))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(Drp2/10))                                             # dropout 3
    model.add(Conv2D(64, (3,3), padding='same' ,activation= 'relu' 
                      , kernel_initializer='he_uniform', kernel_constraint=MaxNorm(3)   ))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(Drp3/10))           
    model.add(Flatten())
    model.add(Dense(128, activation= 'relu' , kernel_initializer='he_uniform', kernel_constraint=MaxNorm(3) ) )  
    model.add(Dropout(Drp3/10))                                             # dropout 4
    model.add(Dense(2, activation= 'softmax' , kernel_initializer='he_uniform' )) 
    opt = SGD(learning_rate=0.0001, momentum=0.99)
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy']) 
    return model

# Create new CNN model
def create_cnn3():
    model = tf.keras.models.Sequential()
    model.add(Input(shape=(IMG_HEIGHT,IMG_WIDTH, 1)))
    model.add(Conv2D(32, 3, strides=2, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(64, 3, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(128, 3, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))
    # Compile model
    model.compile(loss='sparse_categorical_crossentropy',optimizer=RMSprop(), metrics=['accuracy'])
    return model

# #GAN-CNN Model training:
def model_training(x_data, y_data ,x_test, y_test,save_dir, sel_mod ,fig_title):
    tf.random.set_seed(seed)
    np.random.seed(seed)
    print ("Training Data= ",len(x_data) ) 
    print ("Validation Data= ",len(x_test) )
    model = sel_mod
    # 1-Times generated data:
    mcg = ModelCheckpoint(save_dir, monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
    history = model.fit(x_data, y_data, epochs=cnn_epochs , batch_size=cnn_batch_size, verbose=0 
                        ,validation_data=(x_test, y_test), callbacks=[mcg] )


    pyplot.figure()
    pyplot.plot(history.history['accuracy'])
    pyplot.plot(history.history['val_accuracy'])
    pyplot.title(fig_title)
    pyplot.ylabel('Accuracy')
    pyplot.xlabel('Epoch')
    pyplot.legend(['Train', 'Test'], loc='lower right')
    pyplot.grid()
    pyplot.show()
    pyplot.close() 
    
    
for nfolds in range(0,10):
    tf.random.set_seed(seed)
    np.random.seed(seed)
        
    x_img_cl1,x_img_cl2 = class_imgs(train_X, train_Y)
    x_test_img_cl1, x_test_img_cl2 = class_imgs(testX, testY)
    (x_tr,y_tr) = (train_X[nfolds] , train_Y[nfolds])
    (x_ev, y_ev) = (testX[nfolds], testY[nfolds])
    print ('Fold: ' , nfolds )
    print ("fold data shape = ",x_tr.shape, y_tr.shape )
    print ("Test fold data shape = ",x_ev.shape, y_ev.shape )
    cnn_epochs = 600
    model_training(x_tr,y_tr, x_ev, y_ev, '\{0}_CNN1_{1}.h5'.format( subject,  nfolds, ), create_cnn1(), '{}_CNN1 Model accuracy fold {} '.format(  subject, nfolds))        
    model_training(x_tr,y_tr, x_ev, y_ev, '\{0}_CNN2_{1}.h5'.format( subject,  nfolds, ), create_cnn2(), '{}_CNN2 Model accuracy fold {} '.format(  subject, nfolds))        
    cnn_epochs = 100
    model_training(x_tr,y_tr, x_ev, y_ev, '\{0}_CNN3_{1}.h5'.format( subject,  nfolds, ), create_cnn3(), '{}_CNN3 Model accuracy fold {} '.format(  subject, nfolds))        

#%%======================================= 10 fold test ==========================
def CNN_GAN_test(cnn):
    scores = list()
    for f in range(0,10):

        elif cnn == 1:
            model = load_model('\{0}_CNN_{1}.h5'.format( subject,f))
            test_loss, test_acc= model.evaluate(testX[f],testY[f],verbose=0)
            print('CNN: ',f,'  Accuracy',test_acc)
            scores.append(test_acc)
        elif cnn == 2:
            model = load_model('\{0}_CNN2_{1}.h5'.format( subject,f))
            test_loss, test_acc= model.evaluate(testX[f],testY[f],verbose=0)
            print('CNN2: ',f,'  Accuracy',test_acc)
            scores.append(test_acc)
        elif cnn == 3:
            model = load_model('\{0}_CNN3_{1}.h5'.format( subject,f))
            test_loss, test_acc= model.evaluate(testX[f],testY[f],verbose=0)
            print('CNN3: ',f,'  Accuracy',test_acc)
            scores.append(test_acc)

            scores.append(test_acc)    
    print('\n >>>> {0} Accuracy: mean={1} std={2}, n={3}' .format (subject, mean(scores)*100, std(scores)*100, len(scores)))
    print ('*************************************')
    ## box and whisker plots of results
    plt.boxplot(scores)
    plt.show()
    plot_model(model, show_shapes=True, expand_nested=True)
    return scores

sub_cnn1_acc = CNN_GAN_test(cnn=1)
sub_cnn2_acc = CNN_GAN_test(cnn=2)
sub_cnn3_acc = CNN_GAN_test(cnn=3)


