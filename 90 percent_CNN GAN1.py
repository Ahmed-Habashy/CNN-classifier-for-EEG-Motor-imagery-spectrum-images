"""
Created on Sat Oct 15 09:05:29 2022

@author: ahmed
"""
import numpy as np
import os
import tensorflow as tf
import cv2
from  matplotlib import pyplot as plt
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPool2D, Input, InputLayer,Reshape, Conv2DTranspose
from tensorflow.keras.utils import to_categorical 
from tensorflow.keras import backend 
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.constraints import MaxNorm
from sklearn.model_selection import KFold, StratifiedKFold
from numpy import std
from numpy import mean
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from os.path import exists
from tensorflow.keras.utils import plot_model
import sys
from os import makedirs
from numpy import zeros
from numpy import ones
from tensorflow.keras.optimizers import Adam, SGD
from keras.layers import LeakyReLU, BatchNormalization, ReLU
from numpy.random import randn, randint
from matplotlib import pyplot
import matplotlib.image as mpimg

# ============================= Loading Spectrum Images ========================
# defining the input images size    
IMG_WIDTH=64
IMG_HEIGHT=64
subject = "sub_B04"
cnn_batch_size = 9 
cnn_epochs = 500
Ad_times = 1
nfolds = 0
R_nfolds = [ 7,9 ]
seed = 7
cnn_acc = list()
cnn2_acc = list()


img_folder =r'spectrogram\sec_4\{}'.format(subject) 
img_folder_test =r'spectrogram\sec_4\Test\{}'.format(subject)   

def create_dataset(img_folder):       
    img_data_array=[]
    class_name=[]       
    for dir1 in os.listdir(img_folder):
        for file in os.listdir(os.path.join(img_folder, dir1)):
            image_path= os.path.join(img_folder, dir1,  file)
            image=mpimg.imread(image_path)
            image=cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH),interpolation = cv2.INTER_AREA)
            image=np.array(image)
            image = image.astype('float32')
            image /= 255 
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

x_tr,y_tr = create_dataset(img_folder)
x_ev,y_ev = create_dataset(img_folder_test)
x0 = np.concatenate(( x_tr,x_ev ))   
y0 = np.concatenate(( y_tr,y_ev ))  

# splitting the dataset to 10-folds:
def fold_split(xdata,ydata,folds=10): 
    
    trainX = np.empty((int(folds),int (len(xdata)-(len(xdata)/folds)) , IMG_HEIGHT, IMG_WIDTH,3))
    trainY = np.empty((int(folds),int (len(xdata)-(len(xdata)/folds)) ))
    testX = np.empty((int(folds),int (len(xdata)/folds) , IMG_HEIGHT, IMG_WIDTH,3))
    testY = np.empty((int(folds),int (len(xdata)/folds) ))

    sub_fold = StratifiedKFold(folds, shuffle=True, random_state=2) 
    i=0
    # ## enumerate splits
    for train, cv in sub_fold.split(xdata,ydata):
        # select data for train and test
        trainX[i,:,:,:,:], trainY[i,:], testX[i,:,:,:,:], testY[i,:] = xdata[train], ydata[train], xdata[cv], ydata[cv]
        i+=1
    return trainX, trainY, testX, testY


for nfolds in range(0,10):

    # fix random seed for reproducibility
    tf.random.set_seed(seed)
    np.random.seed(seed)
    print (subject)
    print (nfolds)
   
    
    trainX, trainY, testX, testY = fold_split(x0, y0)
    #%% ================================== CNN model =================================
    Drp1= 2
    Drp2= 2
    Drp3= 4
    Drp4= 4
       
    def create_cnn():  
        model= Sequential()
        model.add(Dropout(Drp1/10, input_shape = (IMG_HEIGHT,IMG_WIDTH,3)) )     # dropout 1
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
    model = create_cnn()
    model.summary()
    
    def create_cnn2(): #dropout_rate1=0.0, dropout_rate2=0.0 , momentum=0, weight_constraint=0, 
        model= Sequential()
        model.add(Dropout(Drp1/10, input_shape = (IMG_HEIGHT,IMG_WIDTH,3)) )     # dropout 1
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
        # opt = SGD(learning_rate=0.0001, momentum=0.99)
        opt = Adam(learning_rate=0.0002, beta_1 = 0.5, beta_2 = 0.8) 

        model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy']) 
        return model
    #%% ============================= Training ====================================
    # CNN Model training:
    def model_training(x_data, y_data ,save_dir, sel_mod ,fig_title):
        tf.random.set_seed(seed)
        np.random.seed(seed)
        print ("Training Data= ",len(x_data) )    
        model = sel_mod
        # 1-Times generated data:
        mcg = ModelCheckpoint(save_dir, monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
    
        history = model.fit(x_data, y_data, epochs=cnn_epochs , batch_size=cnn_batch_size, verbose=0 
                            ,validation_data=(testX[nfolds],testY[nfolds]), callbacks=[mcg] )
        plt.figure()
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title(fig_title)
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='lower right')
        plt.grid()
        plt.show()
        plt.close()
        

    model_training(trainX[nfolds], trainY[nfolds],'{0}_CNN_modelf_{1}.h5'.format( subject, nfolds), create_cnn(), '{0}_CNN Model accuracy fold {1}'.format(  subject, nfolds))        
    model_training(trainX[nfolds], trainY[nfolds],'{0}_CNN2_model_f{1}.h5'.format( subject, nfolds), create_cnn2(), '{0}_CNN2 Model accuracy fold {1}'.format(  subject, nfolds))        

#%%======================================= 10 fold test ==========================
# # for subject in subjects:
def CNN_GAN_test(cnn=2):
    scores = list()

    for f in range(0,10):
        if cnn == 2 :
        # load the saved model
           model = load_model('{0}_CNN2_model_f{1}.h5'.format( subject, f))
            test_loss, test_acc= model.evaluate(testX[f],testY[f],verbose=0)
            print('\nTest: ',f,'CNN_ fold Accuracy',test_acc)
            scores.append(test_acc)
        else: cnn == 1:
            model = load_model('{0}_CNN_modelf_{1}.h5'.format( subject,  f))
            test_loss, test_acc= model.evaluate(testX[f],testY[f],verbose=0)
            print('\nTest: ',f,' fold Accuracy',test_acc)
            scores.append(test_acc)
        
            
           

    print('\n >>>> {0} folds Accuracy: mean={1} std={2}, n={3}' .format (subject, mean(scores)*100, std(scores)*100, len(scores)))
    print ('*************************************')
    ## box and whisker plots of results
    plt.boxplot(scores)
    plt.show()
    plot_model(model, show_shapes=True, expand_nested=True)
    return scores

cnn_acc = CNN_GAN_test(cnn=1)
cnn2_acc = CNN_GAN_test(cnn=2)
