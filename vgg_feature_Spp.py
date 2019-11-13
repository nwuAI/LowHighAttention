#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib
from SpatialPyramidPooling import SpatialPyramidPooling
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from keras.models import Model
#from theano import *
import cv2
import numpy as np
import scipy as sp
import sys
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras import backend as K
K.set_image_dim_ordering('tf')
import h5py
import time
import os
import tensorflow as tf
import pandas
from keras.applications.vgg16 import preprocess_input
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'





#def get_features(model, layer, X_batch): #
    #
    #get_features = K.function([model.layers[0].input, K.learning_phase()], [model.layers[layer].output])#
    #features = get_features([X_batch,0])
    #print(features.shape)
    #print([model.layers[layer].output])

    #return features


def VGG_16(weights_path=None):

 
   

    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(224,224,3)))
    model.add(Conv2D(64, (3, 3), activation='relu',name="conv1-1"))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(64, (3, 3), activation='relu',name="conv1-2"))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128, (3, 3), activation='relu',name="conv2-1"))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128, (3, 3), activation='relu',name="conv2-2"))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, (3, 3), activation='relu',name="conv3-1"))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, (3, 3), activation='relu',name="conv3-2"))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, (3, 3), activation='relu',name="conv3-3"))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation='relu',name="conv4-1"))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation='relu',name="conv4-2"))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation='relu',name="conv4-3"))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation='relu',name="conv5-1"))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation='relu',name="conv5-2"))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation='relu',name="conv5-3"))
    #model.add(MaxPooling2D((2,2), strides=(2,2)))
    model.add(SpatialPyramidPooling([1,4,7],name="Sppool"))
    # model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    
    

    if weights_path:
        model.load_weights(weights_path)

    return model


if __name__ == '__main__':
    dataPath = "train"
    filename = 'img_train_list4.txt'
    outputname1 = 'img_train_block5_sppool_120000.h5'
    outputname2 = 'img_train_block5_sppool_120000.h5'
    error = 'error.txt'
    f_w = open(error, 'a')
    print (filename)
    print (outputname1)
    #f_out = h5py.File(outputname, 'w')
    #f_out = pandas.to_csv(outputname)
    mean_pixel = [103.939, 116.779, 123.68]  #

    # load pretrained model 
    base_model = VGG_16('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')
    base_model.summary()
    model = Model(inputs= base_model.input,outputs=base_model.get_layer('Sppool').output)
    img_list = [] #
    with open(filename) as f:
        with h5py.File(outputname1,'a') as f_out:
        
            for line in f:
                line = line.strip().split('\n')
                img_list.append(line[0])
            for item in img_list:
                #print(img_list)
                print ("process " + item + '.jpg')
                name = dataPath + "/"+item + '.jpg'
                print(name)
                try:
                    im = cv2.resize(cv2.imread(name),(224,224))#,interpolation=cv2.INTER_CUBIC)#.astype(np.float32)
                    im = im/255#归一化处理
                except:
                    f_w.write(item + '\n')
                    continue
                for c in range(3):
                    im[:, :, c] = im[:, :, c] - mean_pixel[c]
                #######
                #im = im.transpose((2, 0, 1))
                im = image.img_to_array(im)
                print('###############',im.shape)
                #im = im.transpose((1, 0, 2))
                #print('++++++++++++++=',im.shape)
                im = np.expand_dims(im, axis=0)
                print('--------------',im.shape)
                #im = preprocess_input(im)
                start = time.time()
                print('////////////////',start)
                #get_features = K.function([base_model.layers[0].input], [base_model.layers[30].output])
                #features = get_features([im])
                features = model.predict(im)
                #feat = features
                print('features[0].shape is:######################',features[0].shape)
                feat = features[0].reshape(66,512)
                print('features.shape is:##########################',feat.shape)
               
                
        
                print ('%s feature extracted in %f  seconds.' % (name, time.time() - start))
                
                f_out.create_dataset(name=item,data=feat)
              
