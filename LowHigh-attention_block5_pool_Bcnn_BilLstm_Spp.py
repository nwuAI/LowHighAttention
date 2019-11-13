from keras.models import Sequential, Model
from keras.layers.embeddings import Embedding
from keras.layers import Activation, Dense, Permute, Flatten, Dropout, Reshape,RepeatVector,Lambda
import tensorflow as tf
from keras.layers.recurrent import LSTM
from keras.layers.convolutional import Convolution2D, MaxPooling2D, Convolution1D, MaxPooling1D, AveragePooling1D
from keras.preprocessing.sequence import pad_sequences
from keras.layers.wrappers import TimeDistributed
from keras.callbacks import History
from keras.layers import Input,multiply
from keras import optimizers
from keras.layers.normalization import BatchNormalization
from keras.layers.wrappers import Bidirectional
from SpatialPyramidPooling import SpatialPyramidPooling

from keras.layers import concatenate,dot,add
import numpy as np
from data_helper_block5_pool_Spp import  load_data
from result_calculator import *
from keras import backend as K
#K.set_image_data_format('channels_first')
#K.set_image_data_format('channels_first')
#from sklearn.metrics import average_precision_score, precision_score, recall_score
from keras.utils.vis_utils import plot_model

from keras.utils import np_utils
import h5py

import os

#os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin'

def batch_dot(cnn_ab):
    return K.batch_dot(cnn_ab[0],cnn_ab[1],axes=[1,1])
    
def sign_sqrt(x):
    return K.sign(x)*K.sqrt(K.abs(x) + 1e-10)
    
def l2_norm(x):
    return K.l2_normalize(x,axis=-1)


if __name__ == '__main__':
    print ('loading data...')
    
    x, img_x, y, valid_x, valid_img_x, valid_y, test_x, test_img_x, test_y, vocabulary, vocabulary_inv, hashtagVoc, hashtagVoc_inv, maxlen = load_data()


    #print('The shape:',img_x.shape)
    vocab_size = len(vocabulary_inv) + 1
    hashtag_size = len(hashtagVoc_inv)

    #print(valid_y,'\n',valid_x,'\n',valid_img_x)

    print('-')
    print('Vocab size:', vocab_size, 'unique words')
    print('Hashtag size:', hashtag_size, 'unique hashtag')
    print('Max length:', maxlen, 'words')
    print('-')
    print('Here\'s what a "hashtag" tuple looks like (x, img_x, label):')
    #print(x[0])#, img_x[0].shape, y[0])
    print('The img_x[0] is:',img_x[0].shape)
   # print(y[0])
    print('-')
    print('-')
    print('input_mention_tweet: integer tensor of shape (samples, max_length)')
    print('shape:', x.shape)
    print('-')
    print('-')
    print('input_label: integer tensor of shape (samples, hashtag_size)')
    print('shape:', y.shape)
    print('-')
    
    embedding_dim =300
    nb_epoch = 100
    batch_size = 100

    feat_dim = 512
    w = 7
    num_region = 66

    # build model
    print ("Build model...")

    tweet = Input(shape=(maxlen,), dtype='int32')#input_1
    
    embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=maxlen, mask_zero=False)(tweet)
    lstm = Bidirectional(LSTM(embedding_dim, return_sequences=True, input_shape=(maxlen, embedding_dim)))(embedding)
    #lstm = LSTM(embedding_dim, return_sequences=True, input_shape=(maxlen, embedding_dim))(embedding)
    dropout = Dropout(0.5)(lstm)
    #img = Input(shape=(1, 14, 14,512))
    
    
    
    img = Input(shape=(66,512))#input_2 
    #img = SpatialPyramidPooling([1,2,7])(img)
    #print('img shape is:-----------',img)

    #--------Image Attention
    img_att = BatchNormalization(axis=-1)(img)
    img_att = Activation('relu')(img_att)
    
    #img_att = TimeDistributed(Dense(1))(img_att)
    img_att_pro = Activation('softmax')(img_att)
    print('img_att_pro shape--------------:',img_att_pro.shape)
    #img_att_pro = Flatten()(img_att)

    img_new = multiply([img_att_pro, img])#dot([img_att_pro, img_att], axes=(1, 1))#49.512s
    print('img_new shape---------------:',img_new.shape)
      
    '''
    The End input is:[img,tweet]
    '''
    #BCNN----->
    cnn_out_a = img_new
    cnn_out_b = cnn_out_a
    cnn_out_dot = Lambda(batch_dot)([cnn_out_a,cnn_out_b])
    print('cn_out_dot shape is:-----------',cnn_out_dot.shape)
    
    sign_sqrt_out = Lambda(sign_sqrt)(cnn_out_dot)
    l2_norm_out = Lambda(l2_norm)(sign_sqrt_out)

    


    img_dense = TimeDistributed(Dense(embedding_dim))(l2_norm_out)
    print('img_dense shape:--------',img_dense.shape)#(?, 512, 300)
    #img_dense = TimeDistributed(Dense(embedding_dim))(img)
    tweet_avg = AveragePooling1D(pool_size=maxlen)(dropout)
    
    tweet_avg = Flatten()(tweet_avg)
    print('tweet_avg shape:---------',tweet_avg.shape)
   
    tweet_avg_dense = Dense(embedding_dim)(tweet_avg)#(?, 300)
    print('tweet_avg_dense shape:---------',tweet_avg_dense.shape)
    
    tweet_repeat = RepeatVector(512)(tweet_avg_dense)#(?, 49, 300)
    print('tweet_repeat shape:---------',tweet_repeat.shape)
   
    att_1 = concatenate([tweet_repeat, img_dense],axis=-1)#(?, 512, 600)
    print('att_1 shape:---------',att_1.shape)
    #att_1 = merge([tweet_avg_dense, img_dense], mode='concat')
   
    att_1 = Activation('tanh')(att_1)
    att_1 = TimeDistributed(Dense(1))(att_1)
    att_1 = Activation('softmax')(att_1)#(?, 512, 1)
    print('att_1 shape:---------',att_1.shape)
    att_1_pro = Flatten()(att_1)
    print('att_1_pro shape:---------',att_1_pro)
    #img_new = merge([att_1_pro, img_dense], mode='dot', dot_axes=(1, 1))
    
    img_new = dot([att_1_pro, img_dense], axes=(1, 1))# (?, 300)
    print('img_new shape:---------',img_new.shape)
    # img->text      
    
    img_new_dense = Dense(embedding_dim)(img_new)#(?, 300)
    print('img_new_dense shape:---------',img_new_dense.shape)
    
    img_new_repeat = RepeatVector(maxlen)(img_new_dense)#(?, 24, 300)
    print('img_new_repeat shape:---------',img_new_repeat.shape)
    
    tweet_dense = TimeDistributed((Dense(embedding_dim)))(dropout) #(?, 24, 300)
    
#att_2 = merge([img_new_repeat, tweet_dense], mode='concat')
    print('tweet_dense shape:---------',tweet_dense.shape)
    
    att_2 = concatenate([img_new_repeat, tweet_dense], axis=-1)
    print('att_2 shape:---------',att_2.shape)
    
    att_2 = Activation('tanh')(att_2)
    att_2 = TimeDistributed(Dense(1))(att_2)
    print('att_2 shape:---------',att_2.shape)
    
    att_2 = Activation('softmax')(att_2)
    
    att_2_pro = Flatten()(att_2)
    print('att_2_pro shape:---------',att_2_pro[1])

    #tweet_new = merge([att_2_pro, dropout], mode='dot', dot_axes=(1, 1))
    
    tweet_new = dot([att_2_pro, dropout], axes=(1, 1))#(?, 300)
    print('tweet_new shape:---------',tweet_new.shape)
    # set gate
    img_new_tanh = Dense(embedding_dim, activation='tanh')(img_new)
    tweet_new_tanh = Dense(embedding_dim, activation='tanh')(tweet_new) #(?, 300)
    print('tweet_new_tanh shape:---------',tweet_new_tanh.shape)
    #merge = merge([img_new_tanh, tweet_new_tanh], mode='concat')
    
    #img_new_tanh_T = Reshape(-1)(img_new_tanh)
    #print('img_new_tanh_T shape is :-------',img_new_tanh_T.shape)
    merge = concatenate([img_new_tanh, tweet_new_tanh],axis=-1)
   # merge = dot([img_new_tanh, tweet_new_tanh],axes=[1,1])
    #merge = Dense(600,activation='relu')(merge)
    #add = add([img_new_tanh, tweet_new_tanh])
    
    #z = Dense(1, activation='sigmoid')(merge)
    #output = Dense(hashtag_size, activation='softmax')(merge)
    
    output = Dense(hashtag_size, activation='sigmoid')(merge)
    
    model = Model(inputs=[tweet, img], output=output)
    #adam = optimizers.Adam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)   
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=["accuracy"])
        
    print (model.summary())
    print ("finished building model")
    #plot_model(model, show_shapes=True, to_file='Sppp_bcnn_bilstm_model.png')

    # starts training
    y = np_utils.to_categorical(y, hashtag_size)
    
    
    print ("starts training")
    best_f1 = 0
    topK = [1, 2, 3, 4, 5]
    for j in range(nb_epoch):

        history = History()
        
        #print('the img_x shape is:',img_x.shape)
        #print("the x shape is:",x.shape)
        #print('THE [x,img_x] shape: ',np.array([x,img_x]).shape)
        
        #input_x = np.array([x,img_x])
        #img_x = Reshape((1,14,14,512))(img_x)
        input_x = [x,img_x]#type is list
        #print('input_x[0][0] is----------:',input_x[0][0])
        #print('input_x[0][1] is----------:',input_x[0][1])
        #print('input_x[1] is----------:',input_x[1][0])
        #print('img_x[0] is:-------',img_x[0])
        #print('img_x[1] is:-------',img_x[1])
        #print('img_x[0].shape is:-------------',img_x[0].shape)
        
       # print('input_x[0][0].shape is:-------------',input_x[0][0].shape)
        #print('input_x[0][1].shape is:-------------',input_x[0][1].shape)
       # print('input_x[1][0].shape is:-------------',input_x[1][0].shape)
       # print('input_x.type is:---------',type(input_x))
        #print('y shape is:',y.shape)
        #print('input_x shape is :',(np.array(input_x).shape))
        model.fit([x,img_x],y,batch_size=batch_size,epochs=1,verbose=1,callbacks=[history])
        #print("33333333333333")
        #model.save_weights('my_model_weights.h5')
        print (history.history)
        print (len(history.history))
        y_pred = model.predict([valid_x, valid_img_x], batch_size=batch_size, verbose=0)
        
        y_pred = np.argsort(y_pred, axis=1)
        precision = precision_score(valid_y, y_pred)
        recall = recall_score(valid_y, y_pred)
        F1 = 2 * (precision * recall) / (precision + recall)
        print ("Epoch:", (j + 1), "precision:", precision, "recall:", recall, "f1 score:", F1)

        if best_f1 < F1:
            best_f1 = F1
            y_pred = model.predict([test_x, test_img_x], batch_size=batch_size, verbose=0)

            y_pred = np.argsort(y_pred, axis=1)
            for k in topK:
                precision = precision_score(test_y, y_pred, k=k)
                recall = recall_score(test_y, y_pred, k=k)
                hscore = hits_score(test_y, y_pred, k=k)
                F1 = 2 * (precision * recall) / (precision + recall)
                print ("\t", "top:", k, "Test: precision:", precision, "recall:", recall, "f1 score:", F1, "hits score:", hscore)


