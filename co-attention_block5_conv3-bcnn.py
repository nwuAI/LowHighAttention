from keras.models import Sequential, Model
from keras.layers.embeddings import Embedding
from keras.layers import Activation, Dense, Permute, Flatten, Dropout, Reshape,RepeatVector,Lambda
import tensorflow as tf
import numpy as np
from keras.layers.recurrent import LSTM,GRU
from keras.layers.convolutional import Convolution2D, MaxPooling2D, Convolution1D, MaxPooling1D, AveragePooling1D
from keras.preprocessing.sequence import pad_sequences
from keras.layers.wrappers import TimeDistributed
from keras.callbacks import History
from keras.layers import Input
from keras import optimizers
from keras.layers.normalization import BatchNormalization
from keras.layers.wrappers import Bidirectional
from sklearn.decomposition import PCA
from keras.callbacks import EarlyStopping

'''
批正则化
'''
from keras.layers import concatenate,dot
import numpy as np
from data_helper_block5_conv3 import  load_data
from result_calculator import *
from keras import backend as K
#K.set_image_data_format('channels_first')
#K.set_image_data_format('channels_first')
#from sklearn.metrics import average_precision_score, precision_score, recall_score
from keras.utils.vis_utils import plot_model

from keras.utils import np_utils
import h5py

import os
'''import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import keras.backend.tensorflow_backend as KTF
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.02
session = tf.Session(config=config)
KTF.set_session(session )
'''
#os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin'
def batch_dot(cnn_ab):
    return K.batch_dot(cnn_ab[0], cnn_ab[1], axes=[1, 1])
 
 
def sign_sqrt(x):
    return K.sign(x) * K.sqrt(K.abs(x) + 1e-10)
 
 
def l2_norm(x):
    return K.l2_normalize(x, axis=-1)

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
    nb_epoch = 70
    batch_size = 1000

    feat_dim = 512
    w = 14
    num_region = 512

    # build model
    print ("Build model...")
    tweet = Input(shape=(maxlen,), dtype='int32')#input_1
    
    embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=maxlen, mask_zero=False)(tweet)
    lstm = LSTM(embedding_dim, return_sequences=True, input_shape=(maxlen, embedding_dim))(embedding)
    
    #lstm = LSTM(embedding_dim, return_sequences=True)(lstm)
    #lstm = LSTM(embedding_dim)(lstm)

    
    #lstm = Bidirectional(LSTM(embedding_dim,return_sequences=True,input_shape=(maxlen,embedding_dim)))(embedding)
    
    #gru = GRU(embedding_dim, return_sequences=True, input_shape=(maxlen, embedding_dim))(embedding)
    dropout = Dropout(0.5)(lstm)
    #img = Input(shape=(1, 14, 14,512))
    
    
    img = Input(shape=(14,14,512))#input_2 
    cnn_out_a = img
    cnn_out_shape = img.shape
    cnn_out_a = Reshape([cnn_out_shape[1]*cnn_out_shape[2],
                         cnn_out_shape[-1]])(cnn_out_a)
    print("cnn_out_a.shape is:---------",cnn_out_a.shape)#(,196,512)
    cnn_out_b = cnn_out_a
    
    cnn_out_dot = Lambda(batch_dot)([cnn_out_a, cnn_out_b])
    print("cnn_out_dot.shape is :--------",cnn_out_dot.shape) #(512.512)
    #cnn_out_dot = Reshape([cnn_out_shape[-1]*cnn_out_shape[-1]])(cnn_out_dot)
    sign_sqrt_out = Lambda(sign_sqrt)(cnn_out_dot)
    print("sign_sqrt_out.shape is:------",sign_sqrt_out.shape)# (512,512
    l2_norm_out = Lambda(l2_norm)(sign_sqrt_out)
    print("the l2_norm_out.shape is:-----------",l2_norm_out.shape)#(512,512
    
    
    #l2_norm_out = np.concatenate(l2_norm_out,aixs=1)
    #pca = PCA(n_components = 300)
    #pca.fit_transform(l2_norm_out)
    #img_pca = pca.transform(l2_norm_out)
   # print("img_pca.shape is:--------",img_pca.shape)
      
    '''
    The End input is:[img,tweet]
    '''
    
    #img_reshape = Reshape((14* 14,512))(l2_norm_out)#(196,512)
    
    
    #img_reshape = Reshape((7*7,512))(img_reshape)

    #print('img_reshape is :',img_reshape.shape)
    #img_permute = Permute((2, 1))(img_reshape)#(512,196)
    
    

    # text->img
    # 
    
    img_dense = TimeDistributed(Dense(embedding_dim))(l2_norm_out)
    #img_dense = TimeDistributed(Dense(embedding_dim))(img)
    tweet_avg = AveragePooling1D(pool_size=maxlen)(dropout)
    
    tweet_avg = Flatten()(tweet_avg)

   
    tweet_avg_dense = Dense(embedding_dim)(tweet_avg)
    
    tweet_repeat = RepeatVector(512)(tweet_avg_dense)
    
   
    att_1 = concatenate([tweet_repeat, img_dense],axis=-1)
    #att_1 = merge([tweet_avg_dense, img_dense], mode='concat')
   
    att_1 = Activation('tanh')(att_1)
    att_1 = TimeDistributed(Dense(1))(att_1)
    att_1 = Activation('softmax')(att_1)
    att_1_pro = Flatten()(att_1)

    #img_new = merge([att_1_pro, img_dense], mode='dot', dot_axes=(1, 1))
    img_new = dot([att_1_pro, img_dense], axes=(1, 1))

    # img->text      
    img_new_dense = Dense(embedding_dim)(img_new) 
    img_new_repeat = RepeatVector(maxlen)(img_new_dense)
    tweet_dense = TimeDistributed((Dense(embedding_dim)))(dropout)
#att_2 = merge([img_new_repeat, tweet_dense], mode='concat')
    att_2 = concatenate([img_new_repeat, tweet_dense], axis=-1)
    att_2 = Activation('tanh')(att_2)
    att_2 = TimeDistributed(Dense(1))(att_2)
    att_2 = Activation('softmax')(att_2)
    att_2_pro = Flatten()(att_2)

    #tweet_new = merge([att_2_pro, dropout], mode='dot', dot_axes=(1, 1))
    tweet_new = dot([att_2_pro, dropout], axes=(1, 1))

    # set gate
    img_new_tanh = Dense(embedding_dim, activation='tanh')(img_new)
    tweet_new_tanh = Dense(embedding_dim, activation='tanh')(tweet_new)
    #merge = merge([img_new_tanh, tweet_new_tanh], mode='concat')
    

    
    
    merge = concatenate([img_new_tanh, tweet_new_tanh],axis=-1) #The Final feature for the Classfication machine  (merage)
    
    
    #z = Dense(1, activation='sigmoid')(merge)
    output = Dense(hashtag_size, activation='softmax')(merge)
    
    #output = Dense(hashtag_size, activation='sigmoid')(l2_norm_out)
    
    
    
    model = Model(inputs=[tweet, img], output=output)
    #adam = optimizers.Adam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)  
    

     
    #sgd = optimizers.SGD(lr=0.01, momentum=0.9, decay=0.0, nesterov=True)
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=["accuracy"])
        
    print (model.summary())
    print ("finished building model")
   # plot_model(model, show_shapes=True, to_file='model.png')

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
        
        #early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
        model.fit([x,img_x],y,batch_size=batch_size,epochs=1,verbose=1,callbacks=[history])
        #print("33333333333333")
        #model.save_weights('my_model_weights.h5')
        print (history.history)
        #print (len(history.history))
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

