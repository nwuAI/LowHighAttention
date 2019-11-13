from keras.models import Sequential, Model
from keras.layers.embeddings import Embedding
from keras.layers import Activation, Dense, Permute, Flatten, Dropout, Reshape, RepeatVector, Lambda
import tensorflow as tf
from keras.layers.recurrent import LSTM
from keras.layers.convolutional import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D, AveragePooling1D
from keras.preprocessing.sequence import pad_sequences
from keras.layers.wrappers import TimeDistributed
from keras.callbacks import History
from keras.layers import Input
from keras import optimizers
from keras.layers.normalization import BatchNormalization
from keras.layers.wrappers import Bidirectional
from keras.layers import multiply
from keras import regularizers


from keras.layers import concatenate, dot
import numpy as np
# from data_helper_pool_guiyi import  load_data
from data_helper_pool_guiyi import load_data
from result_calculator import *
from keras import backend as K
# K.set_image_data_format('channels_first')
# K.set_image_data_format('channels_first')
# from sklearn.metrics import average_precision_score, precision_score, recall_score
from keras.utils.vis_utils import plot_model
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.utils import np_utils
import h5py

import os
import tensorflow as tf


# os.environ["CUDA_VISIBLE_DEVICES"]="0" #
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.95 #
# session = tf.Session(config=config)
#
# # session
# K.set_session(session )

# os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin'

def batch_dot(cnn_ab):
    return K.batch_dot(cnn_ab[0], cnn_ab[1], axes=[1, 1])


def sign_sqrt(x):
    return K.sign(x) * K.sqrt(K.abs(x) + 1e-10)


def l2_norm(x):
    return K.l2_normalize(x, axis=-1)


if __name__ == '__main__':
    print('loading 90000 data...')

    x, img_x, y, valid_x, valid_img_x, valid_y, test_x, test_img_x, test_y, vocabulary, vocabulary_inv, hashtagVoc, hashtagVoc_inv, maxlen = load_data()
    print('_________________________', img_x[0])
    # shuffle the data
    # [x,img_x],y = shuffle([x,img_x],y)
    # print('The shape:',img_x.shape)
    vocab_size = len(vocabulary_inv) + 1
    hashtag_size = len(hashtagVoc_inv)

    # print(valid_y,'\n',valid_x,'\n',valid_img_x)
    print('-')
    print('Vocab size:', vocab_size, 'unique words')
    print('Hashtag size:', hashtag_size, 'unique hashtag')
    print('Max length:', maxlen, 'words')
    print('-')
    print('Here\'s what a "hashtag" tuple looks like (x, img_x, label):')
    # print(x[0])#, img_x[0].shape, y[0])
    print('The img_x[0] is:', img_x[0].shape)
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

    embedding_dim = 300
    nb_epoch = 200
    batch_size = 256
    feat_dim = 512
    w = 7

    # build model
    print("Build no-attention model...")

    tweet = Input(shape=(maxlen,), dtype='int32')  # input_1

    embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=maxlen, mask_zero=False)(tweet)

    lstm = Bidirectional(LSTM(embedding_dim, return_sequences=True, input_shape=(maxlen, embedding_dim)))(embedding)
   
    dropout = Dropout(0.5)(lstm)
    

    img = Input(shape=(7, 7, 512))  # input_2

  

    # -------Image Bcnn start
    cnn_out_a = img
    cnn_out_shape = img.shape
    cnn_out_a = Reshape([cnn_out_shape[1] * cnn_out_shape[2],
                         cnn_out_shape[-1]])(cnn_out_a)
    print("cnn_out_a.shape is:---------", cnn_out_a.shape)  # (,196,512)
    cnn_out_b = cnn_out_a

    cnn_out_dot = Lambda(batch_dot)([cnn_out_a, cnn_out_b])
    print("cnn_out_dot.shape is :--------", cnn_out_dot.shape)  # (512.512)
    # cnn_out_dot = Reshape([cnn_out_shape[-1]*cnn_out_shape[-1]])(cnn_out_dot)
    sign_sqrt_out = Lambda(sign_sqrt)(cnn_out_dot)
    print("sign_sqrt_out.shape is:------", sign_sqrt_out.shape)  # (512,512
    l2_norm_out = Lambda(l2_norm)(sign_sqrt_out)
    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    print("the l2_norm_out is:-----------", l2_norm_out)
    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    print("the l2_norm_out.shape is:-----------", l2_norm_out.shape)  # (512,512

    # -------Image Bcnn end

    img_dense = Dense(600)(l2_norm_out)  # ,300
    print('img_dense shape------------', img_dense.shape)
    # img_dense = TimeDistributed(Dense(embedding_dim))(img)
    tweet_avg = AveragePooling1D(pool_size=maxlen)(dropout)  # 600
    print('tweet_avg shape:------', tweet_avg.shape)

    tweet_avg = Flatten()(tweet_avg)  # 600
    print('tweet_avg  Flatten shape:------', tweet_avg.shape)

    tweet_avg_dense = Dense(600)(tweet_avg)

    tweet_repeat = RepeatVector(512)(tweet_avg_dense)  # 49.600
    print('tweet_repeat shape is------------:', tweet_repeat.shape)

    cnn_out_a = l2_norm_out
    cnn_out_shape = img_dense.shape
    print('cnn_out_a shape is:---------', cnn_out_a.shape)  # 49.300

    cnn_out_b = tweet_repeat

    cnn_out_dot = Lambda(batch_dot)([cnn_out_a, cnn_out_b])
    print("cnn_out_dot.shape is :--------", cnn_out_dot.shape)  # (512.512)
    # cnn_out_dot = Reshape([cnn_out_shape[-1]*cnn_out_shape[-1]])(cnn_out_dot)
    sign_sqrt_out = Lambda(sign_sqrt)(cnn_out_dot)
    print("sign_sqrt_out.shape is:------", sign_sqrt_out.shape)  # (512,300
    img_tweet_new = Lambda(l2_norm)(sign_sqrt_out)
    print("the l2_norm_out_1.shape is:-----------", img_tweet_new.shape)  # (300.300


    # set gate    Image  Tweet Fusion

    img_tweet_new_tanh = Activation('tanh')(img_tweet_new)  # Dense(512*300, activation='tanh')(img_tweet_new)
    print('img_tweet_new_tanh shape:------------', img_tweet_new_tanh.shape)

    # merge = Conv1D(100,1, activation='tanh')(img_tweet_new_tanh)
    print('merge shape==============:', img_tweet_new_tanh.shape)
   

    merge_dense = Conv1D(10, 1, activation='tanh')(img_tweet_new_tanh)  # 307200
  
    merge_flatten = Flatten()(merge_dense)

    output = Dense(hashtag_size, activation='softmax')(merge_flatten)

    model = Model(inputs=[tweet, img], output=output)
    # adam = optimizers.Adam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(optimizer='Nadam', loss='categorical_crossentropy', metrics=["accuracy"])

    print(model.summary())
    print("finished building model")
    # plot_model(model, show_shapes=True, to_file='model.png')

    # starts training
    y = np_utils.to_categorical(y, hashtag_size)

    print("starts training")
    best_f1 = 0
    topK = [1, 2, 3, 4, 5]
    for j in range(nb_epoch):

        history = History()

        

        model.fit([x, img_x], y, batch_size=batch_size, epochs=1, verbose=2, callbacks=[history])
        
        # model.save_weights('my_model_weights.h5')
        print(history.history)
        print(len(history.history))
        y_pred = model.predict([valid_x, valid_img_x], batch_size=batch_size, verbose=2)

        y_pred = np.argsort(y_pred, axis=1)
        # argsort
        precision = precision_score(valid_y, y_pred)
        recall = recall_score(valid_y, y_pred)
        F1 = 2 * (precision * recall) / (precision + recall)
        print("Epoch:", (j + 1), "Valid_precision:", precision, "Valid_recall:", recall, "Valid_f1 score:", F1)

        if best_f1 < F1:
            best_f1 = F1
            y_pred = model.predict([test_x, test_img_x], batch_size=batch_size, verbose=2)

            y_pred = np.argsort(y_pred, axis=1)
            for k in topK:
                precision = precision_score(test_y, y_pred, k=k)
                recall = recall_score(test_y, y_pred, k=k)
                hscore = hits_score(test_y, y_pred, k=k)
                F1 = 2 * (precision * recall) / (precision + recall)
                print("\t", "top:", k, "Test: precision:", precision, "recall:", recall, "f1 score:", F1, "hits score:",
                      hscore)


