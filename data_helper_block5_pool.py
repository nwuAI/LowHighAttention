# -*- encoding=utf-8 -*-
import numpy as np
from collections import Counter
import itertools
import os
import pickle as p
import codecs
import h5py
from keras.preprocessing.sequence import pad_sequences
#from IJCAI2017.LSTM_att.Str2Byte import StrToBytes
import keras.backend as K
#K.set_image_data_format('channels_last')



trainFile = "train30000New.txt"
validFile = "valid7500.txt"
testFile = "test7500.txt"

dataPath1 = ""
h5_file1 = "img_train_block5_pool_30000.h5"
h5_file2 = "img_test_block5_pool.h5"
h5_file3 = "img_valid_block5_pool.h5"

f = h5py.File(h5_file1, 'r')
f1 = h5py.File(h5_file2,'r')
f2 = h5py.File(h5_file3,'r')




def load_data():
    maxlen = 0
        
    x, img_x, y, maxlen = load_traindata(maxlen, trainFile)
    valid_x, valid_img_x, valid_y, maxlen = load_validdata(maxlen, validFile)
    test_x, test_img_x, test_y, maxlen = load_testdata(maxlen, testFile)

    print ("Train set size: ", len(x), len(img_x), len(y))
    print ("Valid set size: ", len(valid_x), len(valid_img_x), len(valid_y))
    print ("Test set size: ", len(test_x), len(test_img_x), len(test_y))
    print ("Max word len: ", maxlen)

    vocabulary, vocabulary_inv, hashtagVoc, hashtagVoc_inv = build_vocab(x, y, valid_x, valid_y, test_x, test_y)
    x, img_x, y = build_input_data(x, img_x, y, vocabulary, hashtagVoc, maxlen)
    valid_x, valid_img_x, valid_y = build_valid_data(valid_x, valid_img_x, valid_y, vocabulary, hashtagVoc, maxlen)
    test_x, test_img_x, test_y = build_test_data(test_x, test_img_x, test_y, vocabulary, hashtagVoc, maxlen)
    '''
    print x[0],img_x[0],y[0]
    print valid_x[0],valid_img_x[0],valid_y[0]
    print test_x[0],test_img_x[0],test_y[0]
    '''
    #print("valid_x:",valid_x)
   # print("valid_y:",valid_y)
    #print("valid_img_x:",valid_img_x)



    return [x, img_x, y, valid_x, valid_img_x, valid_y, test_x, test_img_x, test_y, vocabulary, vocabulary_inv,
            hashtagVoc, hashtagVoc_inv, maxlen]


def build_input_data(x, img_x, y, vocabulary, hashtagVoc, maxlen):
      
    x = np.asarray([[vocabulary[w] for w in t if w in vocabulary] for t in x])
    x = pad_sequences(x, maxlen=maxlen).astype(np.int32) #padding the sequence   shape:(76972, 24)
    for i in img_x:
        if i.shape !=(7,7,512):
            print('type:{} shape:{}'.format(type(i),i.shape))
            print(i)
    
    print('ok')
    #img_x = np.stack(img_x,axis=0)  #shape: (76972, )
    img_x = np.asarray(img_x)
    y = np.asarray([hashtagVoc[h] for h in y], dtype=np.int32)

    
     
    return [x, img_x, y]


def build_test_data(x, img_x, y, vocabulary, hashtagVoc, maxlen):
    x = np.asarray([[vocabulary[w] for w in t if w in vocabulary] for t in x])
    x = pad_sequences(x, maxlen=maxlen).astype(np.int32)
    img_x = np.asarray(img_x)
    y = np.asarray([[hashtagVoc[h] for h in hlist] for hlist in y])

    #print('Test img_x is :',img_x)
    return [x, img_x, y]
	
def build_valid_data(x, img_x, y, vocabulary, hashtagVoc, maxlen):
    x = np.asarray([[vocabulary[w] for w in t if w in vocabulary] for t in x])
    x = pad_sequences(x, maxlen=maxlen).astype(np.int32)
    img_x = np.asarray(img_x)
    y = np.asarray([[hashtagVoc[h] for h in hlist] for hlist in y])

    return [x, img_x, y]


def build_vocab(x, y, valid_x, valid_y, test_x, test_y):
    vocabFileName = 'vocabulary_fullNew.pkl'

    if os.path.isfile(vocabFileName):
        print ("loading vocabulary...")
        vocabulary, vocabulary_inv, hashtagVoc, hashtagVoc_inv = p.load(open(vocabFileName,'rb'))
        #print('-----------------------')
        #print('hashtagVoc_inv :',hashtagVoc_inv)
        #print("vocabulary_inv:",vocabulary_inv)

        vocabulary_inv = vocabulary_inv[:500000]
        vocabulary = {x: i + 1 for i, x in enumerate(vocabulary_inv)}
        #print(vocabulary)
    else: 
        print ("calculating vocabulary...")
        text = []
        for x_t in [x, valid_x, test_x]:
            for t in x_t:
                for w in t:
                    text.append(w)

        print (len(text))  #The len:4090042
        word_counts = Counter(text)

        vocabulary_inv = [x[0] for x in word_counts.most_common()]
        
        vocabulary = {x: i + 1 for i, x in enumerate(vocabulary_inv)}
        #print('字典长度是：',len(vocabulary))
        hashtaglist = []
        for h in y:
            hashtaglist.append(h)
        for hlist in valid_y + test_y:
            for h in hlist:
                hashtaglist.append(h)

        hashtag_count = Counter(hashtaglist)
        hashtagVoc_inv = [x[0] for x in hashtag_count.most_common()]
        hashtagVoc = {x: i for i, x in enumerate(hashtagVoc_inv)}
              
        p.dump([vocabulary, vocabulary_inv, hashtagVoc, hashtagVoc_inv], open(vocabFileName, "wb"))
    return [vocabulary, vocabulary_inv, hashtagVoc, hashtagVoc_inv]


def load_testdata(maxlen, Filename):
    print (Filename)
    x = []
    img_x = []
    y = []
    index = 0
    #使用生成器读取文件

    with open(dataPath1 + Filename,'r',encoding='utf-8') as infile:
        for line in infile:
            line = line.strip().split('\t')
            hashtagList = line[2].split('||')
            tweet = line[1].split(' ')
            tweet = [w.rstrip() for w in tweet if w != '']
            maxlen = max(len(tweet), maxlen)
            x.append(tweet)
            y.append(hashtagList)

            data = f1.get(line[0])
            #print('data.type is:-----',type(data))
            #print('data.shape is:-----',data.shape)
            if data == None:
                data = np.zeros(shape=(7,7,512))
            np_data = np.array(data)
            img_x.append(np_data)
            #img_x.append(line[0])
        return x, img_x, y, maxlen

def load_validdata(maxlen, Filename): 
    print (Filename)
    x = []
    img_x = []
    y = []
    index = 0
    with open(dataPath1 + Filename,'r',encoding='utf-8') as infile:
        for line in infile:
            line = line.strip().split('\t')           
            hashtagList = line[2].split('||') #
            tweet = line[1].split(' ')
            tweet = [w.rstrip() for w in tweet if w != '']
            maxlen = max(len(tweet), maxlen)
            x.append(tweet)
            y.append(hashtagList)
            # image matrix store in list      
            data = f2.get(line[0])      
            if data == None:
                data = np.zeros(shape=(7, 7,512))
            np_data = np.array(data)
            img_x.append(np_data)
            #img_x.append(line[0])
        return x, img_x, y, maxlen




def load_traindata(maxlen, Filename):
    print (Filename)
    x = []
    img_x = []
    y = []
    index=0
    cur_hashtag_count = 0
    #print(dataPath + Filename)
    with open(dataPath1 + Filename,'r',encoding='utf-8') as infile:
    #print(dataPath + Filename+'finished')
        for line in infile:
            index+=1
           # print(index)
            if index%100000==0:
                print (index)
           # print(line)
            line = line.strip().split('\t')
            hashtagList = line[2].split('||')
            cur_hashtag_count = len(hashtagList)
           # print('cur_hashtag_count is:-------',cur_hashtag_count)
            tweet = line[1].split(' ')
            tweet = [w.rstrip() for w in tweet if w != '']
            maxlen = max(len(tweet), maxlen)
            for i in range(cur_hashtag_count):
                x.append(tweet)
            for h in hashtagList:
                y.append(h)
            for i in range(cur_hashtag_count):
                data = f.get(line[0])#(14, 14, 512)
                #print('data.type is:-----',type(data))
                #print('data.shape is:-----',data.shape)#(14, 14, 512)
    
                if data == None:
                    data = np.zeros(shape=(7, 7,512))
                #print(data.shape)
                np_data = np.array(data)#(14, 14, 512)
                #print('np_data.shape is:=======',np_data.shape)
                img_x.append(np_data)    
               # print('img_x type is:----------',type(img_x))#<class 'list'>
                #print('img_x.shape is:--------',img_x[0].shape)#img_x.shape is:-------- (14, 14, 512)
                #print('x shape is:-----------',x.shape)         
                #img_x.append(line[0])
                #print('the np data 0is:',np_data)
        print( 'return')
        return x, img_x, y, maxlen


if __name__ == '__main__':
    #print("second going on !")
    load_data()
    #f.close()
   # f1.cloes()
    #f2.close()
    print("Data_helper Finished !")
    
