# -*- coding: utf-8 -*-
import numpy as np
import time
import os
np.random.seed(1337)

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Flatten
from keras.layers import Conv1D, Embedding
from keras.models import Sequential
from keras.layers import merge
from keras.models import Model
from sklearn.utils import shuffle
import random
from keras.models import model_from_json  
from gensim.models import word2vec

MAX_SEQUENCE_LENGTH = 15 

def nfromm(m, n, unique=True):
    """
    从[0, m)中产生n个随机数
    :param m:
    :param n:
    :param unique:
    :return:
    """
    if unique:
        box = [i for i in range(m)]
        out = []
        for i in range(n):
            index = random.randint(0, m - i - 1)

            # 将选中的元素插入的输出结果列表中
            out.append(box[index])

            # 元素交换，将选中的元素换到最后，然后在前面的元素中继续进行随机选择。
            box[index], box[m - i - 1] = box[m - i - 1], box[index]
        return out
    else:
        # 允许重复
        out = []
        for _ in range(n):
            out.append(random.randint(0, m - 1))
        return out


def getsubset(x,y):
    t1=[[],[]]
    t2=[[],[]]
    t3=[[],[]]

    for i in range(len(y)):
        if y[i][0]==1:
            t1[0].append(x[0][i])
            t2[0].append(x[1][i])
            t3[0].append(y[i])
        else:
            t1[1].append(x[0][i])
            t2[1].append(x[1][i])
            t3[1].append(y[i])

    
    num=(int)(len(t1[1])*0.8)############################

    index=nfromm(len(t1[1]),num)

    tt1=[]
    tt2=[]
    tt3=[]

    for i in range(len(index)):
        tt1.append(t1[1][index[i]])
        tt2.append(t2[1][index[i]])
        tt3.append(t3[1][index[i]])


    index=nfromm(len(t1[0]),num)

    for i in range(len(index)):
        tt1.append(t1[0][index[i]])
        tt2.append(t2[0][index[i]])
        tt3.append(t3[0][index[i]])
        

    t=[np.array(tt1),np.array(tt2)]

    return t,np.array(tt3)

projects=['android-backup-extractor-20140630',"AoI30","areca-7.4.7","freeplane-1.3.12","grinder-3.6","jedit","jexcelapi_2_6_12","junit-4.10","pmd-5.2.0","weka"]
#projects=["areca-7.4.7","freeplane-1.3.12","jedit","jexcelapi_2_6_12","junit-4.10","pmd-5.2.0","weka"]
basepath="D:/TSE/python/featureenvy/"
ff1=0.0

model_num=1####################################################################


def train():#select optional model
    for kk in range(len(projects)):#len(projects)
        ss=time.time()

        print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
        print(projects[kk])
        print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")


        distances = []  # TrainSet
        labels = []  # 0/1
        texts = [] # ClassNameAndMethodName
        MAX_SEQUENCE_LENGTH = 15 
        EMBEDDING_DIM = 200 # Dimension of word vector

        embedding_model=word2vec.Word2Vec.load(basepath+"new_model.bin")

        with open(basepath+"data_compare/"+projects[kk]+"/train_Distances.txt",'r') as file_to_read:
        #with open("D:/data/7#Fold/train-weka"+"/train_distances.txt",'r') as file_to_read:
            for line in file_to_read.readlines():
                values = line.split()
                distance = values[:2]
                distances.append(distance)
                label =values[2:]
                labels.append(label)

            
        with open(basepath+"data_compare/"+projects[kk]+"/train_Names.txt",'r') as file_to_read:
        #with open("D:/data/7#Fold/train-weka"+"/train_names.txt",'r') as file_to_read:
            for line in file_to_read.readlines():
                texts.append(line)

        print('Found %s train_distances.' % len(distances))

        tokenizer = Tokenizer(num_words=None)
        tokenizer.fit_on_texts(texts)
        sequences = tokenizer.texts_to_sequences(texts)
        word_index = tokenizer.word_index
        data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)   
        distances = np.asarray(distances)
        labels = to_categorical(np.asarray(labels))

        print('Shape of train_data tensor:', data.shape)
        print('Shape of train_label tensor:', labels.shape)

        x_train = []
        x_train_names = data
        x_train_dis = distances 
        x_train_dis = np.expand_dims(x_train_dis, axis=2)

        x_train.append(x_train_names)
        x_train.append(np.array(x_train_dis))
        y_train = np.array(labels)
        
        for index in range(model_num):########################
            print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
            print(projects[kk],'---',index+1)
            print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
            print ("start time: "+time.strftime("%Y/%m/%d  %H:%M:%S"))

            x_train,y_train=getsubset(x_train,y_train)

            nb_words = len(word_index)
            embedding_matrix = np.zeros((nb_words+1, EMBEDDING_DIM))
            for word, i in word_index.items():
                embedding_vector = embedding_model.wv[word]
                if embedding_vector is not None:
                    embedding_matrix[i] = embedding_vector 

            embedding_layer = Embedding(nb_words + 1,
                                EMBEDDING_DIM,
                                input_length=MAX_SEQUENCE_LENGTH,
                                weights=[embedding_matrix],
                                trainable=False)

            print('Training model.')

            model_left = Sequential()
            model_left.add(embedding_layer)
            model_left.add(Conv1D(128, 1, padding = "same", activation='tanh'))
            model_left.add(Conv1D(128, 1, activation='tanh'))
            model_left.add(Conv1D(128, 1, activation='tanh'))
            model_left.add(Flatten())

            model_right = Sequential()
            model_right.add(Conv1D(128, 1, input_shape=(2,1), padding = "same", activation='tanh'))
            model_right.add(Conv1D(128, 1, activation='tanh'))
            model_right.add(Conv1D(128, 1, activation='tanh'))
            model_right.add(Flatten())

            output = merge.Concatenate()([model_left.output, model_right.output]) 

            output=Dense(128, activation='tanh')(output)
            output=Dense(2,activation='sigmoid')(output)

            input_left=model_left.input
            input_right=model_right.input

            model=Model([input_left,input_right],output)

            model.compile(loss='binary_crossentropy',optimizer='Adadelta',metrics=['accuracy'])

            model.fit(x_train, y_train,epochs=3,verbose=2)


            #json_string = model.to_json()
            #open(basepath+"models_compare/"+projects[kk]+"_"+str(index)+".json",'w').write(json_string)
            #model.save_weights(basepath+"models_compare/"+projects[kk]+"_"+str(index)+'.h5')
        print('########################',time.time()-ss)
train()
