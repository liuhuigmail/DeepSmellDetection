from keras.models import Sequential, load_model, Model, model_from_json
from keras.layers import Conv1D, MaxPooling1D, Embedding,LSTM,GaussianNoise,Masking
from keras.layers import Dense, Flatten, merge, Dropout, Reshape, Input,BatchNormalization,Activation
from keras import regularizers
from sklearn import metrics
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import keras
import os
import shutil
import preprocess
import time
import numpy as np
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

total_y_pre=[]
total_y_test=[]

EMBEDDING_DIM=200
MAX_SEQUENCE_LENGTH = 50
MAX_JACCARD_LENGTH = 30
INC_BATCH_SIZE = 80000

W2V_MODEL_DIR = 'D:/TSE/python/largeclass/new_model.bin'
TRAIN_SET_DIR = 'D:/TSE/largeclass/data'

tokenizer = preprocess.get_tokenizer()
all_word_index = tokenizer.word_index
embedding_matrix = preprocess.get_embedding_matrix(all_word_index, W2V_MODEL_DIR, dim=EMBEDDING_DIM)

MODEL_NUMBER=5
SUBSETSIZE=0.8

def eval(y_pre,y_test):
        tp,tn,fp,fn=0,0,0,0
        for i in range(len(y_pre)):
                total_y_pre.append(y_pre[i])
                total_y_test.append(y_test[i])
                if y_pre[i]>=0.5:
                        if y_test[i]==1:
                                tp+=1
                        else:
                                fp+=1
                else:
                        if y_test[i]==1:
                                fn+=1
                        else:
                                tn+=1

        print("tp : ",tp)
        print("tn : ",tn)
        print("fp : ",fp)
        print("fn : ",fn)
        P=tp*1.0/(tp+fp)
        R=tp*1.0/(tp+fn)
        print("Precision : ",P)
        print("Recall : ",R)
        print("F1 : ",2*P*R/(P+R))

        a=tp+fp
        b=tp+fn
        c=tn+fp
        d=tn+fn
        print("MCC : ",(tp*tn-fp*fn)/((a*b*c*d)**0.5))
        print("AUC : ",metrics.roc_auc_score(y_test,y_pre))

        return 2*P*R/(P+R)

def getModels():
        models=[]
        for i in range(MODEL_NUMBER):
                method_a = Input(shape=(MAX_SEQUENCE_LENGTH,EMBEDDING_DIM),name='method_a')
                metric_a = Input(shape=(12,),name='metric_a')
                masking_layer = Masking(mask_value=0,input_shape=(MAX_SEQUENCE_LENGTH,EMBEDDING_DIM))
                lstm_share = LSTM(output_dim=2,activation='sigmoid',init='uniform')
                embedding_a = masking_layer(method_a)
                lstm_a = lstm_share(embedding_a)
                dense_share2 = Dense(12,activation='tanh',init='uniform')
                mtrdense_a= dense_share2(metric_a)
                m_j_merged_a = keras.layers.concatenate([lstm_a,mtrdense_a],axis=-1)
                dense1_a = Dense(4,activation='tanh',init='zero')(m_j_merged_a)
                total_dropout = Dropout(0.6)(dense1_a)
                total_output = Dense(1,activation='sigmoid',name='output')(total_dropout)
                model = Model(inputs=[method_a,metric_a],output=total_output)
                sgd = keras.optimizers.SGD(lr=0.1, momentum=0.9, decay=0.0, nesterov=False)
                model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['binary_accuracy'])
                models.append(model)
        print("Moldes Number:",len(models))
        return models

def getData(projectname):
        x_train, y_train= preprocess.get_xy_train(projectname,tokenizer=tokenizer, mn_maxlen=MAX_SEQUENCE_LENGTH,embedding_matrix=embedding_matrix)
        x_test, y_test = preprocess.get_xy_test(projectname,tokenizer=tokenizer, maxlen=MAX_SEQUENCE_LENGTH,embedding_matrix=embedding_matrix)
        return x_train,y_train,x_test,y_test

def train(projectname,models,x_train,y_train):
        for k in range(MODEL_NUMBER):
                mn0,mn1,metrics0,metrics1=[],[],[],[]
                print("Training ",k,"th model...")
                for i in range(y_train.shape[0]):
                        if y_train[i]==0:
                                mn0.append(x_train[0][i])
                                metrics0.append(x_train[1][i])
                        else:
                                mn1.append(x_train[0][i])
                                metrics1.append(x_train[1][i])

                size=(int)(len(mn1)*SUBSETSIZE)
                temp_set=[]
                mn0=np.array(mn0)
                mn1=np.array(mn1)
                metrics0=np.array(metrics0)
                metrics1=np.array(metrics1)

                indices0=np.arange(mn0.shape[0])
                indices1=np.arange(mn1.shape[0])

                #np.random.shuffle(indices0)
                #np.random.shuffle(indices1)
                indices0=shuffle(indices0)
                indices1=shuffle(indices1)

                mn0=mn0[indices0[:size]]
                mn1=mn1[indices1[:size]]
                metrics0=metrics0[indices0[:size]]
                metrics1=metrics1[indices1[:size]]
                temp_set=[]
                for i in range(size):
                        temp_set.append([mn0[i],metrics0[i],0])
                        temp_set.append([mn1[i],metrics1[i],1])

                np.random.shuffle(temp_set)
                
                y=[]
                mn=[]
                metrics=[]
                for i in range(len(temp_set)):
                        mn.append(temp_set[i][0])
                        metrics.append(temp_set[i][1])
                        y.append(temp_set[i][2])

                mn=np.array(mn)
                metrics=np.array(metrics)

                x=[mn,metrics]
                y=np.array(y)

                models[k].fit(x, y,epochs=10,batch_size=5,verbose=0)
                json_string = models[k].to_json()
                open('D:/TSE/python/largeclass/model/'+projectname+'-'+(str)(k)+'.json','w').write(json_string)
                models[k].save_weights('D:/TSE/python/largeclass/model/'+projectname+'-'+(str)(k)+'.h5')
        return models

def test(models,x_test,y_test):
        mn0,mn1,metrics0,metrics1=[],[],[],[]
        for i in range(y_test.shape[0]):
                if y_test[i]==0:
                        mn0.append(x_test[0][i])
                        metrics0.append(x_test[1][i])
                else:
                        mn1.append(x_test[0][i])
                        metrics1.append(x_test[1][i])

        size=(int)(len(mn0)/0.9329*0.0671+0.5)

        temp_set=[]
        mn0=np.array(mn0)
        mn1=np.array(mn1)
        metrics0=np.array(metrics0)
        metrics1=np.array(metrics1)

        indices0=np.arange(mn0.shape[0])
        indices1=np.arange(mn1.shape[0])

        np.random.shuffle(indices0)
        np.random.shuffle(indices1)


        mn1=mn1[indices1[:size]]
        metrics1=metrics1[indices1[:size]]

        temp_set=[]
        for i in range(mn0.shape[0]):
                temp_set.append([mn0[i],metrics0[i],0])
        for i in range(size):
                temp_set.append([mn1[i],metrics1[i],1])

        np.random.shuffle(temp_set)
        print('---',len(temp_set),'---')
        y=[]
        mn=[]
        metrics=[]
        for i in range(len(temp_set)):
                mn.append(temp_set[i][0])
                metrics.append(temp_set[i][1])
                y.append(temp_set[i][2])

        mn=np.array(mn)
        metrics=np.array(metrics)

        x=[mn,metrics]
        y=np.array(y)
        predict=[]
        for i in range(MODEL_NUMBER):
                predict.append(models[i].predict(x))
        y_pre=[]
        for i in range(y.shape[0]):
                t=0.0
                for j in range(MODEL_NUMBER):
                        t+=predict[j][i]
                y_pre.append(t/MODEL_NUMBER)
        return eval(y_pre,y)


projects=['android-backup-extractor-20140630',"AoI30","areca-7.4.7","freeplane-1.3.12","grinder-3.6","jedit","jexcelapi_2_6_12","junit-4.10","pmd-5.2.0","weka"]


for i in range(len(projects)):

        print("Build Models")
        models=getModels()
        print("Get Data")
        x_train,y_train,x_test,y_test=getData(projects[i])
        print("Start Training")
        models=train(projects[i],models,x_train,y_train)
        print('*'*80)
        print(projects[i])
        f1=test(models,x_test,y_test)



print('*'*80)
print("Final")
eval(total_y_pre,total_y_test)


print("Done")
