from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import pandas as pd
from keras.utils.np_utils import to_categorical
from gensim import models
from gensim.models import word2vec
from keras.layers import Dense, Flatten
from keras.layers import Conv1D, Embedding
from keras.models import Sequential
from keras.layers import merge
from keras.models import Model
from sklearn.utils import shuffle
import random
from keras.models import model_from_json  
import os
from sklearn import metrics

MAX_CLASSNAME_LENGTH=5
MAX_PACKAGENAME_LENGTH=7
MAX_SEQUENCE_LENGTH=19
MODEL_NUMBER=1
EMBEDDING_DIM=200

def classname_process(classname):
    for ch in classname:
        if ch.isupper():
            classname=classname.replace(ch," "+ch.lower(),1)
        elif ch.isdigit():
            classname=classname.replace(ch," "+ch)
    name=classname.strip().split(' ')
    sentence=[]
    for word in name:
        if word!='':
            sentence.append(word)
    
    if MAX_CLASSNAME_LENGTH<len(sentence):
        return sentence[:MAX_CLASSNAME_LENGTH]
    sentence=['*']*(MAX_CLASSNAME_LENGTH-len(sentence))+sentence
    return sentence

def packagename_process(packagename):
    for ch in packagename:
        if ch.isupper():
            packagename=packagename.replace(ch,ch.lower(),1)
    name=packagename.strip().split('.')
    sentence=[]
    for word in name:
        if word!='':
            sentence.append(word)

    if MAX_PACKAGENAME_LENGTH<len(sentence):
        return sentence[:MAX_PACKAGENAME_LENGTH]
    sentence=['*']*(MAX_PACKAGENAME_LENGTH-len(sentence))+sentence
    return sentence

def generate_corpus(df):
    texts=[]
    for i in range(df.shape[0]):
        texts+=classname_process(df.iloc[i,3])
        texts+=packagename_process(df.iloc[i,4])
        texts+=packagename_process(df.iloc[i,5])
    content=""
    for i in range(len(texts)):
        content+=texts[i]
        if (i+1)%19==0:
            content+='\n'
        else:
            content+=' '
    f=open('D:/TSE/python/missplaceclass/embedding_model/corpus.txt','w')
    f.write(content)
    f.close()

#生成w2v模型，需要先运行generate_corpus产生语料库
def train_embedding_model(tokenizer):
    texts=[]
    word_index = tokenizer.word_index
    for word, i in word_index.items():
        texts.append(word)
    model = word2vec.Word2Vec(min_count=0, window=20, size=200, iter = 10)
    model.build_vocab([texts])
    model.train([texts], total_examples = model.corpus_count, epochs = 10)
    model.save('D:/TSE/python/missplaceclass/embedding_model/new_model_1.bin')

def get_tokenizer(df):
    texts=[]
    for i in range(df.shape[0]):
        texts+=classname_process(df.iloc[i,3])
        texts+=packagename_process(df.iloc[i,4])
        texts+=packagename_process(df.iloc[i,5])
    tokenizer = Tokenizer(num_words=None)
    tokenizer.fit_on_texts(texts)
    return tokenizer

def build_model(tokenizer):
    embedding_model=word2vec.Word2Vec.load('D:/TSE/python/missplaceclass/embedding_model/new_model_1.bin')
    word_index = tokenizer.word_index
    nb_words = len(word_index)
    embedding_matrix = np.zeros((nb_words+1, EMBEDDING_DIM))
    for word, i in word_index.items():
        if word=='false':
            print(word)
        embedding_vector = embedding_model.wv[word]
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector 
    models=[]
    for i in range(MODEL_NUMBER):
        embedding_layer = Embedding(nb_words + 1,
            EMBEDDING_DIM,
            input_length=MAX_SEQUENCE_LENGTH,
            weights=[embedding_matrix],
            trainable=False)
        model_left = Sequential()
        model_left.add(embedding_layer)
        model_left.add(Conv1D(128, 1, padding = "same", activation='tanh'))
        model_left.add(Conv1D(128, 1, activation='tanh'))
        model_left.add(Conv1D(128, 1, activation='tanh'))
        model_left.add(Flatten())

        model_right = Sequential()
        model_right.add(Conv1D(128, 1, input_shape=(8,1), padding = "same", activation='tanh'))
        model_right.add(Conv1D(128, 1, activation='tanh'))
        model_right.add(Conv1D(128, 1, activation='tanh'))
        model_right.add(Flatten())

        output = merge.Concatenate()([model_left.output, model_right.output]) 
        output=Dense(128, activation='tanh')(output)
        output=Dense(1,activation='sigmoid')(output)
        input_left=model_left.input
        input_right=model_right.input

        model=Model([input_left,input_right],output)
        model.compile(loss='binary_crossentropy',optimizer='Adadelta',metrics=['accuracy'])
        models.append(model)
    return models

def get_data(df,tokenizer):
    texts=[]
    for i in range(df.shape[0]):
        text=''
        cname=classname_process(df.iloc[i,3])
        pname1=packagename_process(df.iloc[i,4])
        pname2=packagename_process(df.iloc[i,5])
        for j in range(len(cname)):
            text+=cname[j]+" "
        for j in range(len(pname1)):
            text+=pname1[j]+" "
        for j in range(len(pname2)):
            text+=pname2[j]+" "
        texts.append(text.strip(' '))
    sequences = tokenizer.texts_to_sequences(texts)
    word_index = tokenizer.word_index
    txt=pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    metric=np.expand_dims(df.iloc[:,6:14], axis=2)
    label=df.iloc[:,14]
    return txt,metric,label
    
def get_data_casestudy(df,tokenizer):
    texts=[]
    for i in range(df.shape[0]):
        text=''
        cname=classname_process(df.iloc[i,3])
        pname1=packagename_process(df.iloc[i,4])
        pname2=packagename_process(df.iloc[i,5])
        for j in range(len(cname)):
            text+=cname[j]+" "
        for j in range(len(pname1)):
            text+=pname1[j]+" "
        for j in range(len(pname2)):
            text+=pname2[j]+" "
        texts.append(text.strip(' '))
    sequences = tokenizer.texts_to_sequences(texts)
    word_index = tokenizer.word_index
    txt=pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    metric=np.expand_dims(df.iloc[:,6:14], axis=2)
    return [txt,metric]

def train(df,tokenizer,projectname):
    print('Getting Training Data...')
    train_txt,train_metric,train_label=get_data(df,tokenizer)
    train_txt=np.array(train_txt)
    train_metric=np.array(train_metric)
    train_label=np.array(train_label)
    print('The Size of Training Set :',train_label.shape[0])
    print('Building Models...')
    models=build_model(tokenizer)
    print('Models Built')
    for i in range(MODEL_NUMBER):
        index=np.arange(train_label.shape[0])
        np.random.shuffle(index)
        size=(int)(train_label.shape[0]*0.4+0.5)

        txt=train_txt[index[:size]]
        metric=train_metric[index[:size]]
        label=train_label[index[:size]]

        models[i].fit([txt,metric],label,epochs=3,verbose=2)

        json_string = models[i].to_json()
        #open('D:/TSE/python/missplaceclass/models/'+projectname+'-'+(str)(i)+'.json','w').write(json_string)
        #models[i].save_weights('D:/TSE/python/missplaceclass/models/'+projectname+'-'+(str)(i)+'.h5')
    return models

def load_models(projectname):
    models=[]
    for i in range(MODEL_NUMBER):
        model=model_from_json(open('D:/TSE/python/missplaceclass/models/'+projectname+'-'+(str)(i)+'.json').read())
        model.load_weights('D:/TSE/python/missplaceclass/models/'+projectname+'-'+(str)(i)+'.h5')
        models.append(model)
    return models

def test_each(df,tokenizer,models):
    print('Getting Testing Data...')
    test_txt,test_metric,test_label=get_data(df,tokenizer)
    predicts=[]
    for i in range(MODEL_NUMBER):
        predicts.append(models[i].predict([test_txt,test_metric]))
    pro=[]
    for i in range(test_label.shape[0]):
        temp=0.0
        for j in range(MODEL_NUMBER):
            temp+=predicts[j][i]
        pro.append(temp/MODEL_NUMBER)
    eval_each(pro,np.array(test_label))
    return np.array(pro),np.array(test_label)

def eval_each(y_pre,y_test):
    tp,tn,fp,fn=0,0,0,0
    for i in range(len(y_pre)):
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

def case_study_test(df,tokenizer,models,projectname):
    classes=df.drop_duplicates(['classqualifiedname'])
    cnt=0
    record=[]
    for i in range(classes.shape[0]):
        clazz=classes.iloc[i,2]
        items=df[df.classqualifiedname==clazz]
        x=get_data_casestudy(items,tokenizer)

        pre=[]
        for j in range(MODEL_NUMBER):
            pre.append(models[j].predict(x))
        res=[]
        for j in range(pre[0].shape[0]):
            temp=0.0
            for k in range(MODEL_NUMBER):
                temp+=pre[k][j]
            res.append(temp/MODEL_NUMBER)

        pos=np.argmax(res)

        if res[pos]<=0.5:
            continue


        if items.iloc[pos,4].startswith(items.iloc[pos,5]) or items.iloc[pos,5].startswith(items.iloc[pos,4]):
            continue
        print('Class : ',clazz,'Target : ',items.iloc[pos,5],res[pos])
        #record=np.concatenate((record,np.array([projectname,clazz,items.iloc[pos,5],res[pos]])),axis=0)
        record.append([projectname,clazz,items.iloc[pos,5],res[pos]])
        cnt+=1

    print(cnt,classes.shape[0])
    return record

def eval(y_pre,y_test,target_correct):
    tp,tn,fp,fn=0,0,0,0
    for i in range(len(y_pre)):
        if y_pre[i]>0.5:
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

    print('Target Correct : ',target_correct)
    print('Accuracy : ',target_correct*1.0/tp)

    #return 2*P*R/(P+R)

def test(df_classes,df_items,tokenizer,models):
    
    classes=df_items.drop_duplicates('classqualifiedname')
    score=[]
    y=[]
    target_correct=0
    tp,tn,fp,fn=0,0,0,0

    for i in range(classes.shape[0]):
        clazz=classes.iloc[i,2]
        label=df_classes[df_classes.qualifiedclassname==clazz].iloc[0,1]
        y.append(label)
        target=df_classes[df_classes.qualifiedclassname==clazz].iloc[0,2]
        items=df_items[df_items.classqualifiedname==clazz]
        x=get_data_casestudy(items,tokenizer)
        pre=[]
        for i in range(MODEL_NUMBER):
            pre.append(models[i].predict(x))

        pre_y=[]
        res=[]
        for i in range(pre[0].shape[0]):
            temp1=0.0
            temp2=0
            for j in range(MODEL_NUMBER):
                temp1+=pre[j][i]
                #if pre[j][i]>=0.5:
                    #temp2+=1
            temp1/=MODEL_NUMBER
            res.append(temp1)
            '''
            if temp2>=3:
                pre_y.append(1)
            else:
                pre_y.append(0)
            '''
        pos=np.argmax(res)
        if res[pos]>0.5 and label==1:
            if items.iloc[pos,5]==target:
                target_correct+=1
        score.append(res[pos])

    eval(score,y,target_correct)
    return np.array(score),np.array(y),target_correct
        




