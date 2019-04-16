import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import preprocessing,metrics
from sklearn.utils import shuffle
from sklearn.neural_network import MLPClassifier
from sklearn import neighbors
from sklearn import svm
from sklearn.externals import joblib
import time

MODEL_NUMBER=5
SUBSET_SIZE=0.8

tr=[]
trp=[]

def train(projectName):
    print(projectName)
    df=pd.read_csv("D:/TSE/python/longmethod/data_semi.csv",encoding='ISO-8859-1')

    
    df=df[(df.projectname!=projectName)]


    data0=df[(df.Label==0)]
    data1=df[(df.Label==1)]

    num=(int)(data1.shape[0]*SUBSET_SIZE)
    
    models=[]
    for i in range(MODEL_NUMBER):
        print("training ",i+1,"th model")
        data0=shuffle(data0)
        data1=shuffle(data1)

        train_set0=data0.iloc[:num,:]
        train_set1=data1.iloc[:num,:]

        data=train_set0.append(train_set1)
        data=shuffle(data)

        x=data.iloc[:,5:14]
        y=np.array(data.iloc[:,14])

        #x=preprocessing.scale(x,axis=1,with_mean=True,with_std=True,copy=True)

        clf=MLPClassifier(hidden_layer_sizes=(16,8,4), max_iter=200,)
        
        #clf=RandomForestClassifier(n_estimators=100)

        clf.fit(x,y)

        #save model
        #joblib.dump(clf,"D:/TSE/python/longmethod/model/"+projectName+"_"+str(i)+".joblib")
        
        
        models.append(clf)

    return models

def eval(tp,tn,fp,fn):
    print("tp : ",tp)
    print("tn : ",tn)
    print("fp : ",fp)
    print("fn : ",fn)
    if tp==0 or tn==0 or fp==0 or fn==0:
        return 1
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
    
    return 2*P*R/(P+R)

def test(models,projectName):
    print(projectName)
    df=pd.read_csv("D:/TSE/python/longmethod/data_semi.csv",encoding='ISO-8859-1')

    df=df[(df.projectname==projectName)]
    predicts=[]
    predicts_proba=[]
    for i in range(MODEL_NUMBER):
        clf=models[i]
        x=df.iloc[:,5:14]
        
        #x=preprocessing.scale(x,axis=1,with_mean=True,with_std=True,copy=True)

        predict=clf.predict(x)
        predict_proba=clf.predict_proba(x)

        predicts.append(predict)
        predicts_proba.append(predict_proba)
    result=[]
    for i in range(len(predicts[0])):
        total=0
        for j in range(MODEL_NUMBER):
            total=total+predicts[j][i]
        if total>=3:
            result.append(1)
        else:
            result.append(0)

    rp=[]
    for i in range(len(predicts_proba[0])):
        total=0
        for j in range(MODEL_NUMBER):
            total=total+predicts_proba[j][i][1]
        rp.append(total/MODEL_NUMBER)
        trp.append(total/MODEL_NUMBER)

    y=np.array(df.iloc[:,14])
    print('*'*80)
    print("AUC : ",metrics.roc_auc_score(y,rp))
    print('*'*80)
    tp,tn,fp,fn=0,0,0,0

    for i in range(len(y)):
        tr.append(y[i])
        if result[i]==y[i]:
            if result[i]==0:
                tn=tn+1
            else:
                tp=tp+1
        else:
            if result[i]==0:
                fn=fn+1
            else:
                fp=fp+1
    
    return tp,tn,fp,fn

def load_models(projectName):
    models=[]

    for i in range(MODEL_NUMBER):
        #clf=joblib.load('D:/Longmethod/model/4677/'+projectName+"_"+str(i)+'.joblib')
        clf=joblib.load("D:/TSE/python/longmethod/model/"+projectName+"_"+str(i)+".joblib")

        models.append(clf)

    return models


projects = ['areca-7.4.7','freeplane-1.3.12','jedit','junit-4.10','pmd-5.2.0','weka','android-backup-extractor-20140630','grinder-3.6','AoI30','jexcelapi_2_6_12']

ttp,ttn,tfp,tfn=0,0,0,0
for i in range(10):

    print("------------------------------------")
    ss=time.time()
    models=train(projects[i])
    print('#####################',time.time()-ss)
    #models=load_models(projects[i])
    ss=time.time()
    tp,tn,fp,fn=test(models,projects[i])
    print(time.time()-ss)
    ttp=ttp+tp
    ttn=ttn+tn
    tfp=tfp+fp
    tfn=tfn+fn
    eval(tp,tn,fp,fn)
print("------------------------------------")
print("Final Evaluation:")
ans=eval(ttp,ttn,tfp,tfn)
print("AUC : ",metrics.roc_auc_score(tr,trp))


