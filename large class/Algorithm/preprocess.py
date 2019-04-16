import keras.preprocessing.sequence as s
from keras.preprocessing.text import Tokenizer
import numpy as np
from keras.utils.np_utils import to_categorical
#from compiler.ast import flatten
from gensim import models
import os


import collections
def flatten(x):
    result = []
    for el in x:
        if isinstance(x, collections.Iterable) and not isinstance(el, str):
            result.extend(flatten(el))
        else:
            result.append(el)
    return result


def get_data(path,maxlen,tokenizer,embedding_matrix,projectname):
    mn_datas = []
    train_projects = []
    for project in sorted(os.listdir(path)):
        if project==projectname:
            continue
        train_projects.append(project)
        mn_path = path+"/"+project+"/mn_train.txt"
        with open(mn_path) as f:
            for line in f:
                #print line
                data_mn_list = line.strip('\n').strip('\r').split('.')
                #print data_mn_list
                identifiers = []
                tp_sequs = tokenizer.texts_to_sequences(data_mn_list)
                #print tp_sequs
                for tp_sequ in tp_sequs:
                    if len(tp_sequ)>0:
                        #print tp_sequ
                        embeddings = []
                        for tp in tp_sequ:
                            embedding = embedding_matrix[tp]
                            embeddings.append(embedding)
                        identifier_embedding = sum(embeddings)/len(embeddings)
                        #print('shape of identifiers:',np.asarray(identifier_embedding).shape)
                        identifiers.append(identifier_embedding)
                        #print identifier_embedding
                #print('shape of identifiers:',np.asarray(identifiers).shape)
                #input()
                #print identifiers
                mn_datas.append(identifiers)
    #print mn_datas[0]
    mn_datas = s.pad_sequences(mn_datas,maxlen=maxlen,dtype='float32')
    #print mn_datas[0][49]
    #input()
    print ('training ',train_projects)
    return mn_datas

def get_labels(path,projectname):
    labels =[]
    for project in sorted(os.listdir(path)):
        if project==projectname:
            continue

        lb_path = path+"/"+project+"/lb_train.txt"
        with open(lb_path) as f:
            for line in f:
                labels.append(int(line))
    labels = np.asarray(labels)
    #labels = labels.reshape((labels.shape[0],1))

    nb_labels_ONE = 0
    nb_labels_ZERO = 0
    for i in labels:
        if(i==0):
            nb_labels_ZERO=nb_labels_ZERO+1
        if(i==1):
            nb_labels_ONE=nb_labels_ONE+1
    print ('nb_labels_ONE: ',nb_labels_ONE)
    print ('nb_labels_ZERO: ',nb_labels_ZERO)
    return labels

def get_metrics(path,projectname):
    metrics_datas = []
    for project in sorted(os.listdir(path)):
        if project==projectname:
            continue
        mt_path = path+"/"+project+"/mt_train.txt"
        with open(mt_path) as f:
            for line in f:
                metrics = line.split()
                metrics_data = []
                for metric in metrics:
                    metrics_data.append(float(metric))
                metrics_datas.append(metrics_data)
    #print metrics_datas
    return np.asarray(metrics_datas)


def get_val(jc_path):
    f = open(jc_path)
    indices = np.arrange(len(f.readlines()))
    np.random.shuffle(indices)
    f.close()

def get_xy_train(projectname, mn_maxlen, tokenizer,embedding_matrix):
    path='D:/TSE/python/largeclass/data/'
    mn_datas = get_data(path, mn_maxlen, tokenizer,embedding_matrix,projectname)
    metrics_datas = get_metrics(path,projectname)
    labels = get_labels(path,projectname)

    """np_validation_samples = int(0.2*mn_datas1.shape[0])

    mn_datas = mn_datas1[:-np_validation_samples].repeat(100,axis=0)
    metrics_datas = metrics_datas1[:-np_validation_samples].repeat(100,axis=0)
    labels = labels1[:-np_validation_samples].repeat(100,axis=0)"""
    '''
    print(type(mn_datas),type(metrics_datas),type(labels))
    print('Shape of name tensor:', mn_datas.shape)
    print('Shape of metrics tensor:', metrics_datas.shape)
    print('Shape of label tensor:', labels.shape)
    print('*'*80)
    '''
    np.random.seed(0)
    indices = np.arange(mn_datas.shape[0])
    np.random.shuffle(indices)
    mn_datas = np.asarray(mn_datas)[indices]
    metrics_datas = np.asarray(metrics_datas)[indices]
    labels = np.asarray(labels)[indices]
    #np_validation_samples = int(0.8*mn_datas.shape[0])

    x_train = []
    x_train.append(mn_datas)
    x_train.append(metrics_datas)
    y_train = labels
    """x_val = []
    x_val.append(mn_datas1[-np_validation_samples:])
    x_val.append(metrics_datas1[-np_validation_samples:])
    y_val = labels1[-np_validation_samples:]
    #return x_train,y_train,x_val,y_val"""
    return x_train,y_train

def get_xy_test(test_project, maxlen, tokenizer,embedding_matrix):
    path='D:/TSE/python/largeclass/data/'
    labels =[]
    mn_datas = []
    metrics_datas = []
    lb_path = path+"/"+test_project+"/lb_train.txt"
    with open(lb_path) as f:
        for line in f:
            labels.append(int(line))

    mn_path = path+"/"+test_project+"/mn_train.txt"
    with open(mn_path) as f:
        for line in f:
            data_mn_list = line.strip('\n').strip('\r').strip().split('.')
            #print data_mn_list
            identifiers = []
            tp_sequs = tokenizer.texts_to_sequences(data_mn_list)
            #print tp_sequs
            for tp_sequ in tp_sequs:
                if len(tp_sequ):
                    embeddings = []
                    for tp in tp_sequ:
                        embedding = embedding_matrix[tp]
                        embeddings.append(embedding)
                    identifier_embedding = sum(embeddings)/len(embeddings)
                    #print('shape of identifiers:',np.asarray(identifier_embedding).shape)
                    identifiers.append(identifier_embedding)
            #print('shape of identifiers:',np.asarray(identifiers).shape)
            #input()
            mn_datas.append(identifiers)

    mt_path = path+"/"+test_project+"/mt_train.txt"
    with open(mt_path) as f:
        for line in f:
            metrics = line.split()
            metrics_data = []
            for metric in metrics:
                metrics_data.append(float(metric))
            metrics_datas.append(metrics_data)

    labels = np.asarray(labels)
    #labels = labels.reshape((labels.shape[0],1))

    nb_labels_ONE = 0
    nb_labels_ZERO = 0
    for i in labels:
        if(i==0):
            nb_labels_ZERO=nb_labels_ZERO+1
        if(i==1):
            nb_labels_ONE=nb_labels_ONE+1
    '''
    print ('nb_labels_ONE: ',nb_labels_ONE)
    print ('nb_labels_ZERO: ',nb_labels_ZERO)
    '''
    mn_datas = s.pad_sequences(mn_datas,maxlen=maxlen,dtype='float32')
    #print metrics_datas
    metrics_datas = np.asarray(metrics_datas)

    """print 'Shape of name tensor:', mn_datas.shape
    print 'Shape of metrics tensor:', metrics_datas.shape
    print 'Shape of label tensor:', labels.shape"""

    """np.random.seed(0)
    indices = np.arange(mn_datas.shape[0])
    np.random.shuffle(indices)

    mn_datas1 = np.asarray(mn_datas)[indices]
    metrics_datas1 = np.asarray(metrics_datas)[indices]
    labels1 = np.asarray(labels)[indices]
    #print indices
    print mn_datas1.shape
    print metrics_datas1.shape
    print labels1.shape
    for e,i in enumerate(indices):
        if mn_datas1[e].all()!= mn_datas[i].all():
            print 'error!'
            return None,None
        if metrics_datas1[e].all()!=metrics_datas[i].all():
            print 'error!'
            return None,None
        if labels1[e]!=labels[i]:
            print 'error!'
            return None,None"""

    x_val = []
    x_val.append(mn_datas)
    x_val.append(metrics_datas)
    y_val = labels
    """x_val = []
    x_val.append(mn_datas1[-np_validation_samples:])
    x_val.append(metrics_datas1[-np_validation_samples:])
    y_val = labels1[-np_validation_samples:]
    #return x_train,y_train,x_val,y_val"""
    return x_val,y_val

def get_test_data(path, maxlen, tokenizer):
    texts_first = []
    texts_second = []
    for test_index in sorted(os.listdir(path)):
        test_class_path = path+test_index+'/'
        mn_path = test_class_path+'mn_train.txt'
        print('in '+mn_path)

        with open(mn_path) as f:
            for line in f:
                identifiers = line.split('.')
                identifier0 = identifiers[0]
                identifier1 = identifiers[1]
                words0 = identifier0.split()
                words1 = identifier1.strip('\b').split()
                words0 = " ".join(words0)
                words1 = " ".join(words1)
                texts_first.append(words0)
                texts_second.append(words1)
    sequences_first = tokenizer.texts_to_sequences(texts_first)
    sequences_second = tokenizer.texts_to_sequences(texts_second)
    data1 = s.pad_sequences(sequences_first, maxlen=maxlen)
    data2 = s.pad_sequences(sequences_second, maxlen=maxlen)
    return data1,data2

def get_x_pre(mn_path, jc_path, mn_maxlen, jc_maxlen, tokenizer):
    data1,data2 = get_data(mn_path, mn_maxlen, tokenizer)
    jaccard1,jaccard2 = get_jaccard(jc_path, jc_maxlen)
    x_pre = []
    x_pre.append(data1)
    x_pre.append(data2)
    x_pre.append(jaccard1)
    x_pre.append(jaccard2)
    return x_pre

def get_embedding_matrix(all_word_index, model_path, dim):
    print('Preparing embedding matrix.')
    # prepare embedding matrix
    #nb_words = min(MAX_NB_WORDS, len(word_index))
    embedding_matrix = np.zeros((len(all_word_index) + 1, dim))
    w2v_model = models.word2vec.Word2Vec.load(model_path)
    for word, i in all_word_index.items():
        #f i > MAX_NB_WORDS:
        #    continue
        try:
            embedding_vector = w2v_model[word]
        except KeyError:
            continue

            """if embedding_vector is not None:
            # words not found in embedding index will be all-zeros."""
        embedding_matrix[i] = embedding_vector # word_index to word_embedding_vector ,<20000(nb_words)
    return embedding_matrix

def get_tokenizer():
    path='D:/TSE/python/largeclass/data'
    texts = []
    for sett in sorted(os.listdir(path)):
        full_path = path+'/'+sett+'/full_mn/mn_full.txt'
        f = open(full_path)
        for line in f:
            texts.append(line)
        f.close()

    tokenizer = Tokenizer(num_words=None)
    tokenizer.fit_on_texts(texts)
    #all_word_index = tokenizer.word_index
    return tokenizer
