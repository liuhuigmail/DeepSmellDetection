# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 14:51:07 2018

@author: xzf0724----2
从数据库提取数据
"""

import pymysql
import numpy as np
import random
import time

from gensim.models import word2vec

#连接SQL
conn=pymysql.connect(host="127.0.0.1",user="root",passwd="12345",db="testdata",charset="utf8") 
cur = conn.cursor()

MAX_NUM_WORD = 5 #最多单词数
words = []
Train_path = 'D:/TSE/python/featureenvy/data_compare/'#train dataset path
Test_path = 'D:/TSE/python/featureenvy/data_compare/'#test dataset path

def participle(name):#按大小写分词
    for ch in name:
        if(ch.isupper()):
            name = name.replace(ch, " "+ch.lower(), 1)
        else:
            if(ch == "_" or ch == "$"):
                name = name.replace(ch," ",1)
            else:
                if(ch.isdigit()):
#                    print(ch)
                    name = name.replace(ch, " "+ch)
#    print(name)
    name1 = name.strip().split(" ")
    sentence = []
    for word in name1:
        if(word != ''):
            sentence.append(word)
#    print(sentence)
    return sentence
def preprocessingNames(sentence):
    sentence = participle(sentence)
#    print(sentence)
    num = len(sentence)
    sentence1 = ''
    if(num < MAX_NUM_WORD):
        for i in range(MAX_NUM_WORD-num):
            sentence1 += '*'+' '
        sentence1 = sentence1.strip().split(" ")
        sentence1 += sentence
#        print("num<6-------sentence----%s"%sentence1)
    if(num > MAX_NUM_WORD):
        num = MAX_NUM_WORD
        num1 = MAX_NUM_WORD
        sentence1 = ''
        for word in sentence:
            if(num1 == MAX_NUM_WORD): sentence1 += word 
            else: sentence1 += ' '+word
            num1 = num1-1
            if(num1 == 0): break
        sentence = sentence1.strip().split(" ")
#        print('preprocessingNames-----%s'%sentence1)
    if(num == MAX_NUM_WORD):
        sentence1 = sentence
    return sentence1
def getModel():
    model = word2vec.Word2Vec.load('new_model20180327.bin')           
    return model
def getVectors(sentence,model):
    num = len(sentence)
    if(num>MAX_NUM_WORD):
        num = MAX_NUM_WORD
        num1 = MAX_NUM_WORD
        sentence1 = ''
        for word in sentence:
            if(num1 == MAX_NUM_WORD): sentence1 += word 
            else: sentence1 += ' '+word
            num1 = num1-1
            if(num1 == 0): break
        sentence = sentence1.split(" ")
            
    vector1 = np.array([model[word] for word in sentence])
    vector1 = vector1.reshape(num*200,1)

    vector2 = np.array([[0.0]*200*(MAX_NUM_WORD-num)])
    vector2 = vector2.reshape(200*(MAX_NUM_WORD-num),1)

    vectors = np.vstack((vector2, vector1))
#    vectors = sequence.pad_sequences(vectors, dtype = 'float32')
    return vectors
def writeNamesIntoTrainTxt(methodName,className,targetClassName,projectname):
    
    fileName = Train_path+projectname + '/train_Names.txt'
#    print ('Writting file: ', fileName)
    txt = open(fileName, 'a', encoding = 'utf-8')
    for word in methodName:
        txt.write(str(word)+' ')
    for word in className:
        txt.write(str(word)+' ')
    for word in targetClassName:
        txt.write(str(word)+' ')
   
    txt.write('\n')
    txt.close()

###############################################################################################
def writeNamesIntoTrainTEMP(methodName,className,targetClassName,projectname):
    fileName = Train_path+projectname + '/temp_Names.txt'
    txt = open(fileName, 'a', encoding = 'utf-8')
    for word in methodName:
        txt.write(str(word)+' ')
    for word in className:
        txt.write(str(word)+' ')
    for word in targetClassName:
        txt.write(str(word)+' ')
   
    txt.write('\n')
    txt.close()




def writeNamesIntoTestTxt(methodName,className,targetClassName,projectname):
    
    fileName = Test_path +projectname+ '/test_Names.txt'
#    print ('Writting file: ', fileName)
    txt = open(fileName, 'a', encoding = 'utf-8')
    for word in methodName:
        txt.write(str(word)+' ')
    for word in className:
        txt.write(str(word)+' ')
    for word in targetClassName:
        txt.write(str(word)+' ')
   
    txt.write('\n')
    txt.close()
    
def writeDistanceIntoTrainFile(distance1,distance2,n,projectname):

    fileName = Train_path +projectname+ '/train_Distances.txt'
#    print ('Writting file: ', fileName)
    txt = open(fileName, 'a', encoding = 'utf-8')
    txt.write(str(distance1)+' ')
    txt.write(str(distance2)+' ')
    txt.write(str(n)+'\n')
    txt.close()


###########################################################################################
def writeDistanceIntoTrainTEMP(distance1,distance2,projectname):

    fileName = Train_path +projectname+ '/temp_Distances.txt'
#    print ('Writting file: ', fileName)
    txt = open(fileName, 'a', encoding = 'utf-8')
    txt.write(str(distance1)+' ')
    txt.write(str(distance2)+' ')
    txt.write('\n')
    txt.close()


    
def writeDistanceIntoTestFile(distance1,distance2,n):

    fileName = Test_path +projectname+ '/test_Distances.txt'
#    print ('Writting file: ', fileName)
    txt = open(fileName, 'a', encoding = 'utf-8')
    txt.write(str(distance1)+' ')
    txt.write(str(distance2)+' ')
    txt.write(str(n)+'\n')
    txt.close()
def writeRealNameIntoFile(methodName,className,targetClassName):
#    fileName = 'data/word2vec.200d' + '.txt'
#    print ('Writting file: ', fileName)
#    txt = open(fileName, 'a', encoding = 'utf-8')
    global words
    for word in methodName:  
        words.append(word)
    for word in className:
        words.append(word)
    for word in targetClassName:
        words.append(word)
        
def writeWordAndVectorIntofile():
    global words
    fileName = 'data/word2vecNocopy.200d' + '.txt'
#    print ('Writting file: ', fileName)
    txt = open(fileName, 'a', encoding = 'utf-8')
    model = getModel()
    print(words.count)
    new_words = set(words)
    
#    print(new_words)
    for word in new_words:
#        print(word)
        txt.write(str(word))
        vector = model[word]
        for num in vector:
            txt.write(' '+str(num))
        txt.write('\n')
    txt.close()
def writeDatasetIntoTXT(projectname):
    n=0
#where classQualifiedName not like 'de.masters_of_disaster%' and classQualifiedName not like'installer%' and classQualifiedName not like'net.sourceforge.jarbundler%' and classQualifiedName not like'org.gjt.sp%' and classQualifiedName not like'org.jedit%' OR classname = 'Native2ASCIIEncodingTest' OR classname = 'GenerateTocXML';
    #cur.execute("select * from classinfo where classQualifiedName not like 'weka%';")

    names=[]
    dis=[]

    cur.execute("select * from classinfo where projectname='%s';"%projectname)
    rowsOfClassInfo = cur.fetchall()
    jin_total=0
    for row in rowsOfClassInfo:
#        m = random.randint(1,50)
#        classID = row[0]
        classQualifiedName = row[1]
        className = row[2]
        #print(classQualifiedName)
        cur.execute("select * from methodinfo where MethodOfClass='%s';"%classQualifiedName)
        rowsOfMethodInfo = cur.fetchall()
        for row1 in rowsOfMethodInfo:
            methodID = row1[0]
            methodName = row1[1]
            methodParameters = row1[2]
            methodOfClass = row1[3]
            #get distance
            cur.execute("select * from distanceValue where methodname='%s' "%methodName+"and methodParameters = '%s' "%methodParameters+"and methodOfClass = '%s';"%methodOfClass)
            rowOfDiatance = cur.fetchall()
            distance1 =1.0
            for row4 in rowOfDiatance:
#                methodname = row4[1]
                if(row4[4] == methodOfClass):
                    distance1 = row4[5]
            #############################################################################################
            cur.execute("select * from distancevalue_intersection where methodname='%s' "%methodName+"and methodParameters = '%s' "%methodParameters+"and methodOfClass = '%s';"%methodOfClass)
            rowsinterdis1 = cur.fetchall()
            inter_dis1=0.0
            for rowinterdis1 in rowsinterdis1:
                if rowinterdis1[4]==methodOfClass:
                    inter_dis1=rowinterdis1[5]

            #get Relations Class
            cur.execute("select * from relations where MethodID='%d';"%methodID)
            rowsOfRelations = cur.fetchall()

            for row2 in rowsOfRelations:
                classIDR = row2[1]
                cur.execute("select * from classinfo where ClassID ='%s';"%classIDR)
                rowsOfClass = cur.fetchall()
                if(rowsOfClass.count != 0):
                    for row3 in rowsOfClass:
                        targetClassName = row3[2]###
                        targetQualifiedName = row3[1]

                        if(targetQualifiedName != classQualifiedName):
#                            print("targetClassName---%s"%targetClassName)
#                            print("methodName----%s"%methodName)
                            methodName1 = preprocessingNames(methodName)
#                            print("methodName1---%s"%methodName1)
                            className1 = preprocessingNames(className)
                            targetClassName1 = preprocessingNames(targetClassName)
                            distance2 =1.0
                            for row4 in rowOfDiatance:
                                if(row4[4] == targetClassName):
                                    distance2 = row4[5]
#                            writeRealNameIntoFile(methodName1,className1,targetClassName1)

                            #######################################################################################
                            inter_dis2=0.0
                            for rowinterdis1 in rowsinterdis1:
                                if rowinterdis1[4]==targetClassName:
                                    inter_dis2=rowinterdis1[5]



                            if inter_dis1<=inter_dis2:
                                continue

                            if(distance1 == 1.0 and distance2 == 1.0):
                                s = random.randint(1,10)
                                if(s != 3):
                                    continue
                            """
                            if(m == 5):
                                n+=1
                                writeNamesIntoTestTxt(methodName1,className1,targetClassName1)
                                writeDistanceIntoTestFile(distance1,distance2,0)
                                writeNamesIntoTestTxt(methodName1,targetClassName1,className1)
                                writeDistanceIntoTestFile(distance2,distance1,1)
                            
                            else:
                            """
                            #writeNamesIntoTrainTxt(methodName1,className1,targetClassName1,projectname)
                            #names.append([methodName1,className1,targetClassName1])
                            #writeDistanceIntoTrainFile(distance1,distance2,0,projectname)
                            #dis.append([distance1,distance2])
                            writeNamesIntoTrainTEMP(methodName1,className1,targetClassName1,projectname)
                            writeDistanceIntoTrainTEMP(distance1,distance2,projectname)
                            #writeNamesIntoTrainTxt(methodName1,targetClassName1,className1,projectname)
                            #names.append([methodName1,targetClassName1,className1])
                            #writeDistanceIntoTrainFile(distance2,distance1,1,projectname)
                            #dis.append([distance2,distance1,1])
                            
        jin_total+=1
        if jin_total%100==0:
            print(jin_total)
    #return names,dis
    

projects=['android-backup-extractor-20140630',"AoI30","areca-7.4.7","freeplane-1.3.12","grinder-3.6","jedit","jexcelapi_2_6_12","junit-4.10","pmd-5.2.0","weka"]
if __name__=='__main__':
    print ("程序开始时间:"+time.strftime("%Y/%m/%d  %H:%M:%S"))
    
    '''
    for i in range(len(projects)):#len(projects)

        print('*'*80)
        print(projects[i])
        writeDatasetIntoTXT(projects[i])
        #names.append(name1)
        #distances.append(dis1)
        

    '''
    for i in range(len(projects)):#len(projects)
        print('-'*80)
        print(projects[i])
        names=[]
        distances=[]
        for j in range(len(projects)):
            if i==j:
                continue
            f1=open('D:/TSE/python/featureenvy/data_compare/'+projects[j]+'/temp_Names.txt','r')
            f2=open('D:/TSE/python/featureenvy/data_compare/'+projects[j]+'/temp_Distances.txt','r')
            lines1=f1.readlines()
            lines2=f2.readlines()
            names+=lines1
            distances+=lines2

        for j in range(len(names)):#len(names)
            methodName1,className1,targetClassName1=[],[],[]
            for k in range(5):
                methodName1.append(names[j].split(' ')[k])
                className1.append(names[j].split(' ')[5+k])
                targetClassName1.append(names[j].split(' ')[10+k])
            distance1,distance2=distances[j].split(' ')[0],distances[j].split(' ')[1]
            
            writeNamesIntoTrainTxt(methodName1,className1,targetClassName1,projects[i])
            writeDistanceIntoTrainFile(distance1,distance2,0,projects[i])
            writeNamesIntoTrainTxt(methodName1,targetClassName1,className1,projects[i])
            writeDistanceIntoTrainFile(distance2,distance1,1,projects[i])
            






        

    
    print ("程序运行结束时间:"+time.strftime("%Y/%m/%d  %H:%M:%S"))
#    writeWordAndVectorIntofile()