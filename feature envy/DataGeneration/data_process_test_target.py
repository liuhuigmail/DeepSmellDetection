# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 14:51:07 2018

@author: xzf0724----test1
从数据库提取数据
"""

import pymysql
import random
import time

#连接SQL
conn=pymysql.connect(host="127.0.0.1",user="root",passwd="12345",db="testdata",charset="utf8") 
cur = conn.cursor()
#Train_path = 'D:/TSE/python/featureenvy/data/areca-7.4.7/'#train dataset path
project="weka"
Test_path = "D:/TSE/python/featureenvy/data/"+project+"/"#test dataset path
FILENAME = 'D:/TSE/featureenvy/CompareProjects-ori/MoveMethodNameAndState_'+project+'.txt'

MAX_NUM_WORD = 5 #最多单词数
words = []
values = []#方法名和其所在方法名
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


def writeNamesIntoTestTxt(methodName,className,targetClassName,methodId):
    
    fileName = Test_path + 'test_Names'+str(methodId)+'.txt'
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
    
def writeDistanceIntoTestFile(distance1,distance2,n,methodId):

    fileName = Test_path + 'test_Distances'+str(methodId)+'.txt'
#    print ('Writting file: ', fileName)
    txt = open(fileName, 'a', encoding = 'utf-8')
    txt.write(str(distance1)+' ')
    txt.write(str(distance2)+' ')
    txt.write(str(n)+'\n')
    txt.close()
    
def writeMethodIdIntoTestFile(methodId,label):

    fileName = Test_path + 'test_ClssId.txt'
#    print ('Writting file: ', fileName)
    txt = open(fileName, 'a', encoding = 'utf-8')
    txt.write(str(methodId)+' ')
    txt.write(str(label)+'\n')
    txt.close()      
    
def writeTargetClassNameIntoTestFile(targetClassName):
    fileName = Test_path + 'targetClasses.txt'
#    print ('Writting file: ', fileName)
    txt = open(fileName, 'a', encoding = 'utf-8')
    for word in targetClassName:
        txt.write(str(word)+' ')
    txt.write('\n')
    txt.close()     
def extractMehtodName():
    f = open(FILENAME, 'r', encoding = 'utf-8')
    for line in f:
        value = line.split()
        values.append(value)
#    for sentence in values:
#        print(sentence[1])
        
def writeDatasetIntoTXT():
#    num=0
#    n=0
    for sentence in values:
        methodName = sentence[0]
        parameters = sentence[1]
        classQualifiedName = sentence[2]
#        oneOfTagetClassName = sentence[3]
        if(sentence[4]!='0'and sentence[4]!='1'):
            print('---------------------------------------------')
            parameters = sentence[1]+' '+sentence[2]+' '+sentence[3]
            classQualifiedName = sentence[4]
            label = sentence[6]
        else:
            parameters = sentence[1]
            classQualifiedName = sentence[2]
            label = sentence[4]
        print('label---',label)
        #print('classQualifiedName----'+classQualifiedName)
        #if(label == '1'):
        flag = 0
        cur.execute("select * from classinfo where classQualifiedName ='%s';"%classQualifiedName)
        rowsOfClassInfo = cur.fetchall()
        for row in rowsOfClassInfo:
#        m = random.randint(1,30)
#                classID = row[0]
            className = row[2]
            #print(classQualifiedName)
            #flag = 0
            cur.execute("select * from methodinfo where methodName='%s'"%methodName+" and methodOfClass = '%s'"%classQualifiedName+" and methodparameters = '%s';"%parameters)
            rowsOfMethodInfo = cur.fetchall()
            
            for row1 in rowsOfMethodInfo:
                
                methodID = row1[0]
                methodName = row1[1]
                methodParameters = row1[2]
                methodOfClass = row1[3]
                #print('methodName----%s'%methodName)
                
            #get distance
                cur.execute("select * from distanceValue where methodname='%s' "%methodName+"and methodParameters = '%s' "%methodParameters+"and methodOfClass = '%s';"%methodOfClass)
                rowOfDiatance = cur.fetchall()
                distance1 =1.0
                for row4 in rowOfDiatance:
                    methodname = row4[1]
                    if(row4[4] == methodOfClass):
                        distance1 = row4[5]
            #get Relations Class
                #print('methodName-----',methodName)
                cur.execute("select * from relations where MethodID='%d';"%methodID)
                rowsOfRelations = cur.fetchall()
                
                for row2 in rowsOfRelations:
                    
                    classIDR = row2[1]
                    #methodIDR = row2[3]
                    #print('methodIDR---',methodIDR)
                    #print('classIDR---',classIDR)
                    cur.execute("select * from classinfo where ClassID ='%s';"%classIDR)
                    rowsOfClass = cur.fetchall()
                    #print('rowsOfClass.count---',rowsOfClass.count)
                    if(rowsOfClass.count != 0):
                        for row3 in rowsOfClass:
                            print("*******")
                            #targetClassId = row[0]
                            targetClassName = row3[2]###
                            targetQualifiedName = row3[1]
                            
                            if(targetQualifiedName != classQualifiedName):
                                #print("targetClassName---%s"%targetClassName)
                                #print("methodName----%s"%methodName)
                                methodName1 = preprocessingNames(methodName)
#                                print("methodName1---%s"%methodName1)
                                className1 = preprocessingNames(className)
                                targetClassName1 = preprocessingNames(targetClassName)
                                distance2 =1.0
                                for row4 in rowOfDiatance:
                                    if(row4[4] == targetClassName):
                                        distance2 = row4[5]
#                            writeRealNameIntoFile(methodName1,className1,targetClassName1)
#                                r = random.randint(1,2)
                                
                                if(label == '1'):
                                    if(flag == 0):
                                        writeMethodIdIntoTestFile(methodID,label)
                                        writeTargetClassNameIntoTestFile(className1)
                                        flag = 1
                                    writeNamesIntoTestTxt(methodName1,targetClassName1,className1,methodID)
                                    writeDistanceIntoTestFile(distance2,distance1,1,methodID)
                                    className = targetClassName
                                    targetQualifiedName = classQualifiedName
                                    distance1 = distance2
                                    label = '0'
                                else:
                                    if(flag == 0):
                                        writeMethodIdIntoTestFile(methodID,0)
                                        flag = 1
                                    writeNamesIntoTestTxt(methodName1,className1,targetClassName1,methodID)
                                    writeDistanceIntoTestFile(distance1,distance2,0,methodID)
                                
                                                       
#    print(num)
#    print(n)
    
if __name__=='__main__':
    extractMehtodName()
    writeDatasetIntoTXT()
    print ("程序运行结束时间:"+time.strftime("%Y/%m/%d  %H:%M:%S"))
#    writeWordAndVectorIntofile()