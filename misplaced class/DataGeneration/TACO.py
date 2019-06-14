import pymysql
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import metrics

conn=pymysql.connect(host="127.0.0.1",user="root",passwd="12345",db="misplaceclass",charset="utf8") 
cur = conn.cursor()



def get_class_txt(classname):
    cur.execute("select code from taco_temp where classname='%s';"%classname)
    temp=cur.fetchall()
    if len(temp)==0:
        return ""
    return temp[0][0]

def get_package_txt(packagename,classname):
    cur.execute("select code from taco_temp where classname!='%s'"%classname+" and package='%s';"%packagename)
    temp=cur.fetchall()
    t=""
    for i in range(len(temp)):
        t+=temp[i][0]
    return t

def get_related_package(classname):
    cur.execute("select * from taco_relate where classname='%s';"%classname)
    temp=cur.fetchall()
    ori=temp[0][1]
    t=[]
    for i in range(len(temp)):
        t.append(temp[i][3])
    return ori,t

def cal_sim(ctxt,ptxt):
    vectorizer = TfidfVectorizer()
    tfidf=vectorizer.fit_transform([ctxt,ptxt])

    d=(int)((tfidf.shape[0]*tfidf.shape[1])**0.2)
    svd=TruncatedSVD(d)

    lsa = make_pipeline(svd, Normalizer(copy=False))
    res_lsa=lsa.fit_transform(tfidf)

    return cosine_similarity(res_lsa)[0,1]

def get_closed_package(related_packages,ctxt,classname):
    pos=0
    max_sim=0
    for j in range(len(related_packages)):
        p=related_packages[j]
        ptxt="package "+get_package_txt(p,classname)
        s=cal_sim(ctxt,ptxt)
        if s>max_sim:
            max_sim=s
            pos=j
    
    return related_packages[pos],max_sim

def get_real_target(classname):
    cur.execute("select classname2 from taco_package where classname1='%s';"%classname)
    temp=cur.fetchall()
    classname2=temp[0][0]
    t=classname2.split('.')
    ans=""
    for i in range(len(t)-1):
        if i>0:
            ans+="."
        ans+=t[i]
    return ans

def get_label(classname):
    cur.execute("select classname2 from taco_package where classname1='%s';"%classname)
    temp=cur.fetchall()
    if len(temp)==0:
        return 0
    classname2=temp[0][0]
    if classname==classname2:
        return 0
    return 1


def get_package_name(classname2):
    t=classname2.split('.')
    ans=""
    for i in range(len(t)-1):
        if i>0:
            ans+="."
        ans+=t[i]
    return ans

def test(projectname):
    res=[]
    targets=[]
    cur.execute("select distinct(classname) from taco_relate where projectname='%s';"%projectname)
    classes=cur.fetchall()
    for i in range(len(classes)):
        classname=classes[i][0]
        ctxt="code "+get_class_txt(classname)
        ori_package,related_packages=get_related_package(classname)

        ori_ptxt=get_package_txt(ori_package,classname)
        ori_sim=cal_sim(ctxt,ori_ptxt)

        predict_tar,tar_sim=get_closed_package(related_packages,ctxt,classname)

        res.append(tar_sim-ori_sim)
        if tar_sim-ori_sim<=0:
            targets.append(" ")
        else:
            targets.append(predict_tar)
        '''
        if (i+1)%100==0:
            print(i+1,len(classes))
        '''
    return eval(projectname,res,targets)

def eval(projectname,res,targets):
    labels=[]
    target_correct=0
    cur.execute("select distinct(classname) from taco_relate where projectname='%s';"%projectname)
    classes=cur.fetchall()
    tempres=[]
    for i in range(len(res)):
        if res[i]>0:
            tempres.append(res[i])
    tempres.sort()
    threshold=0
    pos=(int)(len(tempres)/2)
    #print(pos)
    if len(tempres)%2==0:
        #print("-----------------",tempres[pos],tempres[pos+1])
        threshold=(tempres[pos]+tempres[pos+1])*1.0/2
    else:
        #print("-----------------",tempres[pos+1])
        threshold=tempres[pos+1]
    tp,tn,fp,fn=0,0,0,0
    for i in range(len(classes)):
        classname=classes[i][0]
        label=get_label(classname)
        labels.append(label)
        if label==0:
            if res[i]>threshold:
                fp+=1
            else:
                tn+=1
        else:
            if res[i]>threshold:
                tp+=1
                if get_real_target(classname)==targets[i]:
                    target_correct+=1
            else:
                fn+=1
    print("Threshold:",threshold)
    print_result(tp,fp,tn,fn,labels,res,target_correct)
    return tp,fp,tn,fn,labels,res,target_correct

def print_result(tp,fp,tn,fn,labels,res,target_correct):
    print("tp : ",tp)
    print("tn : ",tn)
    print("fp : ",fp)
    print("fn : ",fn)

    if (tp+fp)==0:
        P=0
    else:
        P=tp*1.0/(tp+fp)

    if tp+fn==0:
        R=0
    else:
        R=tp*1.0/(tp+fn)
    print("Precision : ",P)
    print("Recall : ",R)
    if P+R==0:
        print("F1 : ",0)
    else:
        print("F1 : ",2*P*R/(P+R))

    a=tp+fp
    b=tp+fn
    c=tn+fp
    d=tn+fn
    print("MCC : ",(tp*tn-fp*fn)/((a*b*c*d)**0.5))
    print("AUC : ",metrics.roc_auc_score(labels,res))
    print('Target Correct : ',target_correct)
    if tp==0:
        print('Accuracy0 : 0')
    else:
        print('Accuracy : ',target_correct*1.0/tp)

ttp,tfp,ttn,tfn=0,0,0,0
tlabels=[]
tres=[]
tcor=0


projects=["junit-4.10","jexcelapi_2_6_12",'android-backup-extractor-20140630',"AoI30","areca-7.4.7","freeplane-1.3.12","grinder-3.6","jedit","pmd-5.2.0","weka"]

for i in range(len(projects)):#len(projects)

    project=projects[i]
    print('*'*80)
    print(project)
    print('Testing...')
    tp,fp,tn,fn,labels,res,target_correct=test(project)
    #print_result(tp,fp,tn,fn,labels,res,target_correct)
    ttp+=tp
    tfp+=fp
    ttn+=tn
    tfn+=fn
    tlabels+=labels
    tres+=res
    tcor+=target_correct

print('*'*80)
print("Fianl")
tlabels=np.array(tlabels)
res=np.array(res)
print_result(ttp,tfp,ttn,tfn,tlabels,tres,tcor)
