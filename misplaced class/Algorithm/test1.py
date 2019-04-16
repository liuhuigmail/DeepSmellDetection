import numpy as np
import pandas as pd
import preprocessing
from gensim import models
from gensim.models import word2vec
import time

projects=['android-backup-extractor-20140630',"AoI30","areca-7.4.7","freeplane-1.3.12","grinder-3.6","jedit","jexcelapi_2_6_12","junit-4.10","pmd-5.2.0","weka"]
df=pd.read_csv('D:/TSE/python/missplaceclass/input.csv')
df_classes=pd.read_csv('D:/TSE/python/missplaceclass/test_class.csv')
df_items=pd.read_csv('D:/TSE/python/missplaceclass/test_items.csv')

print('Building Tokenizer...')
tokenizer=preprocessing.get_tokenizer(df)

score=np.zeros(shape=(0))
label=np.zeros(shape=(0))
target_correct=0



for i in range(len(projects)):#len(projects)
    
    project=projects[i]
    print('*'*80)
    print(project)
    ss=time.time()
    models=preprocessing.train(df[df.projectname!=project],tokenizer,project)
    print('###########################',time.time()-ss)

    #models=preprocessing.load_models(project)
    ss=time.time()
    tscore,tlabel,t_target_correct=preprocessing.test(df_classes,df_items[df_items.projectname==project],tokenizer,models)
    print('###########################',time.time()-ss)
    score=np.concatenate((score,tscore.reshape(-1)),axis=0)
    label=np.concatenate((label,tlabel.reshape(-1)),axis=0)
    target_correct+=t_target_correct


print('*'*80)
print('Final')
preprocessing.eval(score,label,target_correct)

