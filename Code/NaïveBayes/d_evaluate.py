# -*- coding: utf-8 -*-
import nltk
import json
import re
from nltk.stem.porter import PorterStemmer
porter_stemmer = PorterStemmer() 
f1=open('E:\\KEA_code\\data\\Semeval\\corpus\\1\\test1_24.json','r')
lines1=f1.readlines()
#f2=open('E:\\KEA_code\\NB\\KEA-3.0\\key\\key1.txt','r')
f2=open('E:\\KEA_code\\NB\\KEA-3.0\\key\\key1_ref.txt','r')
lines2=f2.readlines()
STP=0.0
SFP=0.0
SFN=0.0
SP=0.0
SR=0.0
SF=0.0
SUM=0

for line1,line2 in zip(lines1,lines2):	
    key_au=re.findall(r"\"keywords\":\"(.*?)\", \"",line1)#Keypharses tagged by the author
#    print(key_au)
    if len(key_au):
        key1=re.split(";",key_au[0]) 
    else:
        print('error')
    n=len(key1)
    
    key_ex=re.split("; ",line2)
    key2=[]
    for ke in key_ex:
        key2.append(ke)#Extracted keypharses
    l=len(key2)

    TP=0.0
    FP=0.0
    if l>3:
        for i in range(3):#TOPN
            ke2=key2[i]
            if ke2 in key1:
                TP+=1
            else:
                FP+=1
    else:
        for i in range(l):#TOPN
            ke2=key2[i]
            if ke2 in key1:
                TP+=1
            else:
                FP+=1
    FN=len(key1)-TP
    STP+=TP
    SFP+=FP
    SFN+=FN
    SUM+=len(key1)
     
SRecall=STP/(STP+SFN)
SPrecision=STP/(STP+SFP)
SF1=2*SRecall*SPrecision/(SRecall+SPrecision)
print(STP)
print(SFP)
print(SFN)   
print(SPrecision)
print(SRecall)
print(SF1)
print(SUM)
f1.close()
f2.close()