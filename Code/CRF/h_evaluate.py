#Calculate the P, R and F1 values of the extraction results


# -*- coding: utf-8 -*-
import nltk
import json
import re
from nltk.stem.porter import PorterStemmer
porter_stemmer = PorterStemmer() 


f1=open('E:\\KEA_code\\crf\\KP20K\\1\\test1_200.json','r')
lines1=f1.readlines()
f2=open('E:\\KEA_code\\crf\\corpus\\test11_jisuan.json','r')
#f2=open('E:\\KEA_code\\crf\\corpus2\\10\\test10_ref_jisuan.json','r')
lines2=f2.readlines()
STP=0.0
SFP=0.0
SFN=0.0
SP=0.0
SR=0.0
SF=0.0
SUM=0
for line1,line2 in zip(lines1,lines2):
    key_au=re.findall(r"\"keywords\": \"(.*?)\", \"",line1)#Keypharses tagged by the author
    if len(key_au):
        ke0=re.split(";",key_au[0]) 
        key1=[]
        for ki in ke0:
            k1=re.split(" ",ki)
            k2=[]
            for kii in k1:
                k3=porter_stemmer.stem(kii)
                k2.append(k3)
            ke3=" ".join(k2)
            key1.append(ke3)
    else:
        print('error')
    n=len(key1)
        
    key_ex=re.split(";",line2.strip())
    key2=[]
    for ke in key_ex:
        key2.append(ke)#Extracted keypharses
    l=len(key2)

    TP=0.0
    FP=0.0
    for i in range(l):#TOPN
        ke2=key2[i]
        if ke2 in key1:
            TP+=1
        else:
            FP+=1
    FN=n-TP
    STP+=TP
    SFP+=FP
    SFN+=FN
    SUM+=n


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
