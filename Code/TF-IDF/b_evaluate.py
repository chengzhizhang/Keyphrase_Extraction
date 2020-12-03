#Calculate the P, R and F1 values of the extraction results

# -*- coding: utf-8 -*-
import nltk
import json
import re
from nltk.stem.porter import PorterStemmer
porter_stemmer = PorterStemmer() 


f1=open('E:\\KEA_code\\data\\Semeval\\a_Semeval.range_244.json','r')
lines1=f1.readlines()
f2=open('E:\\KEA_code\\tfidf\\a_Semeval.range_tfidf_ref.json','r')#Extracted keypharses results
#f2=open('E:\\KEA_code\\tfidf\\a_Semeval.range_tfidf.json','r')
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
        key1=re.split(";",key_au[0]) 
    else:
        print('error')
    n=len(key1)
    
    key_ex=re.split("; ",line2)
    key2=[]
    for ke in key_ex:
        ke1=re.split(": ",ke)
        key2.append(ke1[0])#Extracted keypharses

    TP=0.0
    FP=0.0
    for i in range(3):#TOPN
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
