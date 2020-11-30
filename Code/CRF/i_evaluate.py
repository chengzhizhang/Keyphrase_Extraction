# -*- coding: utf-8 -*-
import nltk
import json
import re
from nltk.stem.porter import PorterStemmer
porter_stemmer = PorterStemmer() 

#Calculate R values under different conditions with reference information added(Appears or does not appear in the title and abstract)

f1=open('E:\\KEA_code\\crf\\KP20K\\1\\test1_200.json','r')
lines1=f1.readlines()
f2=open('E:\\KEA_code\\crf\\corpus\\test11_jisuan.json','r')
lines2=f2.readlines()
STP=0.0 #The extracted keywords are those assigned by the author
STP1=0.0 #The extracted keywords are those assigned by the author and appear in the title and abstract
STP2=0.0 #The extracted keywords are those assigned by the author and do not appear in the title and abstract
SFP=0.0 #The extracted keywords are not those assigned by the author
key_sum=0.0

for line1,line2 in zip(lines1,lines2):	
    key_au=re.findall(r"\"keywords\": \"(.*?)\", \"",line1)#Keywords tagged by the author
    tit=re.findall(r"\"title\":\"(.*?)\",",line1) 
    ab=re.findall(r"\"abstract\":\"(.*?)\",\"",line1)
    ta=ab+tit
    ta1=" . ".join(ta) 
    ta2=re.split(" ",ta1.lower()) 
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
    len_key=len(key1)
    key_ex=re.split(";",line2.strip()) 
    TP1=0.0
    TP2=0.0
    FP=0.0
    ke2=[]
    for ke in key_ex:
        ke2.append(ke)#Extracted keypharses
    l=len(ke2)
    for i in range(l):
        ke3=ke2[i]
        if ke3 in key1:
            TP+=1
            if len(ke3)==1:
                if ke3 in ta2:
                    TP1+=1
            else:
                if ke3 in ta1:
                    TP1+=1
                else:
                    TP2+=1
        else:
            FP+=1
    key_sum+=len_key
    STP+=TP
    STP1+=TP1
    STP2+=TP2
    SFP+=FP


print(STP)
print(STP1)
print(STP2)
print(STP1/key_sum)  #R values for keywords that appear in titles and abstracts
print(STP2/key_sum)  #R value for keywords that do not appear in the title and summary
print(SFP)
f1.close()
f2.close()
