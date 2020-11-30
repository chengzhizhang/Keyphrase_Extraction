# -*- coding: utf-8 -*-
import nltk
import json
import re
from nltk.stem.porter import PorterStemmer
porter_stemmer = PorterStemmer() 

#Calculate R values under different conditions with reference information added(Appears or does not appear in the title and abstract)

f1=open('E:\\KEA_code\\data\\Semeval\\b_Semeval.range_stem_244.json','r')
lines1=f1.readlines()
f2=open('E:\\KEA_code\\tfidf\\a_Semeval.range_tfidf_ref.json','r')
lines2=f2.readlines()
STP=0.0 #The extracted keywords are those assigned by the author
STP1=0.0 #The extracted keywords are those assigned by the author and appear in the title and abstract
STP2=0.0 #The extracted keywords are those assigned by the author and do not appear in the title and abstract
SFP=0.0 #The extracted keywords are not those assigned by the author
key_sum=0.0

for line1,line2 in zip(lines1,lines2):	
    ke=re.findall(r"\"keywords\":\"(.*?)\", \"",line1)#Keywords tagged by the author
    tit=re.findall(r"\"title\":\"(.*?)\",",line1) 
    ab=re.findall(r"\"abstract\":\"(.*?)\",\"",line1) 
    ta=ab+tit
    ta1=" . ".join(ta) 
    ta2=re.split(" ",ta1.lower()) 
    key1=re.split(";",ke[0].lower())
    len_key=len(key1)

    TP=0.0
    TP1=0.0
    TP2=0.0
    FP=0.0
    key_ex=re.split("; ",line2)#The extracted keywords are segmented.
    ke2=[]
    for ke in key_ex:
        ke1=re.split(": ",ke)
        ke2.append(ke1[0])#Extracted keypharses
    for m in range(3):
        ke3=ke2[m]
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
    