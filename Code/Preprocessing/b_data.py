# -*- coding: utf-8 -*-
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import brown
from nltk.stem.porter import PorterStemmer
import numpy as np
import json
import re
import math
#Stop word filtering, Symbol removal and Stemming.
f=open('E:\\KEA_code\\data\\KP20K_2000.json','r')
lines=f.readlines()
#wr=open('E:\\KEA_code\\data\\Semeval\\b_Semeval.stem_244.json','w')#Stop word filtering, Symbol removal and Stemming.
wr=open('E:\\KEA_code\\data\\KP20K.stem.json','w')#Only Stemming
len_text=len(lines)
print(len_text) 
porter_stemmer = PorterStemmer() 
N=['JJ','JJR','JJS','NN','NNS']

english_punctuations = [',', '.', ':', ';', '``','?', '（','）','(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%','\\','\"','}','{','.']
stops = set(stopwords.words("english"))
for i in range(len_text):y
    output=[]
    s=[]
    z=[]
    m=[]
    k=[]
    id0=re.findall(r"\"id\": \"(.*?)\",",lines[i]) #id
    output.append("{\"id\": \""+id0[0]+"\"")
    
    key=re.findall(r"\"keywords\": \"(.*?)\",",lines[i]) #keyphrases
    ke=re.split(";",key[0])
    for ke0 in ke:
        k1=[]
        ke11=word_tokenize(ke0)
        for ke1 in ke11:
            ke2=porter_stemmer.stem(ke1)  #Stemming
            k1.append(ke2.lower())
        ke3=" ".join(k1)
        k.append(ke3)
    ke4=";".join(k)
    output.append("\"keywords\":\""+ke4+"\"")
    
    tit=re.findall(r"\"title\": \"(.*?)\"}",lines[i]) #titles
    tit1=word_tokenize(tit[0])#Tokenizing

    for ti in tit1:
        tit2=porter_stemmer.stem(ti)  #Stemming
#        if tit2 not in english_punctuations: #Symbol removal
#            if tit2 not in stops: #Stop-words filtering
#                s.append(tit2.lower())
        s.append(tit2.lower())
    tit3=" ".join(s)
    if len(tit)>0:
        output.append("\"title\":\""+tit3+"\"")
    
    ab=re.findall(r"\"abstract\": \"(.*?)\", \"",lines[i]) #abstract
    a1=word_tokenize(ab[0])
    for a2 in a1:
        a3=porter_stemmer.stem(a2)
#        if a3 not in english_punctuations:
#            if a3 not in stops:
#                z.append(a3.lower())
        z.append(a3.lower())
    a4=" ".join(z)
    if len(ab)>0:
        output.append("\"abstract\":\""+a4+"\"")
        
    ref=re.findall(r"\"references\": \[(.*?)\],",lines[i]) #references
    if len(ref[0]):
        re00=re.split('\", \"',ref[0])
        for re0 in re00:
            re1=word_tokenize(re0)
            for re2 in re1:
                re3=porter_stemmer.stem(re2)
                if re3 not in english_punctuations: 
                    if re3 not in stops: 
                        m.append(re3.lower())
            m.append(".")
        re4=" ".join(m)
    else:
        re4=" "
    output.append("\"references\": ["+re4+"]}"+"\n")

    wr.writelines(", ".join(output))

f.close()
wr.close()
