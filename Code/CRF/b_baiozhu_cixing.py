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

#Perform part-of-speech filtering and stemming on the data, and return (word form extracted by stem: part of speech of the original word)

f=open('E:\\KEA_code\\data\\KP20K_2000.json','r')
lines=f.readlines()
wr=open('E:\\KEA_code\\crf\\b_KP20K_cixing.json','w')
len_text=len(lines)
print(len_text) 
porter_stemmer = PorterStemmer() 
english_punctuations = [',', ':', ';', '``','?', '（','）','(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%','\\','\"','}','{']
stops = set(stopwords.words("english"))
def tit_stem(con):
    con0=con.replace('\\',' ').lower()
    word_list= nltk.word_tokenize(con0)
    word_pos=nltk.pos_tag(word_list) #Part-of-speech tagging
    ret=[]
    word_list1=[]
    for i in range(len(word_pos)):
        ret1=[]
        ret1.append(word_pos[i][0])
        ret1.append(word_pos[i][1])
        ret.append(ret1)

    for i1 in range(len(ret)):
        ret[i1][0]=porter_stemmer.stem(ret[i1][0]) #Stemming
        word=(":").join(ret[i1])
        word_list1.append(word)
    return word_list1

def con_stem(con):
    con0=con.replace('\\',' ').lower()
    word_list= nltk.word_tokenize(con0)
    word_pos=nltk.pos_tag(word_list) #Part-of-speech tagging
    ret=[]
    word_list1=[]
    for i in range(len(word_pos)):
        if word_pos[i][0] not in english_punctuations: #Symbol removal
            if word_pos[i][0] not in stops: #Stop-words filtering
                ret1=[]
                ret1.append(word_pos[i][0])
                ret1.append(word_pos[i][1])
                ret.append(ret1)
        else:
            continue
    for i1 in range(len(ret)):
        ret[i1][0]=porter_stemmer.stem(ret[i1][0]) #Stemming
        word=(":").join(ret[i1])
        word_list1.append(word)
    return word_list1


for i in range(len_text):
    output=[]
    id0=re.findall(r"\"id\": \"(.*?)\",",lines[i]) 
    output.append("{\"id\": \""+id0[0]+"\"")    
    key=re.findall(r"\"keywords\": \"(.*?)\",",lines[i]) 
    ke0=re.split(";",key[0])
    key0=[]
    for ki in ke0:
        ke1=re.split(" ",ki)
        key1=[]
        for kii in ke1:
            ke2=porter_stemmer.stem(kii)
            key1.append(ke2)
        ke3=" ".join(key1)
        key0.append(ke3)
    key2=";".join(key0)

    output.append("\"keywords\":\""+key2+"\"")
    
    tit=re.findall(r"\"title\": \"(.*?)\"}",lines[i]) 
    tit_list=tit_stem(tit[0])
    tit1=" ".join(tit_list)
    if len(tit)>0:
        output.append("\"title\":\""+tit1+"\"")
    
    ab=re.findall(r"\"abstract\": \"(.*?)\", \"",lines[i]) 
    ab_list=con_stem(ab[0])
    ab1=" ".join(ab_list)
    if len(ab)>0:
        output.append("\"abstract\":\""+ab1+"\"")
    
    ref=re.findall(r"\"references\": \[(.*?)\"\],",lines[i]) 
    if ref:
        ref0=ref[0].replace(r'"','')
        re_list=tit_stem(ref0)
        re1=" ".join(re_list)
        output.append("\"references\": ["+re1+"]}"+"\n")
    else:
        output.append("\"references\": [" "]}"+"\n")

    wr.writelines(", ".join(output))

f.close()
wr.close()