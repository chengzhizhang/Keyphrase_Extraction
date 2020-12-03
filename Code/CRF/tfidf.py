#Calculate the TF*IDF value.

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

#Calculate the TF*IDF value.

f=open('E:\\KEA_code\\data\\KP20K_2000.json','r')
lines=f.readlines()
#wr=open('E:\\KEA_code\\crf\\a_KP20K.tfidf.json','w')
wr=open('E:\\KEA_code\\crf\\a_KP20K.tfidf_ref.json','w')

len_text=len(lines)
print(len_text)
porter_stemmer = PorterStemmer() 
english_punctuations = [',', '.', ':', ';', '``','?', '（','）','(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%','\\','\"','}','{','\'']
stops = set(stopwords.words("english"))

def con_stem(con): #Tokenizing, Stop word filtering, Symbol removal and Stemming.
    con0=con.replace('\\',' ').lower()
    word_list= nltk.word_tokenize(con0)
    word_list1=[]
    for i in range(len(word_list)-1):
        if word_list[i] not in english_punctuations: #Symbol removal
            if word_list[i] not in stops: #Stop-words filtering
                ret1=porter_stemmer.stem(word_list[i])#Stemming
                word_list1.append(ret1)
    return word_list1

#Calculated TF-IDF
def df_com(word_list):
    dict_df={}
    ret2=[]
    for word in word_list: #Document Frequency
        word0=word[0]
        ret2.append(word0)
    word_list1=list(set(ret2))
    for word in word_list1:
        if word in dict_df:
            dict_df[word]=dict_df[word]+1
        else:
            dict_df[word]=1
    return dict_df

def tfidf_com(word_list):
    dict_tf={}
    dict_ti={}
    for i in range(len(word_list)-1):
        word= word_list[i]
        if word in dict_tf:
            dict_tf[word]=dict_tf[word]+1
        else:
            dict_tf[word]=1
    max_key=max(dict_tf,key=dict_tf.get)
    l=float(dict_tf[max_key])
    for key,value in dict_tf.items():
        word=key[0]
        num_tf=value/l
        dict_df=df_com(word_list)
        num_df=math.log((len_text+1)/(float(dict_df[word])+1),2)
        tf_idf=num_tf*num_df
        dict_ti[key]=tf_idf
    output=sorted(dict_ti.items(),key=lambda x:x[1],reverse=True)
    return output


for i in range(len_text):
    tit=re.findall(r"\"title\": \"(.*?)\"}",lines[i])
    ab=re.findall(r"\"abstract\": \"(.*?)\", \"",lines[i])
    ref=re.findall(r"\"references\": \[(.*?)\],",lines[i])

#    con=" ".join(tit+ab).lower()
    con=" ".join(tit+ab+ref).lower() #Control whether reference information is added.
    con_list=con_stem(con.replace(r'"',' '))
    cont=[]
    for word in con_list:
        cont.append(word)
    cont=" ".join(cont)
    output=tfidf_com(con_list)
    
    json_out=[]
    ii=0
    for ii in range(len(output)):
        key=output[ii][0]
        value=output[ii][1]
        if ii==len(output)-1:
            json_out.append("%s:%f"%(key,value))
        else:
            json_out.append("%s:%f, "%(key,value))
    json_out.append('\n')
    wr.writelines(json_out)

wr.close()
f.close()
