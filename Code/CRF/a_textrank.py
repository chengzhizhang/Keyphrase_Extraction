#Calculate the TextRank value.

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

#Calculate the TextRank value.

f=open('E:\\KEA_code\\data\\KP20K_2000.json','r')
lines=f.readlines()
#wr=open('E:\\KEA_code\\crf\\a_KP20K.tr.json','w')
wr=open('E:\\KEA_code\\crf\\a_KP20K.tr_ref.json','w')

len_text=len(lines)
print(len_text)
porter_stemmer = PorterStemmer() 
english_punctuations = [',', '.', ':', ';', '``','?', '（','）','(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%','\\','\"','}','{','\'']
stops = set(stopwords.words("english"))

def con_stem(con):
    con0=con.replace('\\',' ').lower()
    word_list= nltk.word_tokenize(con0)
    word_list1=[]
    for i in range(len(word_list)-1):
        if word_list[i] not in english_punctuations: #Symbol removal
            if word_list[i] not in stops: #Stop-words filtering
                ret1=porter_stemmer.stem(word_list[i])#Stemming
                word_list1.append(ret1)
    return word_list1

def textrank_com(word_list):
    edge_dict={}
    window=3 
    alpha=0.85
    iternum=100 
    tmp_list=[]
    word_list_len=len(word_list)
    for index,word in enumerate(word_list):
        if word not in edge_dict.keys():
            tmp_list.append(word)
            tmp_set=set()
            left=index-window+1 
            right=index+window 
            if left<0: left=0
            if right>=word_list_len: right=word_list_len
            for i in range(left,right):
                if i==index:
                    continue
                tmp_set.add(word_list[i])
            edge_dict[word]=tmp_set

    matrix=np.zeros([len(edge_dict),len(edge_dict)])
    word_index = {} 
    index_dict = {} 

    for i,v in enumerate(edge_dict):
        word_index[v] = i
        index_dict[i] = v

    for key in edge_dict.keys():
        for w in edge_dict[key]:
            matrix[word_index[key]][word_index[w]] = 1
            matrix[word_index[w]][word_index[key]] = 1

    for j in range(matrix.shape[1]):
        sum = 0
        for i in range(matrix.shape[0]):
            sum += matrix[i][j]
        for i in range(matrix.shape[0]):
            matrix[i][j] /=sum

    PR=np.ones([len(edge_dict),1])
    for i in range(iternum):
        PR=(1-alpha)+alpha*np.dot(matrix,PR)

    word_pr={}
    for i in range(len(PR)):
        word_pr[index_dict[i]]=PR[i][0]
    output=sorted(word_pr.items(),key=lambda x:x[1],reverse=True)
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
    output=textrank_com(con_list)

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
