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

#Automatic keyphrase extraction Methods based on TextRank

f=open('E:\\KEA_code\\data\\Semeval\\a_Semeval.range_244.json','r')
lines=f.readlines()
wr=open('E:\\KEA_code\\textrank\\a_Semeval.range_textrank.json','w')
#wr=open('E:\\KEA_code\\textrank\\a_Semeval.range_textrank_ref.json','w')

len_text=len(lines)
print(len_text)
porter_stemmer = PorterStemmer() 
N=['JJ','JJR','JJS','NN','NNS']
english_punctuations = [',', '.', ':', ';', '``','?', '（','）','(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%','\\','\"','}','{']
stops = set(stopwords.words("english"))

def con_stem(con):
    con0=con.replace('\\',' ').lower()
    word_list= nltk.word_tokenize(con0) #Tokenizing
    word_pos=nltk.pos_tag(word_list) #Part-of-speech tagging
    ret=[]
    word_list1=[]
    for i in range(len(word_pos)-1):
        if word_pos[i][0] not in english_punctuations: #Symbol removal
            if word_pos[i][0] not in stops: #Stop-words filtering
                ret1=[]
                ret1.append(word_pos[i][0])
                ret1.append(word_pos[i][1])
                ret.append(ret1)
        else:
            continue
    for i1 in range(len(ret)-1):
        ret[i1][0]=porter_stemmer.stem(ret[i1][0]) #Stemming
        if ret[i1][1] in N:
            ret[i1]=tuple(ret[i1])
            word_list1.append(ret[i1]) #Choose words whose pos are nouns and adjectives.
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

#According to the relationship between the edges, the matrix is constructed
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

def extr(output,content,tit):
    twokey={}
    NL=["NN","NNS"]

    #Combination candidate keyphrases
    output3=[]
    for t in range(len(output)):
        if output[t][0][1] in NL:	#The number of words in the keyphrase is 3.
            for m in range(len(output)):
                for o in range(len(output)):
                    result=[]
                    result.append(output[m][0][0])
                    result.append(output[o][0][0])
                    result.append(output[t][0][0])
                    score=(output[t][1]+output[m][1]+output[o][1])/3
                    result3=" ".join(result)
                    if result3 in tit:
                        twokey[result3]=score
                        output3.append(result3)

    output2=[]
    for t in range(len(output)):
        if output[t][0][1] in NL:	#The number of words in the keyphrase is 2.
            for m in range(len(output)):
                result=[]
                result.append(output[m][0][0])
                result.append(output[t][0][0])
                score=(output[t][1]+output[m][1])/2
                result2=" ".join(result)
                if result2 in tit:
                    twokey[result2]=score
                    output2.append(result2)

    for t in range(len(output)):
        if output[t][0][1] in NL:	#The number of words in the keyphrase is 1.
            twokey[output[t][0][0]]=output[t][1]
    twokey=sorted(twokey.items(),key = lambda x:x[1],reverse = True)[:10]
    json_out=[]
    ii=0
    for ii in range(len(twokey)):
        key=twokey[ii][0]
        value=twokey[ii][1]
        if ii==len(twokey)-1:
            json_out.append("%s: %f"%(key,value))
        else:
            json_out.append("%s: %f; "%(key,value))
    json_out.append('\n')
    wr.writelines(json_out)
    return twokey

for i in range(244):
    tit=re.findall(r"\"title\": \"(.*?)\"}",lines[i])
    ab=re.findall(r"\"abstract\": \"(.*?)\", \"",lines[i])
    ref=re.findall(r"\"references\": \[(.*?)\],",lines[i])
    tit1=[]
    if len(tit):
        tit0=re.split(" ",tit[0].lower())
        for t in tit0:
            t1=porter_stemmer.stem(t)
            tit1.append(t1)
    tit2=" ".join(tit1) #Stem extraction of the title.
#    print(tit2)
    con=" ".join(tit+ab).lower()
#    con=" ".join(tit+ab+ref).lower() #Control whether reference information is added.
    con_list=con_stem(con)
    cont=[]
    for word in con_list:
        cont.append(word[0])
    cont=" ".join(cont)
    output=textrank_com(con_list)
    result=extr(output,cont,tit2)

wr.close()
f.close()