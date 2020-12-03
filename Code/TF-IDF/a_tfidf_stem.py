#Automatic keyphrase extraction Methods based on TF*IDF

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



f=open('E:\\KEA_code\\data\\Semeval\\a_Semeval.range_244.json','r')
lines=f.readlines()
#wr=open('E:\\KEA_code\\tfidf\\a_Semeval.range_tfidf.json','w')
wr=open('E:\\KEA_code\\tfidf\\a_Semeval.range_tfidf_ref.json','w')

len_text=len(lines)
print(len_text)
porter_stemmer = PorterStemmer() 
N=['JJ','JJR','JJS','NN','NNS']
#N=['JJ','JJR','JJS','NN','NNS','VBD','VBG','VBZ','VBP','RB','VB']
english_punctuations = [',', '.', ':', ';', '``','?', '（','）','(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%','\\','\"','}','{']
stops = set(stopwords.words("english"))

def con_stem(con): #Tokenizing, Part-of-speech tagging, Stop word filtering, Symbol removal and Stemming.
    con0=con.replace('\\','')
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

def extr(output,content,tit):
    twokey={}
    NL=["NN","NNS"]

    #Combination candidate keyphrases
    output3=[]
    for t in range(len(output)):
        if output[t][0][1] in NL:	 #The number of words in the keyphrase is 3.
            for m in range(len(output)):
                for o in range(len(output)):
                    result=[]
                    result.append(output[m][0][0])
                    result.append(output[o][0][0])
                    result.append(output[t][0][0])
                    score=(output[t][1]+output[m][1]+output[o][1])/3
                    result3=" ".join(result)
                    if result3 in tit:	#Whether it appears in the title or not (compared to other situations, this is the best result)
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
        if output[t][0][1] in NL:	
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

#    con=" ".join(tit+ab).lower()
    con=" ".join(tit+ab+ref).lower() #Control whether reference information is added.
    con_list=con_stem(con)
    cont=[]
    for word in con_list:
        cont.append(word[0])
    cont=" ".join(cont)
    output=tfidf_com(con_list)
    result=extr(output,cont,tit2)

wr.close()
f.close()
