# -*- coding: utf-8 -*-
import json
import re
from nltk.stem.porter import PorterStemmer

#Annotation: TextRank value, TF*IDF value, position, sequence 5_tag {S,B,M,E,N}, part of speech

porter_stemmer = PorterStemmer() 
f1=open('E:\\KEA_code\\crf\\b_KP20K_cixing.json','r')
lines1=f1.readlines()
f2=open('E:\\KEA_code\\crf\\a_KP20K.tr.json','r')
#f2=open('E:\\KEA_code\\crf\\a_Semeval.tr_ref.json','r')
lines2=f2.readlines()
f3=open('E:\\KEA_code\\crf\\a_KP20K.tfidf.json','r') 
#f3=open('E:\\KEA_code\\crf\\a_Semeval.tfidf_ref.json','r') 
lines3=f3.readlines()
w=open('E:\\KEA_code\\crf\\c_KP20K_bz.json','w')
#w=open('E:\\KEA_code\\crf\\c_Semeval_bz_ref.json','w')
def bz_trtf(tr_llist,tf_list,word):
    TR={}
    for i in range(len(tr_llist)):
        tr2=re.split(":",tr_llist[i])
        tr3=float(tr2[len(tr2)-1])
        if (tr3>=0 and tr3<0.5):
            TR[tr2[0]]="1"
        elif (tr3>=0.5 and tr3<1.0):
            TR[tr2[0]]="2"
        elif (tr3>=1.0 and tr3<1.5):
            TR[tr2[0]]="3"
        elif (tr3>=1.5 and tr3<2.0):
            TR[tr2[0]]="4"
        elif (tr3>=2.0 and tr3<2.5):
            TR[tr2[0]]="5"
        elif (tr3>=2.5 and tr3<3.0):
            TR[tr2[0]]="6"
        elif (tr3>=3.0 and tr3<3.5):
            TR[tr2[0]]="7"
        elif (tr3>=3.5 and tr3<4.0):
            TR[tr2[0]]="8"
        elif (tr3>=4.0 and tr3<4.5):
            TR[tr2[0]]="9"
        elif tr3>=4.5:
            TR[tr2[0]]="10"
        else:
            print(tr3)
    TF={}
    for i in range(len(tf_list)-1):
        tf2=re.split(":",tf_list[i])
        tf3=float(tf2[len(tf2)-1])
        if tf3>=0 and tf3<0.5:
            TF[tf2[0]]="1"
        elif tf3>=0.5 and tf3<1.0:
            TF[tf2[0]]="2"
        elif tf3>=1.0 and tf3<1.5:
            TF[tf2[0]]="3"
        elif tf3>=1.5 and tf3<2.0:
            TF[tf2[0]]="4"
        elif tf3>=2.0 and tf3<2.5:
            TF[tf2[0]]="5"
        elif tf3>=2.5 and tf3<3.0:
            TF[tf2[0]]="6"
        elif tf3>=3.0 and tf3<3.5:
            TF[tf2[0]]="7"
        elif tf3>=3.5 and tf3<4.0:
            TF[tf2[0]]="8"
        elif tf3>=4.0 and tf3<4.5:
            TF[tf2[0]]="9"
        elif tf3>=4.5:
            TF[tf2[0]]="10"
        else:
            print(tf3)
    result=[]
    if word in TR:
        if word in TF:
            result.append(TR[word]+" "+TF[word]+"\n")
        else:
            result.append(TR[word]+" "+"0"+"\n")
    else:
        if word in TF:
            result.append("0"+" "+TF[word]+"\n")
        else:
            result.append("0"+" "+"0"+"\n")
    return result

def bz_location(word_list,key_list,loc,tr_llist,tf_list):
    i=0
    n_len=len(word_list)
    output=[]
    while i<n_len:
        for key1 in key_list:
            if len(key1)==1:
                key2=[]
                key2.append(key1)
            else:
                key2=re.split(" ",key1)
            if word_list[i][0]==key2[0]:
                l=len(key2)
                if l==1:
                    result=[]
                    result.append(word_list[i][0]+" "+word_list[i][1]+" "+loc+" "+"S"+" ")
                    output.append("".join(result+bz_trtf(tr_llist,tf_list,word_list[i][0])))
                    i+=1
                    break
                elif l>1 and (i+l)<n_len:
                    w=[]
                    for wi in range(i,i+l):
                        w.append(word_list[wi][0])
                    word1=" ".join(w)
                    if word1==key1:
                        if l==1:
                            result=[]
                            result.append(word_list[i][0]+" "+word_list[i][1]+" "+loc+" "+"S"+" ")
                            output.append("".join(result+bz_trtf(tr_llist,tf_list,word_list[i][0])))
                            i+=1
                            break
                        elif l==2:
                            result1=[]
                            result2=[]
                            result1.append(word_list[i][0]+" "+word_list[i][1]+" "+loc+" "+"B"+" ")
                            output.append("".join(result1+bz_trtf(tr_llist,tf_list,word_list[i][0])))
                            result2.append(word_list[i+1][0]+" "+word_list[i+1][1]+" "+loc+" "+"E"+" ")
                            output.append("".join(result2+bz_trtf(tr_llist,tf_list,word_list[i+1][0])))
                            i+=2
                            break
                        elif l==3:
                            result3=[]
                            result4=[]
                            result5=[]
                            result3.append(word_list[i][0]+" "+word_list[i][1]+" "+loc+" "+"B"+" ")
                            output.append("".join(result3+bz_trtf(tr_llist,tf_list,word_list[i][0])))
                            result4.append(word_list[i+1][0]+" "+word_list[i+1][1]+" "+loc+" "+"M"+" ")
                            output.append("".join(result4+bz_trtf(tr_llist,tf_list,word_list[i+1][0])))
                            result5.append(word_list[i+2][0]+" "+word_list[i+2][1]+" "+loc+" "+"E"+" ")
                            output.append("".join(result5+bz_trtf(tr_llist,tf_list,word_list[i+2][0])))
                            i+=3
                            break
                        elif l>3:
                            result6=[]
                            result6.append(word_list[i][0]+" "+word_list[i][1]+" "+loc+" "+"S"+" ")
                            output.append("".join(result6+bz_trtf(tr_llist,tf_list,word_list[i][0])))
                            for p in range(1,l-1):
                                result7=[]
                                result7.append(word_list[i+p][0]+" "+word_list[i+p][1]+" "+loc+" "+"M"+" ")
                                output.append("".join(result7+bz_trtf(tr_llist,tf_list,word_list[i+p][0])))
                            result8=[]
                            result8.append(word_list[i+l-1][0]+" "+word_list[i+l-1][1]+" "+loc+" "+"E"+" ")
                            output.append("".join(result8+bz_trtf(tr_llist,tf_list,word_list[i+l-1][0])))
                            i+=l
                            break
                        else:
                            print("错误")
                else:
                    if i<n_len:
                        result10=[]
                        result10.append(word_list[i][0]+" "+word_list[i][1]+" "+loc+" "+"N"+" ")
                        output.append("".join(result10+bz_trtf(tr_llist,tf_list,word_list[i][0])))
                        i+=1
                        break
                    else:
                        break
        else:
            if i<n_len:
                if word_list[i][0] is ".":
                    output.append("\n")
                else:
                    result11=[]
                    result11.append(word_list[i][0]+" "+word_list[i][1]+" "+loc+" "+"N"+" ")
                    output.append("".join(result11+bz_trtf(tr_llist,tf_list,word_list[i][0])))
                i+=1
            else:
                break
    return output

for line1,line2,line3 in zip(lines1,lines2,lines3):	
    title=re.findall(r"\"title\":\"(.*?)\", \"",line1)
    title0=re.split(" ",title[0])
    tit_list=[]
    for ti in title0:
        ti_word=re.split(":",ti)
        tit_list.append(ti_word)
    
    abstract=re.findall(r"\"abstract\":\"(.*?)\", \"",line1)
    ab0=re.split(" ",abstract[0].replace(r'\x',' '))
    ab_list=[]
    for abi in ab0:
        ab_word=re.split(":",abi)
        ab_list.append(ab_word)
    key=re.findall(r"\"keywords\":\"(.*?)\",",line1)
    key0=re.split(";",key[0])
            
    ref=re.findall(r"\"references\": \[(.*?)\]}",line1)
    ref0=re.split(" ",ref[0])
    ref_list=[]
    for rei in ref0:
        ref_word=re.split(":",rei)
        ref_list.append(ref_word)
    abs0=ab_list
#    abs0=ab_list+ref_list #Control whether reference information is added.
    tr1=re.split(", ",line2.strip())
    tf1=re.split(", ",line3.strip())
    
#the titles
    t_loc="1"
    title_biao=bz_location(tit_list,key0,t_loc,tr1,tf1)
    w.writelines(title_biao)
    w.writelines("\n")

#the absreact / the absreact and references
    c_loc="0"
    con_biao=bz_location(abs0,key0,c_loc,tr1,tf1)
    w.writelines(con_biao)
    w.writelines("%%%"+"\n"+"\n")

w.close()
f1.close()
f2.close()
f3.close()
