# -*- coding: utf-8 -*-
import json
import re

#Process the corpus into the format required by KEA

#f=open('E:\\KEA_code\\data\\Semeval\\corpus\\1\\test1_24.json','r')
f=open('E:\\KEA_code\\data\\Semeval\\corpus\\1\\train1.json','r')
lines=f.readlines()
#wr=open('E:\\KEA_code\\NB\\corpus\\10\\test10\\1.txt','w')
#wr=open('E:\\KEA_code\\NB\\corpus\\10\\train10\\25.txt','w')
#wr=open('E:\\KEA_code\\NB\\corpus\\1\\test1_ref\\1.txt','w')
wr=open('E:\\KEA_code\\NB\\corpus\\1\\train1_ref\\25.txt','w')

i=25
for line in lines:
    i+=1
    id0=re.findall(r"\"id\": \"(.*?)\",",line) 
    wr.writelines("id: "+id0[0]+"\n")
    tit=re.findall(r"\"title\":\"(.*?)\",",line)
    wr.writelines("title: "+tit[0]+"\n")
    ab=re.findall(r"\"abstract\":\"(.*?)\", \"",line)
#    wr.writelines("abstract: "+ab[0])
    wr.writelines("abstract: "+ab[0]+"\n")
    ref=re.findall(r"\"references\": \[(.*?)\]}",line) 
    wr.writelines("references: "+ref[0])
    wr.close()
#    wr=open('E:\\KEA_code\\NB\\corpus\\1\\test1_ref\\'+str(i)+'.txt','w')
    wr=open('E:\\KEA_code\\NB\\corpus\\1\\train1_ref\\'+str(i)+'.txt','w')

wr.close()
f.close()
print('finished')