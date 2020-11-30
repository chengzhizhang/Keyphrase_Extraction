# -*- coding: utf-8 -*-
import json
import re

#Process the corpus into the format required by KEA

f=open('E:\\KEA_code\\data\\Semeval\\corpus\\1\\train1.json','r')
lines=f.readlines()
#wr=open('E:\\KEA_code\\NB\\corpus\\10\\train10\\25.key','w')
wr=open('E:\\KEA_code\\NB\\corpus\\1\\train1_ref\\25.key','w')
i=25 
for line in lines:
    i+=1
    key_au=re.findall(r"\"keywords\":\"(.*?)\", \"",line)
    ke1=re.split(";",key_au[0]) 
    for ke2 in ke1:
        wr.writelines(ke2+'\n')
    wr.close()
#    wr=open('E:\\KEA_code\\NB\\corpus\\10\\train10\\'+str(i)+'.key','w')
    wr=open('E:\\KEA_code\\NB\\corpus\\1\\train1_ref\\'+str(i)+'.key','w')

wr.close()
f.close()
print('finished')