# -*- coding: utf-8 -*-
import json
import re

#The corpus is divided into test set and training set.

f=open('E:\\KEA_code\\crf\\c_KP20K_bz.json','r')
#f=open('E:\\KEA_code\\crf\\c_Semeval_bz_ref.json','r')

lines=f.readlines()
w1=open('E:\\KEA_code\\crf\\corpus\\test1.json','w')
w2=open('E:\\KEA_code\\crf\\corpus\\train1.json','w')
#w1=open('E:\\KEA_code\\crf\\corpus2\\10\\test10_ref.json','w')
#w2=open('E:\\KEA_code\\crf\\corpus2\\10\\train10_ref.json','w')
i=1
F=['%%%']
for line in lines:
    line1=re.split('\t',line.strip())
    if i>0 and i<201:
        if line1[0] in F:
            i+=1
        w1.writelines(line)
    else:
        if line1[0] in F:
            i+=1
        w2.writelines(line)

w2.close()
w1.close()
f.close()
