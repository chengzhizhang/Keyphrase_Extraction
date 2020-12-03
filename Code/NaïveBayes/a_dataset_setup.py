#The corpus is divided into ten data sets for ten fold cross validation.

# -*- coding: utf-8 -*-
import json

#The corpus is divided into ten data sets for ten fold cross validation.

f=open('E:\\KEA_code\\data\\Semeval\\b_Semeval.range_stem_244.json','r')
lines=f.readlines()
#w1=open('E:\\KEA_code\\data\\Semeval\\corpus\\1\\test1_24.json','w')
#w2=open('E:\\KEA_code\\data\\Semeval\\corpus\\1\\train1.json','w')
#w1=open('E:\\KEA_code\\data\\Semeval\\corpus\\2\\test25_48.json','w')
#w2=open('E:\\KEA_code\\data\\Semeval\\corpus\\2\\train2.json','w')
#w1=open('E:\\KEA_code\\data\\Semeval\\corpus\\3\\test49_72.json','w')
#w2=open('E:\\KEA_code\\data\\Semeval\\corpus\\3\\train3.json','w')
#w1=open('E:\\KEA_code\\data\\Semeval\\corpus\\4\\test73_96.json','w')
#w2=open('E:\\KEA_code\\data\\Semeval\\corpus\\4\\train4.json','w')
#w1=open('E:\\KEA_code\\data\\Semeval\\corpus\\5\\test97_120.json','w')
#w2=open('E:\\KEA_code\\data\\Semeval\\corpus\\5\\train5.json','w')
#w1=open('E:\\KEA_code\\data\\Semeval\\corpus\\6\\test121_144.json','w')
#w2=open('E:\\KEA_code\\data\\Semeval\\corpus\\6\\train6.json','w')
#w1=open('E:\\KEA_code\\data\\Semeval\\corpus\\7\\test145_168.json','w')
#w2=open('E:\\KEA_code\\data\\Semeval\\corpus\\7\\train7.json','w')
#w1=open('E:\\KEA_code\\data\\Semeval\\corpus\\8\\test169_192.json','w')
#w2=open('E:\\KEA_code\\data\\Semeval\\corpus\\8\\train8.json','w')
#w1=open('E:\\KEA_code\\data\\Semeval\\corpus\\9\\test193_216.json','w')
#w2=open('E:\\KEA_code\\data\\Semeval\\corpus\\9\\train9.json','w')
w1=open('E:\\KEA_code\\data\\Semeval\\corpus\\10\\test217_240.json','w')
w2=open('E:\\KEA_code\\data\\Semeval\\corpus\\10\\train10.json','w')
i=0
for line in lines:
    i+=1
    if i>216 and i<241:
        w1.writelines(line)
    else:
        w2.writelines(line)
w1.close()
w2.close()
f.close()
