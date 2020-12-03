#Select the features of the corpus.

# -*- coding: utf-8 -*-
import json
import re

#Select the characteristics of the corpus.
#Adjust feature sequence

f=open('E:\\KEA_code\\crf\\corpus\\test1.json','r')
#f=open('E:\\KEA_code\\crf\\corpus\\train1.json','r')
#f=open('E:\\KEA_code\\crf\\corpus2\\10\\test10_ref.json','r')
#f=open('E:\\KEA_code\\crf\\corpus2\\10\\train10_ref.json','r')
lines=f.readlines()
w=open('E:\\KEA_code\\crf\\corpus\\test1_yl.json','w')
#w=open('E:\\KEA_code\\crf\\corpus\\train1_yl.json','w')
#w=open('E:\\KEA_code\\crf\\corpus2\\10\\test10_yl_ref.json','w')
#w=open('E:\\KEA_code\\crf\\corpus2\\10\\train10_yl_ref.json','w')
i=0
for line in lines:
    i+=1
    output=[]
    F=['\n']
    if line in F:
        output.append(line)
    else:
        ke=re.split(" ",line.strip()) 
        if len(ke)==6:
            if len(ke[1])>0:
                output.append(ke[0]+' '+ke[1]+' '+ke[2]+' '+ke[4]+' '+ke[5]+' '+ke[3]+'\n')#features selection

            else:
                output.append('\n')
        elif len(ke)==1:
            output.append(ke[0]+' '+"%"+' '+"0"+' '+'0'+' '+"0"+' '+"N"+'\n')#features selection

        else:
            continue

    w.writelines(output)

w.close()
f.close()
