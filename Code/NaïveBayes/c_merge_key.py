#Merge the predicted keywords for each paper into a single file, line by line.

# -*- coding: utf-8 -*-
import os


#w=open('E:\\KEA_code\\NB\\KEA-3.0\\key\\key10.txt','w')
w=open('E:\\KEA_code\\NB\\KEA-3.0\\key\\key2_ref.txt','w')
output=[]
for i in range(1,25):
#    f=open('E:\\KEA_code\\NB\\KEA-3.0\\data\\test10\\'+str(i)+'.key','r')
    f=open('E:\\KEA_code\\NB\\KEA-3.0\\data\\test2_ref\\'+str(i)+'.key','r')
    lines=f.readlines()
    l=len(lines)
    output1=[]
    for i in range(l):
        output1.append(lines[i].strip())
    keywords="; ".join(output1)
    output.append(keywords)
    output.append('\n')

w.writelines(output)
w.close()
f.close()
