# -*- coding: utf-8 -*-
import re

#Writes the predicted words with {S,B,M,E} tags in line order to the new file.

f=open(r'E:\\KEA_code\\crf\\corpus\\test11.txt','r')
lines=f.readlines()
w=open(r'E:\\KEA_code\\crf\\corpus\\test11_tiqu.txt','w')
output=[]
F=['%%%']
for line in lines:
    line1=re.split('\t',line.strip())
    if len(line1)>2: 
        if line1[0] in F:
            output.append('\n')
        else:
            if line1[-1]=='S':
                op=line1[0]+":"+"S"
                output.append(op)
            elif line1[-1]=='B':
                op=line1[0]+":"+"B"
                output.append(op)
            elif line1[-1]=='M':
                op=line1[0]+":"+"M"
                output.append(op)
            elif line1[-1]=='E':
                op=line1[0]+":"+"E"
                output.append(op)
            else:
                continue
    else:
        pass

w.writelines(" ".join(output))
w.close()
f.close()