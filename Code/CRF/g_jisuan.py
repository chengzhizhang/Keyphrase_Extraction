# -*- coding: utf-8 -*-
import re

#Combines the extracted labeled words and writes them to the new file.

f=open(r'E:\\KEA_code\\crf\\corpus\\test11_tiqu.txt','r')
lines=f.readlines()
w=open(r'E:\\KEA_code\\crf\\corpus\\test11_jisuan.json','w')
output=[]
for line in lines:
    line1=re.split(' ',line.strip())
    if len(line1):
        output1=[]
        l=len(line1)
        i=0
        while i<l:
            k=re.split(':',line1[i])
            if len(k)==2 and k[1]=='S':
                output1.append(k[0])
                i+=1
            elif len(k)==2 and k[1]=='B':
                output2=[]
                output2.append(k[0])
                k2=re.split(':',line1[i+1])
                if k2[1]=='M':
                    output2.append(k2[0])
                    k3=re.split(':',line1[i+2])
                    if k3[1]=='M':
                        output2.append(k3[0])
                        k4=re.split(':',line1[i+3])
                        if k4[1]=='E':
                            output2.append(k4[0])
                            ke4=" ".join(output2)
                            output1.append(ke4.lower())
                            i+=4
                        elif k4[1]=='M':
                            output2.append(k4[0])
                            k5=re.split(':',line1[i+4])
                            if k5[1]=='E':
                                output2.append(k5[0])
                                ke5=" ".join(output2)
                                output1.append(ke5.lower())
                                i+=5
                            elif k5[1]=='M':
                                output2.append(k5[0])
                                k6=re.split(':',line1[i+5])
                                if k6[1]=='E':
                                    output2.append(k6[0])
                                    ke6=" ".join(output2)
                                    output1.append(ke6.lower())
                                    i+=6
                                else:
                                    i+=1
                                    print(存在)
                    elif k3[1]=='E':
                        output2.append(k3[0])
                        ke3=" ".join(output2)
                        output1.append(ke3.lower())
                        i+=3
                    else:
                        i+=1
                elif k2[1]=='E':
                    output2.append(k2[0])
                    ke2=" ".join(output2)
                    output1.append(ke2.lower())	
                    i+=2
                else:
                    i+=1
            else:
                i+=1
        out=list(set(output1))#duplicate removal
        key=";".join(out)
        output.append(key)
        output.append('\n')
    else:
        output.append('\n')
w.writelines(output)
w.close()
f.close()
