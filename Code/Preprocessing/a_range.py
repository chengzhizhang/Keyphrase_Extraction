# -*- coding: utf-8 -*-
import random
#Reordering the data in the dataset
f=open('E:\\KEA_code\\data\\Semeval2010_244.json','rb')
lines=f.readlines()
w1=open('E:\\KEA_code\\data\\Semeval\\Semeval.range_244.json','w')
num=[]
con=[]
for i in range(0,244):
    num.append(i)
random.shuffle(num)

for m in num:
    w1.writelines(str(lines[m].strip()))
    w1.writelines('\n')
#    print(con[m])

w1.close()
f.close()
