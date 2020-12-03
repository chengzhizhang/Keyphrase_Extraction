#Reordering the data in the dataset

# -*- coding: utf-8 -*-
import random
#Reordering the data in the dataset
f=open('Data\\Semeval2010_244.json','rb')
lines=f.readlines()
w1=open('Data\\Semeval\\Semeval.range_244.json','w')
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
