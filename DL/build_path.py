# -*- coding: utf-8 -*-
# Create file path
import os
from PATH import MODEL_FOLDER

# file folder names
c1_names = ['datas', 'results']
c2_names = ['PubMed', 'LIS-2000', 'SemEval-2010']
c3_names = ['bin', 'vocab']

# Creating root folder
for name in c1_names:
    path = os.path.join('./', name)
    if not os.path.exists(path):
        os.makedirs(path)
        print(path, ': The path created successfully.')
    else:
        print(path, ': The path already exists.')

# Create secondary directory
for n1 in c1_names:
    for n2 in c2_names:
        path = os.path.join('./%s'%n1, n2)
        if not os.path.exists(path):
            os.makedirs(path)
            print(path, ': The path created successfully.')
        else:
            print(path, ': The path already exists.')

# Create tertiary directory
for n1 in c2_names:
    for n2 in c3_names:
        path = os.path.join(MODEL_FOLDER, n1, n2)
        if not os.path.exists(path):
            os.makedirs(path)
            print(path, ': The path created successfully.')
        else:
            print(path, ': The path already exists.')
