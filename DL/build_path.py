# -*- coding: utf-8 -*-
# Create file path
import os

# Creating root folder
up_names = ['datas', 'results']
for name in up_names:
    path = os.path.join('./', name)
    if not os.path.exists(path):
        os.makedirs(path)
        print(path, ': The path created successfully.')
    else:
        print(path, ': The path already exists.')

# Creating datas subfolder
data_names = ['KP20k', 'LIS-2000', 'SemEval-2010']
for name in data_names:
    path = os.path.join('./datas', name)
    if not os.path.exists(path):
        os.makedirs(path)
        print(path, ': The path created successfully.')
    else:
        print(path, ': The path already exists.')
