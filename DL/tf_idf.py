# -*- coding: utf-8 -*-
import math
from tqdm import tqdm
from collections import Counter


# 计算idf
def cal_idf(datas):
    total_files  = len(datas)
    all_words = set([j for line in datas for j in line])
    temp = {word: 0.0 for word in all_words}
    for word in temp.keys():
        for words in datas:
            if word in words:
                temp[word] += 1.0
    idf = {}
    for k, v in temp.items():
        idf[k] = math.log((total_files + 1) / (v + 1), 2)
    return idf

# 计算tf-idf
def tf_idf(datas):
    idf = cal_idf(datas)
    return_datas = []
    with tqdm(range(len(datas))) as pbar:
        for index in pbar:
            words = datas[index]
            num_words = len(words)
            tf = Counter(words)
            keywords = {}
            for k, v in tf.items():
                keywords[k] = (v / num_words) * idf[k]
            return_datas.append(keywords)
            pbar.set_description("第%s个文档计算TF-IDF完成"%(index + 1))
    return return_datas
