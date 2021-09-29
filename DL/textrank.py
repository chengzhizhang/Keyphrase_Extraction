# -*- coding: utf-8 -*-
from tqdm import tqdm
import numpy as np

# TextRank algorithm implementation
def textrank(datas, window = 3, alpha = 0.85, iternum = 100, threshold=0.0001):
    return_datas = []
    with tqdm(range(len(datas))) as pbar:
        for index in pbar:
            word_list = datas[index]
            edges, nodes = {}, []
            word_list_length = len(word_list)
            # According to the window size, construct the edge set
            for i, word in enumerate(word_list):
                if word not in edges.keys():
                    nodes.append(word)
                    link_nodes = []
                    left, right = i - window + 1, i + window
                    if left < 0: left = 0
                    if right > word_list_length: right = word_list_length
                    for j in range(left, right):
                        if j != i: link_nodes.append(word_list[j])
                    edges[word] = set(link_nodes)
            # According to the relationship between the edges, the matrix is constructed
            word_index, index_dict = {}, {}  # Build node set and index table
            for i, v in enumerate(edges):
                word_index[v] = i
                index_dict[i] = v
            matrix = np.zeros([len(nodes), len(nodes)])
            for key in edges.keys():
                for w in edges[key]:
                    matrix[word_index[key]][word_index[w]] = 1
                    matrix[word_index[w]][word_index[key]] = 1
            # Calculate the relative weight of the point pointed to by node V_i in Out(V_i)
            matrix = matrix/np.sum(matrix, axis=0)
            # Iteration
            score, last_score = np.ones([len(nodes), 1]), np.zeros([len(nodes), 1])
            for i in range(iternum):
                score = (1 - alpha) + alpha * np.dot(matrix, score)
                diff = np.sum(np.abs(score - last_score))
                if diff < threshold: break
                last_score = score
           # Output results
            keywords = {}
            for i in range(len(score)):
                keywords[index_dict[i]] = score[i][0]

            return_datas.append(keywords)
            pbar.set_description("The %s document has been calculated TextRank completed" % (index + 1))
    return return_datas
