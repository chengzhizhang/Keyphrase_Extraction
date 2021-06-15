import argparse
import os

from tqdm import tqdm
import pandas as pd
import numpy as np

from configs import DATASET_PATH
from evaluate import evaluate
from preprocessing import data_preprocessing
from utils import read_datas, save_datas, merge_phrases, merge_datas_toline, get_targets

# Getting parameters from the command line
def init_args():
    parser = argparse.ArgumentParser(description='Key phrase extraction using TF-IDF.')
    # Input/output options
    parser.add_argument('--mode', '-m', default='run', type=str, help='Mode of program operation.')
    parser.add_argument('--dataset_name', '-dn', default='SemEval-2010', type=str, help='Name of the dataset.')
    parser.add_argument('--fields', '-fd', default=['title', 'abstract'], type=str, nargs='+', help='Fields from which key phrases need to be extracted.')
    parser.add_argument('--TOPN', '-n', default=10, type=int,  help='The key phrases are ranking TopN.')
    parser.add_argument('--eTOPNS', '-en', type=int, default=3, nargs="+", help='The key phrases are ranking TopN in evalution.')
    parser.add_argument('--evaluate_path', '-sp', type=str, help='Save path of evaluation datas.')
    opt = parser.parse_args()
    return opt


# Implementation of textrank algorithm
def textrank(datas, content_of_merging_phrases = None, topk = 10,
             is_merge_phrases = True, window = 3, alpha = 0.85, iternum = 100, threshold=0.0001):
    return_datas = []
    with tqdm(range(len(datas))) as pbar:
        for index in pbar:
            word_list = datas[index]
            edges, nodes = {}, []
            word_list_length = len(word_list)
            # According to the window size, the edge set is constructed
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
            word_index, index_dict = {}, {}  # Building node and index table
            for i, v in enumerate(edges):
                word_index[v] = i
                index_dict[i] = v
            matrix = np.zeros([len(nodes), len(nodes)])
            for key in edges.keys():
                for w in edges[key]:
                    matrix[word_index[key]][word_index[w]] = 1
                    matrix[word_index[w]][word_index[key]] = 1
            # Calculate out (V_ i) Middle node V_ The relative weight of the point I points to
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
            # Merge phrases
            if is_merge_phrases:
                rs = merge_phrases(list(keywords.items()), content_of_merging_phrases[index], topk)
                return_datas.append(rs)
            else:
                return_datas.append(keywords)
            pbar.set_description("Textrank calculation for document %s completed" % (index + 1))
    return return_datas

if __name__ == "__main__":
    # Initialization parameters
    args = init_args()
    print("dataset_name:", args.dataset_name)
    print("fields:", args.fields)
    file_path = os.path.join(DATASET_PATH, args.dataset_name+".json")
    save_path = os.path.join("./datas", args.dataset_name, 'textrank/rs.json')

    # Select different modes to run the program
    if args.mode == 'run':
        origin_datas = read_datas(file_path)
        ref_of_merging_phrases = merge_datas_toline(data_preprocessing(origin_datas, ['title'], [2]), 1)
        datas = merge_datas_toline(data_preprocessing(origin_datas, args.fields, [7] * len(args.fields)), 2)
        keywords = textrank(datas, ref_of_merging_phrases, topk=args.TOPN)
        save_datas(keywords, save_path)
    elif args.mode == 'evaluate1':
        origin_datas = read_datas(file_path)
        targets = merge_datas_toline(data_preprocessing(get_targets(origin_datas, args.fields, choose=3), ['keywords'], [3]), 2)
        if type(args.eTOPNS) != list:
            evaluate_results = evaluate(save_path, targets, topk=args.eTOPNS)
            df = pd.DataFrame(np.array([args.eTOPNS] + list(evaluate_results)).reshape(1, 5),
                              columns=['TopN', 'P', 'R', 'F1', 'Keywords_num'])
            if args.evaluate_path != None: df.to_csv(args.evaluate_path)
            print(df)
        else:
            rs = []
            for topk in args.eTOPNS:
                evaluate_results = evaluate(save_path, targets, topk=topk)
                rs.append([topk] + list(evaluate_results))
            df = pd.DataFrame(np.array(rs), columns=['TopN', 'P', 'R', 'F1', 'Keywords_num'])
            if args.evaluate_path != None: df.to_csv(args.evaluate_path)
            print(df)
    elif args.mode == 'evaluate2':
        origin_datas = read_datas(file_path)
        targets = merge_datas_toline(data_preprocessing(get_targets(origin_datas, args.fields, choose=3), ['keywords'], [3]), 2)
        title_abstract = merge_datas_toline(data_preprocessing(origin_datas, ['title', 'abstract'], [2, 2]), 1)
        if type(args.eTOPNS) != list:
            evaluate_results = evaluate(save_path, targets, title_abstract=title_abstract, topk=args.eTOPNS)
            df = pd.DataFrame(np.array([args.eTOPNS] + list(evaluate_results)).reshape(1, 5), columns=['TopN', 'P', 'R', 'F1', 'Keywords_num'])
            if args.evaluate_path != None: df.to_csv(args.evaluate_path)
            print(df)
        else:
            rs = []
            for topk in args.eTOPNS:
                evaluate_results = evaluate(save_path, targets, title_abstract=title_abstract, topk=topk)
                rs.append([topk] + list(evaluate_results))
            df = pd.DataFrame(np.array(rs), columns=['TopN', 'P', 'R', 'F1', 'Keywords_num'])
            if args.evaluate_path != None: df.to_csv(args.evaluate_path)
            print(df)
    else:
        raise RuntimeError("The value of the variable '-m' is in ['run', 'evaluate1', 'evaluate2'].")