import argparse
import math
import os.path
import time
from tqdm import tqdm
from collections import Counter
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
    parser.add_argument('--data_type', '-dt', default='test', type=str, help='Type of data.')
    parser.add_argument('--fields', '-fd', default=['title', 'abstract'], type=str, nargs='+', help='Fields from which key phrases need to be extracted.')
    parser.add_argument('--TOPN', '-n', default=10, type=int,  help='The key phrases are ranking TopN.')
    parser.add_argument('--eTOPNS', '-en', type=int, default=3, nargs="+", help='The key phrases are ranking TopN in evalution.')
    opt = parser.parse_args()
    return opt

# Calculating IDF
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

# Calculating TF-IDF
def tf_idf(datas, content_of_merging_phrases = None, topk = 10,
           is_merge_phrases = True, verbose = True):
    idf = cal_idf(datas)
    return_datas = []
    if verbose:
        with tqdm(range(len(datas))) as pbar:
            for index in pbar:
                words = datas[index]
                num_words = len(words)
                tf = Counter(words)
                keywords = {}
                for k, v in tf.items():
                    keywords[k] = (v / num_words) * idf[k]
                # Merge phrases
                if is_merge_phrases:
                    rs = merge_phrases(list(keywords.items()), content_of_merging_phrases[index], topk=topk)
                    return_datas.append(rs)
                else:
                    return_datas.append(keywords)
                pbar.set_description("TF-IDF calculation for document %s completed"%(index + 1))
    else:
        for index in range(len(datas)):
            words = datas[index]
            num_words = len(words)
            tf = Counter(words)
            keywords = {}
            for k, v in tf.items():
                keywords[k] = (v / num_words) * idf[k]
            # Merge phrases
            if is_merge_phrases:
                rs = merge_phrases(list(keywords.items()), content_of_merging_phrases[index], topk=topk)
                return_datas.append(rs)
            else:
                return_datas.append(keywords)
    return return_datas

if __name__ == "__main__":
    # Initialization parameters
    args = init_args()
    print('='*20, "Initialization information", '='*20)
    print("dataset_name:", args.dataset_name)
    print("data_type:", args.data_type)
    print("fields:", args.fields)
    file_path = os.path.join(DATASET_PATH, "%s/%s.json"%(args.dataset_name, args.data_type))
    corpus_type = "".join([i[0] for i in args.fields]).upper()
    save_path = os.path.join("./datas", args.dataset_name, 'tf_idf/results_%s.json'%corpus_type)

    # Select different modes to run the program
    if args.mode == 'run':
        origin_datas = read_datas(file_path)
        ref_of_merging_phrases = merge_datas_toline(
            data_preprocessing(origin_datas, ['title', 'abstract'], [2, 2]), 1)
        # ref_of_merging_phrases = merge_datas_toline(
        #     data_preprocessing(origin_datas, args.fields, [2] * len(args.fields)), 1)
        datas = merge_datas_toline(
            data_preprocessing(origin_datas, args.fields, [7] * len(args.fields)), 2)
        start_time = time.time()
        keywords = tf_idf(datas, ref_of_merging_phrases, topk=args.TOPN)
        print("\033[32mTotal time(%s): %sS\033[0m" % (args.mode, round(time.time() - start_time, 2)))
        save_datas(keywords, save_path)
        print("\033[32mINFO: Result save path: %s\033[0m"%save_path)
    elif args.mode == 'evaluate':
        start_time = time.time()
        origin_datas = read_datas(file_path)
        targets = merge_datas_toline(data_preprocessing(get_targets(origin_datas, args.fields, choose=3), ['keywords'], [3]), 2)
        if type(args.eTOPNS) != list:
            evaluate_results = evaluate(save_path, targets, topk=args.eTOPNS)
            df = pd.DataFrame(np.array([args.eTOPNS] + list(evaluate_results)).reshape(1, 5),
                              columns=['TopN', 'P', 'R', 'F1', 'Keywords_num'])
            df.to_csv(os.path.join("./results", args.dataset_name, 'tf_idf_%s.csv' % corpus_type))
            df = df.round({"TopN":0, "P":2, "R":2, "F1":2, "Keywords_num":0})
            print(df)
        else:
            rs = []
            for topk in args.eTOPNS:
                evaluate_results = evaluate(save_path, targets, topk=topk)
                rs.append([topk] + list(evaluate_results))
            df = pd.DataFrame(np.array(rs), columns=['TopN', 'P', 'R', 'F1', 'Keywords_num'])
            df = df.round({"TopN": 0, "P": 2, "R": 2, "F1": 2, "Keywords_num": 0})
            df.to_csv(os.path.join("./results", args.dataset_name, 'tf_idf_%s.csv' % corpus_type))
            print(df)
            print("\033[32mTotal time(%s): %sS\033[0m" % (args.mode, round(time.time() - start_time, 2)))
    else:
        raise RuntimeError("The value of the variable '-m' is in ['run', 'evaluate'].")