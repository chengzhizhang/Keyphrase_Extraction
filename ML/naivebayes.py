import argparse
import os
import time
from configs import DATASET_PATH
from evaluate import evaluate
from preprocessing import data_preprocessing
from utils import merge_datas_toline, read_datas, get_targets
import numpy as np
import pandas as pd

# Getting parameters from the command line
def init_args():
    parser = argparse.ArgumentParser(description='Key phrase extraction using NaiveBayes.')
    # Input/output options
    parser.add_argument('--mode', '-m', default='train', type=str, help='Mode of program operation.')
    parser.add_argument('--dataset_name', '-dn', default='SemEval-2010', type=str, help='Name of the dataset.')
    parser.add_argument('--fields', '-fd', default=['title', 'abstract'], type=str, nargs='+', help='Fields from which key phrases need to be extracted.')
    parser.add_argument('--kea_path', '-kp', default="./KEA-3.0", type=str, help='The file path of KEA.')
    parser.add_argument('--TOPN', '-n', default=10, type=int, help='The key phrases are ranking TopN.')
    parser.add_argument('--eTOPNS', '-en', type=int, default=3, nargs="+", help='The key phrases are ranking TopN in evalution.')
    opt = parser.parse_args()
    return opt

# Preprocess datas
def preprocess_datas(path, fields, mode, save_folder):
    corpus_type = "".join([i[0] for i in fields]).upper()
    ids = merge_datas_toline(data_preprocessing(read_datas(path), ['id'], [1]), 1)
    texts = merge_datas_toline(data_preprocessing(read_datas(path), fields, [2] * len(fields)), 1)
    targets = merge_datas_toline(data_preprocessing(read_datas(path), ['keywords'], [3]), 1)
    # Construction of corpus folder
    save_path = os.path.join(os.path.abspath(save_folder), mode, corpus_type)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    else:
        for file_name in os.listdir(save_path):
            os.remove(os.path.join(save_path, file_name))
    # Save training corpus
    for i in range(len(texts)):
        with open(os.path.join(save_path, ids[i]+".txt"), "w", encoding="utf-8") as fp:
            fp.write(texts[i])
        if mode == 'train':
            with open(os.path.join(save_path, ids[i] + ".key"), "w", encoding="utf-8") as fp:
                fp.write(targets[i])
    print("\033[34mINFO: Dataset construction completed.\033[0m")

# Training model
def train(root_path, dataset_name, mode, fields, kea_path):
    corpus_type = "".join([i[0] for i in fields]).upper()
    path = os.path.join(os.path.abspath(root_path), mode, corpus_type)
    # Switch to Kea path, and only switch once
    os.chdir(os.path.abspath(kea_path))
    # Run the program
    state = os.system("java KEAModelBuilder -l {0} -m AKE_Model_{1}_{2}"
                      .format(os.path.join(root_path, path), dataset_name, corpus_type))
    if state == 0:
        print("\033[34mINFO: Model training completed!\033[0m")
    else:
        print("\033[31mINFO: Model training failed!\033[0m")

# Prediction model
def predict(root_path, dataset_name, mode, fields, kea_path, topN=3):
    corpus_type = "".join([i[0] for i in fields]).upper()
    path = os.path.join(os.path.abspath(root_path), mode, corpus_type)
    # Switch to Kea path, and only switch once
    os.chdir(os.path.abspath(kea_path))
    # Delete files with suffix. Key
    for i in os.listdir(path):
        if i.endswith(".key"): os.remove(os.path.join(path, i))
    # Run the program
    state = os.system("java KEAKeyphraseExtractor -l {0} -m AKE_Model_{1}_{2} -n {3}"
                      .format(path, dataset_name, corpus_type, topN))
    if state == 0:
        print("\033[34mINFO: Test set result prediction completed!\033[0m")
    else:
        print("\033[34mINFO: Test set result prediction failed!\033[0m")
    os.chdir(os.path.abspath('../'))

# Read prediction results
def read_results(path, targets):
    targets = [line.split("\n") for line in targets]
    targets = {i[0]: i[1:] for i in targets}
    rev_pred, rev_targets, ids = [], [], []
    for name in os.listdir(path):
        if name.endswith(".key"):
            file_path, id = os.path.join(os.path.abspath(path), name), name.split(".key")[0]
            with open(file_path, "r") as fp:
                rev_pred.append([i.strip() for i in fp.readlines()])
            rev_targets.append(targets[id])
            ids.append(id)
    return rev_pred, rev_targets, ids

# Assessment results
def evalute_results(save_path, mode, fields, targets, topk=3):
    corpus_type = "".join([i[0] for i in fields]).upper()
    path = os.path.join(os.path.abspath(save_path), mode, corpus_type)
    rev_pred, rev_targets, ids = read_results(path, targets)
    evaluate_results = evaluate(rev_pred, rev_targets, topk=topk)
    return evaluate_results

if __name__ == '__main__':
    # Initialization parameters
    args = init_args()
    print('=' * 20, "Initialization information", '=' * 20)
    print("dataset_name:", args.dataset_name)
    print("fields:", args.fields)
    file_path = os.path.join(DATASET_PATH, "%s/%s.json" % (args.dataset_name, args.mode))
    corpus_type = "".join([i[0] for i in args.fields]).upper()
    save_path = os.path.join("./datas", args.dataset_name, 'naivebayes')

    # Select different modes to run the program
    if args.mode == 'train':
        preprocess_datas(file_path, args.fields, args.mode, save_path)
        start_time = time.time()
        train(save_path, args.dataset_name, args.mode, args.fields, args.kea_path)
        print("\033[32mTotal time(%s): %sS\033[0m" % (args.mode, round(time.time() - start_time, 2)))
    elif args.mode == 'test' or args.mode == 'dev':
        preprocess_datas(file_path, args.fields, args.mode, save_path)
        start_time = time.time()
        predict(save_path, args.dataset_name, args.mode, args.fields, args.kea_path, args.TOPN)
        print("\033[32mTotal time(%s): %sS\033[0m" % (args.mode, round(time.time() - start_time, 2)))
        # Calculate performance merics
        origin_datas = read_datas(file_path)
        targets = merge_datas_toline(
            data_preprocessing(get_targets(origin_datas, args.fields, choose=3), ['id', 'keywords'], [1, 3]), 1)
        if type(args.eTOPNS) != list:
            results = evalute_results(save_path, args.mode, args.fields, targets, topk=args.eTOPNS)
            df = pd.DataFrame(np.array([args.eTOPNS] + list(results)).reshape(1, 5), columns=['TopN', 'P', 'R', 'F1', 'Keywords_num'])
            df.to_csv(os.path.join("./results", args.dataset_name, 'naivebayes_%s.csv' % corpus_type))
            print(df)
        else:
            rs = []
            for topk in args.eTOPNS:
                evaluate_results = evalute_results(save_path, args.mode, args.fields, targets, topk=topk)
                rs.append([topk] + list(evaluate_results))
            df = pd.DataFrame(np.array(rs), columns=['TopN', 'P', 'R', 'F1', 'Keywords_num'])
            df.to_csv(os.path.join("./results", args.dataset_name, 'naivebayes_%s.csv' % corpus_type))
            print(df)
    else:
        raise RuntimeError("The value of the variable '-m' is in ['train', 'dev', 'test', 'evaluate'.")