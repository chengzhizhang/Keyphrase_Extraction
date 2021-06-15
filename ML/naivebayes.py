import argparse
import os
from sklearn.model_selection import KFold
from tqdm import tqdm

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
    parser.add_argument('--mode', '-m', default='data', type=str, help='Mode of program operation.')
    parser.add_argument('--dataset_name', '-dn', default='SemEval-2010', type=str, help='Name of the dataset.')
    parser.add_argument('--fields', '-fd', default=['title', 'abstract'], type=str, nargs='+', help='Fields from which key phrases need to be extracted.')
    parser.add_argument('--kea_path', '-kp', default="./KEA-3.0", type=str, help='The file path of KEA.')
    parser.add_argument('--TOPN', '-n', default=10, type=int, help='The key phrases are ranking TopN.')
    parser.add_argument('--eTOPNS', '-en', type=int, default=3, nargs="+", help='The key phrases are ranking TopN in evalution.')
    parser.add_argument('--evaluate_path', '-sp', type=str, help='Save path of evaluation datas.')
    opt = parser.parse_args()
    return opt

# Partition data set
def build_data_sets(path, fields, save_folder):
    # Preprocessed datas
    ids = merge_datas_toline(data_preprocessing(read_datas(path), ['id'], [1]), 1)
    texts = merge_datas_toline(data_preprocessing(read_datas(path), fields, [2] * len(fields)), 1)
    targets = merge_datas_toline(data_preprocessing(read_datas(path), ['keywords'], [3]), 1)
    # Construction of training corpus folder
    root_train_path = os.path.join(os.path.abspath(save_folder), "train")
    if not os.path.exists(root_train_path):
        os.mkdir(root_train_path)
    else:
        for name in os.listdir(root_train_path):
            for file_name in os.listdir(os.path.join(root_train_path, name)):
                os.remove(os.path.join(root_train_path, name, file_name))
            os.rmdir(os.path.join(root_train_path, name))
    # Construction of test corpus folder
    root_test_path = os.path.join(os.path.abspath(save_folder), "test")
    if not os.path.exists(root_test_path):
        os.mkdir(root_test_path)
    else:
        for name in os.listdir(root_test_path):
            for file_name in os.listdir(os.path.join(root_test_path, name)):
                os.remove(os.path.join(root_test_path, name, file_name))
            os.rmdir(os.path.join(root_test_path, name))
    #The dataset is divided into ten fold cross
    kf = KFold(n_splits=10, shuffle=True, random_state=1)
    for index, (train_index, test_index) in enumerate(kf.split(range(len(texts)))):
        train_path = os.path.join(root_train_path, str(index))
        if not os.path.exists(train_path): os.mkdir(train_path)
        test_path = os.path.join(root_test_path, str(index))
        if not os.path.exists(test_path): os.mkdir(test_path)
        # Save training corpus
        for i in train_index:
            with open(os.path.join(train_path, ids[i]+".txt"), "w", encoding="utf-8") as fp:
                fp.write(texts[i])
            with open(os.path.join(train_path, ids[i] + ".key"), "w", encoding="utf-8") as fp:
                fp.write(targets[i])
        # Save test corpus
        for i in test_index:
            with open(os.path.join(test_path, ids[i] + ".txt"), "w", encoding="utf-8") as fp:
                fp.write(texts[i])
    print("Dataset construction completed.")

# Training model
def train(train_datas_path, kea_path):
    root_path, flag = os.path.join(os.path.abspath(train_datas_path), "train"), True
    paths = os.listdir(root_path)
    with tqdm(range(len(paths))) as pbar:
        for index in pbar:
            name = paths[index]
            # Call Kea training model
            if flag:
                # Switch to Kea path, and only switch once
                os.chdir(os.path.abspath(kea_path))
                flag = False
            # Run the program
            state = os.system("java KEAModelBuilder -l {0} -m configs{1}".format(os.path.join(root_path, name),name))
            if state == 0:
                pbar.set_description("The training set {0} is trained and the generated model name is model {1}".format(name, name))
            else:
                pbar.set_description("Training set %s failed"%name)

# Prediction model
def predict(test_datas_path, kea_path, topN=3):
    root_path, flag = os.path.join(os.path.abspath(test_datas_path), "test"), True
    paths = os.listdir(root_path)
    with tqdm(range(len(paths))) as pbar:
        for index in pbar:
            name = paths[index]
            # Call Kea training model
            if flag:
                # Switch to Kea path, and only switch once
                os.chdir(os.path.abspath(kea_path))
                flag = False
            test_path = os.path.join(root_path, name)
            # Delete files with suffix. Key
            for i in os.listdir(test_path):
                if i.endswith(".key"): os.remove(os.path.join(test_path, i))
            # Run the program
            state = os.system("java KEAKeyphraseExtractor -l {0} -m configs{1} -n {2}".format(test_path, name, topN))
            if state == 0:
                pbar.set_description("Test set %s prediction complete"%name)
            else:
                pbar.set_description("Test set %s prediction failed"%name)

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
def evalute_results(path, targets, title_abstract=None, topk=3):
    results = []
    if title_abstract == None:
        for name in os.listdir(path):
            rev_pred, rev_targets, _ = read_results(os.path.join(os.path.abspath(path), name), targets)
            # Evaluate extraction results
            evaluate_results = evaluate(rev_pred, rev_targets, title_abstract=title_abstract, topk=topk)
            results.append(list(evaluate_results))
    else:
        temps = {}
        for line in title_abstract:
            line = line.split("\n")
            temps[line[0]] = "\n".join(line[1:])
        for name in os.listdir(path):
            rev_pred, rev_targets, ids = read_results(os.path.join(os.path.abspath(path), name), targets)
            tit_abs = [temps.get(id) for id in ids]
            evaluate_results = evaluate(rev_pred, rev_targets, title_abstract=tit_abs, topk=topk)
            results.append(list(evaluate_results))
    results = np.array(results)
    avg_results = np.average(results[:, 0: 3], axis=0)
    total_words = np.sum(results[:, 3], axis=0)
    return avg_results.tolist() + [total_words.tolist()]

if __name__ == '__main__':
    # Initialization parameters
    args = init_args()
    print("dataset_name:", args.dataset_name)
    print("fields:", args.fields)
    file_path = os.path.join(DATASET_PATH, args.dataset_name + ".json")
    save_path = os.path.join("./datas", args.dataset_name, 'naivebayes')

    # Select different modes to run the program
    if args.mode == 'data':
        build_data_sets(file_path, args.fields, save_path)
    elif args.mode == 'train':
        train(save_path, args.kea_path)
    elif args.mode == 'test':
        predict(save_path, args.kea_path, args.TOPN)
    elif args.mode == 'evaluate1':
        origin_datas = read_datas(file_path)
        targets = merge_datas_toline(data_preprocessing(get_targets(origin_datas, args.fields, choose=3), ['id', 'keywords'], [1, 3]), 1)
        if type(args.eTOPNS) != list:
            results = evalute_results(os.path.join(save_path, 'test'), targets, topk=args.eTOPNS)
            df = pd.DataFrame(np.array([args.eTOPNS] + list(results)).reshape(1, 5), columns=['TopN', 'P', 'R', 'F1', 'Keywords_num'])
            if args.evaluate_path != None: df.to_csv(args.evaluate_path)
            print(df)
        else:
            rs = []
            for topk in args.eTOPNS:
                evaluate_results = evalute_results(os.path.join(save_path, 'test'), targets, topk=topk)
                rs.append([topk] + list(evaluate_results))
            df = pd.DataFrame(np.array(rs), columns=['TopN', 'P', 'R', 'F1', 'Keywords_num'])
            if args.evaluate_path != None: df.to_csv(args.evaluate_path)
            print(df)
    elif args.mode == 'evaluate2':
        origin_datas = read_datas(file_path)
        title_abstract = merge_datas_toline(data_preprocessing(origin_datas, ['id', 'title', 'abstract'], [1, 2, 2]), 1)
        targets = merge_datas_toline(data_preprocessing(get_targets(origin_datas, args.fields, choose=3), ['id', 'keywords'], [1, 3]), 1)
        if type(args.eTOPNS) != list:
            results = evalute_results(os.path.join(save_path, 'test'), targets, title_abstract=title_abstract, topk=args.eTOPNS)
            df = pd.DataFrame(np.array([args.eTOPNS] + list(results)).reshape(1, 5),
                              columns=['TopN', 'P', 'R', 'F1', 'Keywords_num'])
            if args.evaluate_path != None: df.to_csv(args.evaluate_path)
            print(df)
        else:
            rs = []
            for topk in args.eTOPNS:
                evaluate_results = evalute_results(os.path.join(save_path, 'test'), targets, title_abstract=title_abstract, topk=topk)
                rs.append([topk] + list(evaluate_results))
            df = pd.DataFrame(np.array(rs), columns=['TopN', 'P', 'R', 'F1', 'Keywords_num'])
            if args.evaluate_path != None: df.to_csv(args.evaluate_path)
            print(df)
    else:
        raise RuntimeError("The value of the variable '-m' is in ['data', 'train', 'test', 'evaluate1', 'evaluate2'].")