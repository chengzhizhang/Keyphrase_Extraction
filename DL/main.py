# -*- coding: utf-8 -*-
import argparse
import os
import time
from PATH import MODEL_FOLDER, DATAS_FOLDER
from model import train, test
from preprocess import build_data_sets

# Get the absolute path under the current working directory
WORKING_PATH = os.path.dirname(os.getcwd())
NAMES = ['text', 'label', 'POS', 'LEN', 'WFF', 'WFOF', 'TI', 'TR', 'WFR', 'IWOR', 'IWOT']

# Get parameters from the command line
def init_args():
    parser = argparse.ArgumentParser(description='Key phrase extraction using BILSTM-CRF.')
    # Input/output options
    parser.add_argument('--mode', '-m', default='train', type=str, help='Mode of program operation.')
    parser.add_argument('--dataset_name', '-dn', default='SemEval-2010', type=str, help='Name of the dataset.')
    parser.add_argument('--fields', '-fd', default=['title', 'abstract'], type=str, nargs='+',
                        help='Fields from which key phrases need to be extracted.')
    parser.add_argument('--features', '-fs', default=['CEM', 'TI', 'TR', 'WFOF', 'LEN'], type=str, nargs='+',
                        help='Fields from which key phrases need to be extracted.')
    opt = parser.parse_args()
    return opt

if __name__ == '__main__':
    # Initialization parameters
    args, features = init_args(), ['CEM']
    print('=' * 20, "Initialization information", '=' * 20)
    print("dataset_name:", args.dataset_name)
    print("fields:", args.fields)

    # File path
    dataset_path = os.path.join(WORKING_PATH, 'Dataset', args.dataset_name)
    file_folder = os.path.join(DATAS_FOLDER, args.dataset_name)
    vocab_folder = os.path.join(MODEL_FOLDER, args.dataset_name)
    # Fields contained in the dataset
    corpus_type = "".join([i[0] for i in args.fields]).upper()
    if args.features[0] != 'None': features += args.features
    print("features:", features)
    if args.mode == 'train':
        # Building datasets
        build_data_sets(dataset_path, args.fields, file_folder, vocab_folder)
        # start time
        s_time = time.time()
        train(file_folder, args.dataset_name, corpus_type, features)
        print("\033[031mINFO: Total time of model training:%sS\033[0m"%round(time.time() - s_time, 3))
    elif args.mode == 'test' or args.mode == 'dev':
        s_time = time.time()
        p, r, f1 = test(file_folder, args.dataset_name, corpus_type, args.mode, features)
        print("\033[031mINFO: Total time of model test:%sS\033[0m"%round(time.time() - s_time, 3))

