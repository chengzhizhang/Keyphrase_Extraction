# -*- coding: utf-8 -*-
import argparse
import json
import os
import time

from train import run
from preprocess import build_data_sets

# Get the absolute path under the current working directory
WORKING_PATH = os.path.dirname(os.getcwd())


# Get parameters from the command line
def init_args():
    parser = argparse.ArgumentParser(description='Key phrase extraction using BILSTM-CRF.')
    # Input/output options
    parser.add_argument('--dataset_name', '-dn', default='SemEval-2010', type=str, help='Name of the dataset.')
    parser.add_argument('--fields', '-fd', default=['title', 'abstract'], type=str, nargs='+', help='Fields from which key phrases need to be extracted.')
    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    # Initialization parameters
    args = init_args()
    print("dataset_name:", args.dataset_name)
    print("fields:", args.fields)

    dataset_path = os.path.join(WORKING_PATH, 'dataset', args.dataset_name+".json")
    save_path = os.path.join('./datas', args.dataset_name)

    # Construction datas
    configs, is_add_ref = None, False
    if len(args.fields) == 2:
        is_add_ref = False
        configs = json.load(open('./configs/%s/config_wr.json' % args.dataset_name, 'r', encoding='utf-8'))
    elif len(args.fields) == 3:
        is_add_ref = True
        configs = json.load(open('./configs/%s/config_wor.json' % args.dataset_name, 'r', encoding='utf-8'))
    build_data_sets(dataset_path, args.fields, save_path, embedding_dim=configs['word_embedding_dim'])

    # model training
    s_time = time.time()
    run(save_path, args.dataset_name, is_add_ref)
    print("Total time of model training:", time.time()-s_time)
