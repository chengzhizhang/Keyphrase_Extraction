import argparse
import json
import os
import re
from sklearn.model_selection import KFold
from tqdm import tqdm
import pandas as pd
import numpy as np

from configs import DATASET_PATH
from evaluate import evaluate
from preprocessing import data_preprocessing
from textrank import textrank
from tf_idf import tf_idf
from utils import merge_datas_toline, read_datas, get_targets


# Get parameters from command line
def init_args():
    parser = argparse.ArgumentParser(description='Key phrase extraction using CRF.')
    # Input/output options
    parser.add_argument('--mode', '-m', default='train', type=str, help='Mode of program operation.')
    parser.add_argument('--dataset_name', '-dn', default='SemEval-2010', type=str, help='Name of the dataset.')
    parser.add_argument('--fields', '-fd', default=['title', 'abstract'], type=str, nargs='+', help='Fields from which key phrases need to be extracted.')
    parser.add_argument('--features', '-fs', default=['pos', 'tr', 'tf', 'p'], type=str, nargs='+', help='Fields from which key phrases need to be extracted.')
    parser.add_argument('--crf_path', '-cp', default="./CRF++", type=str, help='The file path of CRF++.')
    parser.add_argument('--evaluate_path', '-sp', type=str, help='Save path of evaluation datas.')
    opt = parser.parse_args()
    return opt

# Tag keywords
def mark_keyword(keywords, words_and_pos):
    words, text, pos = [], None, []
    if type(words_and_pos[0]) == list:
        for line in words_and_pos:
            for k, v in line:
                words.append(k)
                pos.append(v)
            words.append("\n")
            pos.append("\n")
        text = " ".join(words)
    else:
        for k, v in words_and_pos:
            words.append(k)
            pos.append(v)
        text = " ".join(words)
    # Initial marker variable
    position, text_len  = ['N'] * len(words), len(text)
    for keyword in keywords:
        keyword_length, kw_num = len(keyword), len(keyword.split(" "))
        index_list = [(i.start(), i.end()) for i in re.finditer(keyword.replace("+", "\+"), text)]
        for start, end in index_list:
            if len(text) == keyword_length \
                   or (end == text_len and text[start - 1] == ' ') \
                   or (start == 0 and text[end] == ' ') \
                   or (text[start - 1] == ' ' and text[end] == ' '):
                p_index = len(text[0: start].split(" ")) - 1
                if kw_num == 1:
                    position[p_index] = 'S'
                elif kw_num == 2:
                    position[p_index] = 'B'
                    position[p_index + 1] = 'E'
                else:
                    position[p_index] = 'B'
                    for i in range(kw_num - 2):
                        position[p_index + i + 1] = 'M'
                    position[p_index + kw_num - 1] = 'E'
    return words, pos, position

# Construct datas
def build_datas(path, fields, features):
    # Calculate textrank and TF-IDF
    textrank_words, tf_idf_words = None, None
    if 'tr' in features:
        datas_words = merge_datas_toline(data_preprocessing(read_datas(path), fields, [6] * len(fields)), 2)
        textrank_words = textrank(datas_words, is_merge_phrases=False)
    if 'tf' in features:
        datas_words = merge_datas_toline(data_preprocessing(read_datas(path), fields, [6] * len(fields)), 2)
        tf_idf_words = tf_idf(datas_words, is_merge_phrases=False)
    # Tagging part of speech
    choose = []
    for field in fields:
        if field == 'title': choose.append(4)
        elif field == 'abstract': choose.append(8)
        elif field == 'references': choose.append(4)
        else: raise RuntimeError('The value of field is in [title abstract references]')
    pos_words = data_preprocessing(read_datas(path), fields, choose)
    # Get ID and keywords
    ids_keywords = data_preprocessing(get_targets(read_datas(path), fields, choose=3), ["id", "keywords"], [1, 3])
    # Merge information
    return_datas = []
    for index, line in enumerate(ids_keywords):
        id, keywords, pos_words_inline = line['id'], line['keywords'], pos_words[index]
        if len(keywords) == 0: continue
        data, end =  {}, None
        keywords = sorted(keywords, key=lambda item: len(item.strip().split(" ")), reverse=False)
        for i, (key, value) in enumerate(pos_words_inline.items()):
            words, pos, position = mark_keyword(keywords, value)
            temps, flag = [], 0
            if key == 'title': flag = 1
            for j, word in enumerate(words):
                if word != '\n' and word != '.':
                    temp, end = [word], ['%%%']
                    single_pos, single_position = pos[j], position[j]
                    if 'pos' in features:
                        temp.append(single_pos)
                        end.append('%%%')
                    if 'p' in features:
                        temp.append(str(flag))
                        end.append('0')
                    if 'tr' in features:
                        tr = textrank_words[index]
                        if word in tr.keys(): temp.append(str(int(tr[word]//0.5 + 1)))
                        else: temp.append('0')
                        end.append('0')
                    if 'tf' in features:
                        tf = tf_idf_words[index]
                        if word in tf.keys(): temp.append(str(int(tf[word] * 10 // 0.5 + 1)))
                        else: temp.append('0')
                        end.append('0')
                    temp.append(single_position)
                    temps.append(" ".join(temp))
                else:
                    temps.append("\n")
            data[key] = temps
        end.append("N\n\n")
        data['%%%'] = [" ".join(end)]
        return_datas.append([id, data, keywords])
    return return_datas

# Partition data set
def build_data_sets(path, fields, save_folder, features=None):
    # Data saving path
    save_folder = os.path.join(os.path.abspath(save_folder), "datas")
    if not os.path.exists(save_folder): os.mkdir(save_folder)
    # Building training corpus folder
    root_train_path = os.path.join(os.path.abspath(save_folder), "train")
    if not os.path.exists(root_train_path):
        os.mkdir(root_train_path)
    else:
        for file_name in os.listdir(root_train_path):
            os.remove(os.path.join(root_train_path, file_name))
    # Building test corpus folder
    root_test_path = os.path.join(os.path.abspath(save_folder), "test")
    if not os.path.exists(root_test_path):
        os.mkdir(root_test_path)
    else:
        for file_name in os.listdir(root_test_path):
            os.remove(os.path.join(root_test_path, file_name))
    # Initialize feature set
    if features == None: features = ['pos', 'tr', 'tf', 'p']
    datas = build_datas(path, fields, features=features)
    # The dataset is divided into ten fold cross
    kf = KFold(n_splits=10, shuffle=True, random_state=1)
    for index, (train_index, test_index) in enumerate(kf.split(range(len(datas)))):
        # Save training set
        train_texts = []
        for i in train_index:
            id, data = datas[i][0], datas[i][1]
            temp = []
            for key, value in data.items():
                temp.append("\n".join(value).replace('\n\n', '\n'))
            train_texts.append("\n\n".join(temp).replace('\n\n\n', '\n\n'))
        with open(os.path.join(root_train_path, str(index)), "w", encoding="utf-8") as fp:
            fp.write("\n".join(train_texts))
        # Save test set
        test_texts, test_kewords = [], []
        for i in test_index:
            id, data, keywords = datas[i][0], datas[i][1], datas[i][2]
            temp = []
            for key, value in data.items():
                temp.append("\n".join(value).replace('\n\n', '\n'))
            text = "\n\n".join(temp).replace('\n\n\n', '\n\n')
            test_texts.append("\n".join([i[0: -1] for i in text.split("\n")]))
            test_kewords.append(json.dumps({id: keywords}))
        with open(os.path.join(root_test_path, str(index)), "w", encoding="utf-8") as fp:
            fp.write("\n".join(test_texts))
        with open(os.path.join(root_test_path, str(index) + ".json"), "w", encoding="utf-8") as fp:
            fp.write("\n".join(test_kewords))
    print("Dataset partition completed.")

# Training model
def train(path, crf_path):
    root_path = os.path.join(os.path.abspath(path), "datas", "train")
    flag = True
    # Save training information
    info_folder = os.path.join(os.path.abspath(path), "info")
    if not os.path.exists(info_folder): os.mkdir(info_folder)
    # Save training model
    model_folder = os.path.join(os.path.abspath(path), "configs")
    if not os.path.exists(model_folder): os.mkdir(model_folder)
    paths = os.listdir(root_path)
    with tqdm(range(len(paths))) as pbar:
        for index in pbar:
            name = paths[index]
            # Call crf++ training model
            if flag:
                # Switch to crf++ path and only once
                os.chdir(os.path.abspath(crf_path))
                flag = False
            # Run the program
            data_path = os.path.join(root_path, name)
            info_path = os.path.join(info_folder, name)
            model_path = os.path.join(model_folder, name)
            if os.path.exists(info_path): os.remove(info_path)
            if os.path.exists(model_path): os.remove(model_path)
            cmd = "crf_learn.exe template {0} {1} >>{2}".format(data_path, model_path, info_path)
            state = os.system(cmd)
            if state == 0:
                pbar.set_description("The training set {0} is trained and the generated model name is model {1}".format(name, name))
            else:
                pbar.set_description("Training set %s failed"%name)

# prediction model
def predict(path, crf_path):
    root_path = os.path.abspath(path)
    test_root_path = os.path.join(root_path, "datas", "test")
    flag =  True
    paths = os.listdir(test_root_path)
    with tqdm(range(len(paths))) as pbar:
        for index in pbar:
            name = paths[index]
            # Call CRF + + to predict results
            if flag:
                # Switch to CRF + + path only once
                os.chdir(os.path.abspath(crf_path))
                flag = False
            if len(name) != 1: continue
            data_path = os.path.join(test_root_path, name)
            model_path = os.path.join(root_path, "configs", name)
            result_path = os.path.join(test_root_path, "result"+name+".txt")
            if os.path.exists(result_path): os.remove(result_path)
            # Run the program
            cmd = "crf_test.exe -m {0} {1} >>{2}".format(model_path, data_path, result_path)
            state = os.system(cmd)
            if state == 0:
                pbar.set_description("Test set %s prediction complete"%name)
            else:
                pbar.set_description("Test set %s prediction failed"%name)

# Read prediction results
def read_results(path, title_abstract=None):
    # Construct the dictionary corresponding to title and summary
    temps = {}
    if title_abstract != None:
        for line in title_abstract:
            line = line.split("\n")
            temps[line[0]] = "\n".join(line[1:])
    # Path of test datas
    root_path = os.path.join(os.path.abspath(path), "datas", "test")
    targets, preds, tit_abs = [], [], []
    for name in os.listdir(root_path):
        target, tit_ab = [], []
        if name.endswith(".json"):
            rs = read_datas(os.path.join(root_path, name))
            for line in rs:
                id, value = list(line.keys())[0], list(line.values())[0]
                target.append(value)
                if title_abstract != None: tit_ab.append(temps[id])
            targets.append(target)
            if title_abstract != None: tit_abs.append(tit_ab)
        elif name.endswith(".txt"):
            text_tags, keywords = [], []
            # Read file results
            with open(os.path.join(root_path, name), 'r', encoding='utf-8') as fp:
                temp = []
                for i in fp.readlines():
                    line = i.strip().split('\t')
                    head, tail = line[0], line[-1]
                    temp.append((head, tail))
                    if head == "%%%":
                        text_tags.append(temp)
                        temp = []
            # Combined keywords
            for text in text_tags:
                txts, tags, keyword = [], [], []
                for t1, t2 in text:
                    txts.append(t1)
                    tags.append(t2)
                B_index = [i for i, tag in enumerate(tags) if (tag == 'S' or tag == 'B')]
                for index in B_index:
                    if index == len(tags) - 1:
                        keyword.append(txts[index])
                    elif tags[index] == 'S':
                        keyword.append(txts[index])
                    else:
                        temp = [txts[index]]
                        for i in range(index + 1, len(tags)):
                            if tags[i] == 'M' or tags[i] == 'E':
                                temp.append(txts[i])
                            elif tags[i] == 'B' or tags[i] == 'N':
                                break
                        keyword.append(" ".join(temp))
                keyword = list(set(keyword))
                keywords.append(keyword)
            preds.append(keywords)
    return preds, targets, tit_abs

# Assessment results
def evaluate_results(path, topk=1000, title_abstract=None):
    results = []
    if title_abstract == None:
        rev_pred, rev_targets, _ = read_results(path, title_abstract)
        for pred, target in zip(rev_pred, rev_targets):
            # Evaluation extraction results
            evaluate_results = evaluate(pred, target, topk=topk)
            results.append(list(evaluate_results))
    else:
        rev_pred, rev_targets, tit_abs = read_results(path, title_abstract)
        for i in range(len(rev_pred)):
            # Evaluation extraction results
            evaluate_results = evaluate(rev_pred[i], rev_targets[i], title_abstract=tit_abs[i], topk=topk)
            results.append(list(evaluate_results))
    results = np.array(results)
    avg_results = np.average(results[:, 0: 3], axis=0)
    total_words = np.sum(results[:, 3], axis=0)
    return avg_results.tolist() + [total_words.tolist()]

if __name__ == "__main__":
    # Initialization parameters
    args = init_args()
    print("dataset_name:", args.dataset_name)
    print("fields:", args.fields)
    print("features", args.features)
    file_path = os.path.join(DATASET_PATH, args.dataset_name + ".json")
    save_path = os.path.join("./datas", args.dataset_name, 'crf')

    # Select different modes to run the program
    if args.mode == 'data':
        build_data_sets(file_path, args.fields, save_path, features=args.features)
    elif args.mode == 'train':
        train(save_path, args.crf_path)
    elif args.mode == 'test':
        predict(save_path, args.crf_path)
    elif args.mode == 'evaluate1':
        results = evaluate_results(save_path)
        df = pd.DataFrame(np.array(list(results)).reshape(1, 4), columns=['P', 'R', 'F1', 'Keywords_num'])
        if args.evaluate_path != None: df.to_csv(args.evaluate_path)
        print(df)
    elif args.mode == 'evaluate2':
        title_abstract = merge_datas_toline(data_preprocessing(read_datas(file_path), ['id', 'title', 'abstract'], [1, 2, 2]), 1)
        results = evaluate_results(save_path, title_abstract=title_abstract)
        df = pd.DataFrame(np.array(list(results)).reshape(1, 4), columns=['P', 'R', 'F1', 'Keywords_num'])
        if args.evaluate_path != None: df.to_csv(args.evaluate_path)
        print(df)
    else:
        raise RuntimeError("The value of the variable '-m' is in ['data', 'train', 'test', 'evaluate'].")