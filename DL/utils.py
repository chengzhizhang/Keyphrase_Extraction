# -*- coding: utf-8 -*-
import json
import pandas as pd
from tqdm import tqdm

names = ['text', 'label', 'POS', 'LEN', 'WFF', 'WFOF', 'TI', 'TR', 'WFR', 'IWOR', 'IWOT']
def parse_tags(texts, preds):
    keywords = []
    for text, pred in zip(texts, preds):
        text = text.split(' ')
        K_B_index = [i for i, tag in enumerate(pred) if tag == 'K_B']
        for index in K_B_index:
            if index == len(pred) - 1:
                keywords.append(text[index])
            else:
                temp = [text[index]]
                for i in range(index + 1, len(pred)):
                    if pred[i] == 'K_I':
                        if i + 1 == len(pred):
                            temp.append(text[i])
                        elif pred[i+1] != '0' and pred[i-1] != '0':
                            temp.append(text[i])
                    elif pred[i] == 'K_B' or pred[i] == '0':
                        break
                keywords.append(" ".join(temp))
    return list(set(keywords))

# Calculate the P, R and F1 values of the extraction datas
def evaluate(y_preds, y_targets):
    STP, STE, STA, NUM =  0.0, 0.0, 0.0, 0.0
    for index, y_pred in enumerate(y_preds):
        y_true = y_targets[index]
        NUM += len(y_true)
        TP = len(set(y_pred) & set(y_true))
        STP += TP
        STE += len(y_pred)
        STA += len(y_true)
    # print(STP, STE, STA)
    p = (STP / STE) if STE != 0 else 0
    r = (STP / STA) if STA != 0 else 0
    f1 = ((2 *  r * p) / (r + p)) if (r + p) != 0 else 0
    p = round(p * 100, 2)
    r = round(r * 100, 2)
    f1 = round(f1 * 100, 2)
    return p, r, f1, NUM

def performance_metrics(pred_tags, info_path):
    texts = pd.read_csv(info_path.replace("_info.json", ".txt"),
                        sep=' <sep> ', engine='python', names=names).text
    info = read_json_datas(info_path)
    target_keywords, pred_keywords = [], []
    for i in range(len(info)):
        key, item = tuple(tuple(info[i].items())[0])
        target_keyword = item[0]
        pred = pred_tags[item[1]: item[2]]
        text = texts[item[1]: item[2]].tolist()
        pred_keyword = parse_tags(text, pred)
        target_keywords.append(target_keyword)
        pred_keywords.append(pred_keyword)
        # print(pred_keyword, target_keyword)
    metrics = evaluate(pred_keywords, target_keywords)
    return metrics

# Read file
def read_json_datas(path):
    datas = []
    with open(path, "r", encoding="utf-8") as fp:
        for i in fp.readlines():
            datas.append(json.loads(i))
    return datas

# save data
def save_datas(obj, path):
    datas = []
    for line in obj:
        data = {}
        for k, v in line:
            data[k] = v
        datas.append(data)
    with open(path, "w", encoding="utf-8") as fp:
        with tqdm(range(len(datas))) as pbar:
            for i in pbar:
                pbar.set_description("The %s document is saved" % (i + 1))
                fp.write(json.dumps(datas[i])+"\n")

# Read words from file
def read_text(path):
    words = []
    with open(path, 'r', encoding='utf-8') as fp:
        for word in fp.readlines():
            word = word.strip()
            words.append(word)
    return words

# string escape
def string_escape(string):
    string = string.replace("\\", "\\\\")
    string = string.replace("(", "\(")
    string = string.replace(")", "\)")
    string = string.replace("{", "\{")
    string = string.replace("}", "\}")
    string = string.replace("[", "\[")
    string = string.replace("]", "\]")
    string = string.replace("-", "\-")
    string = string.replace(".", "\.")
    string = string.replace("+", "\+")
    string = string.replace("=", "\=")
    string = string.replace("*", "\*")
    string = string.replace("?", "\?")
    string = string.replace("^", "\^")
    string = string.replace("$", "\$")
    return string