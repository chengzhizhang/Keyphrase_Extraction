# -*- coding: utf-8 -*-
import json
import numpy as np
from tqdm import tqdm

def parse_tags(texts, preds):
    keywords = []
    for text, pred in zip(texts, preds):
        K_B_index = [i for i, tag in enumerate(pred) if tag == 'K_B']
        for index in K_B_index:
            if index == len(pred) - 1:
                keywords.append(text[index])
            else:
                temp = [text[index]]
                for i in range(index + 1, len(pred)):
                    if pred[i] == 'K_I':
                        temp.append(text[i])
                    elif pred[i] == 'K_B' or pred[i] == '0':
                        break
                keywords.append(" ".join(temp))
    return list(set(keywords))

def cal_metrics(pred_keywords, target_keywords):
    # 计算P, R
    p_list, r_list = [], []
    for pred, target in zip(pred_keywords, target_keywords):
        num_pred_correct = len(set(pred) & set(target))
        num_pred = len(set(pred))
        num_target = len(set(target))
        p = (num_pred_correct / num_pred) if num_pred != 0 else 0
        r = (num_pred_correct / num_target) if num_target != 0 else 0
        p_list.append(p)
        r_list.append(r)
    P = np.average(p_list)
    R = np.average(r_list)
    # 计算F1
    F1 = (2 * P * R) / (P + R) if (P + R) != 0 else 0
    return round(P * 100, 3), round(R * 100, 3), round(F1 * 100, 3)

def performance_metrics(pred_tags, test_info, is_add_ref):
    if is_add_ref == False:
        target_keywords, pred_keywords = [], []
        for i in range(len(test_info)):
            key, item = test_info[i]
            target = item[1]
            texts = item[2]
            preds = pred_tags[item[0][0]: item[0][1]]
            keywords = parse_tags(texts, preds)
            target_keywords.append(target)
            pred_keywords.append(keywords)
        metrics = cal_metrics(pred_keywords, target_keywords)
        return metrics
    else:
        target_keywords, pred_keywords_in_tit_abs, pred_keywords = [], [], []
        for i in range(len(test_info)):
            key, item = test_info[i]
            # 目标关键词
            target = item[1]
            # 源文本
            texts = item[2]
            # 出现在标题和摘要中的预测
            preds_in_tit_abs = pred_tags[item[0][0]: item[0][1]]
            # 所有文档信息的预测
            preds_all = pred_tags[item[0][0]: item[0][0]+len(texts)]
            # 在标题和摘要中预测的关键词
            keywords_in_tit_abs = parse_tags(texts, preds_in_tit_abs)
            # 在所有文档信息中预测的关键词
            keywords_all = parse_tags(texts, preds_all)
            target_keywords.append(target)
            pred_keywords_in_tit_abs.append(keywords_in_tit_abs)
            pred_keywords.append(keywords_all)
        metrics_all = cal_metrics(pred_keywords, target_keywords)
        # 计算在标题和摘要中预测结果的性能
        p_list, r_list = [], []
        for pred, pred_in_tit_abs, target in zip(pred_keywords_in_tit_abs, pred_keywords, target_keywords):
            num_pred_correct = len(set(pred) & set(target))
            num_pred = len(set(pred_in_tit_abs))
            num_target = len(set(target))
            p = (num_pred_correct / num_pred) if num_pred != 0 else 0
            r = (num_pred_correct / num_target) if num_target != 0 else 0
            p_list.append(p)
            r_list.append(r)
        P = np.average(p_list)
        R = np.average(r_list)
        # 计算F1
        F1 = (2 * P * R) / (P + R) if (P + R) != 0 else 0
        metrics_in_tit_abs = (round(P * 100, 3), round(R * 100, 3), round(F1 * 100, 3))
        return metrics_all, metrics_in_tit_abs

# 读取文件
def read_json_datas(path):
    datas = []
    with open(path, "r", encoding="utf-8") as fp:
        for i in fp.readlines():
            datas.append(json.loads(i))
    return datas

# 保存数据
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
                pbar.set_description("第%s个文档保存完成" % (i + 1))
                fp.write(json.dumps(datas[i])+"\n")

# 从文件中读取单词
def read_text(path):
    words = []
    with open(path, 'r', encoding='utf-8') as fp:
        for word in fp.readlines():
            word = word.strip()
            words.append(word)
    return words