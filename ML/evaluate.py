
from utils import  read_datas

# Calculate the P, R and F1 values of the extraction datas
def evaluate_in_targets(path_or_datas, targets, topk = 3):
    keywords = None
    if type(path_or_datas) == str:
        keywords = read_datas(path_or_datas)
    elif type(path_or_datas) == list:
        keywords = path_or_datas
    STP, SFP, SFN, NUM =  0.0, 0.0, 0.0, 0.0
    for index, words in enumerate(keywords):
        y_pred, y_true = None, targets[index]
        if len(y_true) == 0: continue
        NUM += len(y_true)
        if type(path_or_datas) == str:
            words = sorted(words.items(), key=lambda item: (-item[1], item[0]))
            y_pred = [i[0] for i in words]
        elif type(path_or_datas) == list:
            y_pred = words
        if len(y_pred) == 0: continue
        y_pred = y_pred[0: topk]
        STP += len(set(y_pred) & set(y_true))
        SFP += len(y_pred) - len(set(y_pred) & set(y_true))
        SFN += len(y_true) - len(set(y_pred) & set(y_true))
    recall = STP / (STP + SFN)
    precision = STP / (STP + SFP)
    f1 = 2 *  recall * precision / (recall + precision)
    return precision, recall, f1, NUM

# Calculate the P, R and F1 values of the extraction datas in title and abstract
def evaluate_in_title_and_abstract(path_or_datas, targets, title_abstract, topk = 3):
    keywords = None
    if type(path_or_datas) == str:
        keywords = read_datas(path_or_datas)
    elif type(path_or_datas) == list:
        keywords = path_or_datas
    STP, SFP, SFN, NUM, tt =  0.0, 0.0, 0.0, 0.0, 0.0
    for index, words in enumerate(keywords):
        y_pred, y_true = None, targets[index]
        if len(y_true) == 0: continue
        if type(path_or_datas) == str:
            words = sorted(words.items(), key=lambda item: (-item[1], item[0]))
            y_pred = [i[0] for i in words]
        elif type(path_or_datas) == list:
            y_pred = words
        if len(y_pred) == 0: continue
        y_pred = y_pred[0: topk]
        y_pred_in_ab_tit = [i for i in y_pred if i in title_abstract[index]]
        STP += len(set(y_pred_in_ab_tit) & set(y_true))
        SFP += len(y_pred) - len(set(y_pred) & set(y_true))
        SFN += len(y_true) - len(set(y_pred) & set(y_true))
        NUM += len(y_true)
    recall = STP / (STP + SFN)
    precision = STP / (STP + SFP)
    f1 = 2 *  recall * precision / (recall + precision)
    return precision, recall, f1, NUM

# Calculate the P, R and F1 values of the extraction datas
def evaluate(path_or_datas, targets, title_abstract = None, topk = 3):
    if title_abstract == None:
        return evaluate_in_targets(path_or_datas, targets, topk)
    else:
        return evaluate_in_title_and_abstract(path_or_datas, targets, title_abstract, topk)