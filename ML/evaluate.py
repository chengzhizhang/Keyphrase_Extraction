from utils import  read_datas

# Calculate the P, R and F1 values of the extraction datas
def evaluate(path_or_datas, targets, topk = 3):
    keywords = None
    if type(path_or_datas) == str:
        keywords = read_datas(path_or_datas)
    elif type(path_or_datas) == list:
        keywords = path_or_datas

    STP, STE, STA, NUM =  0.0, 0.0, 0.0, 0.0
    for index, words in enumerate(keywords):
        y_pred, y_true = None, targets[index]
        # if len(y_true) == 0: continue
        NUM += len(y_true)
        if type(path_or_datas) == str:
            words = sorted(words.items(), key=lambda item: (-item[1], item[0]))
            y_pred = [i[0] for i in words]
        elif type(path_or_datas) == list:
            y_pred = words
        # if len(y_pred) == 0: continue
        y_pred = y_pred[0: topk]
        TP = len(set(y_pred) & set(y_true))
        STP += TP
        STE += len(y_pred)
        STA += len(y_true)
    p = (STP / STE) if STE != 0 else 0
    r = (STP / STA) if STA != 0 else 0
    f1 = ((2 * r * p) / (r + p)) if (r + p) != 0 else 0
    p = round(p * 100, 2)
    r = round(r * 100, 2)
    f1 = round(f1 * 100, 2)
    # print(STP, STE, STA)
    return p, r, f1, NUM

