import json
from tqdm import tqdm
from preprocessing import tokenize_pos

# Read file
def read_datas(path):
    datas = []
    with open(path, "r", encoding="utf-8") as fp:
        for i in fp.readlines():
            datas.append(json.loads(i))
    return datas

#Save datas
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
                pbar.set_description("Document %s saved" % (i + 1))
                fp.write(json.dumps(datas[i])+"\n")

# Merge data from multiple fields into one row
def merge_datas_toline(datas, choose):
    def inner_func(data, choose):
        # Merge sentences from multiple fields and use newline spaces
        if choose == 1:
            texts = []
            for value in list(data.values()):
                if type(value) == str:
                    texts.append(value)
                else:
                    for line in value:
                        texts.append(line)
            return "\n".join(texts)
        elif choose == 2:
            words = []
            for value in list(data.values()):
                if type(value[0]) == tuple or type(value[0]) == str:
                    for word in value:
                        words.append(word)
                else:
                    for line in value:
                        for word in line:
                            words.append(word)
            return words
        elif choose == 3:
            for data in datas:
                words = []
                for value in list(data.values()):
                    if type(value[0]) == list:
                        for i in value: words += i
                    else:
                        words += value
                return_datas.append(words)
        else:
            raise RuntimeError("The value of the variable ’choose‘ is 1-2")

    return_datas = []
    for data in datas:
        return_datas.append(inner_func(data, choose))
    return return_datas

# Merge phrases
def merge_phrases(keywords, texts, topk=10):
    outputs, words_num = {}, len(keywords)
    NL = ['NN', 'NNS']
    #Combination candidate keyphrases
    for t in range(words_num):
        if keywords[t][0][1] in NL:
            # The number of words in the keyphrase is 1.
            phrase, score = keywords[t][0][0], keywords[t][1]
            outputs[phrase] = score
            for m in range(words_num):
                # The number of words in the keyphrase is 2.
                phrase = " ".join([keywords[m][0][0], keywords[t][0][0]])
                score = (keywords[m][1] + keywords[t][1]) / 2
                # Whether it appears in the title or not (compared to other situations, this is the best result)
                if phrase in texts: outputs[phrase] = score
                for o in range(words_num):
                    # The number of words in the keyphrase is 3.
                    phrase = " ".join([keywords[o][0][0], keywords[m][0][0], keywords[t][0][0]])
                    score = (keywords[o][1] + keywords[m][1] + keywords[t][1]) / 3
                    # Whether it appears in the title or not (compared to other situations, this is the best result)
                    if phrase in texts: outputs[phrase] = score
    # Sort
    outputs = sorted(outputs.items(), key = lambda item: (-item[1], item[0]))[0: topk]
    return outputs

# Get source text datas
def get_targets(datas, fields, choose=3):
    rev_datas = []
    for line in datas:
        id, keywords, title, abstract, references = None, None, None, None, None
        texts = []
        for key, val in line.items():
            if key == 'id': id = val
            elif key == 'keywords': keywords = [tokenize_pos(key, choose) for key in val]
            elif key == 'title': title = val
            elif key == 'abstract': abstract = val
            elif key == 'references': references = val
            else: raise RuntimeError("error!")
            # Get texts
            if key in fields:
                if type(val) == str:
                    texts.append(tokenize_pos(val, choose))
                elif type(val) == list:
                    texts.extend([tokenize_pos(i, choose) for i in val])
        texts = "\n".join(texts)
        keywords = [key for key in keywords if key in texts]
        rev_datas.append({
            'id': id,
            'keywords': keywords,
            'title': title,
            'abstract': abstract,
            'references': references
        })
    return datas
