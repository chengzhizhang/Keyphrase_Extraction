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
                if len(value) == 0: continue
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
    keywords = {item[0][0]: [item[0][1], item[1]] for item in keywords}
    outputs, words_num = {}, len(keywords)
    # The number of words in the keyphrase is 1.
    for k, v in keywords.items():
        if v[0] in ['NN', 'NNS']: outputs[k] = v[1]
    # The number of words in the keyphrase is 2.
    NL = ['NN', 'NNS', 'JJ', 'JJR', 'JJS']
    text_words = texts.split(" ")
    for i in range(len(text_words)-1):
        f_word = text_words[i]
        l_word = text_words[i+1]
        if f_word not in keywords.keys() \
            or l_word not in keywords.keys()\
            or keywords[f_word][0] not in NL\
            or keywords[l_word][0] not in NL: continue
        phrase = " ".join([f_word, l_word])
        value = (keywords[f_word][1] + keywords[l_word][1])/2
        outputs[phrase] = value
    # The number of words in the keyphrase is 3.
    text_words = texts.split(" ")
    for i in range(len(text_words)-2):
        f_word = text_words[i]
        m_word = text_words[i+1]
        l_word = text_words[i+2]
        if f_word not in keywords.keys() \
            or m_word not in keywords.keys() \
            or l_word not in keywords.keys() \
            or keywords[f_word][0] not in NL \
            or keywords[m_word][0] not in NL \
            or keywords[l_word][0] not in NL: continue
        phrase = " ".join([f_word, m_word,l_word])
        value = (keywords[f_word][1] + keywords[m_word][1] +keywords[l_word][1])/3
        outputs[phrase] = value
    # Sort
    outputs = sorted(outputs.items(), key = lambda item: (-item[1], item[0]))[0: topk]
    return outputs

# Get source text datas
def get_targets(datas, fields, choose=3):
    rev_datas = []
    for line in datas:
        texts = []
        for key in line.keys():
            if key == 'keywords':
                line['keywords'] = [tokenize_pos(key, choose) for key in line['keywords']]
            else:
                # Get texts
                if key in fields:
                    val = line[key]
                    if type(val) == str:
                        texts.append(tokenize_pos(val, choose))
                    elif type(val) == list:
                        texts.extend([tokenize_pos(i, choose) for i in val])
        # text = "\n".join(texts)
        # line['keywords'] = [key for key in line['keywords'] if key in text]
        rev_datas.append(line)
    return datas

# Read file
def read_json_datas(path):
    datas = []
    with open(path, "r", encoding="utf-8") as fp:
        for i in fp.readlines():
            datas.append(json.loads(i))
    return datas

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

def to_string(val):
    if type(val) == str:
        return val
    else:
        return str(int(val))