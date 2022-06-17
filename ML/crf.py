import argparse
import json
import math
import os
import re
import time
import nltk
import pandas as pd
import numpy as np
from tqdm import tqdm
from configs import DATASET_PATH
from evaluate import evaluate
from textrank import textrank
from tf_idf import tf_idf
from collections import Counter
from nltk import PorterStemmer, sent_tokenize, word_tokenize
from utils import read_datas,  read_json_datas, string_escape, to_string

stop_words = nltk.corpus.stopwords.words('english')
english_punctuations = [',', '.', ':', ';', '``','?', '（','）','(', ')',
                        '[', ']', '&', '!', '*', '@', '#', '$', '%','\\','\"','}','{']
stemmer = PorterStemmer()

# Get parameters from command line
def init_args():
    parser = argparse.ArgumentParser(description='Key phrase extraction using CRF.')
    # Input/output options
    parser.add_argument('--mode', '-m', default='train', type=str, help='Mode of program operation.')
    parser.add_argument('--dataset_name', '-dn', default='SemEval-2010', type=str, help='Name of the dataset.')
    parser.add_argument('--fields', '-fd', default=['title', 'abstract'], type=str, nargs='+',
                        help='Fields from which key phrases need to be extracted.')
    parser.add_argument('--features', '-fs', default=['POS', 'LEN', 'WFOF', 'WFF',
                        'WFR', 'IWOT','IWOR', 'TI', 'TR'], type=str, nargs='+',
                        help='Fields from which key phrases need to be extracted.')
    parser.add_argument('--crf_path', '-cp', default="./CRF++", type=str, help='The file path of CRF++.')
    parser.add_argument('--evaluate_path', '-sp', type=str, help='Save path of evaluation datas.')
    opt = parser.parse_args()
    return opt

# Feature discretization
def numerical_interval(val, max_val, min_val):
    v = round((val - min_val)/(max_val - min_val)*10, 0)
    return int(v)

# Tag keywords
def mark_keyword(keywords, tokens):
    text = " ".join(tokens)
    # Initial marker variable
    position, text_len  = ['N'] * len(tokens), len(text)
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
    return tokens, position

# Construct datas
def build_datas(path, mode, fields, save_folder):
    # Datas annotation and features combination
    global js
    wfof_dict_list, wff_dict_list, wfr_dict_list, iwor_dict_list, iwot_dict_list = [], [], [], [], []
    cache_path = os.path.join(os.path.abspath(save_folder), "cache_%s"%mode)
    if os.path.exists(cache_path): js = json.load(open(cache_path, 'r', encoding='utf-8'))
    p_id, id_datas = 0, []
    with tqdm(enumerate(read_json_datas(path))) as pbar1:
        for index, text in  pbar1:
            # Sentence segmentation
            sentences = []
            for field in fields:
                content = text[field]
                if type(content) == str:
                    sentences.extend(sent_tokenize(content))
                elif type(content) == list:
                    for item in content:
                        sentences.extend(sent_tokenize(item))
            # Processing keywords
            keywords = []
            for keyphase in text['keywords']:
                keyphase = keyphase.lower().strip()
                tokens = [stemmer.stem(token.strip())
                          for token in word_tokenize(keyphase)
                          if token.strip() not in english_punctuations]
                keywords.append(" ".join(tokens))
            # ========================================================================
            wfof_dict, wff_dict, wfr_dict, iwor_dict, iwot_dict = {}, {}, {}, {}, {}
            if not os.path.exists(cache_path):
                # feature 1: word first occurrence in full_text (position)
                plain_text = text['full_text'].lower().strip()
                for word in word_tokenize(plain_text):
                    if word in english_punctuations + stop_words: continue
                    item = re.search(string_escape(word), plain_text)
                    if item != None:
                        wfof_dict[stemmer.stem(word)] = int(round(item.start() / len(plain_text), 1) * 10)
                wfof_dict_list.append(wfof_dict)
                # feature 2: Full text word frequency count
                wff_dict = Counter([stemmer.stem(token.strip())
                            for sentence in sent_tokenize(text['full_text'])
                            for token in word_tokenize(sentence.lower().strip())
                            if token.strip() not in english_punctuations + stop_words])
                wff_dict_list.append(wff_dict)
                # feature 3: reference titles word frequency count
                wfr_dict = Counter([stemmer.stem(token.strip())
                            for text in text['references']
                            for token in word_tokenize(text.lower().strip())
                            if token.strip() not in english_punctuations + stop_words])
                wfr_dict_list.append(wfr_dict)
                # feature 4: Does it appear in the title of the reference
                plain_text = " ".join(text['references']).lower().strip()
                for word in word_tokenize(plain_text):
                    if word in english_punctuations + stop_words: continue
                    word = stemmer.stem(word.strip().lower())
                    iwor_dict[word] = 1
                iwor_dict_list.append(iwor_dict)
                # feature 5: Does it appear in the title
                plain_text = text['title'].lower().strip()
                for word in word_tokenize(plain_text):
                    if word in english_punctuations + stop_words: continue
                    word = stemmer.stem(word.strip().lower())
                    iwot_dict[word] = 1
                iwot_dict_list.append(iwot_dict)
            else:
                wfof_dict = js["wfof_dict_list"][index]
                wfr_dict = js["wfr_dict_list"][index]
                wff_dict = js["wff_dict_list"][index]
                iwor_dict = js["iwor_dict_list"][index]
                iwot_dict = js["iwot_dict_list"][index]
            # fs.6-7 pos and length
            pos_list, length_list = [], []
            tokens_list, labels_list = [], []
            for sentence in sentences:
                sentence = sentence.lower().strip()
                words = word_tokenize(sentence)
                if len(words) <= 3: continue
                # 词性和单词长度
                pos, length = {}, {}
                for i, (w, p) in enumerate(nltk.pos_tag(words)):
                    # Stem extraction
                    w_stem = stemmer.stem(w)
                    # Computational features
                    pos[w_stem] = p  # 词性
                    length[w_stem] = len(w_stem)  # 长度
                pos_list.append(pos)
                length_list.append(length)
                # 标签标注
                tokens = [stemmer.stem(token.strip()) for token in words
                          if token.strip() not in english_punctuations + stop_words]
                _, labels = mark_keyword(keywords, tokens)
                labels_list.append(labels)
                tokens_list.append(tokens)
            features = {
                'wff': wff_dict,
                'wfof': wfof_dict,
                'wfr': wfr_dict,
                'iwot': iwot_dict,
                'iwor':  iwor_dict,
                'pos': pos_list,
                'length': length_list
            }
            id_datas.append([text['id']+str(p_id), tokens_list, labels_list, keywords, features])
            # Document ID suffix
            p_id += 1
            pbar1.set_description("The %s-th document processing is completed!"%p_id)
    if not os.path.exists(cache_path):
        json.dump({
            "wfof_dict_list": wfof_dict_list,
            "wff_dict_list": wff_dict_list,
            "wfr_dict_list": wfr_dict_list,
            "iwor_dict_list": iwor_dict_list,
            "iwot_dict_list": iwot_dict_list
        }, open(cache_path, 'w', encoding='utf-8'))
    # ============================================================
    # Datas transferred to TF-IDF and textrank
    texts_words = []
    for data in id_datas:
        tokens_list = data[1]
        tokens = [token for tokens in tokens_list for token in tokens]
        texts_words.append(tokens)
    # features 8-9: The TF-IDF and textrank features are calculated
    tf_idf_dict = tf_idf(texts_words, is_merge_phrases=False, verbose=False)
    textrank_dict = textrank(texts_words, is_merge_phrases=False, verbose=False)
    # Merge informations
    return_datas, id_keywords = {}, {}
    with tqdm(enumerate(id_datas)) as pbar2:
        for d_index, data in pbar2:
            p_id, rev_data = data[0], []
            keywords = data[3]
            for i, (tokens, labels) in enumerate(zip(data[1], data[2])):
                if len(tokens) <= 3: continue
                POS = []
                WFF = np.zeros(shape=(len(tokens)))
                WFR = np.zeros(shape=(len(tokens)))
                WFOF = np.zeros(shape=(len(tokens)))
                IWOT = np.zeros(shape=(len(tokens)))
                IWOR = np.zeros(shape=(len(tokens)))
                LEN = np.zeros(shape=(len(tokens)))
                TI = np.zeros(shape=(len(tokens)))
                TR = np.zeros(shape=(len(tokens)))
                for j in range(len(tokens)):
                    token = tokens[j]
                    try:
                        POS.append(data[4]['pos'][i][token])
                        LEN[j] = data[4]['length'][i][token]
                        WFF[j] = data[4]['wff'][token]
                        WFOF[j] = data[4]['wfof'][token]
                        TI[j] = numerical_interval(
                            tf_idf_dict[d_index][token],
                            max(tf_idf_dict[d_index].values()),
                            min(tf_idf_dict[d_index].values()))
                        TR[j] = numerical_interval(
                            textrank_dict[d_index][token],
                            max(textrank_dict[d_index].values()),
                            min(textrank_dict[d_index].values()))
                        WFR[j] = data[4]['wfr'][token]
                        IWOR[j] = data[4]['iwor'][token]
                        IWOT[j] = data[4]['iwot'][token]
                    except:
                        pass
                assert len(tokens) == len(POS)
                rev_data.append([" ".join(tokens), labels, POS, LEN, WFF, WFOF, TI, TR, WFR, IWOR, IWOT])
            return_datas[p_id] = rev_data
            id_keywords[p_id] = keywords
            pbar2.set_description("The %s-th document feature is completed"%(d_index+1))
    return return_datas, id_keywords

# Preprocess datas
def process_datas(path, mode, fields, save_folder, features=None):
    # 获取数据
    id_datas, id_keywords = build_datas(path, mode, fields, save_folder)
    # Data saving path
    corpus_type = "".join([i[0] for i in fields]).upper()
    save_folder = os.path.join(os.path.abspath(save_folder), corpus_type)
    if not os.path.exists(save_folder): os.mkdir(save_folder)
    # Save training set
    if mode == 'train':
        save_str = []
        for key, data in id_datas.items():
            for item in data:
                fs = []
                if item[0] == None: continue
                fs.append(item[0].split(' '))
                if "POS" in features: fs.append(item[2])
                if "LEN" in features: fs.append(item[3])
                if "WFF" in features: fs.append(item[4])
                if "WFOF" in features: fs.append(item[5])
                if "TI" in features: fs.append(item[6])
                if "TR" in features: fs.append(item[7])
                if "WFR" in features: fs.append(item[8])
                if "IWOR" in features: fs.append(item[9])
                if "IWOT" in features: fs.append(item[10])
                fs.append(item[1])
                df = pd.DataFrame(fs).T.round(0)
                for index, jj in df.iterrows():
                    if len(jj.tolist()[0]) < 2: continue
                    save_str.append(" ".join([to_string(i) for i in jj.tolist()]))
                save_str.append("")
            ender = ['%%%']
            if "POS" in features:
                ender.append('%%%')
                ender.extend(['0'] * (len(features)-1))
            else:
                ender.extend(['0']*len(features))
            ender.append('N')
            save_str.append(" ".join(ender))
            save_str.append("")
        with open(os.path.join(save_folder, mode), "w", encoding="utf-8") as fp:
            fp.write("\n".join(save_str).strip())
    # Save test set
    else:
        save_str = []; test_kewords = []
        for key, data in id_datas.items():
            keywords = id_keywords[key]
            for item in data:
                fs = []
                fs.append(item[0].split(' '))
                if "POS" in features: fs.append(item[2])
                if "LEN" in features: fs.append(item[3])
                if "WFF" in features: fs.append(item[4])
                if "WFOF" in features: fs.append(item[5])
                if "TI" in features: fs.append(item[6])
                if "TR" in features: fs.append(item[7])
                if "WFR" in features: fs.append(item[8])
                if "IWOR" in features: fs.append(item[9])
                if "IWOT" in features: fs.append(item[10])
                df = pd.DataFrame(fs).T.round(0)
                for index, jj in df.iterrows():
                    if len(jj.tolist()[0]) < 2: continue
                    save_str.append(" ".join([to_string(i) for i in jj.tolist()]))
                save_str.append("")
            ender = ['%%%']
            if "POS" in features:
                ender.append('%%%')
                ender.extend(['0'] * (len(features) - 1))
            else:
                ender.extend(['0'] * len(features))
            save_str.append(" ".join(ender))
            save_str.append("")
            test_kewords.append(json.dumps({key: keywords}))
        with open(os.path.join(save_folder, mode), "w", encoding="utf-8") as fp:
            fp.write("\n".join(save_str).strip())
        with open(os.path.join(save_folder, mode + ".json"), "w", encoding="utf-8") as fp:
            fp.write("\n".join(test_kewords))
    print("\033[34mINFO: Dataset construction completed!\033[0m")

# Training model
def train(root_path, mode, fields, crf_path, features, n = 4):
    corpus_type = "".join([i[0] for i in fields]).upper()
    file_path = os.path.join(os.path.abspath(root_path), corpus_type, mode)
    if ('full_text' in fields) or (('body1' in fields) and ('body2' in fields)):
        cache_path = os.path.join(os.path.abspath(root_path), corpus_type, 'cache')
        if not os.path.exists(cache_path): os.mkdir(cache_path)
        datas, file_count = [], []
        with open(file_path, 'r', encoding='utf-8') as fp:
            for i, line in enumerate(fp.readlines()):
                datas.append(line)
                if line.startswith("%%%"): file_count.append(i)
        size = math.ceil(len(file_count) / n)
        start_index, end = 0, 0
        for epoch in range(n):
            end += size
            if end >= len(file_count): end = len(file_count) - 1
            end_index = file_count[end] + 2
            sub_file_path = os.path.join(os.path.abspath(cache_path), "train"+str(epoch))
            with open(sub_file_path, 'w', encoding='utf-8') as fp:
                for i in datas[start_index: end_index]: fp.write(i)
            start_index = end_index
            # Save training information
            info_path = os.path.join(os.path.abspath(cache_path), "info"+str(epoch))
            if os.path.exists(info_path): os.remove(info_path)
            # Save training model
            model_path = os.path.join(os.path.abspath(cache_path), "model"+str(epoch))
            if os.path.exists(model_path): os.remove(model_path)
            # Switch to crf++ path and only once
            os.chdir(os.path.abspath(crf_path))
            # Call crf++ training model
            template_path = "./templates/template%s" % len(features)
            cmd = "crf_learn.exe -f 10 %s %s %s >>%s" % (template_path, sub_file_path, model_path, info_path)
            state = os.system(cmd)
            if state == 0:
                print("\033[34mINFO: Model%s training completed!\033[0m"%epoch)
            else:
                print("\033[31mINFO: Model%s training failed!\033[0m"%epoch)
            os.chdir(os.path.abspath('../'))
    else:
        # Save training information
        info_path = os.path.join(os.path.abspath(root_path), corpus_type, "info")
        if os.path.exists(info_path): os.remove(info_path)
        # Save training model
        model_path = os.path.join(os.path.abspath(root_path), corpus_type, "model")
        if os.path.exists(model_path): os.remove(model_path)
        # Switch to crf++ path and only once
        os.chdir(os.path.abspath(crf_path))
        # Call crf++ training model
        template_path = "./templates/template%s" % len(features)
        # cmd = "crf_learn.exe %s %s %s >>%s"%(template_path, file_path, model_path, info_path)
        # if ('full_text' in fields) or (('body1' in fields) and ('body2' in fields)):
        cmd = "crf_learn.exe -f 10 %s %s %s >>%s"%(template_path, file_path, model_path, info_path)
        # cmd = "crf_learn.exe %s %s %s"%(template_path, file_path, model_path)
        state = os.system(cmd)
        if state == 0:
            print("\033[34mINFO: Model training completed!\033[0m")
        else:
            print("\033[31mINFO: Model training failed!\033[0m")
        os.chdir(os.path.abspath('../'))

# Call CRF + + to predict results
def predict(path, mode, fields, crf_path, n = 4):
    corpus_type = "".join([i[0] for i in fields]).upper()
    file_path = os.path.join(os.path.abspath(path), corpus_type, mode)
    if ('full_text' in fields) or (('body1' in fields) and ('body2' in fields)):
        cache_path = os.path.join(os.path.abspath(path), corpus_type, 'cache')
        for epoch in range(n):
            model_path = os.path.join(os.path.abspath(cache_path), 'model'+str(epoch))
            result_path = os.path.join(os.path.abspath(cache_path), "preds%s.txt"%epoch)
            if os.path.exists(result_path): os.remove(result_path)
            # Switch to CRF + + path
            os.chdir(os.path.abspath(crf_path))
            # Run the program
            cmd = "crf_test.exe -m %s %s >>%s" % (model_path, file_path, result_path)
            # cmd = "crf_test.exe -m %s %s"%(model_path, file_path)
            state = os.system(cmd)
            if state == 0:
                print("\033[34mINFO: Test set result prediction completed!\033[0m")
            else:
                print("\033[34mINFO: Test set result prediction failed!\033[0m")
            os.chdir(os.path.abspath('../'))
    else:
        model_path = os.path.join(os.path.abspath(path), corpus_type, 'model')
        result_path = os.path.join(os.path.abspath(path), corpus_type, "preds.txt")
        if os.path.exists(result_path): os.remove(result_path)
        # Switch to CRF + + path
        os.chdir(os.path.abspath(crf_path))
        # Run the program
        cmd = "crf_test.exe -m %s %s >>%s"%(model_path, file_path, result_path)
        # cmd = "crf_test.exe -m %s %s"%(model_path, file_path)
        state = os.system(cmd)
        if state == 0:
            print("\033[34mINFO: Test set result prediction completed!\033[0m")
        else:
            print("\033[34mINFO: Test set result prediction failed!\033[0m")
        os.chdir(os.path.abspath('../'))

# Read prediction results
def read_results(target_path, preds_path):
    targets, preds = [], []
    # target
    for line in read_datas(target_path):
        id, value = list(line.keys())[0], list(line.values())[0]
        targets.append(value)
    # Read file results
    text_tags = []
    with open(preds_path, 'r', encoding='utf-8') as fp:
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
        preds.append(keyword)
    return preds, targets

# Assessment results
def evaluate_results(path, fields, topk=1000, n=4):
    corpus_type = "".join([i[0] for i in fields]).upper()
    root_path = os.path.join(os.path.abspath(path), corpus_type)
    target_path = os.path.join(root_path, 'test.json')
    if ('full_text' in fields) or (('body1' in fields) and ('body2' in fields)):
        cache_path = os.path.join(os.path.abspath(path), corpus_type, 'cache')
        rs = []
        for epoch in range(n):
            # Path of test datas
            preds_path = os.path.join(cache_path, "preds%s.txt"%epoch)
            rev_pred, rev_targets = read_results(target_path, preds_path)
            # Evaluation extraction results
            evaluate_results = evaluate(rev_pred, rev_targets, topk=topk)
            rs.append(evaluate_results)
        rs = np.mean(np.array(rs), axis=0)
        return rs
    else:
        preds_path = os.path.join(root_path, "preds.txt")
        rev_pred, rev_targets = read_results(target_path, preds_path)
        # Evaluation extraction results
        evaluate_results = evaluate(rev_pred, rev_targets, topk=topk)
        return evaluate_results

if __name__ == "__main__":
    # Initialization parameters
    args = init_args()
    print('=' * 20, "Initialization information", '=' * 20)
    print("dataset_name:", args.dataset_name)
    print("fields:", args.fields)
    print("features", args.features)
    file_path = os.path.join(DATASET_PATH, "%s/%s.json"%(args.dataset_name, args.mode))
    corpus_type = "".join([i[0] for i in args.fields]).upper()
    save_path = os.path.join("./datas", args.dataset_name, 'crf')

    # Select different modes to run the program
    if args.mode == 'train':
        process_datas(file_path, args.mode, args.fields, save_path, features=args.features)
        start_time = time.time()
        train(save_path, args.mode, args.fields, args.crf_path, features=args.features)
        print("\033[32mTotal time(%s): %sS\033[0m" % (args.mode, round(time.time() - start_time, 2)))
    elif args.mode == 'test':
        process_datas(file_path, args.mode, args.fields, save_path, features=args.features)
        start_time = time.time()
        predict(save_path, args.mode, args.fields, args.crf_path)
        print("\033[32mTotal time(%s): %sS\033[0m" % (args.mode, round(time.time() - start_time, 2)))
        results = evaluate_results(save_path, args.fields)
        df = pd.DataFrame(np.array(list(results)).reshape(1, 4), columns=['P', 'R', 'F1', 'Keywords_num'])
        df.to_csv(os.path.join("./results", args.dataset_name, 'crf_%s.csv' % corpus_type))
        print(df)
    else:
        raise RuntimeError("The value of the variable '-m' is in ['data', 'train', 'test', 'evaluate'].")
