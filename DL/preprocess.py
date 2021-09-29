# -*- coding: utf-8 -*-
import json
import os
import re
import nltk
from collections import Counter
import numpy as np
from nltk import sent_tokenize, word_tokenize, PorterStemmer
from tqdm import tqdm

from textrank import textrank
from tf_idf import tf_idf
from utils import read_json_datas, string_escape

stop_words = nltk.corpus.stopwords.words('english')
english_punctuations = [',', '.', ':', ';', '``','?', '（','）','(', ')',
                '[', ']', '&', '!', '*', '@', '#', '$', '%','\\','\"','}','{']
stemmer = PorterStemmer()

# Mark Keywords
def mark_keyword(keywords, words):
    # Initial marker variable
    text = " ".join(words)
    position, text_len  = ['0'] * len(words), len(text)
    for keyword in keywords:
        keyword_length, kw_num = len(keyword), len(keyword.split(" "))
        index_list = [(i.start(), i.end()) for i in re.finditer(keyword.replace("+", "\+"), text)]
        for start, end in index_list:
            if text_len == keyword_length \
                   or (end == text_len and text[start - 1] == ' ') \
                   or (start == 0 and text[end] == ' ') \
                   or (text[start - 1] == ' ' and text[end] == ' '):
                p_index = len(text[:start].split(" ")) - 1
                if kw_num == 1:
                    position[p_index] = 'K_B'
                else:
                    position[p_index] = 'K_B'
                    for i in range(1, kw_num):
                        position[p_index + i] = 'K_I'
    return " ".join(words), position

# Construct datas
def build_datas(path, fields, mode, save_folder):
    # Datas annotation and features combination
    global js
    wfof_dict_list, wff_dict_list, \
    wfr_dict_list, iwor_dict_list, \
    iwot_dict_list = [], [], [], [], []
    cache_path = os.path.join(save_folder, "cache_%s"%mode)
    if os.path.exists(cache_path): js = json.load(
        open(cache_path, 'r', encoding='utf-8'))
    p_id, id_datas = 0, []
    with tqdm(enumerate(read_json_datas(path))) as pbar1:
        for index, text in pbar1:
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
            # ============================================================
            wfof_dict, wff_dict, wfr_dict, \
            iwor_dict, iwot_dict = {}, {}, {}, {}, {}
            if not os.path.exists(cache_path):
                # feature 1: word first occurrence in full_text (position)
                plain_text = text['full_text'].lower().strip()
                for word in word_tokenize(plain_text):
                    if word in english_punctuations + stop_words: continue
                    item = re.search(string_escape(word), plain_text)
                    if item != None:
                        wfof_dict[stemmer.stem(word)] = item.start()/len(plain_text)
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
                    word = stemmer.stem(word.strip())
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
                # 词性和词长
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
    # ======================================================
    # Datas transferred to TF-IDF and textrank
    texts_words = []
    for data in id_datas:
        tokens_list = data[1]
        tokens = [token for tokens in tokens_list for token in tokens]
        texts_words.append(tokens)
    # features 8-9: The TF-IDF and textrank features are calculated
    tf_idf_dict = tf_idf(texts_words)
    textrank_dict = textrank(texts_words)
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
                        TI[j] = tf_idf_dict[d_index][token]
                        TR[j] = textrank_dict[d_index][token]
                        WFR[j] = data[4]['wfr'][token]
                        IWOR[j] = data[4]['iwor'][token]
                        IWOT[j] = data[4]['iwot'][token]
                    except:
                        pass
                assert len(tokens) == len(POS)
                rev_data.append([" ".join(tokens), labels, POS, LEN, WFF, WFOF, TI, TR, WFR, IWOR, IWOT])
            return_datas[p_id] = rev_data
            id_keywords[p_id] = keywords
            pbar2.set_description("The %s-th document feature is completed" % (d_index + 1))
    return return_datas, id_keywords

# 5. save datas
def save_datas(datas, keywords, data_type, save_folder):
    POS_VOCAB, WORD_VOCAB, CHAR_VOCAB = [], [], []
    texts, curr_index, last_index, infos = [], 0, 0, {}
    for key, data in datas.items():
        count, sentences = 0, []
        for item in data:
            temp, features, sentence = [], [], item[0]
            if sentence == None: continue
            POS_VOCAB.extend(item[2])
            WORD_VOCAB.extend(word_tokenize(sentence))
            CHAR_VOCAB.extend([i for i in sentence])
            for index in range(1, len(item)):
                if type(item[index]) == list:
                    val = " ".join(item[index])
                    features.append(val)
                else:
                    val = " ".join(np.round(item[index], 8).astype(np.str).tolist())
                    features.append(val)
            sentences.append(sentence)
            temp.append(sentence)
            temp.extend(features)
            texts.append(temp)
            count += 1
        infos[key] = [keywords[key]]
        infos[key].append(last_index)
        curr_index += count
        last_index = curr_index
        infos[key].append(curr_index)
    with open(os.path.join(save_folder, data_type+".txt"), "w", encoding="utf-8") as fp:
        for text in texts: fp.write(" <sep> ".join(text) + '\n')
    with open(os.path.join(save_folder, data_type+"_info.json"), "w", encoding="utf-8") as fp:
        for key, val in infos.items():
            v = json.dumps({key: val})
            fp.write(v + '\n')
    print("\033[34mINFO: %s data saving completed!\033[0m" % data_type)
    return POS_VOCAB, WORD_VOCAB, CHAR_VOCAB

# Partition data-set
def build_data_sets(path, fields, save_folder, vocab_folder):
    # 1. Building configuration folder
    root_vocab_path = os.path.join(os.path.abspath(vocab_folder), "vocab")
    if not os.path.exists(root_vocab_path): os.mkdir(root_vocab_path)
    else:
        for file_name in os.listdir(root_vocab_path):
            os.remove(os.path.join(root_vocab_path, file_name))
    # 2. Construction datas
    train_path = os.path.join(path, 'train.json')
    test_path = os.path.join(path, 'test.json')
    train_datas, train_keywords = build_datas(train_path, fields, 'train', save_folder)
    test_datas, test_keywords = build_datas(test_path, fields, 'test', save_folder)
    POS_VOCAB1, WORD_VOCAB1, CHAR_VOCAB1 = save_datas(train_datas, train_keywords, "train", save_folder)
    POS_VOCAB2, WORD_VOCAB2, CHAR_VOCAB2 = save_datas(test_datas, test_keywords, "test", save_folder)
    WORD_VOCAB = WORD_VOCAB1 + WORD_VOCAB2
    CHAR_VOCAB = CHAR_VOCAB1 + CHAR_VOCAB2
    POS_VOCAB = POS_VOCAB1 + POS_VOCAB2
    # 3. Save word dictionary
    WORD_VOCAB = ["[PAD]", "[UNK]"] + list(set(WORD_VOCAB))
    with open(os.path.join(root_vocab_path, 'word_vocab.txt'), "w", encoding="utf-8") as fp:
        fp.write("\n".join(WORD_VOCAB))
    # Save character dictionary
    CHAR_VOCAB = ["[PAD]", "[UNK]"] + list(set(CHAR_VOCAB))
    with open(os.path.join(root_vocab_path, 'char_vocab.txt'), "w", encoding="utf-8") as fp:
        fp.write("\n".join(CHAR_VOCAB))
    # Save Dictionary of part of speech categories
    POS_VOCAB = ["[PAD]", "[UNK]"] + list(set(POS_VOCAB))
    with open(os.path.join(root_vocab_path, 'pos_vocab.txt'), "w", encoding="utf-8") as fp:
        fp.write("\n".join(POS_VOCAB))
    print("\033[34mINFO: Dataset partition completed\033[0m")

