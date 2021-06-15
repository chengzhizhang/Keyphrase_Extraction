# -*- coding: utf-8 -*-
import json
import os
import re
import nltk
from collections import Counter

from gensim.models import Word2Vec
from nltk import sent_tokenize, word_tokenize, PorterStemmer
from sklearn.model_selection import KFold

from textrank import textrank
from tf_idf import tf_idf
from utils import read_json_datas

english_punctuations = [',', '.', ':', ';', '``','?', '（','）','(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%','\\','\"','}','{']
porter_stemmer = PorterStemmer()

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
def build_datas(path, fields):
    #======================================================================================
    # Datas transferred to TF-IDF and textrank
    texts_words = []
    for line in read_json_datas(path):
        words = []
        for key, value in line.items():
            if key in fields:
                if type(value) == str:
                    tokens = [porter_stemmer.stem(token.strip().lower()) for token in word_tokenize(value)
                     if token.strip().lower() not in english_punctuations]
                    words.extend(tokens)
                elif type(value) == list:
                    for sen in value:
                        tokens = [porter_stemmer.stem(token.strip().lower()) for token in word_tokenize(sen)
                                  if token.strip().lower() not in english_punctuations]
                        words.extend(tokens)
        texts_words.append(words)
    # The TF-IDF and textrank features are calculated
    tf_idf_dict = tf_idf(texts_words)
    textrank_dict = textrank(texts_words)
    #=======================================================================================

    # Datas annotation and features combination
    id_datas, id_keywords = {}, {}
    p_id, pos_index = 0, 0
    for ii, line in enumerate(read_json_datas(path)):
        texts, rank = [], 0,
        last_pos_index = pos_index
        # Sentence segmentation
        for key, value in line.items():
            if key in fields:
                if type(value) == str:
                    if "." in value:
                        for s in sent_tokenize(value):
                            texts.append(s)
                            if key in ['title', 'abstract']: rank += 1
                            pos_index += 1
                    else:
                        texts.append(value)
                        pos_index += 1
                        if key in ['title', 'abstract']: rank += 1
                elif type(value) == list:
                    for i in value:
                        if "." in value:
                            for s in sent_tokenize(i):
                                texts.append(s)
                                if key in ['title', 'abstract']: rank += 1
                                pos_index += 1
                        else:
                            texts.append(i)
                            pos_index += 1
                            if key in ['title', 'abstract']: rank += 1
        # Citation titles_word frequency count
        word_freq_in_reference = Counter([porter_stemmer.stem(token.strip().lower()) for text in texts[rank: ]
                     for token in word_tokenize(text) if token.strip().lower() not in english_punctuations])
        # Full text_word frequency count
        word_freq_in_fulltext = Counter([porter_stemmer.stem(token.strip().lower()) for text in texts
                     for token in word_tokenize(text) if token.strip().lower() not in english_punctuations])

        # Processing texts
        sentences_stem = []
        TEXTRANK, TF_IDF = [], []
        POS, LOC, LEN, FREQ_REF, FREQ_FT = [], [], [], [], []
        for text in texts:
            sentence_stem = []
            # Text features
            POS_fs, LOC_fs, LEN_fs, FREQ_in_ref_fs = [], [], [], []
            TEXTRANK_fs, TF_IDF_fs, FREQ_in_fulltext_fs = [], [], []
            # Punctuation / lowercase
            tokens = [token.strip().lower() for token in word_tokenize(text)
                      if token.strip().lower() not in english_punctuations]
            len_tokens = len(tokens)
            for i, (token, pos) in enumerate(nltk.pos_tag(tokens)):
                # Stem extraction
                token_stem = porter_stemmer.stem(token)
                # Computational features
                POS_fs.append(pos)
                LOC_fs.append(i/len_tokens)
                LEN_fs.append(len(token_stem))
                try:
                    TEXTRANK_fs.append(textrank_dict[ii][token_stem])
                    TF_IDF_fs.append(tf_idf_dict[ii][token_stem])
                    FREQ_in_ref_fs.append(word_freq_in_reference[token_stem])
                    FREQ_in_fulltext_fs.append(word_freq_in_fulltext[token_stem])
                except KeyError:
                    TEXTRANK_fs.append(0.0)
                    TF_IDF_fs.append(0.0)
                    FREQ_in_ref_fs.append(0.0)
                    FREQ_in_fulltext_fs.append(0.0)
                sentence_stem.append(token_stem)
            # The sentence length is greater than 0
            if len(sentence_stem) != 0:
                POS.append(POS_fs)
                LOC.append(LOC_fs)
                LEN.append(LEN_fs)
                TEXTRANK.append(TEXTRANK_fs)
                TF_IDF.append(TF_IDF_fs)
                FREQ_REF.append(FREQ_in_ref_fs)
                FREQ_FT.append(FREQ_in_fulltext_fs)
                sentences_stem.append(sentence_stem)
        id_datas[line['id']+str(p_id)] = [sentences_stem,
                                          [last_pos_index, last_pos_index+rank, len(sentences_stem)],
                                          POS, LOC, LEN, TF_IDF, TEXTRANK, FREQ_FT, FREQ_REF]

        # Processing keywords
        keywords_stem, keywords = [], []
        for keyphase in line['keywords']:
            keyword_stem, keyword = [], []
            tokens = [token.strip().lower() for token in word_tokenize(keyphase)
                      if token.strip().lower() not in english_punctuations]
            for token in tokens:
                keyword.append(token)
                keyword_stem.append(porter_stemmer.stem(token))
            keywords.append(" ".join(keyword))
            keywords_stem.append(" ".join(keyword_stem))
        id_keywords[line['id']+str(p_id)] = keywords_stem
        # Document ID suffix
        p_id += 1

    # Merge informations
    return_datas = {}
    for id, keywords in id_keywords.items():
        data = id_datas[id]
        keywords = sorted(keywords, key=lambda item: len(item.strip().split(" ")), reverse=False)
        rev_data = []
        for i, text in enumerate(data[0]):
            words, labels = mark_keyword(keywords, text)
            rev_data.append([data[0][i], labels,
                             data[2][i], data[3][i], data[4][i], data[5][i],
                             data[6][i], data[7][i], data[8][i]])
        return_datas[id] = [data[0], data[1], rev_data]
    return return_datas, id_keywords

# Partition data-set
def build_data_sets(path, fields, save_folder, embedding_dim):
    # 1. Building training corpus folder
    root_train_path = os.path.join(os.path.abspath(save_folder), "train")
    if not os.path.exists(root_train_path):
        os.mkdir(root_train_path)
    else:
        for file_name in os.listdir(root_train_path):
            os.remove(os.path.join(root_train_path, file_name))
    # 2. Building test corpus folder
    root_test_path = os.path.join(os.path.abspath(save_folder), "test")
    if not os.path.exists(root_test_path):
        os.mkdir(root_test_path)
    else:
        for file_name in os.listdir(root_test_path):
            os.remove(os.path.join(root_test_path, file_name))
    # 3. Building configuration folder
    root_config_path = os.path.join(os.path.abspath(save_folder), "vocab")
    if not os.path.exists(root_config_path):
        os.mkdir(root_config_path)
    else:
        for file_name in os.listdir(root_config_path):
            os.remove(os.path.join(root_config_path, file_name))

    # 4. Construction datas
    datas, id_keywords = build_datas(path, fields)

    # 5. Cross the data-set by 10 fold
    POS_VOCAB, WORD_VOCAB, CHAR_VOCAB = [], [], []
    kf = KFold(n_splits=10, shuffle=True, random_state=1)
    for index, (train_index, test_index) in enumerate(kf.split(range(len(datas)))):
        # Save training set
        rank, last_rank, start_point = 0, 0, 0
        train_texts, train_infos = [], []
        for i, (id, data) in enumerate(datas.items()):
            if i not in train_index.tolist(): continue
            for item in data[2]:
                WORD_VOCAB.extend(item[0])
                POS_VOCAB.extend(item[2])
                CHAR_VOCAB.extend([str(i) for word in item[0] for i in word])
                # text, label, POS, LOC, LEN, TF_IDF5, TEXTRANK, FREQ_FT, FREQ_REF
                text = " ".join([str(i) for i in item[0]])
                label = " ".join([str(i) for i in item[1]])
                pos = " ".join([str(i) for i in item[2]])
                loc = " ".join([str(i) for i in item[3]])
                length = " ".join([str(i) for i in item[4]])
                tfidf = " ".join([str(i) for i in item[5]])
                textrank = " ".join([str(i) for i in item[6]])
                freq_ft = " ".join([str(i) for i in item[7]])
                freq_ref = " ".join([str(i) for i in item[8]])
                train_texts.append("\t".join([str(rank), text, label, pos, loc,
                                              length, tfidf, textrank, freq_ft, freq_ref]))
                rank += 1
            end_point = start_point + (data[1][1] - data[1][0])
            train_infos.append(json.dumps((id, [[start_point, end_point], id_keywords[id], data[0]])))
            start_point += data[1][-1]
        with open(os.path.join(root_train_path, str(index)), "w", encoding="utf-8") as fp:
            fp.write("\n".join(train_texts))
        with open(os.path.join(root_train_path, str(index) + ".json"), "w", encoding="utf-8") as fp:
            fp.write("\n".join(train_infos))

        # Save test set
        rank, last_rank, start_point = 0, 0, 0
        test_texts, test_infos = [], []
        for i, (id, data) in enumerate(datas.items()):
            if i not in test_index.tolist(): continue
            for item in data[2]:
                WORD_VOCAB.extend(item[0])
                POS_VOCAB.extend(item[2])
                CHAR_VOCAB.extend([str(i) for word in item[0] for i in word])
                # text, label, POS, LOC, LEN, TF_IDF5, TEXTRANK, FREQ_FT, FREQ_REF
                text = " ".join([str(i) for i in item[0]])
                label = " ".join([str(i) for i in item[1]])
                pos = " ".join([str(i) for i in item[2]])
                loc = " ".join([str(i) for i in item[3]])
                length = " ".join([str(i) for i in item[4]])
                tfidf = " ".join([str(i) for i in item[5]])
                textrank = " ".join([str(i) for i in item[6]])
                freq_ft = " ".join([str(i) for i in item[7]])
                freq_ref = " ".join([str(i) for i in item[8]])
                test_texts.append("\t".join([str(rank), text, label, pos, loc,
                                              length, tfidf, textrank, freq_ft, freq_ref]))
                rank += 1
            end_point = start_point + (data[1][1] - data[1][0])
            test_infos.append(json.dumps((id, [[start_point, end_point], id_keywords[id], data[0]])))
            start_point += data[1][-1]
        with open(os.path.join(root_test_path, str(index)), "w", encoding="utf-8") as fp:
            fp.write("\n".join(test_texts))
        with open(os.path.join(root_test_path, str(index) + ".json"), "w", encoding="utf-8") as fp:
            fp.write("\n".join(test_infos))
    # Save word dictionary
    WORD_VOCAB = ["[PAD]", "[UNK]"] + list(set(WORD_VOCAB))
    with open(os.path.join(root_config_path, 'word_vocab.txt'), "w", encoding="utf-8") as fp:
        fp.write("\n".join(WORD_VOCAB))
    # Save character dictionary
    CHAR_VOCAB = ["[PAD]", "[UNK]"] + list(set(CHAR_VOCAB))
    with open(os.path.join(root_config_path, 'char_vocab.txt'), "w", encoding="utf-8") as fp:
        fp.write("\n".join(CHAR_VOCAB))
    # Save Dictionary of part of speech categories
    POS_VOCAB = ["[PAD]", "[UNK]"] + list(set(POS_VOCAB))
    with open(os.path.join(root_config_path, 'pos_vocab.txt'), "w", encoding="utf-8") as fp:
        fp.write("\n".join(POS_VOCAB))

    print("Data-set partition completed")


if __name__ == '__main__':
    name = 'SemEval-2010'
    file_path = '../dataset/SemEval-2010.json'
    fields = ['title', 'abstract']
    #fields = ['title', 'abstract', 'references']
    save_folder = './datas/%s'%name

    configs = json.load(open('./configs/%s/config_wr.json'%name, 'r', encoding='utf-8'))
    build_data_sets(file_path, fields, save_folder, embedding_dim=configs['word_embedding_dim'])


