import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Symbol removal, stop word filtering, part of speech reduction, part of speech tagging
POS = ['JJ','JJR','JJS','NN','NNS']
english_punctuations = [',', '.', ':', ';', '``','?', '（','）','(', ')', '[', ']',
                        '&', '!', '*', '@', '#', '$', '%','\\','\"','}','{']
english_punctuations_not_period = [',', ':', ';', '``','?', '（','）','(', ')',
                                   '[', ']', '&', '!', '*', '@', '#', '$', '%','\\','\"','}','{']
stops_words = set(stopwords.words("english"))
porter_stemmer = PorterStemmer()

# Segmentation, word tagging, de deactivation, etc
def tokenize_pos(data, choose):
    def inner_func(text, choose):
        # Do nothing
        if choose == 1:
            return text
        # Word segmentation, stemming, merging and string return
        elif choose == 2:
            words = []
            new_text = text.replace('\\', '').lower()
            word_list = nltk.word_tokenize(new_text)  # Tokenizing
            for word in word_list:
                w = porter_stemmer.stem(word)  # Stemming
                words.append(w)
            return " ".join(words)
        # Word segmentation, filter punctuation,
        # stem extraction, return word eg: [word1, word2,......]
        elif choose == 3:
            words = []
            new_text = text.replace('\\', '').lower()
            word_list = nltk.word_tokenize(new_text)  # Tokenizing
            for word in word_list:
                # Choose words whose pos are nouns and adjectives.
                if word not in english_punctuations:  # Symbol removal
                    w = porter_stemmer.stem(word)  # Stemming
                    words.append(w)
            return " ".join(words)
        # Word segmentation, part of speech tagging, stem extraction,
        # return word and part of speech tuple eg: [(word1, pos1), (word2, pos2),...]
        elif choose == 4:
            rev_datas = []
            new_text = text.replace('\\', '').lower()
            word_list = nltk.word_tokenize(new_text)  # Tokenizing
            word_pos = nltk.pos_tag(word_list)  # Part-of-speech tagging
            for word, pos in word_pos:
                word = porter_stemmer.stem(word)  # Stemming
                rev_datas.append((word, pos))
            return rev_datas
        # Word segmentation, part of speech tagging, removing stop words and filtering punctuation,
        # stem extraction, return tuple eg of word and part of speech: [(word1, pos1), (word2, pos2),...]
        elif choose == 5:
            rev_datas = []
            new_text = text.replace('\\', '').lower()
            word_list = nltk.word_tokenize(new_text)  # Tokenizing
            word_pos = nltk.pos_tag(word_list)  # Part-of-speech tagging
            for word, pos in word_pos:
                # Choose words whose pos are nouns and adjectives.
                if word not in english_punctuations and word not in stops_words:  # Symbol removal
                    word = porter_stemmer.stem(word)  # Stemming
                    rev_datas.append((word, pos))
            return rev_datas
        # Word segmentation, stop words removal and punctuation filtering,
        # stem extraction, return word eg: [word1, word2,...]
        elif choose == 6:
            rev_datas = []
            new_text = text.replace('\\', '').lower()
            word_list = nltk.word_tokenize(new_text)  # Tokenizing
            for word in word_list:
                # Choose words whose pos are nouns and adjectives.
                if word not in english_punctuations and word not in stops_words:  # Symbol removal
                    word = porter_stemmer.stem(word)  # Stemming
                    rev_datas.append(word)
            return rev_datas
        # Word segmentation, part of speech tagging, removing stop words and filtering punctuation,
        # stem extraction, and selecting the words with specified part of speech,
        # return the tuple eg of words and part of speech: [(word1, pos1), (word2, pos2),...]
        elif choose == 7:
            rev_datas = []
            new_text = text.replace('\\', '').lower()
            word_list = nltk.word_tokenize(new_text)  # Tokenizing
            word_pos = nltk.pos_tag(word_list)  # Part-of-speech tagging
            for word, pos in word_pos:
                # Choose words whose pos are nouns and adjectives.
                if word not in english_punctuations and word not in stops_words and pos in POS:  # Symbol removal
                    word = porter_stemmer.stem(word)  # Stemming
                    rev_datas.append((word, pos))
            return rev_datas
        # Word segmentation, part of speech tagging, removal of stop words,
        # stem extraction, return the tuple eg of word and part of speech: [(word1, pos1), (word2, pos2),...]
        elif choose == 8:
            rev_datas = []
            new_text = text.replace('\\', '').lower()
            word_list = nltk.word_tokenize(new_text)  # Tokenizing
            word_pos = nltk.pos_tag(word_list)  # Part-of-speech tagging
            for word, pos in word_pos:
                # Choose words whose pos are nouns and adjectives.
                if word not in english_punctuations_not_period and word not in stops_words:  # Symbol removal
                    word = porter_stemmer.stem(word)  # Stemming
                    rev_datas.append((word, pos))
            return rev_datas
        else:
            raise RuntimeError("The value of the variable ’choose‘ is 1-6")

    # External operations
    return_datas = []
    if type(data) == list:
        for text in data:
            return_datas.append(inner_func(text, choose))
    else:
        return_datas = inner_func(data, choose)
    return return_datas

# Data preprocessing
def data_preprocessing(datas, fields, chooses):
    return_datas = []
    for data in datas:
        row = {}
        for field, choose in zip(fields, chooses):
            rs = tokenize_pos(data[field], choose)
            row[field] = rs
        return_datas.append(row)
    return return_datas
