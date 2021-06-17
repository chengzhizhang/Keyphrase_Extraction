# -*- coding: utf-8 -*-
import torch
import numpy as np
import pandas as pd
from crf import CRF
from torch import optim, nn
from utils import read_text
from torch.utils.data import Dataset, DataLoader
torch.manual_seed(2021)

# Text processing object
class TextDataSet(Dataset):

    def __init__(self, data_path, word_vocab_path, char_vocab_path,
                 pos_vocab_path, sentence_max_len, word_max_len, tags_path='./datas/tags'):
        super(TextDataSet, self).__init__()
        self.data = pd.read_csv(data_path, sep='\t',
                    names=['rank', 'text', 'label',
                           'pos', 'loc', 'len',
                           'tf', 'tr', 'freq_ft', 'freq_ref'])
        self.word_max_len = word_max_len
        self.sentence_max_len = sentence_max_len
        self.tags = {key:i for i, key in enumerate(read_text(tags_path))}
        self.word_vocab = {key:i for i, key in enumerate(read_text(word_vocab_path))}
        self.char_vocab = {key:i for i, key in enumerate(read_text(char_vocab_path))}
        self.pos_vocab = {key:i for i, key in enumerate(read_text(pos_vocab_path))}

    def __getitem__(self, item):
        text_chars = [i for i in self.data['text'].iloc[item].split(' ')]
        text = [self.word_vocab[i] for i in self.data['text'].iloc[item].split(' ')]
        label = [self.tags[i] for i in self.data['label'].iloc[item].split(' ')]
        word_ps = [self.pos_vocab[i] for i in self.data['pos'].iloc[item].split(' ')]
        word_lc = [float(i) for i in self.data['loc'].iloc[item].split(' ')]
        word_ln = [float(i) for i in self.data['len'].iloc[item].split(' ')]
        word_tf = [float(i) for i in self.data['tf'].iloc[item].split(' ')]
        word_tr = [float(i) for i in self.data['tr'].iloc[item].split(' ')]
        freq_ft = [float(i) for i in self.data['freq_ft'].iloc[item].split(' ')]
        freq_rf = [float(i) for i in self.data['freq_ref'].iloc[item].split(' ')]
        sentence_len = len(text)
        if sentence_len >= self.sentence_max_len:
            text = text[:self.sentence_max_len]
            label = label[:self.sentence_max_len]
            word_ps = word_ps[:self.sentence_max_len]
            word_lc = word_lc[:self.sentence_max_len]
            word_ln = word_ln[:self.sentence_max_len]
            word_tf = word_tf[:self.sentence_max_len]
            word_tr = word_tr[:self.sentence_max_len]
            freq_ft = freq_ft[:self.sentence_max_len]
            freq_rf = freq_rf[:self.sentence_max_len]
        else:
            zero_pad = [0] * (self.sentence_max_len - sentence_len)
            minus_one_pad = [-1] * (self.sentence_max_len - sentence_len)
            text.extend(zero_pad)
            label.extend(zero_pad)
            word_ps.extend(zero_pad)
            word_lc.extend(minus_one_pad)
            word_ln.extend(minus_one_pad)
            word_tf.extend(minus_one_pad)
            word_tr.extend(minus_one_pad)
            freq_ft.extend(minus_one_pad)
            freq_rf.extend(minus_one_pad)
        # Construct character encoding sequence
        chars_seq = np.zeros(shape=(self.sentence_max_len, self.word_max_len))
        for ii in range(min([len(text_chars), self.sentence_max_len])):
            chars = [self.char_vocab[i] for i in text_chars[ii]]
            chars_len = len(chars)
            if chars_len >= self.word_max_len:
                chars = chars[:self.word_max_len]
            else:
                zero_pad = [0] * (self.word_max_len - chars_len)
                chars.extend(zero_pad)
            assert len(chars) == self.word_max_len
            chars_seq[ii] = chars
        # Convert the data type to numpy
        word_ids = np.array(text)
        char_ids = np.array(chars_seq)
        label = np.array(label)
        # Features
        word_ps = np.array(word_ps)
        word_lc = np.array(word_lc)
        word_ln = np.array(word_ln)
        word_tf = np.array(word_tf)
        word_tr = np.array(word_tr)
        freq_ft = np.array(freq_ft)
        freq_rf = np.array(freq_rf)
        # Attention mask
        attention_mask = word_ids > 0
        # Feature packaging
        features = np.vstack([word_ps, word_lc, word_ln,
                       word_tf, word_tr, freq_ft, freq_rf])
        return word_ids, char_ids, attention_mask, label, features, np.array(sentence_len)

    def __len__(self):
        return self.data.shape[0]

# Character-level encoding
class CharEncode(nn.Module):

    def __init__(self, vocab_size, embed_dim, hidden_dim,
                 num_layers=2, is_bidirectional=True, dropout=0.5,
                 device=torch.device('cpu')):
        super(CharEncode, self).__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        self.bidire = 2 if is_bidirectional else 1

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.bilstm = nn.LSTM(embed_dim, hidden_dim,
                              num_layers=num_layers, bidirectional=is_bidirectional)
        self.out = nn.Linear(self.bidire*hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, input):
        x = self.embedding(input)
        (B, S, W, E) = x.size()
        x = x.reshape(-1, W, E)
        x = x.transpose(0 ,1)
        _, (hn, cn) = self.bilstm(x)
        x = torch.cat([hn[-2], hn[-1]], dim=-1)
        x = self.dropout(x)
        x = self.out(x)
        x = x.reshape(-1, S, E)
        return x

# BilSTM-CRF configs
class BiLSTM_CRF(nn.Module):

    def __init__(self, word_vocab_path, char_vocab_path, word_embedding_dim,
                 char_embedding_dim, hidden_dim, seq_len=128, is_add_ref=False,
                 use_crf=True, lstm_num_layers=2, pretrain_path=None,
                 is_bidirectional=True, dropout=0.5, tags_path='./datas/tags', device=torch.device('cpu')):

        super(BiLSTM_CRF, self).__init__()
        # parameters
        self.hidden_dim = hidden_dim
        self.is_add_ref = is_add_ref
        self.use_crf = use_crf
        self.tagset_size = pd.read_csv(tags_path, header=None, sep='\n').shape[0]
        self.word_vocab_size = pd.read_csv(word_vocab_path, header=None, sep='\n').shape[0]
        self.char_vocab_size = pd.read_csv(char_vocab_path, header=None, sep='\n').shape[0]

        self.num_directional = 2 if is_bidirectional else 1
        self.target_size = self.tagset_size + 2 if self.use_crf else self.tagset_size

        self.input_to_lstm_dim = word_embedding_dim + char_embedding_dim + 7 \
                if is_add_ref else word_embedding_dim + char_embedding_dim + 6
        self.lstm_to_fc_dim = self.hidden_dim * self.num_directional + char_embedding_dim + 7 \
                if is_add_ref else self.hidden_dim * self.num_directional + char_embedding_dim + 6

        # Word embedding
        self.word_embedding = nn.Embedding(self.word_vocab_size, word_embedding_dim)
        self.char_embedding = CharEncode(self.char_vocab_size, char_embedding_dim,
                                         char_embedding_dim, device=device)

        # Processing layers
        self.bilstm = nn.LSTM(self.input_to_lstm_dim, hidden_dim,
                              num_layers=lstm_num_layers, bidirectional=is_bidirectional)
        self.hidden2tag = nn.Linear(self.lstm_to_fc_dim, self.target_size)
        self.crf_layer = CRF(target_size=self.tagset_size, device=device)

        # Prevent over fitting
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm1 = nn.LayerNorm(normalized_shape=[seq_len, self.input_to_lstm_dim])
        self.layer_norm2 = nn.LayerNorm(normalized_shape=[seq_len, self.lstm_to_fc_dim])
        self.batch_norm = nn.BatchNorm1d(num_features=seq_len)

        # Loss function object
        self.loss_fuc_crf = self.crf_layer.neg_log_likelihood_loss
        self.loss_fuc_cel = nn.CrossEntropyLoss()

        # Load pre-training word vector
        if pretrain_path != None:
            vector = torch.zeros(self.word_embedding.weight.data.shape)
            pretrain_vector = pd.read_csv(pretrain_path, names=['word', 'vec'], sep=';')
            for index, vec in enumerate(pretrain_vector['vec']):
                vector[index] = torch.as_tensor([float(i) for i in vec.split(' ')])
            self.word_embedding.weight.data = torch.as_tensor(vector)

    # Get LSTM outputs
    def __lstm_feats(self, input):

        features = input['features']
        features = features.transpose(0, 1).unsqueeze(-1)

        # features
        word_ps = features[0]
        word_lc = features[1]
        word_ln = features[2]
        word_tf = features[3]
        word_tr = features[4]
        freq_ft = features[5]
        freq_rf = features[6]

        # Encoding (word and character encoding)
        word_embed = self.word_embedding(input['word_ids'])
        char_embed = self.char_embedding(input['char_ids'])

        # Combined feature vector
        x = torch.cat([word_embed, char_embed, word_ps, word_lc,
                       word_ln, word_tf, word_tr, freq_ft], dim=-1)
        if self.is_add_ref:
            x = torch.cat([word_embed, char_embed, word_ps, word_lc,
                       word_ln, word_tf, word_tr, freq_ft, freq_rf], dim=-1)
        # x = self.dropout(x)
        # x = self.layer_norm1(x)

        # Enter into BILSTM
        x = x.transpose(0, 1)
        lstm_out, _ = self.bilstm(x)
        lstm_out = lstm_out.transpose(0, 1)

        x = torch.cat([lstm_out, char_embed, word_ps, word_lc,
                       word_ln, word_tf, word_tr, freq_ft], dim=-1)
        if self.is_add_ref:
            x = torch.cat([lstm_out, char_embed, word_ps, word_lc,
                       word_ln, word_tf, word_tr, freq_ft, freq_rf], dim=-1)

        x = self.dropout(x)
        # x = self.batch_norm(x)
        # x = self.layer_norm2(x)
        feats = self.hidden2tag(x)
        return feats

    # Calculate outputs
    def __outputs(self, feats, mask=None):
        if self.use_crf:
            return self.crf_layer(feats, mask)
        else:
            return None, torch.argmax(torch.softmax(feats, dim=-1), dim=-1)

    # Calculate loss values
    def loss(self, input, labels):
        mask = input['attention_mask']
        feats = self.__lstm_feats(input)
        if self.use_crf:
            return self.loss_fuc_crf(feats, mask, labels)
        else:
            return self.loss_fuc_cel(
                feats.reshape(-1, self.target_size), labels.reshape(-1))


    def forward(self, input):
        mask = input['attention_mask']
        feats = self.__lstm_feats(input)
        outputs = self.__outputs(feats, mask)
        return outputs
