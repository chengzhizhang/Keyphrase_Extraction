# -*- coding: utf-8 -*-
import os
import torch
import json
import pandas as pd
import numpy as np
from numpy import average
from torch import optim, nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from model import BiLSTM_CRF, TextDataSet
from utils import read_json_datas, performance_metrics
from torch.utils.tensorboard import SummaryWriter
torch.manual_seed(2021)

# Time step counter
steps = 1
def train_func(train_data_path, test_data_path, vocab_path, config_path, test_info_path, is_add_ref):
    writer = SummaryWriter('./log/runs')
    # Load configuration file
    test_info = read_json_datas(test_info_path)
    configs = json.load(open(config_path, 'r', encoding='utf-8'))
    print("params:", configs)
    ix2tag = {i:key for i, key in enumerate(
            pd.read_csv(configs['tags_path'], header=None, sep='\n').values.flatten().tolist())}
    train_dataset = TextDataSet(
        data_path = train_data_path,
        word_vocab_path = os.path.join(vocab_path, 'word_vocab.txt'),
        char_vocab_path = os.path.join(vocab_path, 'char_vocab.txt'),
        pos_vocab_path = os.path.join(vocab_path, 'pos_vocab.txt'),
        sentence_max_len = configs['sentence_max_len'],
        word_max_len = configs['word_max_len'])
    test_dataset = TextDataSet(
        data_path = test_data_path,
        word_vocab_path = os.path.join(vocab_path, 'word_vocab.txt'),
        char_vocab_path = os.path.join(vocab_path, 'char_vocab.txt'),
        pos_vocab_path = os.path.join(vocab_path, 'pos_vocab.txt'),
        sentence_max_len=configs['sentence_max_len'],
        word_max_len=configs['word_max_len'])

    # Data Loader
    train_loader = DataLoader(train_dataset, batch_size = configs['batch_size'])
    test_loader = DataLoader(test_dataset, batch_size = configs['batch_size'])

    # Create BiLSTM-CRF model object
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BiLSTM_CRF(
        word_vocab_path = os.path.join(vocab_path, 'word_vocab.txt'),
        char_vocab_path = os.path.join(vocab_path, 'char_vocab.txt'),
        #pretrain_path = os.path.join(vocab_path, 'word2vec.txt'),
        word_embedding_dim = configs['word_embedding_dim'],
        char_embedding_dim = configs['char_embedding_dim'],
        hidden_dim = configs['hidden_dim'],
        seq_len = configs['sentence_max_len'],
        is_add_ref = is_add_ref,
        use_crf = bool(configs['use_crf']),
        lstm_num_layers = configs['lstm_num_layers'],
        is_bidirectional = bool(configs['is_bidirectional']),
        dropout = configs['dropout'],
        tags_path = configs['tags_path'],
        device=device
    ).to(device)
    print(model)

    optimizer = optim.Adam(model.parameters(), lr = configs['lr'], weight_decay = configs['weight_decay'])
    # Learning rate decay
    scheduler = ReduceLROnPlateau(optimizer, factor = configs['optim_scheduler_factor'],
                                  patience = configs['optim_scheduler_patience'],
                                  verbose = bool(configs['optim_scheduler_verbose']))
    best_p, best_r, best_f1 = 0, 0, 0
    best_p_ta, best_r_ta, best_f1_ta = 0, 0, 0
    best_model = None
    global steps
    for epoch in range(configs['epochs']):
        model.train()
        losses = []
        with tqdm(train_loader) as pbar_train:
            for word_ids, char_ids, attention_mask, labels, features, _ in pbar_train:
                x = {
                    'word_ids': torch.as_tensor(word_ids, dtype=torch.long).to(device),
                    'char_ids': torch.as_tensor(char_ids, dtype=torch.long).to(device),
                    'attention_mask': torch.as_tensor(attention_mask).to(device),
                    'features': torch.as_tensor(features, dtype=torch.float).to(device),
                }
                y = torch.as_tensor(labels, dtype=torch.long).to(device)
                loss = model.loss(x, y)
                model.zero_grad()
                loss.backward()
                # Gradient clipping
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=10, norm_type=2)
                optimizer.step()
                losses.append(loss.item())
                pbar_train.set_description("loss:%s"%round(loss.item(),3))
        scheduler.step(np.average(losses))

        model.eval()
        preds, targets = [], []
        with tqdm(test_loader) as pbar_test:
            for word_ids, char_ids, attention_mask, labels, features, sentence_len in pbar_test:
                x = {
                    'word_ids': torch.as_tensor(word_ids, dtype=torch.long).to(device),
                    'char_ids': torch.as_tensor(char_ids, dtype=torch.long).to(device),
                    'attention_mask': torch.as_tensor(attention_mask).to(device),
                    'features': torch.as_tensor(features, dtype=torch.float).to(device),
                }
                _, outputs = model(x)
                for ix, (pred, tag) in enumerate(zip(outputs, labels)):
                    preds.append([ix2tag[i] for i in pred.detach().cpu().tolist()[0: sentence_len[ix]]])
                    targets.append([ix2tag[i] for i in tag.detach().cpu().tolist()[0: sentence_len[ix]]])
                pbar_test.set_description("finish")
        if is_add_ref == False:
            p, r, f1 = performance_metrics(preds, test_info, is_add_ref)
            if f1 > best_f1:
                best_model = model
                best_p, best_r, best_f1 = p, r, f1
            # record information
            writer.add_scalar('Train/Loss', round(np.average(losses)), steps)
            writer.add_scalar('test/p', p,  steps)
            writer.add_scalar('test/r', r,  steps)
            writer.add_scalar('test/f1', f1,  steps)
            writer.add_scalar('params/lr', optimizer.state_dict()['param_groups'][0]['lr'],  steps)
            print("epoch: %s, loss: %s, p: %s, r: %s, f1: %s" % (epoch+1, round(np.average(losses), 3), p, r, f1))
        else:
            (p, r, f1), (p_ta, r_ta, f1_ta) = performance_metrics(preds, test_info, is_add_ref)
            if f1 > best_f1:
                best_model = model
                best_p, best_r, best_f1 = p, r, f1
                best_p_ta, best_r_ta, best_f1_ta = p_ta, r_ta, f1_ta
            print("epoch: %s, loss: %s, p: %s, r: %s, f1: %s" % (epoch+1, np.mean(losses), p, r, f1))
            print("p_ta: %s, r_ta: %s, f1_ta: %s" % (p_ta, r_ta, f1_ta))
            # record information
            writer.add_scalar('Train/Loss', round(np.average(losses)), steps)
            writer.add_scalar('test/p', p, steps)
            writer.add_scalar('test/r', r, steps)
            writer.add_scalar('test/f1', f1, steps)
            writer.add_scalar('test/p_ta', p_ta, steps)
            writer.add_scalar('test/r_ta', r_ta, steps)
            writer.add_scalar('test/f1_ta', f1_ta, steps)
            writer.add_scalar('params/lr', optimizer.state_dict()['param_groups'][0]['lr'], steps)
        steps += 1
    print("best_p: %s, best_r: %s, best_f1: %s"%(best_p, best_r, best_f1))
    print("best_p_ta: %s, best_r_ta: %s, best_f1_ta: %s" % (best_p_ta, best_r_ta, best_f1_ta))
    return best_p, best_r, best_f1, best_p_ta, best_r_ta, best_f1_ta, best_model


# 10 fold cross validation
def run(path, save_name, is_add_ref = False, folds=10):
    train_folder = os.path.join(path, 'train')
    test_folder = os.path.join(path, 'test')
    config_path = os.path.join('./configs/%s/config_wr.json'%save_name) \
                  if is_add_ref else os.path.join('./configs/%s/config_wor.json'%save_name)
    ave_p, ave_r , ave_f1 = [], [], []
    ave_p_ta, ave_r_ta, ave_f1_ta = [], [], []
    for fold in range(folds):
        print("=-"*30, 'fold:', fold+1, "=-"*30)
        if is_add_ref:
            best_p, best_r, best_f1, \
            best_p_ta, best_r_ta, best_f1_ta, \
            best_model = train_func(train_data_path = os.path.join(train_folder, str(fold)),
                  test_data_path = os.path.join(test_folder, str(fold)),
                  test_info_path = os.path.join(test_folder, str(fold)+".json"),
                  vocab_path = os.path.join(path, 'vocab'),
                  config_path = config_path,
                  is_add_ref = is_add_ref)
            ave_p.append(best_p)
            ave_r.append(best_r)
            ave_f1.append(best_f1)
            ave_p_ta.append(best_p_ta)
            ave_r_ta.append(best_r_ta)
            ave_f1_ta.append(best_f1_ta)
            #torch.save(configs.state_dict(),open("./configs/%s_%s_ref_%s.pkl" % (save_name, 'with', round(best_f1, 3)),'wb'))
        else:
            best_p, best_r, best_f1, \
            _, _, _, \
            best_model = train_func(train_data_path = os.path.join(train_folder, str(fold)),
                  test_data_path = os.path.join(test_folder, str(fold)),
                  test_info_path = os.path.join(test_folder, str(fold)+".json"),
                  vocab_path = os.path.join(path, 'vocab'),
                  config_path = config_path,
                  is_add_ref = is_add_ref)
            ave_p.append(best_p)
            ave_r.append(best_r)
            ave_f1.append(best_f1)
            #torch.save(configs.state_dict(),open("./configs/%s_%s_ref_%s.pkl" % (save_name, 'no', round(best_f1, 3)), 'wb'))
        print("=-"*65)
    if is_add_ref:
        print("not in title and abstract and with refs:", [average(ave_p), average(ave_r), average(ave_f1)])
        print("in title and abstract and with refs:", [average(ave_p_ta), average(ave_r_ta), average(ave_f1_ta)])
        pd.DataFrame(np.array([average(ave_p), average(ave_r), average(ave_f1)]).T).to_csv(
            "./results/%s_%s_ref_no_in_title_and_abs.csv"%(save_name, 'with'), index=False, header=False)
        pd.DataFrame(np.array([average(ave_p_ta), average(ave_r_ta), average(ave_f1_ta)]).T).to_csv(
            "./results/%s_%s_ref_in_title_and_abs.csv"%(save_name, 'with'), index=False, header=False)
    else:
        print("not in title and abstract and no refs:", [average(ave_p), average(ave_r), average(ave_f1)])
        pd.DataFrame(np.array([average(ave_p), average(ave_r), average(ave_f1)]).T).to_csv(
            "./results/%s_%s_ref_no_in_title_and_abs.csv" % (save_name, 'no'), index=False, header=False)
