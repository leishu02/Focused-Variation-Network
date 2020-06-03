import torch
import random
import numpy as np
from config import Config
from reader import Reader
from reader import pad_sequences
from network import get_network
from network import cuda_, nan
from metric import Evaluator

from torch.optim import Adam
from torch.autograd import Variable
import argparse, time, os
import logging

import sys
import ast

from itertools import cycle, islice
import itertools

def cal(s):
    sum = 0
    for i, v in enumerate(s):
        sum += pow(2, i) * v
    return int(sum)

class Model:
    def __init__(self, cfg):
        self.reader = Reader(cfg)
        self.m = get_network(cfg, self.reader.vocab)
        self.EV = Evaluator  # evaluator class
        if cfg.cuda:
            self.m = self.m.cuda()
        self.base_epoch = -1
        self.cfg = cfg


    def _convert_batch(self, py_batch, act_idx_dict = None, personality_idx_dict = None, value_idx_dict = None):
        kw_ret = {}
        x = None
        gt_y = None
        batch_size = len(py_batch['slot_seq'])
        slot_np = pad_sequences(py_batch['slot_seq'], self.cfg.slot_max_ts, padding='post',truncating='post').transpose((1, 0))

        slot_value_np = pad_sequences(py_batch['slot_value_seq'], self.cfg.slot_max_ts, padding='post',truncating='post').transpose((1, 0))  # (seqlen, batchsize)
        text_np = pad_sequences(py_batch['text_seq'], self.cfg.text_max_ts, padding='post',truncating='post').transpose((1, 0))
        delex_text_np = pad_sequences(py_batch['delex_text_seq'], self.cfg.text_max_ts, padding='post',truncating='post').transpose((1, 0))
        slot_len = np.array(py_batch['slot_seq_len'])

        slot_value_len = np.array(py_batch['slot_value_seq_len'])
        text_len = np.array(py_batch['text_seq_len'])
        delex_text_len = np.array(py_batch['delex_text_seq_len'])
        go_np = pad_sequences(py_batch['go'], 1, padding='post',truncating='post').transpose((1, 0))
        go = cuda_(torch.autograd.Variable(torch.from_numpy(go_np).long()), self.cfg)

        act_idx = cuda_(Variable(torch.from_numpy(np.asarray(py_batch['slot_idx'])).float()), self.cfg)
        act_flatten_idx_list = [ cal(s) for s in py_batch['slot_idx']]
        act_flatten_idx_np = np.zeros((batch_size, pow(2, self.cfg.act_size)))
        for i, v in enumerate(act_flatten_idx_list):
            act_flatten_idx_np[i,v] = 1
        act_flatten_idx = cuda_(Variable(torch.from_numpy(np.asarray(act_flatten_idx_np)).float()), self.cfg)

        if self.cfg.domain == 'personage':
            personality_np = pad_sequences(py_batch['personality_seq'], self.cfg.personality_size, padding='post',
                                           truncating='post').transpose((1, 0))
            personality_len = np.array(py_batch['personality_seq_len'])
            personality_idx = cuda_(Variable(torch.from_numpy(np.asarray(py_batch['personality_idx'])).long()), self.cfg)
            personality_flatten_idx_np = np.zeros((batch_size, self.cfg.personality_size))
            for i, v in enumerate(py_batch['personality_idx']):
                personality_flatten_idx_np[i,v] = 1
            personality_flatten_idx = cuda_(Variable(torch.from_numpy(np.asarray(personality_flatten_idx_np)).float()), self.cfg)
            kw_ret['condition'] = torch.cat([act_flatten_idx, personality_flatten_idx], dim=-1)
            kw_ret['personality_np'] = personality_np  # seqlen, batchsize
            kw_ret['personality_seq'] = cuda_(Variable(torch.from_numpy(personality_np).long()), self.cfg)  # seqlen, batchsize
            kw_ret['personality_len'] = personality_len  # batchsize
            kw_ret['personality_idx'] = personality_idx
        else:
            kw_ret['condition'] = cuda_(Variable(torch.from_numpy(np.asarray(py_batch['unique'])).float()), self.cfg)


        kw_ret['act_flatten_idx'] = act_flatten_idx
        kw_ret['slot_np'] = slot_np  # seqlen, batchsize
        kw_ret['slot_value_np'] = slot_value_np  # seqlen, batchsize
        kw_ret['text_np'] = text_np  # seqlen, batchsize
        kw_ret['delex_text_np'] = delex_text_np  # seqlen, batchsize
        kw_ret['slot_len'] = slot_len  # batchsize
        kw_ret['slot_value_len'] = slot_value_len  # batchsize
        kw_ret['text_len'] = text_len  # batchsize
        kw_ret['delex_text_len'] = delex_text_len  # batchsize
        kw_ret['go_np'] = go_np
        kw_ret['go'] = go
        kw_ret['act_idx'] = act_idx
        
        if self.cfg.domain == 'e2e':
            for k in self.cfg.key_order:
                kw_ret[k] = cuda_(Variable(torch.from_numpy(np.asarray(py_batch[k])).long()), self.cfg)
        
        if act_idx_dict:
            act_encoding = []
            for i in py_batch['slot_idx']:
                if str(i) not in act_idx_dict:
                    assert()
                else:
                    min_str = str(i)
                dist = act_idx_dict[min_str]
                if self.cfg.sample == 'random':
                    sample = np.random.choice(self.cfg.codebook_size, 1, p=dist)
                    sample = sample.tolist()
                else:
                    samples = np.random.choice(self.cfg.codebook_size, 1000, p=dist)
                    unique, counts = np.unique(np.asarray(samples), return_counts=True)
                    sorted_list = sorted(zip(unique, counts), key=lambda x:-x[1])
                    sample = [sorted_list[0][0]]
                act_encoding.append(sample)
            kw_ret['act_sample_idx'] = cuda_(Variable(torch.from_numpy(np.asarray(act_encoding))).long(), self.cfg)
            #print (kw_ret['act_sample_idx'].size())

        if personality_idx_dict:
            personality_encoding = []
            for i in py_batch['personality_idx']:
                dist = personality_idx_dict[str(i)]
                if self.cfg.sample == 'random':
                    sample = np.random.choice(self.cfg.codebook_size, 1, p=dist)
                    sample = sample.tolist()
                else:
                    if self.cfg.value_codebook_vocab:
                        samples = np.random.choice(len(self.reader.vocab), 1000, p=dist)
                    else:
                        samples = np.random.choice(self.cfg.codebook_size, 1000, p=dist)
                    unique, counts = np.unique(np.asarray(samples), return_counts=True)
                    sorted_list = sorted(zip(unique, counts), key=lambda x:-x[1])
                    sample = [sorted_list[0][0]]
                personality_encoding.append(sample)
            #print (len(act_encoding), len(personality_encoding))
            #print (act_encoding, personality_encoding)
            kw_ret['personality_sample_idx'] = cuda_(Variable(torch.from_numpy(np.asarray(personality_encoding))).long(), self.cfg)

        if value_idx_dict:
            for k in self.cfg.key_order:
                _encoding = []
                for i in py_batch[k]:
                    dist = value_idx_dict[k][str(i)]
                    if self.cfg.sample == 'random':
                        sample = np.random.choice(self.cfg.codebook_size, 1, p=dist)
                        sample = sample.tolist()
                    else:
                        if self.cfg.value_codebook_vocab:
                            samples = np.random.choice(len(self.reader.vocab), 1000, p=dist)
                        else:
                            samples = np.random.choice(self.cfg.codebook_size, 1000, p=dist)
                        unique, counts = np.unique(np.asarray(samples), return_counts=True)
                        sorted_list = sorted(zip(unique, counts), key=lambda x:-x[1])
                        sample = [sorted_list[0][0]]
                    _encoding.append(sample)
                    # print (len(act_encoding), len(personality_encoding))
                    # print (act_encoding, personality_encoding)
                kw_ret[k+'_sample_idx'] = cuda_(
                    Variable(torch.from_numpy(np.asarray(_encoding))).long(), self.cfg)
                #print (kw_ret[k+'_sample_idx'].size())

        if self.cfg.network == 'classification':
            if self.cfg.remove_slot_value == True:
                x = cuda_(Variable(torch.from_numpy(delex_text_np).long()), self.cfg)
            else:
                x = cuda_(Variable(torch.from_numpy(text_np).long()), self.cfg)
                gt_y_np = np.asarray(py_batch['personality_idx'])
                gt_y = cuda_(Variable(torch.from_numpy(gt_y_np).long()), self.cfg)
        elif 'seq2seq' in self.cfg.network:
            if self.cfg.remove_slot_value == True:
                x = cuda_(Variable(torch.from_numpy(slot_np).long()), self.cfg)#seqlen, batchsize
                gt_y = cuda_(Variable(torch.from_numpy(delex_text_np).long()), self.cfg)#seqlen, batchsize
            else:
                x = cuda_(Variable(torch.from_numpy(slot_value_np).long()), self.cfg)#seqlen, batchsize
                gt_y = cuda_(Variable(torch.from_numpy(text_np).long()), self.cfg)#seqlen, batchsize
        elif 'VQVAE' in self.cfg.network or 'CVAE' in self.cfg.network:
            if self.cfg.remove_slot_value == True:
                x = cuda_(Variable(torch.from_numpy(slot_np).long()), self.cfg)#seqlen, batchsize
                gt_y = cuda_(Variable(torch.from_numpy(delex_text_np).long()), self.cfg)#seqlen, batchsize
            else:
                x = cuda_(Variable(torch.from_numpy(slot_value_np).long()), self.cfg)#seqlen, batchsize
                gt_y = cuda_(Variable(torch.from_numpy(text_np).long()), self.cfg)#seqlen, batchsize
        else:
            assert()
    
        return x, gt_y, kw_ret
    
    def _predict_convert_batch(self, py_batch, act_idx_dict = None, personality_idx_dict = None, value_idx_dict = None):
        kw_ret = {}
        x = None
        gt_y = None
        batch_size = len(py_batch['slot_seq'])
        slot_np = pad_sequences(py_batch['slot_seq'], self.cfg.slot_max_ts, padding='post',truncating='post').transpose((1, 0))

        slot_value_np = pad_sequences(py_batch['slot_value_seq'], self.cfg.slot_max_ts, padding='post',truncating='post').transpose((1, 0))  # (seqlen, batchsize)
        text_np = pad_sequences(py_batch['text_seq'], self.cfg.text_max_ts, padding='post',truncating='post').transpose((1, 0))
        delex_text_np = pad_sequences(py_batch['delex_text_seq'], self.cfg.text_max_ts, padding='post',truncating='post').transpose((1, 0))
        slot_len = np.array(py_batch['slot_seq_len'])

        slot_value_len = np.array(py_batch['slot_value_seq_len'])
        text_len = np.array(py_batch['text_seq_len'])
        delex_text_len = np.array(py_batch['delex_text_seq_len'])
        go_np = pad_sequences(py_batch['go'], 1, padding='post',truncating='post').transpose((1, 0))
        go = cuda_(torch.autograd.Variable(torch.from_numpy(go_np).long()), self.cfg)

        act_idx = cuda_(Variable(torch.from_numpy(np.asarray(py_batch['slot_idx'])).float()), self.cfg)
        act_flatten_idx_list = [ cal(s) for s in py_batch['slot_idx']]
        act_flatten_idx_np = np.zeros((batch_size, pow(2, self.cfg.act_size)))
        for i, v in enumerate(act_flatten_idx_list):
            act_flatten_idx_np[i,v] = 1
        act_flatten_idx = cuda_(Variable(torch.from_numpy(np.asarray(act_flatten_idx_np)).float()), self.cfg)

        if self.cfg.domain == 'personage':
            personality_np = pad_sequences(py_batch['personality_seq'], self.cfg.personality_size, padding='post',
                                           truncating='post').transpose((1, 0))
            personality_len = np.array(py_batch['personality_seq_len'])
            personality_idx = cuda_(Variable(torch.from_numpy(np.asarray(py_batch['personality_idx'])).long()), self.cfg)
            personality_flatten_idx_np = np.zeros((batch_size, self.cfg.personality_size))
            for i, v in enumerate(py_batch['personality_idx']):
                personality_flatten_idx_np[i,v] = 1
            personality_flatten_idx = cuda_(Variable(torch.from_numpy(np.asarray(personality_flatten_idx_np)).float()), self.cfg)
            kw_ret['condition'] = torch.cat([act_flatten_idx, personality_flatten_idx], dim=-1)
            kw_ret['personality_np'] = personality_np  # seqlen, batchsize
            kw_ret['personality_seq'] = cuda_(Variable(torch.from_numpy(personality_np).long()), self.cfg)  # seqlen, batchsize
            kw_ret['personality_len'] = personality_len  # batchsize
            kw_ret['personality_idx'] = personality_idx
        else:
            kw_ret['condition'] = cuda_(Variable(torch.from_numpy(np.asarray(py_batch['unique'])).float()), self.cfg)


        kw_ret['act_flatten_idx'] = act_flatten_idx
        kw_ret['slot_np'] = slot_np  # seqlen, batchsize
        kw_ret['slot_value_np'] = slot_value_np  # seqlen, batchsize
        kw_ret['text_np'] = text_np  # seqlen, batchsize
        kw_ret['delex_text_np'] = delex_text_np  # seqlen, batchsize
        kw_ret['slot_len'] = slot_len  # batchsize
        kw_ret['slot_value_len'] = slot_value_len  # batchsize
        kw_ret['text_len'] = text_len  # batchsize
        kw_ret['delex_text_len'] = delex_text_len  # batchsize
        kw_ret['go_np'] = go_np
        kw_ret['go'] = go
        kw_ret['act_idx'] = act_idx

        if self.cfg.domain == 'e2e':
            for k in self.cfg.key_order:
                kw_ret[k] = cuda_(Variable(torch.from_numpy(np.asarray(py_batch[k])).long()), self.cfg)
        
        if act_idx_dict and personality_idx_dict:
            act_encoding = []
            personality_encoding = []
            
            for i in py_batch['slot_idx']:
                dist = act_idx_dict[str(i)]
                sample = np.random.choice(self.cfg.codebook_size, 1000, p=dist)
                act_encoding+=sample.tolist()
            for i in py_batch['personality_idx']:
                dist = personality_idx_dict[str(i)]
                if self.cfg.value_codebook_vocab:
                    sample = np.random.choice(len(self.reader.vocab), 1000, p=dist)
                else:
                    sample = np.random.choice(self.cfg.codebook_size, 1000, p=dist)
                personality_encoding+=sample.tolist()
            
            ae_count = np.bincount(np.asarray(act_encoding))
            ae_set = set(act_encoding)
            print (ae_set)
            most_act = np.argmax(ae_count)
            #ae_set = list(ae_set - set([most_act]))
            #print (ae_set)
            pe_count = np.bincount(np.asarray(personality_encoding))
            pe_set = set(personality_encoding)
            print ('pe_set', pe_set)
            most_personality = np.argmax(pe_count)
            #pe_set = list(pe_set - set([most_personality]))
            #print (pe_set)
            unique, counts = np.unique(np.asarray(personality_encoding), return_counts=True)
            if len(unique) >= 10:
                unique = unique[:10]
            print ('unique', len(unique), {u:counts[i] for i, u in enumerate(unique)}) 
            pe_set = list(unique)
            act_encoding = []
            personality_encoding = []
            chunk = int(batch_size/len(unique))
            for i in range(batch_size):
                pe_idx = int(i/chunk)
                if pe_idx >= len(pe_set):
                    pe_idx = len(pe_set)-1
                act_encoding.append(np.array([most_act]))
                personality_encoding.append(np.array([pe_set[pe_idx]]))
                '''
                if i < batch_size/2:
                    act_encoding.append(np.array([most_act]))
                    if len(pe_set) > 0:
                        personality_encoding.append(np.array([pe_set.pop()]))
                    else:
                        personality_encoding.append(np.array([most_personality]))
                else:
                    if len(ae_set) > 0:
                        act_encoding.append(np.array([ae_set.pop()]))
                    else:
                        act_encoding.append(np.array([most_act]))
                    personality_encoding.append(np.array([most_personality]))
                '''   
            print (len(act_encoding), len(personality_encoding))
            print (personality_encoding)
            kw_ret['act_sample_idx'] = cuda_(Variable(torch.from_numpy(np.asarray(act_encoding))).long(), self.cfg)
            kw_ret['personality_sample_idx'] = cuda_(Variable(torch.from_numpy(np.asarray(personality_encoding))).long(), self.cfg) 
                
        if value_idx_dict:
            act_encoding = []
            i = py_batch['slot_idx'][0]
            dist = act_idx_dict[str(i)]
            sample = np.random.choice(self.cfg.codebook_size, 1000000, p=dist)
            unique, counts = np.unique(np.asarray(sample), return_counts=True)
            sorted_list = sorted(zip(unique, counts), key=lambda x:-x[1])
            act_encoding+=sample.tolist()
            if len(sorted_list) >= batch_size:
                sorted_list = sorted_list[batch_size]
            unique = [ s[0]for s in sorted_list]
            most_act = sorted_list[0][0]
            #act_encoding = list(itertools.chain.from_iterable(itertools.repeat(u, batch_size) for u in unique))
            #random.shuffle(act_encoding)
            act_encoding = [most_act for _ in range(batch_size)]#act_encoding[:batch_size]
            #kw_ret['act_sample_idx'] = cuda_(Variable(torch.from_numpy(np.asarray([most_act]*batch_size))).long(), self.cfg).unsqueeze(1) 
            kw_ret['act_sample_idx'] = cuda_(Variable(torch.from_numpy(np.asarray(act_encoding))).long(), self.cfg).unsqueeze(1)   
            
            
            k_count = 0
            chunk = int(batch_size/len(self.cfg.key_order))
            for k in self.cfg.key_order:
                i = py_batch[k][0]
                print ('batch_size', batch_size)
                dist = value_idx_dict[k][str(i)]
                if self.cfg.value_codebook_vocab:
                    sample = np.random.choice(len(self.reader.vocab), 10000, p=dist)
                else:
                    sample = np.random.choice(self.cfg.codebook_size, 10000, p=dist)
                unique, counts = np.unique(np.asarray(sample), return_counts=True)
                sorted_list = sorted(zip(unique, counts), key=lambda x:-x[1])
 
                max_unique = sorted_list[0][0]
                
                if len(sorted_list) >= 3:
                    sorted_list = sorted_list[:3]
                unique = [s[0] for s in sorted_list]
                print (k, 'unique', len(unique), sorted_list) 
                _cand = list(itertools.chain.from_iterable(itertools.repeat(u, batch_size) for u in unique))
                random.shuffle(_cand)
                _encoding = [max_unique]*batch_size
                #_encoding = _encoding[:batch_size]
                print (max_unique,_encoding)
                if k not in ['name']:
                    #_encoding[k_count*chunk:(k_count+1)*chunk] = [max_unique]*chunk
                    _encoding[k_count*chunk:(k_count+1)*chunk] = _cand[:chunk]
                    k_count += 1
                print (k_count, _encoding)
                #kw_ret[k+'_sample_idx'] = cuda_(Variable(torch.from_numpy(np.asarray(_encoding))).long(), self.cfg).unsqueeze(1)
                kw_ret[k+'_sample_idx'] = cuda_(Variable(torch.from_numpy(np.asarray([max_unique]*batch_size))).long(), self.cfg).unsqueeze(1)
                        
            



        if self.cfg.network == 'classification':
            if self.cfg.remove_slot_value == True:
                x = cuda_(Variable(torch.from_numpy(delex_text_np).long()), self.cfg)
            else:
                x = cuda_(Variable(torch.from_numpy(text_np).long()), self.cfg)
        elif 'seq2seq' in self.cfg.network:
            if self.cfg.remove_slot_value == True:
                x = cuda_(Variable(torch.from_numpy(slot_np).long()), self.cfg)#seqlen, batchsize
                gt_y = cuda_(Variable(torch.from_numpy(delex_text_np).long()), self.cfg)#seqlen, batchsize
            else:
                x = cuda_(Variable(torch.from_numpy(slot_value_np).long()), self.cfg)#seqlen, batchsize
                gt_y = cuda_(Variable(torch.from_numpy(text_np).long()), self.cfg)#seqlen, batchsize
        elif 'VQVAE' in self.cfg.network or 'CVAE' in self.cfg.network:
            if self.cfg.remove_slot_value == True:
                x = cuda_(Variable(torch.from_numpy(slot_np).long()), self.cfg)#seqlen, batchsize
                gt_y = cuda_(Variable(torch.from_numpy(delex_text_np).long()), self.cfg)#seqlen, batchsize
            else:
                x = cuda_(Variable(torch.from_numpy(slot_value_np).long()), self.cfg)#seqlen, batchsize
                gt_y = cuda_(Variable(torch.from_numpy(text_np).long()), self.cfg)#seqlen, batchsize
        else:
            assert()
    
        return x, gt_y, kw_ret

    def train(self):
        lr = self.cfg.lr
        prev_min_loss = np.inf
        prev_max_metrics = 0.
        early_stop_count = self.cfg.early_stop_count
        train_time = 0
        for epoch in range(self.cfg.epoch_num):
            sw = time.time()
            if epoch <= self.base_epoch:
                continue
            sup_loss = 0
            sup_cnt = 0
            data_iterator = self.reader.mini_batch_iterator('train')
            optim = Adam(lr=lr, params=filter(lambda x: x.requires_grad, self.m.parameters()), weight_decay=1e-5)
            for iter_num, dial_batch in enumerate(data_iterator):
                for turn_num, turn_batch in enumerate(dial_batch):
                    if self.cfg.truncated:
                        logging.debug('iter %d turn %d' % (iter_num, turn_num))
                    optim.zero_grad()
                    x, gt_y, kw_ret = self._convert_batch(turn_batch)
                    if 'VQVAE' in self.cfg.network:
                        loss, recon_loss, act_loss, personality_loss, act_vq_loss, personality_vq_loss\
                            = self.m(x=x, gt_y=gt_y, mode='train', **kw_ret)
                    elif self.cfg.network == 'classification':
                        loss = self.m(x=x, gt_y=gt_y, mode='train', **kw_ret)
                    elif self.cfg.network == 'controlled_CVAE':
                        loss, recon_loss, KLD, vocab_vq_loss, act_loss, personality_loss\
                            = self.m(x=x, gt_y=gt_y, mode='train', **kw_ret)
                    else:
                        loss, network_loss, kld = self.m(x=x, gt_y=gt_y, mode='train', **kw_ret)

                    loss.backward(retain_graph=False)
                    grad = torch.nn.utils.clip_grad_norm_(self.m.parameters(), self.cfg.grad_clip_norm)
                    optim.step()
                    sup_loss += loss.item()
                    sup_cnt += 1
                    if self.cfg.domain == 'personage' or not self.cfg.remove_slot_value:
                        if 'VQVAE' in self.cfg.network:
                            logging.debug(
                                'loss:{} reconloss:{} actloss:{} otherloss:{} actvqloss:{} othervqloss:{} grad:{}'\
                                    .format(loss.item(), recon_loss.item(), act_loss.item(), personality_loss.item(), \
                                            act_vq_loss.item(), personality_vq_loss.item(), grad))
                        elif self.cfg.network == 'controlled_CVAE':
                            logging.debug(
                                'loss:{} reconloss:{} KLD:{} actloss:{}  grad:{}'.format(
                                    loss.item(), recon_loss.item(), KLD.item(), act_loss.item(),  grad))
                        elif self.cfg.VAE or 'simple_CVAE' in self.cfg.network:
                            logging.debug('loss:{} network:{} kld:{} grad:{}'.format(loss.item(), network_loss.item(), kld.item(),grad))
                        else:
                            logging.debug('loss:{} grad:{}'.format(loss.item(), grad))
                    else:
                        if 'VQVAE' in self.cfg.network:
                            logging.debug(
                                'loss:{} reconloss:{} actloss:{}  actvqloss:{} grad:{}' \
                                    .format(loss.item(), recon_loss.item(), act_loss.item(), \
                                            act_vq_loss.item(), grad))
                        elif self.cfg.network == 'controlled_CVAE':
                            logging.debug(
                                'loss:{} reconloss:{} KLD:{} actloss:{}  grad:{}'.format(
                                    loss.item(), recon_loss.item(), KLD.item(), act_loss.item(),grad))
                        elif self.cfg.VAE or 'simple_CVAE' in self.cfg.network:
                            logging.debug(
                                'loss:{} network:{} kld:{} grad:{}'.format(loss.item(), network_loss.item(), kld.item(),
                                                                           grad))
                        else:
                            logging.debug('loss:{} grad:{}'.format(loss.item(), grad))

            epoch_sup_loss = sup_loss / (sup_cnt + 1e-8)
            train_time += time.time() - sw
            logging.info('Traning time: {}'.format(train_time))
            logging.info('avg training loss in epoch %d sup:%f' % (epoch, epoch_sup_loss))

            valid_sup_loss, valid_unsup_loss = self.validate()
            logging.info('validation loss in epoch %d sup:%f unsup:%f' % (epoch, valid_sup_loss, valid_unsup_loss))
            logging.info('time for epoch %d: %f' % (epoch, time.time() - sw))
            valid_loss = valid_sup_loss + valid_unsup_loss

            #metrics = self.eval(data='dev')
            #valid_metrics = metrics
            #logging.info('valid metric %f ' %(valid_metrics))
            if valid_loss <= prev_min_loss:
            #if valid_metrics >= prev_max_metrics:
                self.save_model(epoch)
                prev_min_loss = valid_loss
                #prev_max_metrics = valid_metrics
                early_stop_count = self.cfg.early_stop_count
            else:
                early_stop_count -= 1
                lr *= self.cfg.lr_decay
                if not early_stop_count:
                    break
                logging.info('early stop countdown %d, learning rate %f' % (early_stop_count, lr))

    def run_metrics(self, data='test'):
        if os.path.exists(self.cfg.result_path):
            self.m.eval()
            ev = self.EV(self.cfg)
            res = ev.run_metrics()
            self.m.train()
        else:
            self.eval(data=data)
        return res

    def personality_predictor(self):
        person_cfg = Config('personage')
        person_cfg.init_handler('classification')
        person_cfg.remove_slot_value = self.cfg.remove_slot_value
        person_cfg.cuda = self.cfg.cuda
        person_cfg.cuda_device = self.cfg.cuda_device
        person_cfg.update()
        self.person_m = get_network(person_cfg, self.reader.vocab)
        path = person_cfg.model_path
        if self.cfg.cuda:
            self.person_m = self.person_m.cuda()
            all_state = torch.load(path)
        else:
            all_state = torch.load(path, map_location=torch.device('cpu'))
        self.person_m.load_state_dict(all_state['lstd'])

    def predict(self, data = 'test'):
        if self.cfg.network != 'classification' and self.cfg.domain == 'personage':
            self.personality_predictor()
            self.person_m.eval()
        if 'VQVAE' in self.cfg.network:
            act_idx_dict, personality_idx_dict, value_idx_dict = self.getDist()
        self.m.eval()
        self.reader.result_file = None
        data_iterator = self.reader.mini_batch_iterator(data)
        mode = 'test'
        for batch_num, dial_batch in enumerate(data_iterator):
            for turn_num, turn_batch in enumerate(dial_batch):
                if 'VQVAE' in self.cfg.network:
                    x, gt_y, kw_ret = self._predict_convert_batch(turn_batch, act_idx_dict, personality_idx_dict, value_idx_dict)
                else:
                    x, gt_y, kw_ret = self._predict_convert_batch(turn_batch)
                pred_y, _, _ = self.m(x=x, gt_y=gt_y, mode=mode, **kw_ret)
                if self.cfg.network != 'classification':
                    batch_size = len(turn_batch['id'])
                    batch_gen = []
                    batch_gen_len = []
                    chunk = int(batch_size/len(self.cfg.key_order))
                    for i in range(batch_size):
                        if i%chunk == 0:
                            print (self.cfg.key_order[int(i/chunk)], 'is freeze')
                        word_list = []
                        for t in pred_y[i]:
                            word = self.reader.vocab.decode(t.item())
                            if '<go' not in word:
                                word_list.append(t.item())
                            if word == 'EOS':
                                break
                        if not word_list or word_list[-1] != self.reader.vocab.encode('EOS'):
                            word_list += [self.reader.vocab.encode('EOS')]
                        batch_gen.append(word_list)
                        batch_gen_len.append(len(word_list))
                        print(' '.join([self.reader.vocab.decode(w) for w in word_list]))
                    #text_np = pad_sequences(batch_gen, self.cfg.text_max_ts, padding='post', truncating='post').transpose((1, 0))
                    #person_x = cuda_(Variable(torch.from_numpy(text_np).long()), self.cfg)
                    #person_kw_ret = {}
                    #person_kw_ret['delex_text_len'] = np.asarray(batch_gen_len)
                    #person_pred = self.person_m(x=person_x, gt_y=None, mode='test', **person_kw_ret)
                    #self.reader.wrap_result(turn_batch, pred_y, person_pred)
                else:
                    pass
                    #self.reader.wrap_result(turn_batch, pred_y)
            break#1 loop
        #if self.reader.result_file != None:
        #    self.reader.result_file.close()
        #ev = self.EV(self.cfg)
        #res = ev.run_metrics()
        self.m.train()
        if self.cfg.network != 'classification' and self.cfg.domain == 'personage':
            self.person_m.train()
        return None

    def eval(self, data='test'):
        if self.cfg.network != 'classification' and self.cfg.domain == 'personage':
            self.personality_predictor()
            self.person_m.eval()
        if 'VQVAE' in self.cfg.network:
            act_idx_dict, personality_idx_dict, value_idx_dict = self.getDist()
        self.m.eval()
        self.reader.result_file = None
        data_iterator = self.reader.mini_batch_iterator(data)
        mode = 'test'
        for batch_num, dial_batch in enumerate(data_iterator):
            for turn_num, turn_batch in enumerate(dial_batch):
                if 'VQVAE' in self.cfg.network:
                    x, gt_y, kw_ret = self._convert_batch(turn_batch, act_idx_dict, personality_idx_dict, value_idx_dict)
                    pred_y, _, _ = self.m(x=x, gt_y=gt_y, mode=mode, **kw_ret)
                else:
                    x, gt_y, kw_ret = self._convert_batch(turn_batch)
                    pred_y = self.m(x=x, gt_y=gt_y, mode=mode, **kw_ret)

                if self.cfg.network == 'classification' or self.cfg.domain != 'personage':
                    self.reader.wrap_result(turn_batch, pred_y)
                else:
                    batch_size = len(turn_batch['id'])
                    batch_gen = []
                    batch_gen_len = []
                    for i in range(batch_size):
                        word_list = []
                        for t in pred_y[i]:
                            word = self.reader.vocab.decode(t.item())
                            if '<go' not in word:
                                word_list.append(t.item())
                            if word == 'EOS':
                                break
                        if not word_list or word_list[-1] != self.reader.vocab.encode('EOS'):
                            word_list += [self.reader.vocab.encode('EOS')]
                        #print(word_list)
                        decoded_sentence = self.reader.vocab.sentence_decode(word_list)
                        logging.debug('%s'%(decoded_sentence))
                        batch_gen.append(word_list)
                        if len(word_list) > self.cfg.text_max_ts:
                            batch_gen_len.append(self.cfg.text_max_ts)
                        else:
                            batch_gen_len.append(len(word_list))
                    text_np = pad_sequences(batch_gen, self.cfg.text_max_ts, padding='post', truncating='post').transpose((1, 0))
                    person_x = cuda_(Variable(torch.from_numpy(text_np).long()), self.cfg)
                    person_kw_ret = {}
                    person_kw_ret['delex_text_len'] = np.asarray(batch_gen_len)
                    person_pred = self.person_m(x=person_x, gt_y=None, mode='test', **person_kw_ret)
                    self.reader.wrap_result(turn_batch, pred_y, person_pred)

        if self.reader.result_file != None:
            self.reader.result_file.close()
        ev = self.EV(self.cfg)
        res = ev.run_metrics()
        self.m.train()
        if self.cfg.network != 'classification' and self.cfg.domain == 'personage':
            self.person_m.train()
        return res
    
    
    def validate(self, data='dev'):
        self.m.eval()
        data_iterator = self.reader.mini_batch_iterator(data)
        sup_loss, unsup_loss = 0, 0
        sup_cnt, unsup_cnt = 0, 0
        for dial_batch in data_iterator:
            for turn_num, turn_batch in enumerate(dial_batch):
                x, gt_y, kw_ret = self._convert_batch(turn_batch)
                if 'VQVAE' in self.cfg.network:
                    loss, recon_loss, act_loss, personality_loss, act_vq_loss, personality_vq_loss \
                        = self.m(x=x, gt_y=gt_y, mode='train', **kw_ret)
                elif 'classification' in self.cfg.network:
                    loss = self.m(x=x, gt_y=gt_y, mode='train', **kw_ret)
                elif self.cfg.network == 'controlled_CVAE':
                    loss, recon_loss, KLD, vocab_vq_loss, act_loss, personality_loss \
                        = self.m(x=x, gt_y=gt_y, mode='train', **kw_ret)
                else:
                    loss, network_loss, kld = self.m(x=x, gt_y=gt_y, mode='train', **kw_ret)
                sup_loss += loss.item()
                sup_cnt += 1
                if self.cfg.domain == 'personage':
                    if 'VQVAE' in self.cfg.network:
                        logging.debug(
                            'loss:{} reconloss:{} actloss:{} personalityloss:{} actvqloss:{} personalityvqloss:{} '.format(
                                loss.item(), recon_loss.item(), act_loss.item(), personality_loss.item(),
                                act_vq_loss.item(), personality_vq_loss.item()))
                    elif self.cfg.network == 'controlled_CVAE':
                        logging.debug(
                            'loss:{} reconloss:{} KLD:{} actloss:{} personalityloss:{} vocabvqloss{}'.format(
                                loss.item(), recon_loss.item(), KLD.item(), act_loss.item(), personality_loss.item(),
                                vocab_vq_loss.item()))
                    elif self.cfg.VAE or 'CVAE' in self.cfg.network:
                        logging.debug(
                            'loss:{} network:{} kld:{} '.format(loss.item(), network_loss.item(), kld.item()))
                    else:
                        logging.debug('loss:{} '.format(loss.item()))
                else:
                    if 'VQVAE' in self.cfg.network:
                        logging.debug(
                            'loss:{} reconloss:{} actloss:{} actvqloss:{}'.format(
                                loss.item(), recon_loss.item(), act_loss.item(),
                                act_vq_loss.item()))
                    elif self.cfg.network == 'controlled_CVAE':
                        logging.debug(
                            'loss:{} reconloss:{} KLD:{} actloss:{}'.format(
                                loss.item(), recon_loss.item(), KLD.item(), act_loss.item(),
                                ))
                    elif self.cfg.VAE or 'CVAE' in self.cfg.network:
                        logging.debug(
                            'loss:{} network:{} kld:{} '.format(loss.item(), network_loss.item(), kld.item()))
                    else:
                        logging.debug('loss:{} '.format(loss.item()))

        sup_loss /= (sup_cnt + 1e-8)
        unsup_loss /= (unsup_cnt + 1e-8)
        self.m.train()
        return sup_loss, unsup_loss
    
    
    def getDist(self, data='train'):
        self.m.eval()
        data_iterator = self.reader.mini_batch_iterator(data)
        
        def calDist(idx, encoding):
            idx_dict = {}
            for idx_s, encoding_s in zip(idx, encoding):
                if str(idx_s) in idx_dict:
                    idx_dict[str(idx_s)] += encoding_s
                else:
                    idx_dict[str(idx_s)] = encoding_s
            for k, v in idx_dict.items():
                idx_dict[k] = v/np.sum(v)
            return idx_dict
        
        act_idxs = []
        personality_idxs = []
        slot_value_idxs = {k:[] for k in self.cfg.key_order}
        act_encoding_s = []
        personality_encoding_s = []
        slot_value_encoding_s = {k: [] for k in self.cfg.key_order}
        for dial_batch in data_iterator:
            for turn_num, turn_batch in enumerate(dial_batch):
                x, gt_y, kw_ret = self._convert_batch(turn_batch)   
                act_idx = kw_ret['act_idx'].cpu().data.numpy()
                if self.cfg.domain == 'personage':
                    personality_idx = kw_ret['personality_idx'].cpu().data.numpy()
                #print (x, gt_y, kw_ret)
                act_encoding, personality_encoding, value_encoding = self.m(x=x, gt_y=gt_y, mode='getDist', **kw_ret)
                act_idxs.append(act_idx)
                act_encoding_s.append(act_encoding.cpu().data.numpy())
                if self.cfg.domain == 'personage':
                    personality_idxs.append(personality_idx)
                    personality_encoding_s.append(personality_encoding.cpu().data.numpy())
                if self.cfg.domain == 'e2e' and not self.cfg.remove_slot_value:
                    for i, k in enumerate(self.cfg.key_order):
                        slot_value_idxs[k].append(kw_ret[k].cpu().data.numpy())
                        slot_value_encoding_s[k].append(value_encoding[i].cpu().data.numpy())

        personality_idx_dict = calDist(np.concatenate(personality_idxs, axis=0), np.concatenate(personality_encoding_s, axis=0)) if self.cfg.domain == 'personage' else None
        if self.cfg.domain == 'e2e' and not self.cfg.remove_slot_value:
            value_idx_dict = {}
            for k in self.cfg.key_order:
                value_idx_dict[k] = calDist(np.concatenate(slot_value_idxs[k], axis=0), np.concatenate(slot_value_encoding_s[k], axis=0))
        else:
            value_idx_dict = None
        act_idx_dict = calDist(np.concatenate(act_idxs, axis=0), np.concatenate(act_encoding_s, axis=0))

        self.m.train()
        return act_idx_dict, personality_idx_dict, value_idx_dict

    def save_model(self, epoch, path=None):
        if not path:
            path = self.cfg.model_path
        all_state = {'lstd': self.m.state_dict(),
                     'config': self.cfg.__dict__,
                     'epoch': epoch}
        torch.save(all_state, path)

    def load_model(self, path=None):
        if not path:
            path = self.cfg.model_path
        if self.cfg.cuda:
            all_state = torch.load(path)
        else:
            all_state = torch.load(path, map_location=torch.device('cpu'))
        self.m.load_state_dict(all_state['lstd'])
        self.base_epoch = all_state.get('epoch', 0)

    def freeze_module(self, module):
        for param in module.parameters():
            param.requires_grad = False

    def unfreeze_module(self, module):
        for param in module.parameters():
            param.requires_grad = True

    def load_glove_embedding(self):
        initial_arr = self.m.encoder.embedding.weight.data.cpu().numpy()
        if self.cfg.glove_path == '':
            embedding_arr =  torch.from_numpy(initial_arr)
        else:
            embedding_arr = torch.from_numpy(self.reader.get_glove_matrix(self.reader.vocab, initial_arr))

        self.m.encoder.embedding.weight.data.copy_(embedding_arr)
        self.m.encoder.embedding.weight.requires_grad = self.cfg.emb_trainable
        if 'seq2seq' in self.cfg.network:
            self.m.decoder.emb.weight.data.copy_(embedding_arr)
            self.m.decoder.emb.weight.requires_grad = self.cfg.emb_trainable
        elif 'VQVAE' or 'CVAE' in self.cfg.network:
            self.m.decoder.emb.weight.data.copy_(embedding_arr)
            self.m.decoder.emb.weight.requires_grad = self.cfg.emb_trainable

    def count_params(self):
        module_parameters = filter(lambda p: p.requires_grad, self.m.parameters())
        param_cnt = sum([np.prod(p.size()) for p in module_parameters if p.requires_grad])
        logging.info('total trainable params: %d' % param_cnt)
        logging.info(self.m)


def main(sys_args):
    parser = argparse.ArgumentParser()
    parser.add_argument('-domain')
    parser.add_argument('-network')
    parser.add_argument('-mode')
    parser.add_argument('-cfg', nargs='*')
    args = parser.parse_args(sys_args)

    cfg = Config(args.domain)
    cfg.init_handler(args.network)
    if args.cfg:
        for pair in args.cfg:
            k, v = tuple(pair.split('='))
            dtype = type(getattr(cfg, k))
            if dtype == type(None):
                raise ValueError()
            if dtype is bool:
                v = False if v == 'False' else True
            else:
                v = dtype(v)
            setattr(cfg, k, v)
    cfg.update()
    logging.debug(str(cfg))
    if cfg.cuda:
        torch.cuda.set_device(cfg.cuda_device)
        logging.debug('Device: {}'.format(torch.cuda.current_device()))
    cfg.mode = args.mode

    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    os.environ['PYTHONHASHSEED'] = str(cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    m = Model(cfg)
    m.count_params()

    ret = None
    if args.mode == 'train':
        m.load_glove_embedding()
        m.train()
        m.load_model()
        ret, _ = m.validate()
        m.eval(data='test')
    elif args.mode == 'adjust':
        m.load_model()
        m.train()
        m.load_model()
        m.eval(data='test')
    elif args.mode == 'test':
        m.load_model()
        ret = m.eval(data='test')
    elif args.mode == 'eval':
        m.load_model()
        ret = m.run_metrics(data='test')
    elif args.mode == 'predict':
        print ('start predicting')
        m.load_model()
        m.predict(data = 'test')


    logging.info('return from main function:: validation loss:%s' %(str(ret)))
    print (ret)

    return ret

if __name__ == '__main__':
    main(sys.argv[1:])
