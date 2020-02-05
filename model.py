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


class Model:
    def __init__(self, cfg):
        self.reader = Reader(cfg)
        self.m = get_network(cfg, self.reader.vocab)
        self.EV = Evaluator  # evaluator class
        if cfg.cuda:
            self.m = self.m.cuda()
        self.base_epoch = -1
        self.cfg = cfg


    def _convert_batch(self, py_batch, act_idx_dict = None, personality_idx_dict = None):
        kw_ret = {}
        x = None
        gt_y = None
        batch_size = len(py_batch['slot_seq'])
        slot_np = pad_sequences(py_batch['slot_seq'], self.cfg.slot_max_ts, padding='post',truncating='post').transpose((1, 0))
        personality_np = pad_sequences(py_batch['personality_seq'], self.cfg.personality_size, padding='post',truncating='post').transpose((1, 0))
        slot_value_np = pad_sequences(py_batch['slot_value_seq'], self.cfg.slot_max_ts, padding='post',truncating='post').transpose((1, 0))  # (seqlen, batchsize)
        text_np = pad_sequences(py_batch['text_seq'], self.cfg.text_max_ts, padding='post',truncating='post').transpose((1, 0))
        delex_text_np = pad_sequences(py_batch['delex_text_seq'], self.cfg.text_max_ts, padding='post',truncating='post').transpose((1, 0))
        slot_len = np.array(py_batch['slot_seq_len'])
        personality_len = np.array(py_batch['personality_seq_len'])
        slot_value_len = np.array(py_batch['slot_value_seq_len'])
        text_len = np.array(py_batch['text_seq_len'])
        delex_text_len = np.array(py_batch['delex_text_seq_len'])
        go_np = pad_sequences(py_batch['go'], 1, padding='post',truncating='post').transpose((1, 0))
        go = cuda_(torch.autograd.Variable(torch.from_numpy(go_np).long()), self.cfg)
        personality_idx = cuda_(Variable(torch.from_numpy(np.asarray(py_batch['personality_idx'])).long()), self.cfg)
        act_idx = cuda_(Variable(torch.from_numpy(np.asarray(py_batch['slot_idx'])).float()), self.cfg)

        kw_ret['slot_np'] = slot_np  # seqlen, batchsize
        kw_ret['slot_value_np'] = slot_value_np  # seqlen, batchsize
        kw_ret['personality_np'] = personality_np  # seqlen, batchsize
        kw_ret['personality_seq'] = cuda_(Variable(torch.from_numpy(personality_np).long()), self.cfg)  # seqlen, batchsize
        kw_ret['text_np'] = text_np  # seqlen, batchsize
        kw_ret['delex_text_np'] = delex_text_np  # seqlen, batchsize
        kw_ret['slot_len'] = slot_len  # batchsize
        kw_ret['slot_value_len'] = slot_value_len  # batchsize
        kw_ret['personality_len'] = personality_len  # batchsize
        kw_ret['text_len'] = text_len  # batchsize
        kw_ret['delex_text_len'] = delex_text_len  # batchsize
        kw_ret['go_np'] = go_np
        kw_ret['go'] = go
        kw_ret['personality_idx'] = personality_idx
        kw_ret['act_idx'] = act_idx
        
        if act_idx_dict and personality_idx_dict:
            act_encoding = []
            personality_encoding = []
            for i in py_batch['slot_idx']:
                dist = act_idx_dict[str(i)]
                sample = np.random.choice(self.cfg.codebook_size, 1, p=dist)
                act_encoding.append(sample)
            for i in py_batch['personality_idx']:
                dist = personality_idx_dict[str(i)]
                sample = np.random.choice(self.cfg.codebook_size, 1, p=dist)
                personality_encoding.append(sample)
            kw_ret['act_sample_idx'] = cuda_(Variable(torch.from_numpy(np.asarray(act_encoding))).long(), self.cfg)
            kw_ret['personality_sample_idx'] = cuda_(Variable(torch.from_numpy(np.asarray(personality_encoding))).long(), self.cfg)

        if self.cfg.network == 'classification':
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
        elif 'VQVAE' in self.cfg.network:
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
                    else:
                        loss, network_loss, kld = self.m(x=x, gt_y=gt_y, mode='train', **kw_ret)
                    loss.backward(retain_graph=False)
                    grad = torch.nn.utils.clip_grad_norm_(self.m.parameters(), self.cfg.grad_clip_norm)
                    optim.step()
                    sup_loss += loss.item()
                    sup_cnt += 1
                    if 'VQVAE' in self.cfg.network:
                        logging.debug(
                            'loss:{} reconloss:{} actloss:{} personalityloss:{} actvqloss:{} personalityvqloss:{} grad:{}'.format(
                                loss.item(), recon_loss.item(), act_loss.item(), personality_loss.item(), act_vq_loss.item(), personality_vq_loss.item(), grad))
                    elif self.cfg.VAE:
                        logging.debug('loss:{} network:{} kld:{} grad:{}'.format(loss.item(), network_loss.item(), kld.item(),grad))
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

    def eval(self, data='test'):
        act_idx_dict, personality_idx_dict = self.getDist()
        self.m.eval()
        self.reader.result_file = None
        data_iterator = self.reader.mini_batch_iterator(data)
        mode = 'test'
        for batch_num, dial_batch in enumerate(data_iterator):
            for turn_num, turn_batch in enumerate(dial_batch):
                x, gt_y, kw_ret = self._convert_batch(turn_batch, act_idx_dict, personality_idx_dict)
                pred_y = self.m(x=x, gt_y=gt_y, mode=mode, **kw_ret)
                self.reader.wrap_result(turn_batch, pred_y)
        if self.reader.result_file != None:
            self.reader.result_file.close()
        ev = self.EV(self.cfg)
        res = ev.run_metrics()
        self.m.train()
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
                else:
                    loss, network_loss, kld = self.m(x=x, gt_y=gt_y, mode='train', **kw_ret)
                sup_loss += loss.item()
                sup_cnt += 1
                if 'VQVAE' in self.cfg.network:
                    logging.debug(
                        'loss:{} reconloss:{} actloss:{} personalityloss:{} actvqloss:{} personalityvqloss:{} '.format(
                            loss.item(), recon_loss.item(), act_loss.item(), personality_loss.item(),
                            act_vq_loss.item(), personality_vq_loss.item()))
                elif self.cfg.VAE:
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
        act_encoding_s = []
        personality_encoding_s = []
        for dial_batch in data_iterator:
            for turn_num, turn_batch in enumerate(dial_batch):
                x, gt_y, kw_ret = self._convert_batch(turn_batch)   
                act_idx = kw_ret['act_idx'].cpu().data.numpy()
                personality_idx = kw_ret['personality_idx'].cpu().data.numpy()
                act_encoding, personality_encoding = self.m(x=x, gt_y=gt_y, mode='getDist', **kw_ret)
                act_idxs.append(act_idx)
                personality_idxs.append(personality_idx)
                act_encoding_s.append(act_encoding.cpu().data.numpy())
                personality_encoding_s.append(personality_encoding.cpu().data.numpy())
        
        personality_idx_dict = calDist(np.concatenate(personality_idxs, axis=0), np.concatenate(personality_encoding_s, axis=0))
        act_idx_dict = calDist(np.concatenate(act_idxs, axis=0), np.concatenate(act_encoding_s, axis=0))

        self.m.train()
        return act_idx_dict, personality_idx_dict

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
        all_state = torch.load(path)
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
        embedding_arr = torch.from_numpy(self.reader.get_glove_matrix(self.reader.vocab, initial_arr))

        self.m.encoder.embedding.weight.data.copy_(embedding_arr)
        self.m.encoder.embedding.weight.requires_grad = self.cfg.emb_trainable
        if 'seq2seq' in self.cfg.network:
            self.m.decoder.emb.weight.data.copy_(embedding_arr)
            self.m.decoder.emb.weight.requires_grad = self.cfg.emb_trainable
        elif 'VQVAE' in self.cfg.network:
            self.m.decoder.emb.weight.data.copy_(embedding_arr)
            self.m.decoder.emb.weight.requires_grad = self.cfg.emb_trainable

    def count_params(self):
        module_parameters = filter(lambda p: p.requires_grad, self.m.parameters())
        param_cnt = sum([np.prod(p.size()) for p in module_parameters if p.requires_grad])
        print('total trainable params: %d' % param_cnt)
        print(self.m)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-domain')
    parser.add_argument('-network')
    parser.add_argument('-mode')
    parser.add_argument('-cfg', nargs='*')
    args = parser.parse_args()

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
    if args.mode == 'train':
        m.load_glove_embedding()
        m.train()
        m.load_model()
        m.eval(data='test')
    elif args.mode == 'adjust':
        m.load_model()
        m.train()
        m.load_model()
        m.eval(data='test')
    elif args.mode == 'test':
        m.load_model()
        m.eval(data='test')
    elif args.mode == 'eval':
        m.load_model()
        m.run_metrics(data='test')
    elif args.mode == 'predict':
        m.load_model()
    elif args.mode == 'rl':
        m.load_model()



if __name__ == '__main__':
    main()
