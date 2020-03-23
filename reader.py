import numpy as np
from nltk.tokenize import word_tokenize
from collections import defaultdict
import logging
import random
import os
import csv
import json
import pickle


class _ReaderBase(object):
    class LabelSet:
        def __init__(self):
            self._idx2item = {}
            self._item2idx = {}
            self._freq_dict = {}

        def __len__(self):
            return len(self._idx2item)

        def _absolute_add_item(self, item):
            idx = len(self)
            self._idx2item[idx] = item
            self._item2idx[item] = idx

        def add_item(self, item):
            if item not in self._freq_dict:
                self._freq_dict[item] = 0
            self._freq_dict[item] += 1

        def construct(self, limit=None):
            l = sorted(self._freq_dict.keys(), key=lambda x: -self._freq_dict[x])
            print('Actual label size %d' % (len(l) + len(self._idx2item)))
            if limit == None:
                limit = len(l) + len(self._idx2item)
            if len(l) + len(self._idx2item) < limit:
                logging.warning('actual label set smaller than that configured: {}/{}'
                                .format(len(l) + len(self._idx2item), limit))
            for item in l:
                if item not in self._item2idx:
                    idx = len(self._idx2item)
                    self._idx2item[idx] = item
                    self._item2idx[item] = idx
                    if len(self._idx2item) >= limit:
                        break

        def encode(self, item):
            return self._item2idx[item]

        def decode(self, idx):
            return self._idx2item[idx]

    class Vocab(LabelSet):
        def __init__(self, init=True):
            _ReaderBase.LabelSet.__init__(self)
            if init:
                self._absolute_add_item('<pad>')  # 0
                self._absolute_add_item('<go>')  # 1
                self._absolute_add_item('<unk>')  # 2
                self._absolute_add_item('EOS_A')  # 3 eos act
                self._absolute_add_item('EOS_P')  # 4 eos personality
                self._absolute_add_item('EOS')  # 5 eos


        def load_vocab(self, vocab_path):
            f = open(vocab_path, 'rb')
            dic = pickle.load(f)
            self._idx2item = dic['idx2item']
            self._item2idx = dic['item2idx']
            self._freq_dict = dic['freq_dict']
            f.close()

        def save_vocab(self, vocab_path):
            f = open(vocab_path, 'wb')
            dic = {
                'idx2item': self._idx2item,
                'item2idx': self._item2idx,
                'freq_dict': self._freq_dict
            }
            pickle.dump(dic, f)
            f.close()

        def sentence_encode(self, word_list):
            return [self.encode(_) for _ in word_list]

        def sentence_decode(self, index_list, eos=None):
            l = [self.decode(_) for _ in index_list]
            if not eos or eos not in l:
                return ' '.join(l)
            else:
                idx = l.index(eos)
                return ' '.join(l[:idx])

        def nl_decode(self, l, eos=None):
            return [self.sentence_decode(_, eos) + '\n' for _ in l]

        def encode(self, item):
            if item in self._item2idx:
                return self._item2idx[item]
            else:
                return self._item2idx['<unk>']

        def decode(self, idx):
            if idx < len(self):
                return self._idx2item[idx]
            else:
                if self.cfg.vocab_size != None:
                    return 'ITEM_%d' % (idx - self.cfg.vocab_size)

    def __init__(self, cfg):
        self.train, self.dev, self.test = [], [], []
        self.vocab = self.Vocab()
        self.result_file = ''
        self.cfg = cfg

    def _construct(self, *args):
        """
        load data, construct vocab and store them in self.train/dev/test
        :param args:
        :return:
        """
        raise NotImplementedError('This is an abstract class, bro')

    def _bucket_by_turn(self, encoded_data):
        turn_bucket = {}
        for dial in encoded_data:
            turn_len = len(dial)
            if turn_len not in turn_bucket:
                turn_bucket[turn_len] = []
            turn_bucket[turn_len].append(dial)
        del_l = []
        for k in turn_bucket:
            if k >= 5: del_l.append(k)
            logging.debug("bucket %d instance %d" % (k, len(turn_bucket[k])))
        # for k in del_l:
        #    turn_bucket.pop(k)
        return turn_bucket

    def _mark_batch_as_supervised(self, all_batches):
        supervised_num = int(len(all_batches) * self.cfg.spv_proportion / 100)
        for i, batch in enumerate(all_batches):
            for dial in batch:
                for turn in dial:
                    turn['supervised'] = i < supervised_num
        return all_batches

    def _construct_mini_batch(self, data):
        all_batches = []
        batch = []
        for dial in data:
            batch.append(dial)
            if len(batch) == self.cfg.batch_size:
                all_batches.append(batch)
                batch = []

        if len(batch) > 0.5 * self.cfg.batch_size:
            all_batches.append(batch)
        elif len(all_batches):
            all_batches[-1].extend(batch)
        else:
            all_batches.append(batch)
        return all_batches

    def _transpose_batch(self, batch):
        dial_batch = []
        turn_num = len(batch[0])
        for turn in range(turn_num):
            turn_l = {}
            for dial in batch:
                this_turn = dial[turn]
                for k in this_turn:
                    if k not in turn_l:
                        turn_l[k] = []
                    turn_l[k].append(this_turn[k])
            dial_batch.append(turn_l)
        return dial_batch

    def mini_batch_iterator(self, set_name):
        name_to_set = {'train': self.train, 'test': self.test, 'dev': self.dev}
        dial = name_to_set[set_name]
        turn_bucket = self._bucket_by_turn(dial)
        all_batches = []
        for k in turn_bucket:
            batches = self._construct_mini_batch(turn_bucket[k])
            all_batches += batches
        self._mark_batch_as_supervised(all_batches)
        random.shuffle(all_batches)
        for i, batch in enumerate(all_batches):
            yield self._transpose_batch(batch)

    def wrap_result(self, turn_batch, pred_y):
        raise NotImplementedError('This is an abstract class, bro')


class Reader(_ReaderBase):
    def __init__(self, cfg):
        super(Reader, self).__init__(cfg)
        self._construct()
        self.result_file = ''

    def _get_tokenized_data(self, raw_data, construct_vocab, remove_slot_value):

        def delexicalize_text(slot_value, text):
            text_str = ' '.join(text)
            for slot, value in slot_value.items():
                if (slot != 'familyFriendly') and (value.lower() in text_str):
                    text_str = text_str.replace(value.lower(), slot+'Variable')
                if (slot == 'familyFriendly'):
                    for neg in ['', 'n\'t ', 'not ']:
                        for ff in ['kid friendly ', 'family friendly' ]:
                            if neg+ff in text_str:
                                text_str.replace(neg+ff, slot+'Variable')
            return text_str.split(' ')

        tokenized_data = defaultdict(list)
        for dial_id, dial in enumerate(raw_data):
            slot_value = dial['diaact']
            text = [w if 'Variable' in w else w.lower() for w in word_tokenize(dial['text'])]
            delex_text =delexicalize_text(slot_value, text)
            personality = dial['personality'].lower()
            slot_value_seq = []
            for s, v in slot_value.items():
                slot_value_seq += self.slot2phrase[s]
                slot_value_seq += [v.lower()]
                #slot_value_seq += ['EOS_'+s]
            slot_seq = []
            for s, v in slot_value.items():
                slot_seq += self.slot2phrase[s]
                if s != 'familyFriendly':
                    slot_seq += [s+'Variable']
                else:
                    slot_seq += [v.lower()]
                #slot_seq += ['EOS_'+s]
            k = ' '.join(slot_value.keys()) + ' ' + personality
            #if dial_id < 1000:
            tokenized_data[k].append({
                    'id': dial_id,
                    'slot_value':slot_value,
                    'slot_value_size':len(slot_value),
                    'slot_seq': slot_seq + ['EOS_A'],
                    'slot_value_seq': slot_value_seq+['EOS_A'],
                    'personality_seq': [personality, 'EOS_P'],
                    'text_seq': text + ['EOS'],
                    'delex_text_seq': delex_text + ['EOS'],
                    'personality_idx': self.personality2idx[personality],
                    'personality': personality,
            })
            if construct_vocab:
                for word in slot_seq + slot_value_seq + text + delex_text:
                    self.vocab.add_item(word)
        return tokenized_data

    def _get_encoded_data(self, tokenized_data):
        encoded_data = {}   
        for ap, dial in tokenized_data.items():
            encoded_dial = []
            for turn_id, turn in enumerate(dial):
                encoded_dial.append({
                    'id': turn['id'],
                    'go': self.vocab.sentence_encode(['<go'+str(turn_id)+'>']),
                    'slot_value': turn['slot_value'],
                    'slot_value_size': turn['slot_value_size'],
                    'slot_seq': self.vocab.sentence_encode(turn['slot_seq']),
                    'slot_seq_len': len(turn['slot_seq']),
                    'slot_value_seq': self.vocab.sentence_encode(turn['slot_value_seq']),
                    'slot_value_seq_len': len(turn['slot_value_seq']),
                    'personality_seq': self.vocab.sentence_encode(turn['personality_seq']),
                    'personality_seq_len': len(turn['personality_seq']),
                    'text_seq': self.vocab.sentence_encode(turn['text_seq']),
                    'text_seq_len': len(turn['text_seq']),
                    'delex_text_seq': self.vocab.sentence_encode(turn['delex_text_seq']),
                    'delex_text_seq_len': len(turn['delex_text_seq']),
                    'personality_idx': turn['personality_idx'],
                    'personality': turn['personality'],
                    'slot_idx': np.array([1. if s in turn['slot_value'].keys() else 0. for s in self.slot_values.keys()])
                })
                '''
                if len(turn['text_seq']) > max_ts:
                    max_ts = len(turn['text_seq'])
                if len(turn['delex_text_seq']) > max_delex_ts:
                    max_delex_ts = len(turn['delex_text_seq'])
                if len(turn['slot_seq']) > max_slot:
                    max_slot = len(turn['slot_seq'])
                if len(turn['slot_value_seq']) > max_slot_value:
                    max_slot_value = len(turn['slot_value_seq'])
                '''
            encoded_data[ap] = encoded_dial
        #print (max_slot, max_slot_value, max_ts, max_delex_ts)
        return encoded_data

    def _split_data(self, encoded_data, split):
        """
        split data into train/dev/test
        :param encoded_data: list
        :param split: tuple / list
        :return:
        """
        if type(split) is list:
            total = sum(split)
            assert (total == 1.0)

            total = len(encoded_data)
            train_split = int(total * split[0])
            idxes = np.random.permutation(total)
            ap_keys = list(encoded_data.keys())
            act_personality_keys = [ap_keys[i] for i in idxes]
            train_keys = act_personality_keys[:train_split]
            valid_keys = act_personality_keys[train_split:]
            train, dev, test = [], [], []
            for ap_key, dials in encoded_data.items():
                if ap_key in train_keys:
                    train += [[d] for d in dials]
                    #train.append(dials)
                elif ap_key in valid_keys:
                    dev += [[d] for d in dials]
                    #dev.append(dials)
                else:
                    assert()

        else:#split by length of slots
            train, dev, test = [], [], []
            for dial in encoded_data:
                for turn in dial:
                    if turn['slot_value_size'] <= split:
                        train.append(dial)
                    else:
                        test.append(dial)
            dev = train[:2000]
            train = train[2000:]

        print (len(train), len(dev), len(test))
        return train, dev, test


    def _construct(self):
        """
        construct encoded train, dev, test set.
        :param data_json_path:
        :param db_json_path:
        :return:
        """
        raw_data = json.load(open(self.cfg.dialog_path, 'rb'))
        test_data = json.load(open(self.cfg.test_dialog_path, 'rb'))
        self.slot_value_info = json.load(open(self.cfg.slot_path, 'rb'))
        self.slot_values = self.slot_value_info['slot_value']
        self.numslot_value = self.slot_value_info['numslot_value']
        self.slot_cardinality = len(self.slot_values.keys())

        self.personality = json.load(open(self.cfg.personality_path, 'rb'))
        self.personality2idx = {v.lower():i for i, v in enumerate(self.personality)}
        self.idx2personality = {v: k for k, v in self.personality2idx.items()}
        self.slot2phrase = {'name': ['name'], 'food': ['food'], 'customerRating': ['customer', 'rating'],
                            'priceRange': ['price', 'range'], 'area': ['area'], 'eatType': ['eat', 'type'],
                            'familyFriendly': ['family', 'friendly'], 'near': ['near'],
                            }
        construct_vocab = True
        if not os.path.isfile(self.cfg.vocab_path):
            construct_vocab = True
            print('Constructing vocab file...')
        for w in self.personality2idx.keys():
            self.vocab.add_item(w)
        for w in self.slot_values.keys():
            self.vocab.add_item(w+'Variable')
            self.vocab.add_item('EOS_'+w)
        if not self.cfg.remove_slot_value:
            for w in self.slot_values.values():
                self.vocab.add_item(w)

        tokenized_data = self._get_tokenized_data(raw_data, construct_vocab, self.cfg.remove_slot_value)         
        tokenized_test_data = self._get_tokenized_data(test_data, construct_vocab, self.cfg.remove_slot_value)
        
        def findtargetdata(data, p):
            keys = data.keys()
            key_lens = [len(k.split(' ')) for k in keys]
            max_len = max(key_lens)
            for k, l in zip(keys, key_lens):
                if l == max_len and p in k:
                    print (k, p, len(data[k]))
                    return k
        if self.cfg.mode == 'predict':
            k = findtargetdata(tokenized_test_data, 'extravert')
            predict_tokenized_test_data = {}
            predict_tokenized_test_data[k] = tokenized_test_data[k]
                        

        max_variety = max(max([len(v) for v in tokenized_data.values()]), max([len(v) for v in tokenized_test_data.values()]))
        if self.cfg.various_go:
            for i in range(max_variety):
                self.vocab.add_item('<go'+str(i)+'>')

        if construct_vocab:
            self.vocab.construct(self.cfg.vocab_size)
            self.vocab.save_vocab(self.cfg.vocab_path)
        else:
            self.vocab.load_vocab(self.cfg.vocab_path)

        encoded_data = self._get_encoded_data(tokenized_data)
        if self.cfg.mode == 'predict':
            encoded_test_data = self._get_encoded_data(predict_tokenized_test_data)
        else:
            encoded_test_data = self._get_encoded_data(tokenized_test_data)

        self.train, self.dev, _ = self._split_data(encoded_data, self.cfg.split)
        test = []
        for ap_key, dials in encoded_test_data.items():
            test += [[d] for d in dials]
        self.test = test
        random.shuffle(self.train)
        random.shuffle(self.dev)
        random.shuffle(self.test)

    def wrap_result(self, turn_batch, pred_y, person_pred = None):
        field = ['id', 'slot_value', 'slot_seq', 'slot_value_seq', 'personality', 'delex_text', 'text',
                 'pred_delex_text', 'pred_text', 'delex_text_tokens', 'text_tokens',
                 'pred_delex_text_tokens', 'pred_text_tokens','pred_personality']
        results = []
        batch_size = len(turn_batch['id'])
        for i in range(batch_size):
            entry = {}
            entry['id'] = turn_batch['id'][i]
            entry['slot_value'] = json.dumps(turn_batch['slot_value'][i])
            entry['personality'] = turn_batch['personality'][i]
            entry['slot_seq'] = json.dumps([self.vocab.decode(t) for t in turn_batch['slot_seq'][i]])
            entry['slot_value_seq'] = json.dumps([self.vocab.decode(t) for t in turn_batch['slot_value_seq'][i]])
            entry['text_tokens'] = json.dumps([self.vocab.decode(t) for t in turn_batch['text_seq'][i]])
            entry['text'] = ' '.join([self.vocab.decode(t) for t in turn_batch['text_seq'][i]])
            entry['delex_text_tokens'] = json.dumps([self.vocab.decode(t) for t in turn_batch['delex_text_seq'][i]])
            entry['delex_text'] = ' '.join([self.vocab.decode(t) for t in turn_batch['delex_text_seq'][i]])
            if person_pred is not None:
                idx = np.argmax(person_pred[i])
                entry['pred_personality'] = self.idx2personality[idx]

            if self.cfg.network == 'classification':
                idx = np.argmax(pred_y[i])
                entry['pred_personality'] = self.idx2personality[idx]
            elif 'seq2seq' in self.cfg.network:
                word_list = []
                for t in pred_y[i]:
                    word = self.vocab.decode(t.item())
                    if '<go' not in word :
                        word_list.append(word)
                    if word == 'EOS':
                        break
                if self.cfg.remove_slot_value == True:
                    entry['pred_delex_text_tokens'] = json.dumps(word_list)
                    entry['pred_delex_text'] = ' '.join(word_list)
                    entry['pred_text_tokens'] = json.dumps([])
                    entry['pred_text'] = ''
                else:
                    entry['pred_text_tokens'] = json.dumps(word_list)
                    entry['pred_text'] = ' '.join(word_list)
                    entry['pred_delex_text_tokens'] = json.dumps([])
                    entry['pred_delex_text'] = ''
            elif 'VQVAE' in self.cfg.network:
                word_list = []
                for t in pred_y[0][i]:
                    word = self.vocab.decode(t.item())
                    if '<go' not in word:
                        word_list.append(word)
                    if word == 'EOS':
                        break
                if self.cfg.remove_slot_value == True:
                    entry['pred_delex_text_tokens'] = json.dumps(word_list)
                    entry['pred_delex_text'] = ' '.join(word_list)
                    entry['pred_text_tokens'] = json.dumps([])
                    entry['pred_text'] = ''
                else:
                    entry['pred_text_tokens'] = json.dumps(word_list)
                    entry['pred_text'] = ' '.join(word_list)
                    entry['pred_delex_text_tokens'] = json.dumps([])
                    entry['pred_delex_text'] = ''
            '''
            for key in turn_batch:
                if key in field:
                    entry[key] = json.dumps(turn_batch[key][i])
                else:
                    pass #ndarray
            '''
            results.append(entry)
        write_header = False
        if not self.result_file:
            self.result_file = open(self.cfg.result_path, 'w')
            self.result_file.write(str(self.cfg))
            write_header = True

        writer = csv.DictWriter(self.result_file, fieldnames=field)
        if write_header:
            self.result_file.write('START_CSV_SECTION\n')
            writer.writeheader()
        writer.writerows(results)
        return results

    def get_glove_matrix(self, vocab, initial_embedding_np):
        """
        return a glove embedding matrix
        :param self:
        :param glove_file:
        :param initial_embedding_np:
        :return: np array of [V,E]
        """
        if os.path.exists(self.cfg.vocab_emb):
            vec_array = np.load(self.cfg.vocab_emb)
            old_avg = np.average(vec_array)
            old_std = np.std(vec_array)
            logging.info('embedding.  mean: %f  std %f' % (old_avg, old_std))
            return vec_array
        else:
            ef = open(self.cfg.glove_path, 'r')
            cnt = 0
            vec_array = initial_embedding_np
            old_avg = np.average(vec_array)
            old_std = np.std(vec_array)
            vec_array = vec_array.astype(np.float32)
            new_avg, new_std = 0, 0

            for line in ef.readlines():
                line = line.strip().split(' ')
                word, vec = line[0], line[1:]
                vec = np.array(vec, np.float32)
                word_idx = vocab.encode(word)
                if word.lower() in ['unk', '<unk>'] or word_idx != vocab.encode('<unk>'):
                    cnt += 1
                    vec_array[word_idx] = vec
                    new_avg += np.average(vec)
                    new_std += np.std(vec)
            new_avg /= cnt
            new_std /= cnt
            ef.close()
            logging.info('%d known embedding. old mean: %f new mean %f, old std %f new std %f' % (
                cnt, old_avg, new_avg, old_std, new_std))
            np.save(self.cfg.vocab_emb, vec_array)
            return vec_array


def pad_sequences(sequences, maxlen=None, dtype='int32',
                  padding='pre', truncating='pre', value=0.):
    if not hasattr(sequences, '__len__'):
        raise ValueError('`sequences` must be iterable.')
    lengths = []
    for x in sequences:
        if not hasattr(x, '__len__'):
            raise ValueError('`sequences` must be a list of iterables. '
                             'Found non-iterable: ' + str(x))
        lengths.append(len(x))

    num_samples = len(sequences)
    seq_maxlen = np.max(lengths)
    if maxlen is not None:
        maxlen = min(seq_maxlen, maxlen)
    else:
        maxlen = seq_maxlen
    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    x = (np.ones((num_samples, maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if not len(s):
            continue  # empty list/array was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x

