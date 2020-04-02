import csv
import json
import multiprocessing
import os
import subprocess
from collections import defaultdict
from multiprocessing import Pool
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
from io import StringIO


def work(cmd):
    return subprocess.check_output(cmd.split(' '))

def match(gt_list, pred_list):
    gt_matched = [0 for i in range(len(gt_list))]
    pred_matched = [0 for i in range(len(pred_list))]
    for i, g in enumerate(gt_list):
        for j, p in enumerate(pred_list):
            if g == p:
                gt_matched[i] = 1
                pred_matched[j] = 1
    return gt_matched, pred_matched

class Evaluator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.file = open(cfg.result_path, 'r', encoding="utf8")
        self.meta = []
        self.metric_dict = {}
        self.filename = cfg.result_path.split('/')[-1]
        dump_dir = './sheets/' + self.filename.replace('.csv', '.report.txt')
        self.dump_file = open(dump_dir, 'w', encoding="utf8")

    def run_metrics(self):
        data = self.read_result_data()
        personality_labels = ['agreeable', 'disagreeable', 'conscientiousness', 'unconscientiousness', 'extravert']
        full_report = self.classification(data, personality_labels)
        for idx, l in enumerate(personality_labels):
            self.metric_dict[l + '_precision'] = full_report[0][idx]
            self.metric_dict[l + '_recall'] = full_report[1][idx]
            self.metric_dict[l + '_fscore'] = full_report[2][idx]
        self.metric_dict['macro_precision'] = np.mean(full_report[0])
        self.metric_dict['macro_recall'] = np.mean(full_report[1])
        self.metric_dict['macro_fscore'] = np.mean(full_report[2])
        if self.cfg.network == "classification":
            self.dump()
            return self.metric_dict['macro_fscore']
        else:
            if self.cfg.python_path != '':
                e2e_result = self.e2e_metric(data)
                for k, v in e2e_result.items():
                    self.metric_dict[k] = v
            if self.cfg.remove_slot_value:
                success_p, success_r, success_f = self.success_f1_metric(data)
                self.metric_dict['success_precision'] = success_p
                self.metric_dict['success_recall'] = success_r
                self.metric_dict['success_fscore'] = success_f
            self.dump()
            return None #sum(e2e_result.values())

    def classification(self, data, labels):
        pred, truth = [], []
        for row in data:
            pred.append(row['pred_personality'])
            truth.append(row['personality'])
        full_report = precision_recall_fscore_support(truth, pred, average=None, labels = labels)
        return full_report


    def e2e_metric(self, data):
        gen, truth_ap = [], []
        truth_ap_dict = defaultdict(list)
        for row in data:
            ap_key = ' '.join(json.loads(row['slot_value']).keys())+' '+row['personality']
            truth_ap.append(ap_key)
            if self.cfg.remove_slot_value:
                gen.append(row['pred_delex_text'])
                truth_ap_dict[ap_key].append(row['delex_text'])
            else:
                gen.append(row['pred_text'])
                truth_ap_dict[ap_key].append(row['text'])
        cwd = os.getcwd()
        core = multiprocessing.cpu_count()
        chunk_size = int(len(gen)/core)
        chunk_size_list = []
        chunk = [0]
        cmds = []
        for i in range(core):
            if i < core-1:
                chunk.append((i+1)*chunk_size)
                chunk_size_list.append(chunk_size)
            else:
                chunk_size_list.append(len(gen)-chunk_size*(core-1))
                chunk.append(len(gen))
            pred_dir = cwd+'/sheets/' + self.filename.replace('.csv', '_'+str(i)+'_pred.txt')
            truth_dir = cwd+'/sheets/' + self.filename.replace('.csv', '_'+str(i)+'_truth.txt')
            cmds.append(' '.join([self.cfg.python_path, cwd+'/e2e-metrics/measure_scores.py', truth_dir, pred_dir]))
            with open(pred_dir, 'w', encoding="utf8")as outfile:
                for d in gen[chunk[i]:chunk[i+1]]:
                    outfile.write(d+'\n')
            with open(truth_dir, 'w', encoding="utf8")as outfile:
                for ap in truth_ap[chunk[i]:chunk[i+1]]:
                    for d in truth_ap_dict[ap]:
                        outfile.write(d+'\n')
                    outfile.write('\n')
        with Pool(core) as p:
            ret = p.map(work, cmds)

        ret_dict = defaultdict(list)
        for r in ret:
            ret_line = r.decode("utf-8").split('\n')
            for line in ret_line[2:]:
                if ':' in line:
                    tokens = line.split(':')
                    ret_dict[tokens[0]].append(float(tokens[1]))
        ret_dict_merge = {}
        for k, v in ret_dict.items():
            total_score = 0
            for s, w in zip(v, chunk_size_list):
                total_score += s*w
            score = total_score/len(gen)
            ret_dict_merge[k] = score

        return ret_dict_merge

    def retrieve_slot(self, delex_text):
        V_set = {'nameVariable', 'foodVariable', 'nearVariable', 'eatTypeVariable', 'areaVariable',
             'priceRangeVariable', 'customerRatingVariable'}
        output = set(delex_text).intersection(V_set)
        return list(output)

    def success_f1_metric(self, data):#check if agent inform user's request
        dials = self.pack_dial(data)
        tp, fp, fn = 0, 0, 0
        for dial_id in dials:
            truth_pair, gen_pair = [], []
            dial = dials[dial_id]
            for turn_num, turn in enumerate(dial):
                gt_y = json.loads(turn['delex_text_tokens'])
                pred_y = json.loads(turn['pred_delex_text_tokens'])
                truth_pair += self.retrieve_slot(gt_y)
                gen_pair += self.retrieve_slot(pred_y)
            truth_pair = set(truth_pair)
            gen_pair = set(gen_pair)
            for req in gen_pair:
                if req in truth_pair:
                    tp += 1
                else:
                    fp += 1
            for req in truth_pair:
                if req not in gen_pair:
                    fn += 1
        precision, recall = tp / (tp + fp + 1e-8), tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        return precision, recall, f1

    def read_result_data(self):
        while True:
            line = self.file.readline()
            if 'START_CSV_SECTION' in line:
                break
        self.meta.append(line)
        reader = csv.DictReader(self.file)
        data = [_ for _ in reader]
        return data

    def pack_dial(self, data):
        dials = {}
        for turn in data:
            dial_id = int(turn['id'])
            if dial_id not in dials:
                dials[dial_id] = []
            dials[dial_id].append(turn)
        return dials

    def dump(self):
        self.dump_file.write('START_REPORT_SECTION\n')
        for k, v in self.metric_dict.items():
            self.dump_file.write('{}\t{}\n'.format(k, v))

