import csv
import os
import domain_config
import json
from collections import defaultdict

def parse_mr(text_acts):
    slot_value = dict()
    for text in text_acts:
        if text != '':
            left_idx = text.find('[')
            right_idx = text.find(']')
            slot = text[:left_idx]
            value = text[left_idx + 1:right_idx]
            slot_value[slot] = value
    return slot_value

domain = 'e2e' #'personage'
fd = domain_config.domain_path[domain]['fd']

personality_dict = defaultdict(int)
act_slot_dict = defaultdict(int)
act_slot_value_dict = defaultdict(int)
slot_value_map = defaultdict(set)
numslot_value_dict = defaultdict(set)

for name in ['train', 'dev', 'test']:
    fn = domain_config.domain_path[domain][name+'_fn']

    examples = []
    count = defaultdict(int)


    with open(fd + fn, 'r') as tsvfile:
        reader = csv.reader(tsvfile, delimiter=',')
        for l, row in enumerate(reader):
            if l > 0:
                mr = row[0]
                slot_value = parse_mr(mr.split(', '))
                act_slot_dict[str(slot_value.keys())] += 1
                act_slot_value_dict[str(slot_value.items())] += 1
                for k, v in slot_value.items():
                    slot_value_map[k].add(v)
                numslot_value_dict[len(slot_value.keys())].add(str(slot_value.keys()))
                count[len(slot_value.keys())] += 1
                text = row[1]
                if domain == 'personage':
                    personality = row[2]
                    personality_dict[personality] += 1
                    examples.append({'diaact': slot_value, 'text': text, 'personality': personality})
                else:
                    examples.append({'diaact': slot_value, 'text': text})
    print(len(examples))
    print (count)
    print ({k:v/len(examples) for k, v in count.items()})
    with open(fd+name+'.json', 'w') as outfile:
        json.dump(examples, outfile)

slot_value_map = {k:list(v) for k, v in slot_value_map.items()}
numslot_value_dict = {k:list(v) for k, v in numslot_value_dict.items()}

with open(fd+'slot_value.json', 'w') as outfile:
    json.dump({'slot_value': slot_value_map, 'numslot_value': numslot_value_dict, 'distinct_slots': act_slot_value_dict,\
               'distinct_slot_values': act_slot_value_dict}, outfile)

if domain == 'personage':
    print (personality_dict)
    with open(fd + 'personality.json', 'w') as outfile:
        json.dump(list(personality_dict.keys()), outfile)

print (len(act_slot_dict))
print (len(act_slot_value_dict))
print (len(slot_value_map))
for k, v in slot_value_map.items():
    print (k, v)

for k, v in numslot_value_dict.items():
    print (k, v)



