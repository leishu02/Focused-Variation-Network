domain_path = {'multiwoz':{
'fd':'data/dialogue/MULTIWOZ2/',
'fn':'',
'ocean_fn': 'multiwoz_useract_userocean_systemocean.json',
'act_path': 'dialogue_acts.json',
'slot_path': 'ontology.json',
'kbs_path':[
],
'split':['','valListFile.json', 'testListFile.json'],

},

'personage':{#act to text generation
'fd':'data/dialogue/PersonageNLG/',
'train_fn':'stylistic-variation-nlg-corpus.csv',
'test_fn':'personage-nlg-test.csv',
'Lei_dialog_path':'train.json',
'Lei_test_dialog_path':'test.json',
'slot_path':'slot_value.json',
'personality_path':'personality.json',
'split': [0.9, 0.1],
},

'e2e':{#act to text generation
'fd':'data/dialogue/e2e-dataset/',
'train_fn':'trainset.csv',
'test_fn':'testset_w_refs.csv',
'dev_fn':'devset.csv',
'Lei_dialog_path':'train.json',
'Lei_test_dialog_path':'test.json',
'Lei_dev_dialog_path':'dev.json',
'slot_path':'slot_value.json',
'split':None,
},

'fb':{
'fd':'data/personality/',
'fn':'mypersonality_final.csv',
'split':[],
},

'essay':{
'fd':'data/personality/',
'fn':'essays.csv',
'split':[],
},
}