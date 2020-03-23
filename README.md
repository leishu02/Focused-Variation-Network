# Setup

Install requirements:
```
pip isntall -r requirements.txt
```
Install requirements for e2e metrics:
```
pip install -r e2e_metrics/requirements
```

# Execute

## Training
### Train a personality classifier
Trained personality classifier (change on preprocessing). If there is any change in preprocessing, please re-train classifier again before training a generation model
```
python model.py -domain personage -network classifier -mode [train | adjust | test]

```

### Train a generation model
The file `config.py` contains the configuration parameters.
Example command:
```
python model.py -domain personage -network controlled_VQVAE  -mode train -cfg 
cuda=True 
lr = 0.001 #learning rate
lr_decay = 1.0 #learning rate decay on epoch
batch_size=128 #batch size
dropout_rate=0.0 #dropout rate
epoch_num=100 #total number of epoch
early_stop_count=30 #early stop patience
grad_clip_norm=1.0 # grad clip norm
emb_trainable=True # whether embedding layers are trainable or not
encoder_layer_num=1 # **the number of (RNN) layers of encoder
codebook_size=512 #**codebook size
decoder_network='LSTM'#the architecture for decoder network ('LSTM, 'GRU')
teacher_force=50 # teacher force rate in decoding network, range is (1,100)
commitment_cost=0.25 #**commitment cost in VQ loss, range is [0, 1]
text_max_ts = 62 # maximum text steps during inference (text generation), range is greater than 62
beam_search=True # use beam search or not
beam_size=10 # beam size, range is greater than 1
```

## Testing
### Test a generation model
Results(prediction) are saved under 'results/'. Scores are saved under 'sheets/'
```
python model.py -domain personage -network controlled_VQVAE  -mode test -cfg 
cuda=True 
lr = 0.001 #learning rate
lr_decay = 1.0 #learning rate decay on epoch
batch_size=128 #batch size
dropout_rate=0.0 #dropout rate
epoch_num=100 #total number of epoch
early_stop_count=30 #early stop patience
grad_clip_norm=1.0 # grad clip norm
emb_trainable=True # whether embedding layers are trainable or not
encoder_layer_num=1 # the number of (RNN) layers of encoder
codebook_size=512 #codebook size
decoder_network='LSTM'#the architecture for decoder network ('LSTM, 'GRU')
teacher_force=50 # teacher force rate in decoding network, range is (1,100)
commitment_cost=0.25 #commitment cost in VQ loss, range is [0, 1]
text_max_ts = 62 # maximum text steps during inference (text generation), range is greater than 62
beam_search=True # use beam search or not
beam_size=10 # beam size, range is greater than 1
```


# TODO

Lei: add comments on hyperparameters
