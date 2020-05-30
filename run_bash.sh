#!/usr/bin/env bash

#python model.py -domain personage -network controlled_CVAE -mode eval -cfg cuda=True cuda_device=2 beam_search=False python_path='/home/huxu/anaconda3/envs/p3-torch13/bin/python'
#python model.py -domain personage -network simple_CVAE -mode eval -cfg cuda=True cuda_device=2 beam_search=False python_path='/home/huxu/anaconda3/envs/p3-torch13/bin/python'

#python model.py -domain personage -network simple_VQVAE -mode eval -cfg cuda=True cuda_device=2 beam_search=False codebook_size=512 encoder_layer_num=1 text_max_ts=62 python_path='/home/huxu/anaconda3/envs/p3-torch13/bin/python'

#python model.py -domain personage -network focused_VQVAE -mode eval -cfg cuda=True cuda_device=2 beam_search=False codebook_size=1024 encoder_layer_num=3 text_max_ts=62 python_path='/home/huxu/anaconda3/envs/p3-torch13/bin/python'

#python model.py -domain personage -network controlled_VQVAE -mode eval -cfg cuda=True cuda_device=2 beam_search=False codebook_size=1024 encoder_layer_num=3 text_max_ts=62 commitment_cost=0.95 python_path='/home/huxu/anaconda3/envs/p3-torch13/bin/python'

python model.py -domain personage -network simple_seq2seq -mode eval -cfg cuda=True cuda_device=2 beam_search=False emb_size=64 hidden_size=64 epoch_num=1 remove_slot_value=True glove_path='' python_path='/home/huxu/anaconda3/envs/p3-torch13/bin/python'