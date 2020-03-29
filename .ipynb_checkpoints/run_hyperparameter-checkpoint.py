import operator
from numpy import arange
import model


train_args = ['-domain', 'personage', '-network', 'controlled_VQVAE', '-mode', 'train', '-cfg', 'cuda=True', 'python_path=<fill in your python path>']
test_args = ['-domain', 'personage', '-network', 'controlled_VQVAE', '-mode', 'test', '-cfg', 'cuda=True', 'python_path=<fill in your python path>']
hyperparameters_log = dict()

for lr in [0.001]:
    for lr_decay in [1.0]:
        for batch_size in [128]:
            for dropout_rate in arange(0.0, 1.0, 0.1):
                for epoch_num in [100]:
                    for early_stop_count in [30]:
                        for grad_clip_norm in [1.0]:
                            for emb_trainable in ['True', 'False']:
                                for encoder_layer_num in range(1, 5):
                                    for codebook_size in [512, 1024, 2048]:
                                        for decoder_network in ['LSTM', 'GRU']: #may no need to try 'GRU' in all hyperparameter
                                            for teacher_force in range(0, 99, 10):
                                                for commitment_cost in arange(0.0, 1.0, 0.05):
                                                    for text_max_ts in range(62, 102, 4):
                                                        for beam_search in ['True', 'False']:
                                                            hyperparameter = [
                                                                'lr=' + str(lr),
                                                                'lr_decay=' + str(lr_decay),
                                                                'batch_size=' + str(batch_size),
                                                                'dropout_rate=' + str(dropout_rate),
                                                                'epoch_num=' + str(epoch_num),
                                                                'early_stop_count=' + str(early_stop_count),
                                                                'grad_clip_norm=' + str(grad_clip_norm),
                                                                'emb_trainable=' + emb_trainable,
                                                                'encoder_layer_num=' + str(encoder_layer_num),
                                                                'codebook_size=' + str(codebook_size),
                                                                'decoder_network=' + decoder_network,
                                                                'teacher_force=' + str(teacher_force),
                                                                'commitment_cost=' + str(commitment_cost),
                                                                'text_max_ts=' + str(text_max_ts),
                                                                'beam_search=' + beam_search,
                                                            ]
                                                            if beam_search:
                                                                for beam_size in range(5, 30, 5):
                                                                    hyperparameter += ['beam_size='+str(beam_size)]
                                                            else:
                                                                    hyperparameter += ['beam_size=10']

                                                            ret = model.main(train_args + hyperparameter)
                                                            hyperparameters_log[tuple(hyperparameter), ret]


sorted_hypermarameters = sorted(hyperparameters_log.items(), key=operator.itemgetter(1))
best_hyperparameters = sorted_hypermarameters[0]
model.main(test_args + best_hyperparameters)

