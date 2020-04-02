#! /usr/bin/env python
# coding=utf-8
import ailabs.tlogger as tlogger  # TODO pycharm warning - No module
import logging
import sys
import model

train_args = ['-domain', 'personage', '-network', 'controlled_VQVAE', '-mode', 'train', '-cfg', 'cuda=True']
test_args = ['-domain', 'personage', '-network', 'controlled_VQVAE', '-mode', 'test', '-cfg', 'cuda=True']


def wrapper(
        lr=0.001,
        lr_decay=1.0,
        batch_size=128,
        dropout_rate=0.0,
        epoch_num=100,
        early_stop_count=30,
        grad_clip_norm=1.0,
        emb_trainable=True,
        encoder_layer_num=1,
        codebook_size=512,
        decoder_network='LSTM',
        teacher_force=50,
        commitment_cost=0.25,
        text_max_ts=62,
        beam_search=True,
        beam_size=10,
        **kwargs
):
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.DEBUG,
        format='%(message)s'
    )

    logging.info('Starting wrapper with parameters:')
    for parameter, value in locals().items():
        logging.info("  {}: {}".format(parameter, value))
    logging.info("")

    logging.info('Running training')
    hyperparameters = [
        'lr=' + str(lr),
        'lr_decay=' + str(lr_decay),
        'batch_size=' + str(batch_size),
        'dropout_rate=' + str(dropout_rate),
        'epoch_num=' + str(epoch_num),
        'early_stop_count=' + str(early_stop_count),
        'grad_clip_norm=' + str(grad_clip_norm),
        'emb_trainable=' + str(emb_trainable),
        'encoder_layer_num=' + str(encoder_layer_num),
        'codebook_size=' + str(codebook_size),
        'decoder_network=' + decoder_network,
        'teacher_force=' + str(teacher_force),
        'commitment_cost=' + str(commitment_cost),
        'text_max_ts=' + str(text_max_ts),
        'beam_search=' + str(beam_search),
    ]
    if beam_search:
        for beam_size in range(5, 30, 5):
            hyperparameters += ['beam_size=' + str(beam_size)]
    else:
        hyperparameters += ['beam_size=10']

    validation_loss = model.main(train_args + hyperparameters)

    # logging.info("Running evaluation")
    # model.main(test_args + best_hyperparameters)

    logging.info('Saving output to tlogger')
    tlogger.record_tabular('validation_loss', validation_loss)
    tlogger.dump_tabular()

    return validation_loss
