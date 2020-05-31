import logging
import time
import domain_config


class Config:
    def __init__(self, domain):
        self._init_logging_handler()
        self.network = None
        self.cuda_device = 0
        self.seed = 0
        self.spv_proportion = 100
        self.truncated = False

        self.personality_size = 0

        self.domain = domain
        fd = domain_config.domain_path[domain]['fd']
        self.dialog_path = fd+domain_config.domain_path[domain]['Lei_dialog_path']
        self.test_dialog_path =fd+domain_config.domain_path[domain]['Lei_test_dialog_path']
        if self.domain == 'e2e':
            self.dev_dialog_path = fd+domain_config.domain_path[domain]['Lei_test_dialog_path']
        self.slot_path = fd+domain_config.domain_path[domain]['slot_path']
        if self.domain == 'personage':
            self.personality_path = fd+domain_config.domain_path[domain]['personality_path']
            self.personality_size = 5
            self.slot_max_ts = 29
            self.text_max_ts = 62
        elif self.domain == 'e2e':
            self.slot_max_ts = 21
            self.text_max_ts = 82

        self.split = domain_config.domain_path[domain]['split']
        self.python_path = ''

    def init_handler(self, network_type):
        self.network = network_type
        init_method = {
            'classification': self._classification_init,
            'copy_seq2seq': self._copy_seq2seq_init,
            'simple_seq2seq': self._simple_seq2seq_init,
            'simple_VQVAE': self._simple_VQVAE_init,
            'controlled_VQVAE': self._controlled_VQVAE_init,
            'focused_VQVAE': self._focused_VQVAE_init,
            'simple_CVAE': self._CVAE_init,
            'controlled_CVAE': self._controlled_CVAE_init,
        }
        init_method[network_type]()


    def update(self):
        update_method = {
            'classification': self._classification_update,
            'copy_seq2seq': self._copy_seq2seq_update,
            'simple_seq2seq': self._simple_seq2seq_update,
            'simple_VQVAE': self._simple_VQVAE_update,
            'controlled_VQVAE': self._controlled_VQVAE_update,
            'focused_VQVAE': self._focused_VQVAE_update,
            'simple_CVAE': self._CVAE_update,
            'controlled_CVAE': self._controlled_CVAE_update,
        }
        update_method[self.network]()


    def _controlled_VQVAE_init(self):
        self.decoder_network = 'LSTM'
        self.various_go = False
        self.commitment_cost = 0.25
        self.grad_clip_norm = 1.0
        self.max_turn = 200
        self.emb_size = 300
        self.emb_trainable = True
        self.hidden_size = 300
        self.codebook_size = 512
        self.lr = 0.001
        self.lr_decay = 1.0
        self.batch_size = 128
        self.dropout_rate = 0.0
        self.epoch_num = 100  # triggered by early stop
        self.cuda = True
        self.early_stop_count = 30
        self.vocab_size = None
        self.remove_slot_value = True
        self.encoder_layer_num = 1
        self.beam_search = True
        self.beam_size = 10
        self.beam_len_bonus = 0.5
        self.teacher_force = 50
        self.act_size = 8
        self.glove_path = './data/glove.840B.300d.txt'

    def _controlled_VQVAE_update(self):
        self.model_path = './models/controlled_VQVAE_' + self.domain + '_' + self.decoder_network
        self.result_path = './results/controlled_VQVAE_' + self.domain + '_' + self.decoder_network
        self.vocab_emb = './vocabs/embedding_' + self.domain 
        self.vocab_path = './vocabs/' + self.domain 
        if self.remove_slot_value:
            self.model_path += '_delex'
            self.result_path += '_delex'
            self.vocab_path += '_delex'
            self.vocab_emb += '_delex'
        self.model_path += '_CB'+str(self.codebook_size)+'EL'+str(self.encoder_layer_num)+"CC"+str(self.commitment_cost).replace('.','d')
        self.result_path += '_CB'+str(self.codebook_size)+'EL'+str(self.encoder_layer_num)+"CC"+str(self.commitment_cost).replace('.','d')+'TMT'+str(self.text_max_ts)
        if self.beam_search:
            self.result_path += '_beam' + str(self.beam_size)
        self.model_path += '.pkl'
        self.result_path += '.csv'
        self.vocab_path += '.p'
        self.vocab_emb += '.npy'

    def _focused_VQVAE_init(self):
        self.decoder_network = 'LSTM'
        self.various_go = False
        self.commitment_cost = 0.25
        self.grad_clip_norm = 1.0
        self.max_turn = 200
        self.emb_size = 300
        self.emb_trainable = True
        self.hidden_size = 300
        self.codebook_size = 512
        self.lr = 0.001
        self.lr_decay = 1.0
        self.batch_size = 128
        self.dropout_rate = 0.0
        self.epoch_num = 100  # triggered by early stop
        self.cuda = True
        self.early_stop_count = 30
        self.vocab_size = None
        self.remove_slot_value = True
        self.encoder_layer_num = 1
        self.beam_search = True
        self.beam_size = 10
        self.beam_len_bonus = 0.5
        self.teacher_force = 50
        self.act_size = 8
        self.glove_path = './data/glove.840B.300d.txt'

    def _focused_VQVAE_update(self):
        self.model_path = './models/focused_VQVAE_' + self.domain + '_' + self.decoder_network
        self.result_path = './results/focused_VQVAE_' + self.domain + '_' + self.decoder_network
        self.vocab_emb = './vocabs/embedding_' + self.domain
        self.vocab_path = './vocabs/' + self.domain
        if self.remove_slot_value:
            self.model_path += '_delex'
            self.result_path += '_delex'
            self.vocab_path += '_delex'
            self.vocab_emb += '_delex'
        self.model_path += '_CB'+str(self.codebook_size)+'EL'+str(self.encoder_layer_num)
        self.result_path += '_CB'+str(self.codebook_size)+'EL'+str(self.encoder_layer_num)+'TMT'+str(self.text_max_ts)
        if self.beam_search:
            self.result_path += '_beam' + str(self.beam_size)
        self.model_path += '.pkl'
        self.result_path += '.csv'
        self.vocab_path += '.p'
        self.vocab_emb += '.npy'


    def _simple_VQVAE_init(self):
        self.decoder_network = 'LSTM'
        self.various_go = False
        self.commitment_cost = 0.25
        self.grad_clip_norm = 1.0
        self.max_turn = 100
        self.emb_size = 300
        self.emb_trainable = True
        self.hidden_size = 300
        self.codebook_size = 512
        self.lr = 0.001
        self.lr_decay = 1.0
        self.batch_size = 128
        self.dropout_rate = 0.0
        self.epoch_num = 100  # triggered by early stop
        self.cuda = True
        self.early_stop_count = 30
        self.vocab_size = None
        self.remove_slot_value = True
        self.encoder_layer_num = 1
        self.beam_search = True
        self.beam_size = 10
        self.beam_len_bonus = 0.5
        self.teacher_force = 50
        self.act_size = 8
        self.glove_path = './data/glove.840B.300d.txt'

    def _simple_VQVAE_update(self):
        self.model_path = './models/simple_VQVAE_' + self.domain + '_' + self.decoder_network
        self.result_path = './results/simple_VQVAE_' + self.domain + '_' + self.decoder_network
        self.vocab_emb = './vocabs/embedding_' + self.domain 
        self.vocab_path = './vocabs/' + self.domain 
        if self.remove_slot_value:
            self.model_path += '_delex'
            self.result_path += '_delex'
            self.vocab_path += '_delex'
            self.vocab_emb += '_delex'
        self.model_path += '_CB'+str(self.codebook_size)+'EL'+str(self.encoder_layer_num)
        self.result_path += '_CB'+str(self.codebook_size)+'EL'+str(self.encoder_layer_num)+'TMT'+str(self.text_max_ts)
        if self.beam_search:
            self.result_path += '_beam' + str(self.beam_size)
        self.model_path += '.pkl'
        self.result_path += '.csv'
        self.vocab_path += '.p'
        self.vocab_emb += '.npy'

    def _simple_seq2seq_init(self):
        self.VAE = False
        self.various_go = False
        self.grad_clip_norm = 1.0
        self.max_turn = 100
        self.emb_size = 300
        self.emb_trainable = True
        self.hidden_size = 128
        self.lr = 0.001
        self.lr_decay = 1.0
        self.batch_size = 128
        self.dropout_rate = 0.0
        self.epoch_num = 10  # triggered by early stop
        self.cuda = True
        self.early_stop_count = 10
        self.vocab_size = None
        self.remove_slot_value = True
        self.encoder_layer_num = 1
        self.beam_search = False
        self.beam_size = 10
        self.beam_len_bonus = 0.5
        self.teacher_force = 50
        self.act_size = 8
        self.glove_path = './data/glove.840B.300d.txt'
        
        
    def _simple_seq2seq_update(self):
        self.model_path = './models/simple_seq2seq_' + self.domain 
        self.result_path = './results/simple_seq2seq_' + self.domain 
        self.vocab_emb = './vocabs/embedding_' + self.domain
        self.vocab_path = './vocabs/' + self.domain
        if self.remove_slot_value:
            self.model_path += '_delex'
            self.result_path += '_delex'
            self.vocab_path += '_delex'
            self.vocab_emb += '_delex'
        if self.VAE:
            self.model_path += '_VAE'
            self.result_path += '_VAE'
        self.model_path += '_EL'+str(self.encoder_layer_num)
        self.result_path += '_EL'+str(self.encoder_layer_num)+'TMT'+str(self.text_max_ts)
        if self.beam_search:
            self.result_path += '_beam' + str(self.beam_size)
        self.model_path += '.pkl'
        self.result_path += '.csv'
        self.vocab_path += '.p'
        self.vocab_emb += '.npy'


    def _copy_seq2seq_init(self):
        self.VAE = False
        self.various_go = False
        self.grad_clip_norm = 1.0
        self.max_turn = 100
        self.emb_size = 300
        self.emb_trainable = True
        self.hidden_size = 300 
        self.lr = 0.001
        self.lr_decay = 1.0
        self.batch_size = 128
        self.dropout_rate = 0.0
        self.epoch_num = 100  # triggered by early stop
        self.cuda = True
        self.early_stop_count = 30
        self.vocab_size = None
        self.remove_slot_value = True
        self.encoder_layer_num = 1
        self.beam_search = False
        self.beam_size = 10
        self.beam_len_bonus = 0.5
        self.teacher_force = 50
        self.glove_path = './data/glove.840B.300d.txt'

    def _copy_seq2seq_update(self):
        self.model_path = './models/copy_seq2seq_' + self.domain 
        self.result_path = './results/copy_seq2seq_' + self.domain 
        self.vocab_emb = './vocabs/embedding_' + self.domain 
        self.vocab_path = './vocabs/' + self.domain
        if self.remove_slot_value:
            self.model_path += '_delex'
            self.result_path += '_delex'
            self.vocab_path += '_delex'
            self.vocab_emb += '_delex'
        if self.VAE:
            self.model_path += '_VAE'
            self.result_path += '_VAE'
        self.model_path += '_EL'+str(self.encoder_layer_num)
        self.result_path += '_EL'+str(self.encoder_layer_num)+'TMT'+str(self.text_max_ts)
        if self.beam_search:
            self.result_path += '_beam' + str(self.beam_size)
        self.model_path += '.pkl'
        self.result_path += '.csv'
        self.vocab_path += '.p'
        self.vocab_emb += '.npy'


    def _classification_init(self):
        self.VAE = False
        self.various_go = False
        self.vocab_size = None
        self.grad_clip_norm = 1.0
        self.max_turn = 100
        self.emb_size = 300
        self.emb_trainable = True
        self.hidden_size = 256 #xiujun's 60
        self.layer_num = 3
        self.bidirectional = True
        self.remove_slot_value = True
        self.lr = 0.001
        self.lr_decay = 0.9
        self.batch_size = 32
        self.dropout_rate = 0.0
        self.epoch_num = 50  # triggered by early stop
        self.cuda = True
        self.early_stop_count = 10
        self.input_size = 62
        self.output_size = 5 #total 5 personality
        self.glove_path = './data/glove.840B.300d.txt'
        self.act_size = 8


    def _controlled_CVAE_init(self):
        self.VAE = False
        self.various_go = False
        self.commitment_cost = 1.0
        self.grad_clip_norm = 1.0
        self.max_turn = 200
        self.emb_size = 300
        self.emb_trainable = True
        self.hidden_size = 300
        self.lr = 0.001
        self.lr_decay = 1.0
        self.batch_size = 128
        self.dropout_rate = 0.0
        self.epoch_num = 100  # triggered by early stop
        self.cuda = True
        self.early_stop_count = 30
        self.vocab_size = None
        self.remove_slot_value = True
        self.encoder_layer_num = 1
        self.beam_search = True
        self.beam_size = 10
        self.beam_len_bonus = 0.5
        self.teacher_force = 50
        self.act_size = 8
        self.glove_path = './data/glove.840B.300d.txt'
        self.condition_size = pow(2, self.act_size) +self.personality_size

    def _controlled_CVAE_update(self):
        self.model_path = './models/controlled_CVAE_' + self.domain
        self.result_path = './results/controlled_CVAE_' + self.domain
        self.vocab_emb = './vocabs/embedding_' + self.domain
        self.vocab_path = './vocabs/' + self.domain
        if self.remove_slot_value:
            self.model_path += '_delex'
            self.result_path += '_delex'
            self.vocab_path += '_delex'
            self.vocab_emb += '_delex'
        self.model_path += '_EL'+str(self.encoder_layer_num)
        self.result_path += '_EL'+str(self.encoder_layer_num)+'TMT'+str(self.text_max_ts)
        if self.beam_search:
            self.result_path += '_beam' + str(self.beam_size)
        self.model_path += '.pkl'
        self.result_path += '.csv'
        self.vocab_path += '.p'
        self.vocab_emb += '.npy'

    def _CVAE_init(self):
        self.VAE = False
        self.various_go = False
        self.grad_clip_norm = 1.0
        self.max_turn = 200
        self.emb_size = 300
        self.emb_trainable = True
        self.hidden_size = 300
        self.lr = 0.001
        self.lr_decay = 1.0
        self.batch_size = 128
        self.dropout_rate = 0.0
        self.epoch_num = 100  # triggered by early stop
        self.cuda = True
        self.early_stop_count = 30
        self.vocab_size = None
        self.remove_slot_value = True
        self.encoder_layer_num = 1
        self.beam_search = True
        self.beam_size = 10
        self.beam_len_bonus = 0.5
        self.teacher_force = 50
        self.act_size = 8
        self.glove_path = './data/glove.840B.300d.txt'
        self.condition_size = pow(2, self.act_size) +self.personality_size

    def _CVAE_update(self):
        self.model_path = './models/CVAE_' + self.domain
        self.result_path = './results/CVAE_' + self.domain
        self.vocab_emb = './vocabs/embedding_' + self.domain
        self.vocab_path = './vocabs/' + self.domain
        if self.remove_slot_value:
            self.model_path += '_delex'
            self.result_path += '_delex'
            self.vocab_path += '_delex'
            self.vocab_emb += '_delex'
        self.model_path += '_EL'+str(self.encoder_layer_num)
        self.result_path += '_EL'+str(self.encoder_layer_num)+'TMT'+str(self.text_max_ts)
        if self.beam_search:
            self.result_path += '_beam' + str(self.beam_size)
        self.model_path += '.pkl'
        self.result_path += '.csv'
        self.vocab_path += '.p'
        self.vocab_emb += '.npy'

    def _classification_update(self):
        self.model_path = './models/classification_'+self.domain
        self.result_path = './results/classification_'+self.domain
        self.vocab_emb = './vocabs/embedding_' + self.domain
        self.vocab_path = './vocabs/' + self.domain
        if self.remove_slot_value:
            self.model_path += '_delex'
            self.result_path += '_delex'
            self.vocab_path += '_delex'
            self.vocab_emb += '_delex'
        self.model_path += '.pkl'
        self.result_path += '.csv'
        self.vocab_path += '.p'
        self.vocab_emb += '.npy'

    def __str__(self):
        s = ''
        for k, v in self.__dict__.items():
            s += '{} : {}\n'.format(k, v)
        return s

    def _init_logging_handler(self):
        current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        stderr_handler = logging.StreamHandler()
        file_handler = logging.FileHandler('./log/log_{}.txt'.format(current_time))
        logging.basicConfig(handlers=[file_handler])#, stderr_handler
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)


