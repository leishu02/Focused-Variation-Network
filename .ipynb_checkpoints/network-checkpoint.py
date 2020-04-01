import torch
import numpy as np
import random
import math, copy


def cuda_(var, cfg):
    return var.cuda() if cfg.cuda else var


def toss_(p):
    return random.randint(0, 99) <= p


def nan(v):
    if type(v) is float:
        return v == float('nan')
    return np.isnan(np.sum(v.data.cpu().numpy()))


def init_gru(gru):
    gru.reset_parameters()
    for _, hh, _, _ in gru.all_weights:
        for i in range(0, hh.size(0), gru.hidden_size):
            torch.nn.init.orthogonal_(hh[i:i + gru.hidden_size], gain=1)


def init_lstm(lstm):#
    lstm.reset_parameters()
    for _, hh, _, _ in lstm.all_weights:
        for i in range(0, hh.size(0), lstm.hidden_size):
            torch.nn.init.orthogonal_(hh[i:i + lstm.hidden_size], gain=1)


class VectorQuantizerEMA(torch.nn.Module):
    '''
    This use exponential moving averages to update the embedding vectors instead of an auxillary loss.
    This has the advantage that the embedding updates are independent of the choice of optimizer for the encoder,
    decoder and other parts of the architecture.
    For most experiments the EMA version trains faster than the non-EMA version
    '''
    def __init__(self, cfg, decay, epsilon=1e-5, codebook_size=None):
        super(VectorQuantizerEMA, self).__init__()
        self.cfg = cfg
        self._embedding_dim = cfg.hidden_size
        if codebook_size:
            self._num_embeddings = codebook_size
        else:
            self._num_embeddings = cfg.codebook_size

        self._embedding = torch.nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()
        self._commitment_cost = cfg.commitment_cost

        self.register_buffer('_ema_cluster_size', torch.zeros(cfg.codebook_size))
        self._ema_w = torch.nn.Parameter(torch.Tensor(cfg.codebook_size, self._embedding_dim))
        self._ema_w.data.normal_()

        self._decay = decay
        self._epsilon = epsilon

    def forward(self, inputs):
        # inputs from BC
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)

        # Calculate distances
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(self._embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, self._embedding.weight.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = cuda_(torch.zeros(encoding_indices.shape[0], self._num_embeddings), self.cfg)
        encodings.scatter_(1, encoding_indices, 1)

        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + \
                                     (1 - self._decay) * torch.sum(encodings, 0)

            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                    (self._ema_cluster_size + self._epsilon)
                    / (n + self._num_embeddings * self._epsilon) * n)

            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = torch.nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)

            self._embedding.weight = torch.nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)

        # Loss
        e_latent_loss = torch.mean((quantized.detach() - inputs) ** 2)
        loss = self._commitment_cost * e_latent_loss

        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        #
        return loss, quantized, perplexity, encodings


class VectorQuantizer(torch.nn.Module):
    def __init__(self, cfg, codebook_size=None):
        super(VectorQuantizer, self).__init__()
        self.cfg = cfg
        self._embedding_dim = cfg.hidden_size
        if codebook_size:
            self._num_embeddings = codebook_size
        else:
            self._num_embeddings = cfg.codebook_size

        self._embedding = torch.nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1 / self._num_embeddings, 1 / self._num_embeddings)
        self._commitment_cost = cfg.commitment_cost

    def forward(self, inputs):
        # inputs BC
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)

        # Calculate distances
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(self._embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, self._embedding.weight.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = cuda_(torch.zeros(encoding_indices.shape[0], self._num_embeddings), self.cfg)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)

        # Loss
        e_latent_loss = torch.mean((quantized.detach() - inputs) ** 2)
        q_latent_loss = torch.mean((quantized - inputs.detach()) ** 2)
        loss = q_latent_loss + self._commitment_cost * e_latent_loss

        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return loss, quantized, perplexity, encodings


class Vocab_VectorQuantizer(torch.nn.Module):
    def __init__(self, cfg, codebook_size=None):
        super(Vocab_VectorQuantizer, self).__init__()
        self.cfg = cfg
        self.embedding_dim = cfg.hidden_size
        if codebook_size:
            self.num_embeddings = codebook_size
        else:
            self.num_embeddings = cfg.codebook_size

        self.embedding = torch.nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1 / self.num_embeddings, 1 / self.num_embeddings)
        self.commitment_cost = cfg.commitment_cost

    def forward(self, inputs):
        # inputs BC
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self.embedding_dim)

        # Calculate distances
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(self.embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, self.embedding.weight.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = cuda_(torch.zeros(encoding_indices.shape[0], self.num_embeddings), self.cfg)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self.embedding.weight).view(input_shape)

        # Loss
        e_latent_loss = torch.mean((quantized.detach() - inputs) ** 2)
        q_latent_loss = torch.mean((quantized - inputs.detach()) ** 2)
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return loss, quantized, perplexity, encodings


class MultiClass_Classification(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout):
        super(MultiClass_Classification, self).__init__()
        self.dropout_rate = dropout
        self.linear1 = torch.nn.Linear(in_features=input_size, out_features=hidden_size, bias=True)
        self.linear2 = torch.nn.Linear(in_features=hidden_size, out_features=output_size, bias=True)

    def forward(self, x, gt_y, mode):
        linear1_h = torch.nn.functional.relu(self.linear1(x))#x (N, *, in_features) linear1_x (N, *, out_features)
        linear1_h = torch.nn.functional.dropout(linear1_h, self.dropout_rate)
        linear2_h = self.linear2(linear1_h)
        if mode == 'test':
            output = torch.softmax(linear2_h, dim = -1)
            output = output.data.cpu().numpy()
        elif mode == 'train':
            output = torch.nn.functional.cross_entropy(linear2_h, gt_y)
        else:
            assert()
        return output


class MultiLabel_Classification(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout):
        super(MultiLabel_Classification, self).__init__()
        self.dropout_rate = dropout
        self.linear1 = torch.nn.Linear(in_features=input_size, out_features=hidden_size, bias=True)
        self.linear2 = torch.nn.Linear(in_features=hidden_size, out_features=output_size, bias=True)

    def forward(self, x, gt_y, mode):
        linear1_h = torch.nn.functional.relu(self.linear1(x))#x (N, *, in_features) linear1_x (N, *, out_features)
        linear1_h = torch.nn.functional.dropout(linear1_h, self.dropout_rate)
        linear2_h = self.linear2(linear1_h)
        if mode == 'test':
            output = torch.sigmoid(linear2_h)
            output = output.data.cpu().numpy()
        elif mode == 'train':
            output = torch.nn.functional.binary_cross_entropy_with_logits(linear2_h, gt_y)
        else:
            assert()
        return output


class MLP(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout):
        super(MLP, self).__init__()
        self.dropout_rate = dropout
        self.linear1 = torch.nn.Linear(in_features=input_size, out_features=hidden_size, bias=True)
        self.linear2 = torch.nn.Linear(in_features=hidden_size, out_features=output_size, bias=True)

    def forward(self, x):
        linear1_h = torch.nn.functional.relu(self.linear1(x))#x (N, *, in_features) linear1_x (N, *, out_features)
        linear1_h = torch.nn.functional.dropout(linear1_h, self.dropout_rate)
        linear2_h = self.linear2(linear1_h)
        return linear2_h

            
class LSTMDynamicEncoder(torch.nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, n_layers, dropout, cfg, emb=None, bidirectional=True):
        super(LSTMDynamicEncoder, self).__init__()
        self.cfg = cfg
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        if emb:
            self.embedding = emb
        else:
            self.embedding = torch.nn.Embedding(vocab_size, embed_size)
        self.lstm = torch.nn.LSTM(embed_size, hidden_size, n_layers, dropout=dropout, bidirectional=bidirectional)
        init_lstm(self.lstm)

    def forward(self, input_seqs, input_lens, hidden=None, is_embedded = False, enc_out = 'add'):
        """
        forward procedure. No need for inputs to be sorted
        :param input_seqs: Variable of [T,B]
        :param hidden:
        :param input_lens: *numpy array* of len for each input sequence
        :return:
        """
        if is_embedded:
            embedded = input_seqs # [T,B,E]
        else:
            embedded = self.embedding(input_seqs)
        embedded = embedded.transpose(0, 1)  # [B,T,E]
        sort_idx = np.argsort(-input_lens)
        unsort_idx = cuda_(torch.LongTensor(np.argsort(sort_idx)), self.cfg)
        input_lens = input_lens[sort_idx]
        sort_idx = cuda_(torch.LongTensor(sort_idx), self.cfg)
        embedded = embedded[sort_idx].transpose(0, 1)  # [T,B,E]
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lens)
        outputs, hidden = self.lstm(packed, hidden)
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)
        if self.bidirectional:
            if enc_out == 'add':
                outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]
            else:
                pass
        outputs = outputs.transpose(0, 1)[unsort_idx].transpose(0, 1).contiguous()
        h = hidden[0].transpose(0, 1)[unsort_idx].transpose(0, 1).contiguous()
        c = hidden[1].transpose(0, 1)[unsort_idx].transpose(0, 1).contiguous()
        return outputs, (h, c), embedded


class RNN_Decoder(torch.nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size,  dropout_rate, vocab, cfg):
        super(RNN_Decoder, self).__init__()
        self.cfg = cfg
        self.emb = torch.nn.Embedding(vocab_size, embed_size)
        if self.cfg.decoder_network == 'LSTM':
            self.rnn = torch.nn.LSTM(embed_size, hidden_size, 1, dropout=dropout_rate, bidirectional=False)
            init_lstm(self.rnn)
        if self.cfg.decoder_network == 'GRU':
            self.rnn = torch.nn.GRU(embed_size, hidden_size, 1, dropout=dropout_rate, bidirectional=False)
            init_gru(self.rnn)
        self.emb_proj = torch.nn.Linear(hidden_size, embed_size)
        self.proj = torch.nn.Linear(embed_size, vocab_size)
        self.dropout_rate = dropout_rate
        self.vocab = vocab

    def forward(self, m_t_input, last_hidden):
        m_embed = self.emb(m_t_input)
        _in = m_embed
        _out, last_hidden = self.rnn(_in, last_hidden)
        _enc_out = self.emb_proj(_out)
        gen_score = self.proj(_enc_out).squeeze(0)
        proba = torch.nn.functional.softmax(gen_score, dim=1)
        return proba, last_hidden, _enc_out

class Attn_RNN_Decoder(torch.nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size,  dropout_rate, vocab, cfg):
        super(Attn_RNN_Decoder, self).__init__()
        self.cfg = cfg
        self.emb = torch.nn.Embedding(vocab_size, embed_size)
        self.a_attn = Attn(hidden_size)
        self.p_attn = Attn(hidden_size)
        if self.cfg.decoder_network == 'LSTM':
            self.rnn = torch.nn.LSTM(embed_size + 2*hidden_size, hidden_size, 1, dropout=dropout_rate, bidirectional=False)
            init_lstm(self.rnn)
        if self.cfg.decoder_network == 'GRU':
            self.rnn = torch.nn.GRU(embed_size + 2*hidden_size, hidden_size, 1, dropout=dropout_rate, bidirectional=False)
            init_gru(self.rnn)
        self.emb_proj = torch.nn.Linear(hidden_size, embed_size)
        self.proj = torch.nn.Linear(embed_size, vocab_size)
        self.dropout_rate = dropout_rate
        self.vocab = vocab

    def forward(self, m_t_input, last_hidden, act_enc_out, personality_enc_out):
        m_embed = self.emb(m_t_input)
        if self.cfg.decoder_network == 'LSTM':
            a_context = self.a_attn(last_hidden[0], act_enc_out)
            p_context = self.p_attn(last_hidden[0], personality_enc_out)
        else:
            a_context = self.a_attn(last_hidden, act_enc_out)
            p_context = self.p_attn(last_hidden, personality_enc_out)
        _in = torch.cat([m_embed, a_context, p_context], dim=2)
        _out, last_hidden = self.rnn(_in, last_hidden)
        _enc_out = self.emb_proj(_out)
        gen_score = self.proj(_enc_out).squeeze(0)
        proba = torch.nn.functional.softmax(gen_score, dim=1)
        return proba, last_hidden, _enc_out


class VQVAE(torch.nn.Module):
    def __init__(self, cfg, vocab, decay=0, eos_m_token='EOS'):
        super(VQVAE, self).__init__()
        self.cfg = cfg
        self.vocab = vocab
        self.encoder = LSTMDynamicEncoder(len(vocab), cfg.emb_size, cfg.hidden_size, cfg.encoder_layer_num, cfg.dropout_rate, cfg)
        if decay > 0.0:
            self.act_vq_vae = VectorQuantizerEMA(cfg, decay)
            self.personality_vq_vae = VectorQuantizerEMA(cfg, decay)
        else:
            self.act_vq_vae = VectorQuantizer(cfg)
            self.personality_vq_vae = VectorQuantizer(cfg)
        self.decoder = RNN_Decoder(len(vocab), cfg.emb_size, 2*cfg.hidden_size, cfg.dropout_rate, vocab, cfg)
        self.act_predictor = MultiLabel_Classification(cfg.hidden_size, int(cfg.hidden_size/2), cfg.act_size, cfg.dropout_rate)
        self.personality_predictor = MultiClass_Classification(cfg.hidden_size, int(cfg.hidden_size/2), cfg.personality_size, cfg.dropout_rate)
        self.act_mlp = MLP(2*cfg.hidden_size, 4*cfg.hidden_size, cfg.hidden_size, cfg.dropout_rate)
        self.personality_mlp = MLP(2 * cfg.hidden_size, 4 * cfg.hidden_size, cfg.hidden_size, cfg.dropout_rate)
        self.dec_loss = torch.nn.NLLLoss(ignore_index=0, reduction='mean')
        self.max_ts = cfg.text_max_ts
        self.beam_search = cfg.beam_search
        self.teacher_force = cfg.teacher_force
        if self.beam_search:
            self.beam_size = cfg.beam_size
            self.eos_m_token = eos_m_token
            self.eos_token_idx = self.vocab.encode(eos_m_token)

    def forward(self, x, gt_y, mode, **kwargs):
        if mode == 'train' or mode == 'valid':
            loss, recon_loss, act_loss, personality_loss, act_vq_loss, personality_vq_loss = self.forward_turn(x, gt_y, mode, **kwargs)
            return loss, recon_loss, act_loss, personality_loss, act_vq_loss, personality_vq_loss
        elif mode == 'getDist':
            #print ("forward")
            act_encoding, personality_encoding = self.forward_turn(x, gt_y, mode, **kwargs)
            return act_encoding, personality_encoding
        elif mode == 'test':
            pred_y, act_pred, personality_pred = self.forward_turn(x, gt_y, mode, **kwargs)
            return pred_y, act_pred, personality_pred

    def forward_turn(self, x, gt_y, mode, **kwargs):
        #print ("forward turn")
        if self.cfg.remove_slot_value == True:
            x_len = kwargs['slot_len']  # batchsize
            x_np = kwargs['slot_np']  # seqlen, batchsize
            y_len = kwargs['delex_text_len']  # batchsize
            y_np = kwargs['delex_text_np']  # batchsize
        else:
            x_len = kwargs['slot_value_len']  # seqlen, batchsize
            x_np = kwargs['slot_value_np']  # batchsize
            y_len = kwargs['text_len']  # batchsize
            y_np = kwargs['text_np']  # seqlen, batchsize

        personality_idx = kwargs['personality_idx']
        act_idx = kwargs['act_idx']

        batch_size = x.size(1)
        x_enc_out, (h, c), _ = self.encoder(gt_y, y_len)
        z = torch.cat([h[0], h[1]], dim=-1)

        act_z = self.act_mlp(z)
        act_vq_loss, act_quantized, act_perplexity, act_encoding = self.act_vq_vae(act_z)
        personality_z = self.personality_mlp(z)
        personality_vq_loss, personality_quantized, personality_perplexity, personality_encoding = self.personality_vq_vae(personality_z)
        quantized = torch.cat([act_quantized, personality_quantized], dim=-1)
        if mode == 'getDist':
            return act_encoding, personality_encoding
        decoder_c = cuda_(torch.autograd.Variable(torch.zeros(quantized.size())), self.cfg)
 
        text_tm1 = cuda_(torch.autograd.Variable(torch.ones(1, batch_size).long()), self.cfg)  # GO token
        text_length = gt_y.size(0)
        text_dec_proba = []
        text_dec_outs = []
        if mode == 'train':
            if self.cfg.decoder_network == 'LSTM':
                last_hidden = (quantized.unsqueeze(0), decoder_c.unsqueeze(0))
            else:
                last_hidden = quantized.unsqueeze(0)
            act_loss = self.act_predictor(act_quantized, act_idx, mode)
            personality_loss = self.personality_predictor(personality_quantized, personality_idx, mode)
            for t in range(text_length):
                teacher_forcing = toss_(self.teacher_force)
                proba, last_hidden, dec_out = self.decoder(text_tm1, last_hidden)
                if teacher_forcing:
                    text_tm1 = gt_y[t].view(1, -1)
                else:
                    _, text_tm1 = torch.topk(proba, 1)
                    text_tm1 = text_tm1.view(1, -1)
                text_dec_proba.append(proba)
                text_dec_outs.append(dec_out)
            text_dec_proba = torch.stack(text_dec_proba, dim=0)  # [T,B,V]
            pred_y = text_dec_proba
            recon_loss = self.dec_loss( \
                torch.log(pred_y.view(-1, pred_y.size(2))), \
                gt_y.view(-1))
            loss = recon_loss + act_loss + personality_loss + act_vq_loss + personality_vq_loss
            return loss, recon_loss, act_loss, personality_loss, act_vq_loss, personality_vq_loss
        else:
            act_sample_idx = kwargs['act_sample_idx']
            personality_sample_idx = kwargs['personality_sample_idx']
            act_sample_emb = self.act_vq_vae._embedding(act_sample_idx)
            personality_sample_emb = self.personality_vq_vae._embedding(personality_sample_idx)
            sample_quantized = torch.cat([act_sample_emb, personality_sample_emb], dim=-1)
            if self.cfg.decoder_network == 'LSTM':
                last_hidden = (sample_quantized.transpose(0, 1), decoder_c.unsqueeze(0))
            else:
                last_hidden = quantized.transpose(0, 1)
            act_pred = self.act_predictor(act_quantized, act_idx, mode)
            personality_pred = self.personality_predictor(personality_quantized, personality_idx, mode)
            if mode == 'test':
                if not self.cfg.beam_search:
                    text_dec_idx = self.greedy_decode(text_tm1,  last_hidden)

                else:
                    text_dec_idx = self.beam_search_decode(text_tm1,  last_hidden)

                return text_dec_idx, act_pred, personality_pred

    def greedy_decode(self, text_tm1,  last_hidden):
        decoded = []
        for t in range(self.max_ts):
            proba, last_hidden, _ = self.decoder(text_tm1, last_hidden)
            mt_proba, mt_index = torch.topk(proba, 1)  # [B,1]
            mt_index = mt_index.data.view(-1)
            decoded.append(mt_index.clone())
            for i in range(mt_index.size(0)):
                if mt_index[i] >= len(self.vocab):
                    mt_index[i] = 2  # unk
            text_tm1 = cuda_(torch.autograd.Variable(mt_index).view(1, -1), self.cfg)
        decoded = torch.stack(decoded, dim=0).transpose(0, 1)
        decoded = list(decoded)
        return [list(_) for _ in decoded]

    def beam_search_decode_single(self, text_tm1, last_hidden):
        eos_token_id = self.eos_token_idx
        batch_size = text_tm1.size(1)
        if batch_size != 1:
            raise ValueError('"Beam search single" requires batch size to be 1')

        class BeamState:
            def __init__(self, score, last_hidden, decoded, length):
                """
                Beam state in beam decoding
                :param score: sum of log-probabilities
                :param last_hidden: last hidden
                :param decoded: list of *Variable[1*1]* of all decoded words
                :param length: current decoded sentence length
                """
                self.score = score
                self.last_hidden = last_hidden
                self.decoded = decoded
                self.length = length

            def update_clone(self, score_incre, last_hidden, decoded_t):
                decoded = copy.copy(self.decoded)
                decoded.append(decoded_t)
                clone = BeamState(self.score + score_incre, last_hidden, decoded, self.length + 1)
                return clone

        def score_bonus(state, decoded):
            bonus = self.cfg.beam_len_bonus
            return bonus

        def soft_score_incre(score, turn):
            return score

        finished, failed = [], []
        states = []  # sorted by score decreasingly
        dead_k = 0
        states.append(BeamState(0, last_hidden, [text_tm1], 0))
        for t in range(self.max_ts):
            new_states = []
            k = 0
            while k < len(states) and k < self.beam_size - dead_k:
                state = states[k]
                last_hidden, text_tm1 = state.last_hidden, state.decoded[-1]
                proba, last_hidden, _ = self.decoder(text_tm1, last_hidden)
                proba = torch.log(proba)
                mt_proba, mt_index = torch.topk(proba, self.beam_size - dead_k)  # [1,K]
                for new_k in range(self.beam_size - dead_k):
                    score_incre = soft_score_incre(mt_proba[0][new_k].item(), t) + score_bonus(state,
                                                                                                mt_index[0][new_k].item())
                    if len(new_states) >= self.beam_size - dead_k and state.score + score_incre < new_states[-1].score:
                        break
                    decoded_t = mt_index[0][new_k]
                    if decoded_t.item() >= len(self.vocab):
                        decoded_t.item()== 2  # unk
                    if decoded_t.item() == self.eos_token_idx:
                        finished.append(state)
                    else:
                        decoded_t = decoded_t.view(1, -1)
                        new_state = state.update_clone(score_incre, last_hidden, decoded_t)
                        new_states.append(new_state)

                k += 1
            if self.beam_size - dead_k < 0:
                break
            new_states = new_states[:self.beam_size - dead_k]
            new_states.sort(key=lambda x: -x.score)
            states = new_states

            if t == self.max_ts - 1 and not finished:
                finished = failed
                print('FAIL')
                if not finished:
                    finished.append(states[0])

        finished.sort(key=lambda x: -x.score)
        decoded_t = finished[0].decoded
        decoded_t = [_.view(-1).item() for _ in decoded_t]
        #decoded_sentence = self.vocab.sentence_decode(decoded_t, self.eos_m_token)
        #print(decoded_sentence)
        generated = torch.cat(finished[0].decoded, dim=1).data  # [B=1, T]
        return generated

    def beam_search_decode(self, text_tm1, last_hidden):
        decoded = []
        if self.cfg.decoder_network == 'LSTM':
            vars = torch.split(text_tm1, 1, dim=1), torch.split(last_hidden[0], 1, dim=1), torch.split(last_hidden[1], 1, dim=1)
            for i, (text_tm1_s, last_hidden_h_s, last_hidden_c_s) \
                    in enumerate(zip(*vars)):
                decoded_s = self.beam_search_decode_single(text_tm1_s, (last_hidden_h_s, last_hidden_c_s))
                decoded.append(decoded_s)
        else:
            vars = torch.split(text_tm1, 1, dim=1), torch.split(last_hidden, 1, dim=1)
            for i, (text_tm1_s, last_hidden_s) \
                    in enumerate(zip(*vars)):
                decoded_s = self.beam_search_decode_single(text_tm1_s, last_hidden_s)
                decoded.append(decoded_s)

        return [list(_.view(-1)) for _ in decoded]


class Controlled_VQVAE(torch.nn.Module):
    def __init__(self, cfg, vocab, decay=0, eos_m_token='EOS'):
        super(Controlled_VQVAE, self).__init__()
        self.cfg = cfg
        self.vocab = vocab
        self.vocab_vq_vae = Vocab_VectorQuantizer(cfg, len(vocab))
        self.vae_encoder = LSTMDynamicEncoder(len(vocab), cfg.emb_size, cfg.hidden_size, cfg.encoder_layer_num, cfg.dropout_rate, cfg, self.vocab_vq_vae.embedding)
        self.encoder = LSTMDynamicEncoder(len(vocab), cfg.emb_size, cfg.hidden_size, cfg.encoder_layer_num,
                                          cfg.dropout_rate, cfg, self.vocab_vq_vae.embedding)
        #self.personality_attn = Attn(self.cfg.hidden_size)
        #self.slot_attn = Attn(self.cfg.hidden_size)
        if decay > 0.0:
            self.act_vq_vae = VectorQuantizerEMA(cfg, decay)
            self.personality_vq_vae = VectorQuantizerEMA(cfg, decay)
        else:
            self.act_vq_vae = VectorQuantizer(cfg)
            self.personality_vq_vae = VectorQuantizer(cfg)
        self.decoder = Attn_RNN_Decoder(len(vocab), cfg.emb_size, 2*cfg.hidden_size, cfg.dropout_rate, vocab, cfg)
        self.act_predictor = MultiLabel_Classification(cfg.hidden_size, int(cfg.hidden_size/2), cfg.act_size, cfg.dropout_rate)
        self.personality_predictor = MultiClass_Classification(cfg.hidden_size, int(cfg.hidden_size/2), cfg.personality_size, cfg.dropout_rate)
        self.act_mlp = MLP(2*cfg.hidden_size, 4*cfg.hidden_size, cfg.hidden_size, cfg.dropout_rate)
        self.personality_mlp = MLP(2 * cfg.hidden_size, 4 * cfg.hidden_size, cfg.hidden_size, cfg.dropout_rate)
        self.dec_loss = torch.nn.NLLLoss(ignore_index=0, reduction='mean')
        self.max_ts = cfg.text_max_ts
        self.beam_search = cfg.beam_search
        self.teacher_force = cfg.teacher_force
        if self.beam_search:
            self.beam_size = cfg.beam_size
            self.eos_m_token = eos_m_token
            self.eos_token_idx = self.vocab.encode(eos_m_token)

    def forward(self, x, gt_y, mode, **kwargs):
        if mode == 'train' or mode == 'valid':
            loss, recon_loss, act_loss, personality_loss, act_vq_loss, personality_vq_loss = self.forward_turn(x, gt_y, mode, **kwargs)
            return loss, recon_loss, act_loss, personality_loss, act_vq_loss, personality_vq_loss
        elif mode == 'getDist':
            act_encoding, personality_encoding = self.forward_turn(x, gt_y, mode, **kwargs)
            return act_encoding, personality_encoding
        elif mode == 'test':
            pred_y, act_pred, personality_pred = self.forward_turn(x, gt_y, mode, **kwargs)
            return pred_y , act_pred, personality_pred

    def forward_turn(self, x, gt_y, mode, **kwargs):
        if self.cfg.remove_slot_value == True:
            x_len = kwargs['slot_len']  # batchsize
            x_np = kwargs['slot_np']  # seqlen, batchsize
            y_len = kwargs['delex_text_len']  # batchsize
            y_np = kwargs['delex_text_np']  # batchsize
        else:
            x_len = kwargs['slot_value_len']  # seqlen, batchsize
            x_np = kwargs['slot_value_np']  # batchsize
            y_len = kwargs['text_len']  # batchsize
            y_np = kwargs['text_np']  # seqlen, batchsize

        personality_idx = kwargs['personality_idx']
        personality_seq = kwargs['personality_seq']
        personality_len = kwargs['personality_len']
        act_idx = kwargs['act_idx']

        batch_size = x.size(1)
        x_enc_out, (h, c), _ = self.vae_encoder(gt_y, y_len)
        z = torch.cat([h[0], h[1]], dim=-1)
        act_z = self.act_mlp(z)
        act_vq_loss, act_quantized, act_perplexity, act_encoding = self.act_vq_vae(act_z)
        personality_z = self.personality_mlp(z)
        personality_vq_loss, personality_quantized, personality_perplexity, personality_encoding = self.personality_vq_vae(personality_z)
        text_tm1 = cuda_(torch.autograd.Variable(torch.ones(1, batch_size).long()), self.cfg)  # GO token
        text_length = gt_y.size(0)
        text_dec_proba = []
        text_dec_outs = []
        text_quantized_dec_outs = []
        text_vq_loss_s = []
        text_perplexity_s = []

        #act and personality distribution act_encoding (B, codebook_size)
        personality_enc_out, personality_hidden, personality_emb = self.encoder(personality_seq, personality_len, enc_out='cat')
        slot_enc_out, slot_hidden, slot_emb = self.encoder(x, x_len, personality_hidden, enc_out='cat')
        quantized = torch.cat([act_quantized, personality_quantized], dim=-1)
        #decoder_c = cuda_(torch.autograd.Variable(torch.zeros(quantized.size())), self.cfg)
        decoder_c = torch.cat([slot_hidden[1][0], slot_hidden[1][1]], dim =-1)
        '''
        personality_emb_copy = cuda_(torch.autograd.Variable(self.personality_vq_vae._embedding.weight.data.clone()), self.cfg).repeat(batch_size, 1, 1).transpose(0, 1)
        act_emb_copy = cuda_(torch.autograd.Variable(self.act_vq_vae._embedding.weight.data.clone()), self.cfg).repeat(batch_size, 1, 1).transpose(0, 1)
        personality_dist = self.personality_attn(personality_hidden[0][0]+personality_hidden[0][1], personality_emb_copy, return_normalize=True).squeeze(1)
        slot_dist = self.slot_attn(slot_hidden[0][0]+slot_hidden[0][1], act_emb_copy, return_normalize=True).squeeze(1)
        #
        '''
        if mode == 'getDist':
            return act_encoding, personality_encoding
        
        elif mode == 'train':
            if self.cfg.decoder_network == 'LSTM':
                last_hidden = (quantized.unsqueeze(0), decoder_c.unsqueeze(0))
            else:
                last_hidden = quantized.unsqueeze(0)
            act_loss = self.act_predictor(act_quantized, act_idx, mode)
            personality_loss = self.personality_predictor(personality_quantized, personality_idx, mode)
            ##prepare dist from encoding           
            '''
            personality_encoding_dist = getDist(personality_idx, personality_encoding)
            slot_encoding_dist = getDist(act_idx, act_encoding)

            personality_KLdiv = torch.nn.functional.kl_div(personality_dist.log(), personality_encoding_dist)
            slot_KLdiv = torch.nn.functional.kl_div(slot_dist.log(), slot_encoding_dist)
            '''
            ##
            for t in range(text_length):
                teacher_forcing = toss_(self.teacher_force)
                proba, last_hidden, dec_out = self.decoder(text_tm1, last_hidden, slot_enc_out, personality_enc_out)
                if teacher_forcing:
                    text_tm1 = gt_y[t].view(1, -1)
                else:
                    _, text_tm1 = torch.topk(proba, 1)
                    text_tm1 = text_tm1.view(1, -1)
                text_dec_proba.append(proba)
                text_dec_outs.append(dec_out)
                #text vq vae
                text_vq_loss, text_quantized, text_perplexity, _ = self.vocab_vq_vae(dec_out)
                text_vq_loss_s.append(text_vq_loss)
                text_perplexity_s.append(text_perplexity)
                text_quantized_dec_outs.append(text_quantized)
                #
            text_dec_proba = torch.stack(text_dec_proba, dim=0)  # [T,B,V]
            pred_y = text_dec_proba
            recon_loss = self.dec_loss( \
                torch.log(pred_y.view(-1, pred_y.size(2))), \
                gt_y.view(-1))
            #
            vocab_vq_loss = torch.mean(torch.stack(text_vq_loss_s, dim=0))
            vocab_quantized_dec_outs = torch.cat(text_quantized_dec_outs, dim=0)

            #feed text_vq to encoder
            quantized_enc_out, (quantized_h, quantized_c), _ = self.vae_encoder(vocab_quantized_dec_outs, y_len, is_embedded=True)
            quantized_z = torch.cat([quantized_h[0], quantized_h[1]], dim=-1)

            quantized_act_z = self.act_mlp(quantized_z)
            quantized_personality_z = self.personality_mlp(quantized_z)
            quantized_act_loss = self.act_predictor(quantized_act_z, act_idx, mode)
            quantized_personality_loss = self.personality_predictor(quantized_personality_z, personality_idx, mode)
            #

            loss = recon_loss + act_loss + personality_loss + act_vq_loss + personality_vq_loss \
                   + vocab_vq_loss + quantized_act_loss + quantized_personality_loss #+ personality_KLdiv + slot_KLdiv
            return loss, recon_loss, act_loss, personality_loss, act_vq_loss, personality_vq_loss
        else:
            act_sample_idx = kwargs['act_sample_idx']
            personality_sample_idx = kwargs['personality_sample_idx']
            act_sample_emb = self.act_vq_vae._embedding(act_sample_idx)
            personality_sample_emb = self.personality_vq_vae._embedding(personality_sample_idx)
            sample_quantized = torch.cat([act_sample_emb, personality_sample_emb], dim=-1)
            if self.cfg.decoder_network == 'LSTM':
                last_hidden = (sample_quantized.transpose(0, 1), decoder_c.unsqueeze(0))
            else:
                last_hidden = quantized.transpose(0, 1)
            act_pred = self.act_predictor(act_quantized, act_idx, mode)
            personality_pred = self.personality_predictor(personality_quantized, personality_idx, mode)
            if mode == 'test':
                if not self.cfg.beam_search:
                    text_dec_idx = self.greedy_decode(text_tm1,  last_hidden, slot_enc_out, personality_enc_out)

                else:
                    text_dec_idx = self.beam_search_decode(text_tm1,  last_hidden, slot_enc_out, personality_enc_out)

                return text_dec_idx , act_pred, personality_pred

    def greedy_decode(self, text_tm1,  last_hidden, slot_enc_out, personality_enc_out):
        decoded = []
        for t in range(self.max_ts):
            proba, last_hidden, _ = self.decoder(text_tm1, last_hidden, slot_enc_out, personality_enc_out)
            mt_proba, mt_index = torch.topk(proba, 1)  # [B,1]
            mt_index = mt_index.data.view(-1)
            decoded.append(mt_index.clone())
            for i in range(mt_index.size(0)):
                if mt_index[i] >= len(self.vocab):
                    mt_index[i] = 2  # unk
            text_tm1 = cuda_(torch.autograd.Variable(mt_index).view(1, -1), self.cfg)
        decoded = torch.stack(decoded, dim=0).transpose(0, 1)
        decoded = list(decoded)
        return [list(_) for _ in decoded]

    def beam_search_decode_single(self, text_tm1, last_hidden, slot_enc_out, personality_enc_out):
        eos_token_id = self.eos_token_idx
        batch_size = text_tm1.size(1)
        if batch_size != 1:
            raise ValueError('"Beam search single" requires batch size to be 1')

        class BeamState:
            def __init__(self, score, last_hidden, decoded, length):
                """
                Beam state in beam decoding
                :param score: sum of log-probabilities
                :param last_hidden: last hidden
                :param decoded: list of *Variable[1*1]* of all decoded words
                :param length: current decoded sentence length
                """
                self.score = score
                self.last_hidden = last_hidden
                self.decoded = decoded
                self.length = length

            def update_clone(self, score_incre, last_hidden, decoded_t):
                decoded = copy.copy(self.decoded)
                decoded.append(decoded_t)
                clone = BeamState(self.score + score_incre, last_hidden, decoded, self.length + 1)
                return clone

        def score_bonus(state, decoded):
            bonus = self.cfg.beam_len_bonus
            return bonus

        def soft_score_incre(score, turn):
            return score

        finished, failed = [], []
        states = []  # sorted by score decreasingly
        dead_k = 0
        states.append(BeamState(0, last_hidden, [text_tm1], 0))
        for t in range(self.max_ts):
            new_states = []
            k = 0
            while k < len(states) and k < self.beam_size - dead_k:
                state = states[k]
                last_hidden, text_tm1 = state.last_hidden, state.decoded[-1]
                proba, last_hidden, _ = self.decoder(text_tm1, last_hidden, slot_enc_out, personality_enc_out)
                proba = torch.log(proba)
                mt_proba, mt_index = torch.topk(proba, self.beam_size - dead_k)  # [1,K]
                for new_k in range(self.beam_size - dead_k):
                    score_incre = soft_score_incre(mt_proba[0][new_k].item(), t) + score_bonus(state,
                                                                                                mt_index[0][new_k].item())
                    if len(new_states) >= self.beam_size - dead_k and state.score + score_incre < new_states[-1].score:
                        break
                    decoded_t = mt_index[0][new_k]
                    if decoded_t.item() >= len(self.vocab):
                        decoded_t.item()== 2  # unk
                    if decoded_t.item() == self.eos_token_idx:
                        finished.append(state)
                    else:
                        decoded_t = decoded_t.view(1, -1)
                        new_state = state.update_clone(score_incre, last_hidden, decoded_t)
                        new_states.append(new_state)

                k += 1
            if self.beam_size - dead_k < 0:
                break
            new_states = new_states[:self.beam_size - dead_k]
            new_states.sort(key=lambda x: -x.score)
            states = new_states

            if t == self.max_ts - 1 and not finished:
                finished = failed
                print('FAIL')
                if not finished:
                    finished.append(states[0])

        finished.sort(key=lambda x: -x.score)
        decoded_t = finished[0].decoded
        decoded_t = [_.view(-1).item() for _ in decoded_t]
        #decoded_sentence = self.vocab.sentence_decode(decoded_t, self.eos_m_token)
        #print(decoded_sentence)
        generated = torch.cat(finished[0].decoded, dim=1).data  # [B=1, T]
        return generated

    def beam_search_decode(self, text_tm1, last_hidden, slot_enc_out, personality_enc_out):
        decoded = []
        if self.cfg.decoder_network == 'LSTM':
            vars = torch.split(text_tm1, 1, dim=1), torch.split(last_hidden[0], 1, dim=1), torch.split(last_hidden[1], 1, dim=1), \
                    torch.split(slot_enc_out, 1, dim=1), torch.split(personality_enc_out, 1, dim=1)
            for i, (text_tm1_s, last_hidden_h_s, last_hidden_c_s, slot_enc_out_s, personality_enc_out_s) \
                    in enumerate(zip(*vars)):
                decoded_s = self.beam_search_decode_single(text_tm1_s, (last_hidden_h_s, last_hidden_c_s), slot_enc_out_s, personality_enc_out_s)
                decoded.append(decoded_s)
        else:
            vars = torch.split(text_tm1, 1, dim=1), torch.split(last_hidden, 1, dim=1), torch.split(slot_enc_out, 1, dim=1), torch.split(personality_enc_out, 1, dim=1)
            for i, (text_tm1_s, last_hidden_s, slot_enc_out_s, personality_enc_out_s) \
                    in enumerate(zip(*vars)):
                decoded_s = self.beam_search_decode_single(text_tm1_s, last_hidden_s, slot_enc_out_s, personality_enc_out_s)
                decoded.append(decoded_s)

        return [list(_.view(-1)) for _ in decoded]


class Focused_VQVAE(torch.nn.Module):
    def __init__(self, cfg, vocab, decay=0, eos_m_token='EOS'):
        super(Focused_VQVAE, self).__init__()
        self.cfg = cfg
        self.vocab = vocab
        self.vae_encoder = LSTMDynamicEncoder(len(vocab), cfg.emb_size, cfg.hidden_size, cfg.encoder_layer_num,
                                              cfg.dropout_rate, cfg)
        self.encoder = LSTMDynamicEncoder(len(vocab), cfg.emb_size, cfg.hidden_size, cfg.encoder_layer_num,
                                          cfg.dropout_rate, cfg)
        if decay > 0.0:
            self.act_vq_vae = VectorQuantizerEMA(cfg, decay)
            self.personality_vq_vae = VectorQuantizerEMA(cfg, decay)
        else:
            self.act_vq_vae = VectorQuantizer(cfg)
            self.personality_vq_vae = VectorQuantizer(cfg)
        self.decoder = Attn_RNN_Decoder(len(vocab), cfg.emb_size, 2 * cfg.hidden_size, cfg.dropout_rate, vocab, cfg)
        self.act_predictor = MultiLabel_Classification(cfg.hidden_size, int(cfg.hidden_size / 2), cfg.act_size,
                                                       cfg.dropout_rate)
        self.personality_predictor = MultiClass_Classification(cfg.hidden_size, int(cfg.hidden_size / 2),
                                                               cfg.personality_size, cfg.dropout_rate)
        self.act_mlp = MLP(2*cfg.hidden_size, 4*cfg.hidden_size, cfg.hidden_size, cfg.dropout_rate)
        self.personality_mlp = MLP(2 * cfg.hidden_size, 4 * cfg.hidden_size, cfg.hidden_size, cfg.dropout_rate)

        
        self.dec_loss = torch.nn.NLLLoss(ignore_index=0, reduction='mean')
        self.max_ts = cfg.text_max_ts
        self.beam_search = cfg.beam_search
        self.teacher_force = cfg.teacher_force
        if self.beam_search:
            self.beam_size = cfg.beam_size
            self.eos_m_token = eos_m_token
            self.eos_token_idx = self.vocab.encode(eos_m_token)

    def forward(self, x, gt_y, mode, **kwargs):
        if mode == 'train' or mode == 'valid':
            loss, recon_loss, act_loss, personality_loss, act_vq_loss, personality_vq_loss = self.forward_turn(x, gt_y,
                                                                                                               mode,
                                                                                                               **kwargs)
            return loss, recon_loss, act_loss, personality_loss, act_vq_loss, personality_vq_loss
        elif mode == 'getDist':
            act_encoding, personality_encoding = self.forward_turn(x, gt_y, mode, **kwargs)
            return act_encoding, personality_encoding
        elif mode == 'test':
            pred_y, act_pred, personality_pred = self.forward_turn(x, gt_y, mode, **kwargs)
            return pred_y, act_pred, personality_pred

    def forward_turn(self, x, gt_y, mode, **kwargs):
        if self.cfg.remove_slot_value == True:
            x_len = kwargs['slot_len']  # batchsize
            x_np = kwargs['slot_np']  # seqlen, batchsize
            y_len = kwargs['delex_text_len']  # batchsize
            y_np = kwargs['delex_text_np']  # batchsize
        else:
            x_len = kwargs['slot_value_len']  # seqlen, batchsize
            x_np = kwargs['slot_value_np']  # batchsize
            y_len = kwargs['text_len']  # batchsize
            y_np = kwargs['text_np']  # seqlen, batchsize

        personality_idx = kwargs['personality_idx']
        personality_seq = kwargs['personality_seq']
        personality_len = kwargs['personality_len']
        act_idx = kwargs['act_idx']

        batch_size = x.size(1)
        x_enc_out, (h, c), _ = self.vae_encoder(gt_y, y_len)
        z = torch.cat([h[0], h[1]], dim=-1)
        act_z = self.act_mlp(z)
        act_vq_loss, act_quantized, act_perplexity, act_encoding = self.act_vq_vae(act_z)
        personality_z = self.personality_mlp(z)
        personality_vq_loss, personality_quantized, personality_perplexity, personality_encoding = self.personality_vq_vae(
            personality_z)
        text_tm1 = cuda_(torch.autograd.Variable(torch.ones(1, batch_size).long()), self.cfg)  # GO token
        text_length = gt_y.size(0)
        text_dec_proba = []
        text_dec_outs = []
        text_quantized_dec_outs = []
        text_vq_loss_s = []
        text_perplexity_s = []

        # act and personality distribution act_encoding (B, codebook_size)
        personality_enc_out, personality_hidden, personality_emb = self.encoder(personality_seq, personality_len,
                                                                                enc_out='cat')
        slot_enc_out, slot_hidden, slot_emb = self.encoder(x, x_len, personality_hidden, enc_out='cat')
        quantized = torch.cat([act_quantized, personality_quantized], dim=-1)
        # decoder_c = cuda_(torch.autograd.Variable(torch.zeros(quantized.size())), self.cfg)
        decoder_c = torch.cat([slot_hidden[1][0], slot_hidden[1][1]], dim=-1)
        '''
        personality_emb_copy = cuda_(torch.autograd.Variable(self.personality_vq_vae._embedding.weight.data.clone()), self.cfg).repeat(batch_size, 1, 1).transpose(0, 1)
        act_emb_copy = cuda_(torch.autograd.Variable(self.act_vq_vae._embedding.weight.data.clone()), self.cfg).repeat(batch_size, 1, 1).transpose(0, 1)
        personality_dist = self.personality_attn(personality_hidden[0][0]+personality_hidden[0][1], personality_emb_copy, return_normalize=True).squeeze(1)
        slot_dist = self.slot_attn(slot_hidden[0][0]+slot_hidden[0][1], act_emb_copy, return_normalize=True).squeeze(1)
        #
        '''
        if mode == 'getDist':
            return act_encoding, personality_encoding

        elif mode == 'train':
            if self.cfg.decoder_network == 'LSTM':
                last_hidden = (quantized.unsqueeze(0), decoder_c.unsqueeze(0))
            else:
                last_hidden = quantized.unsqueeze(0)
            act_loss = self.act_predictor(act_quantized, act_idx, mode)
            personality_loss = self.personality_predictor(personality_quantized, personality_idx, mode)
            ##prepare dist from encoding
            '''
            personality_encoding_dist = getDist(personality_idx, personality_encoding)
            slot_encoding_dist = getDist(act_idx, act_encoding)

            personality_KLdiv = torch.nn.functional.kl_div(personality_dist.log(), personality_encoding_dist)
            slot_KLdiv = torch.nn.functional.kl_div(slot_dist.log(), slot_encoding_dist)
            '''
            ##
            for t in range(text_length):
                teacher_forcing = toss_(self.teacher_force)
                proba, last_hidden, dec_out = self.decoder(text_tm1, last_hidden, slot_enc_out, personality_enc_out)
                if teacher_forcing:
                    text_tm1 = gt_y[t].view(1, -1)
                else:
                    _, text_tm1 = torch.topk(proba, 1)
                    text_tm1 = text_tm1.view(1, -1)
                text_dec_proba.append(proba)
                text_dec_outs.append(dec_out)

            text_dec_proba = torch.stack(text_dec_proba, dim=0)  # [T,B,V]
            pred_y = text_dec_proba
            recon_loss = self.dec_loss( \
                torch.log(pred_y.view(-1, pred_y.size(2))), \
                gt_y.view(-1))

            loss = recon_loss + act_loss + personality_loss + act_vq_loss + personality_vq_loss
            return loss, recon_loss, act_loss, personality_loss, act_vq_loss, personality_vq_loss
        else:
            act_sample_idx = kwargs['act_sample_idx']
            personality_sample_idx = kwargs['personality_sample_idx']
            act_sample_emb = self.act_vq_vae._embedding(act_sample_idx)
            personality_sample_emb = self.personality_vq_vae._embedding(personality_sample_idx)
            sample_quantized = torch.cat([act_sample_emb, personality_sample_emb], dim=-1)
            if self.cfg.decoder_network == 'LSTM':
                last_hidden = (sample_quantized.transpose(0, 1), decoder_c.unsqueeze(0))
            else:
                last_hidden = quantized.transpose(0, 1)
            act_pred = self.act_predictor(act_quantized, act_idx, mode)
            personality_pred = self.personality_predictor(personality_quantized, personality_idx, mode)
            if mode == 'test':
                if not self.cfg.beam_search:
                    text_dec_idx = self.greedy_decode(text_tm1, last_hidden, slot_enc_out, personality_enc_out)

                else:
                    text_dec_idx = self.beam_search_decode(text_tm1, last_hidden, slot_enc_out, personality_enc_out)

                return text_dec_idx, act_pred, personality_pred

    def greedy_decode(self, text_tm1, last_hidden, slot_enc_out, personality_enc_out):
        decoded = []
        for t in range(self.max_ts):
            proba, last_hidden, _ = self.decoder(text_tm1, last_hidden, slot_enc_out, personality_enc_out)
            mt_proba, mt_index = torch.topk(proba, 1)  # [B,1]
            mt_index = mt_index.data.view(-1)
            decoded.append(mt_index.clone())
            for i in range(mt_index.size(0)):
                if mt_index[i] >= len(self.vocab):
                    mt_index[i] = 2  # unk
            text_tm1 = cuda_(torch.autograd.Variable(mt_index).view(1, -1), self.cfg)
        decoded = torch.stack(decoded, dim=0).transpose(0, 1)
        decoded = list(decoded)
        return [list(_) for _ in decoded]

    def beam_search_decode_single(self, text_tm1, last_hidden, slot_enc_out, personality_enc_out):
        eos_token_id = self.eos_token_idx
        batch_size = text_tm1.size(1)
        if batch_size != 1:
            raise ValueError('"Beam search single" requires batch size to be 1')

        class BeamState:
            def __init__(self, score, last_hidden, decoded, length):
                """
                Beam state in beam decoding
                :param score: sum of log-probabilities
                :param last_hidden: last hidden
                :param decoded: list of *Variable[1*1]* of all decoded words
                :param length: current decoded sentence length
                """
                self.score = score
                self.last_hidden = last_hidden
                self.decoded = decoded
                self.length = length

            def update_clone(self, score_incre, last_hidden, decoded_t):
                decoded = copy.copy(self.decoded)
                decoded.append(decoded_t)
                clone = BeamState(self.score + score_incre, last_hidden, decoded, self.length + 1)
                return clone

        def score_bonus(state, decoded):
            bonus = self.cfg.beam_len_bonus
            return bonus

        def soft_score_incre(score, turn):
            return score

        finished, failed = [], []
        states = []  # sorted by score decreasingly
        dead_k = 0
        states.append(BeamState(0, last_hidden, [text_tm1], 0))
        for t in range(self.max_ts):
            new_states = []
            k = 0
            while k < len(states) and k < self.beam_size - dead_k:
                state = states[k]
                last_hidden, text_tm1 = state.last_hidden, state.decoded[-1]
                proba, last_hidden, _ = self.decoder(text_tm1, last_hidden, slot_enc_out, personality_enc_out)
                proba = torch.log(proba)
                mt_proba, mt_index = torch.topk(proba, self.beam_size - dead_k)  # [1,K]
                for new_k in range(self.beam_size - dead_k):
                    score_incre = soft_score_incre(mt_proba[0][new_k].item(), t) + score_bonus(state,
                                                                                               mt_index[0][
                                                                                                   new_k].item())
                    if len(new_states) >= self.beam_size - dead_k and state.score + score_incre < new_states[-1].score:
                        break
                    decoded_t = mt_index[0][new_k]
                    if decoded_t.item() >= len(self.vocab):
                        decoded_t.item() == 2  # unk
                    if decoded_t.item() == self.eos_token_idx:
                        finished.append(state)
                    else:
                        decoded_t = decoded_t.view(1, -1)
                        new_state = state.update_clone(score_incre, last_hidden, decoded_t)
                        new_states.append(new_state)

                k += 1
            if self.beam_size - dead_k < 0:
                break
            new_states = new_states[:self.beam_size - dead_k]
            new_states.sort(key=lambda x: -x.score)
            states = new_states

            if t == self.max_ts - 1 and not finished:
                finished = failed
                print('FAIL')
                if not finished:
                    finished.append(states[0])

        finished.sort(key=lambda x: -x.score)
        decoded_t = finished[0].decoded
        decoded_t = [_.view(-1).item() for _ in decoded_t]
        # decoded_sentence = self.vocab.sentence_decode(decoded_t, self.eos_m_token)
        # print(decoded_sentence)
        generated = torch.cat(finished[0].decoded, dim=1).data  # [B=1, T]
        return generated

    def beam_search_decode(self, text_tm1, last_hidden, slot_enc_out, personality_enc_out):
        decoded = []
        if self.cfg.decoder_network == 'LSTM':
            vars = torch.split(text_tm1, 1, dim=1), torch.split(last_hidden[0], 1, dim=1), torch.split(last_hidden[1],
                                                                                                       1, dim=1), \
                   torch.split(slot_enc_out, 1, dim=1), torch.split(personality_enc_out, 1, dim=1)
            for i, (text_tm1_s, last_hidden_h_s, last_hidden_c_s, slot_enc_out_s, personality_enc_out_s) \
                    in enumerate(zip(*vars)):
                decoded_s = self.beam_search_decode_single(text_tm1_s, (last_hidden_h_s, last_hidden_c_s),
                                                           slot_enc_out_s, personality_enc_out_s)
                decoded.append(decoded_s)
        else:
            vars = torch.split(text_tm1, 1, dim=1), torch.split(last_hidden, 1, dim=1), torch.split(slot_enc_out, 1,
                                                                                                    dim=1), torch.split(
                personality_enc_out, 1, dim=1)
            for i, (text_tm1_s, last_hidden_s, slot_enc_out_s, personality_enc_out_s) \
                    in enumerate(zip(*vars)):
                decoded_s = self.beam_search_decode_single(text_tm1_s, last_hidden_s, slot_enc_out_s,
                                                           personality_enc_out_s)
                decoded.append(decoded_s)

        return [list(_.view(-1)) for _ in decoded]
    
class VAE(torch.nn.Module):
    def __init__(self, hidden_size, var_size):
        super(VAE, self).__init__()
        self.hidden_size = hidden_size
        self.var_size = var_size
        self.fc1 = torch.nn.Linear(hidden_size, hidden_size)
        self.fc21 = torch.nn.Linear(hidden_size, var_size)
        self.fc22 = torch.nn.Linear(hidden_size, var_size)
        self.fc3 = torch.nn.Linear(var_size, hidden_size)
        self.fc4 = torch.nn.Linear(hidden_size, hidden_size)

    def encode(self, x):
        h1 = torch.nn.functional.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = torch.nn.functional.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.hidden_size))
        z = self.reparameterize(mu, logvar)
        return z, self.decode(z), mu, logvar


class Attn(torch.nn.Module):
    def __init__(self, hidden_size):
        super(Attn, self).__init__()
        self.hidden_size = hidden_size
        self.attn = torch.nn.Linear(self.hidden_size * 2, hidden_size)
        self.v = torch.nn.Parameter(torch.zeros(hidden_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.normal_(mean=0, std=stdv)

    def forward(self, hidden, encoder_outputs, normalize=True, return_normalize=False):
        encoder_outputs = encoder_outputs.transpose(0, 1)  # [B,T,H]
        attn_energies = self.score(hidden, encoder_outputs)
        normalized_energy = torch.nn.functional.softmax(attn_energies, dim=2)  # [B,1,T]
        context = torch.bmm(normalized_energy, encoder_outputs)  # [B,1,H]
        if return_normalize:
            return normalized_energy
        else:
            return context.transpose(0, 1)  # [1,B,H]

    def score(self, hidden, encoder_outputs):
        max_len = encoder_outputs.size(1)
        H = hidden.repeat(max_len, 1, 1).transpose(0, 1)
        energy = torch.tanh(self.attn(torch.cat([H, encoder_outputs], 2)))  # [B,T,2H]->[B,T,H]
        energy = energy.transpose(2, 1)  # [B,H,T]
        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)  # [B,1,H]
        energy = torch.bmm(v, energy)  # [B,1,T]
        return energy


class SimpleDynamicEncoder(torch.nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, n_layers, dropout, cfg, bidirectional=True):
        super(SimpleDynamicEncoder, self).__init__()
        self.cfg = cfg
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.embedding = torch.nn.Embedding(vocab_size, embed_size)
        self.gru = torch.nn.GRU(embed_size, hidden_size, n_layers, dropout=dropout, bidirectional=bidirectional)
        init_gru(self.gru)

    def forward(self, input_seqs, input_lens, hidden=None):
        """
        forward procedure. No need for inputs to be sorted
        :param input_seqs: Variable of [T,B]
        :param hidden:
        :param input_lens: *numpy array* of len for each input sequence
        :return:
        """
        embedded = self.embedding(input_seqs)
        embedded = embedded.transpose(0, 1)  # [B,T,E]
        sort_idx = np.argsort(-input_lens)
        unsort_idx = cuda_(torch.LongTensor(np.argsort(sort_idx)), self.cfg)
        input_lens = input_lens[sort_idx]
        sort_idx = cuda_(torch.LongTensor(sort_idx), self.cfg)
        embedded = embedded[sort_idx].transpose(0, 1)  # [T,B,E]
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lens)
        outputs, hidden = self.gru(packed, hidden)
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)
        if self.bidirectional:
            outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]
        outputs = outputs.transpose(0, 1)[unsort_idx].transpose(0, 1).contiguous()
        hidden = hidden.transpose(0, 1)[unsort_idx].transpose(0, 1).contiguous()
        return outputs, hidden, embedded


class Copy_Decoder(torch.nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, dropout_rate, vocab, cfg):
        super(Copy_Decoder, self).__init__()
        self.cfg = cfg
        self.emb = torch.nn.Embedding(vocab_size, embed_size)
        self.attn_a = Attn(hidden_size)
        self.attn_p = Attn(hidden_size)
        self.gru = torch.nn.GRU(embed_size+2*hidden_size, hidden_size, 1, dropout=dropout_rate, bidirectional=False)
        init_gru(self.gru)
        self.proj = torch.nn.Linear(3*hidden_size, vocab_size)
        self.proj_copy1 = torch.nn.Linear(hidden_size, hidden_size)
        self.proj_copy2 = torch.nn.Linear(hidden_size, hidden_size)
        self.dropout_rate = dropout_rate
        self.vocab = vocab

    def get_sparse_selective_input(self, x_input_np):
        #seqlen, batchsize
        result = np.zeros((x_input_np.shape[0], x_input_np.shape[1], len(self.vocab) + x_input_np.shape[0]),
                          dtype=np.float32)
        result.fill(1e-10)
        for t in range(x_input_np.shape[0]):
            for b in range(x_input_np.shape[1]):
                w = x_input_np[t][b]
                word = self.vocab.decode(w)
                if w == 2 or w >= len(self.vocab):
                    result[t][b][len(self.vocab) + t] = 5.0
                else:
                    if 'EOS' not in word and w != 0:
                        result[t][b][w] = result[t][b][w] + 1.0

        result_np = result.transpose((1, 0, 2))
        result = torch.from_numpy(result_np).float()#batchsize, seqlen, vocabsize+seqlen
        return result

    def forward(self, slot_enc_out, slot_np, personality_enc_out, personality_np, m_t_input, last_hidden):
        sparse_u_input = torch.autograd.Variable(self.get_sparse_selective_input(slot_np), requires_grad=False)  #singal encoded sentence
        m_embed = self.emb(m_t_input)
        a_context = self.attn_a(last_hidden, slot_enc_out)
        p_context = self.attn_p(last_hidden, personality_enc_out)
        gru_in = torch.cat([m_embed, a_context, p_context], dim=2)
        gru_out, last_hidden = self.gru(gru_in, last_hidden)
        gen_score = self.proj(torch.cat([a_context, p_context, gru_out], 2)).squeeze(0)

        u_copy_score = torch.tanh(self.proj_copy2(slot_enc_out.transpose(0, 1)))
        u_copy_score = torch.matmul(u_copy_score, gru_out.squeeze(0).unsqueeze(2)).squeeze(2)
        u_copy_score = u_copy_score.cpu()
        u_copy_score_max = torch.max(u_copy_score, dim=1, keepdim=True)[0]
        u_copy_score = torch.exp(u_copy_score - u_copy_score_max)  # [B,T]
        u_copy_score = torch.log(torch.bmm(u_copy_score.unsqueeze(1), sparse_u_input)).squeeze(
            1) + u_copy_score_max  # [B,V]
        u_copy_score = cuda_(u_copy_score, self.cfg)
        scores = torch.nn.functional.softmax(torch.cat([gen_score, u_copy_score], dim=1), dim=1)
        gen_score, u_copy_score = scores[:, :len(self.vocab)], \
                                  scores[:, len(self.vocab):]
        proba = gen_score + u_copy_score[:, :len(self.vocab)]  # [B,V]
        proba = torch.cat([proba, u_copy_score[:, len(self.vocab):]], 1)
        return proba, last_hidden, gru_out


class Simple_Decoder(torch.nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size,  dropout_rate, vocab, cfg):
        super(Simple_Decoder, self).__init__()
        self.cfg = cfg
        self.emb = torch.nn.Embedding(vocab_size, embed_size)
        self.attn_a = Attn(hidden_size)
        self.attn_p = Attn(hidden_size)
        self.gru = torch.nn.GRU(embed_size+2*hidden_size, hidden_size, 1, dropout=dropout_rate, bidirectional=False)
        init_gru(self.gru)
        self.proj = torch.nn.Linear(3*hidden_size, vocab_size)
        self.dropout_rate = dropout_rate
        self.vocab = vocab

    def forward(self, slot_enc_out, slot_np, personality_enc_out, personality_np, m_t_input, last_hidden):
        m_embed = self.emb(m_t_input)
        a_context = self.attn_a(last_hidden, slot_enc_out)
        p_context = self.attn_p(last_hidden, personality_enc_out)
        gru_in = torch.cat([m_embed, a_context, p_context], dim=2)
        gru_out, last_hidden = self.gru(gru_in, last_hidden)
        gen_score = self.proj(torch.cat([a_context, p_context, gru_out], 2)).squeeze(0)
        proba = torch.nn.functional.softmax(gen_score, dim=1)
        return proba, last_hidden, gru_out


class Seq2Seq(torch.nn.Module):
    def __init__(self, cfg, vocab, eos_m_token='EOS', mode='copy'):
        super(Seq2Seq, self).__init__()
        self.vocab = vocab
        self.cfg = cfg
        self.encoder = SimpleDynamicEncoder(len(vocab), cfg.emb_size, cfg.hidden_size, cfg.encoder_layer_num, cfg.dropout_rate, cfg)
        if cfg.VAE:
            self.vae = VAE(cfg.hidden_size, cfg.hidden_size)
        if mode == 'copy':
            self.decoder = Copy_Decoder(len(vocab), cfg.emb_size, cfg.hidden_size, cfg.dropout_rate, vocab, cfg)
        else:
            self.decoder = Simple_Decoder(len(vocab), cfg.emb_size, cfg.hidden_size, cfg.dropout_rate, vocab, cfg)
        self.max_ts = cfg.text_max_ts
        self.beam_search = cfg.beam_search
        self.teacher_force = cfg.teacher_force
        self.dec_loss = torch.nn.NLLLoss(ignore_index=0, reduction='mean')
        if self.beam_search:
            self.beam_size = cfg.beam_size
            self.eos_m_token = eos_m_token
            self.eos_token_idx = self.vocab.encode(eos_m_token)

    def forward(self, x, gt_y, mode, **kwargs):
        if mode == 'train' or mode == 'valid':
            pred_y, KLD = self.forward_turn(x, gt_y, mode, **kwargs)
            loss, network_loss = self.supervised_loss(pred_y, gt_y, KLD)
            return loss, network_loss, KLD
        elif mode == 'test':
            pred_y = self.forward_turn(x, gt_y, mode, **kwargs)
            return pred_y

    def forward_turn(self, x, gt_y, mode, **kwargs):
        if self.cfg.remove_slot_value == True:
            x_len = kwargs['slot_len']  # batchsize
            x_np = kwargs['slot_np']  # seqlen, batchsize
            y_len = kwargs['delex_text_len']  # batchsize
            y_np = kwargs['delex_text_np']  # batchsize
        else:
            x_len = kwargs['slot_value_len']  # seqlen, batchsize
            x_np = kwargs['slot_value_np']  # batchsize
            y_len = kwargs['text_len']  # batchsize
            y_np = kwargs['text_np']  # seqlen, batchsize

        personality_seq = kwargs['personality_seq']
        personality_len = kwargs['personality_len']#seqlen, batchsize
        personality_np = kwargs['personality_np']#seqlen, batchsize

        batch_size = x.size(1)        
        personality_enc_out, personality_hidden, personality_emb = self.encoder(personality_seq, personality_len)
        slot_enc_out, slot_hidden, slot_emb = self.encoder(x, x_len, personality_hidden)
        last_hidden = slot_hidden[(self.cfg.encoder_layer_num-1)*2:-1]
        
        if self.cfg.VAE:
            z, decode_z, mu, logvar = self.vae(last_hidden)
            last_hidden = z.unsqueeze(0)
            KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        if self.cfg.various_go:
            text_tm1 = kwargs['go']
        else:
            text_tm1 = cuda_(torch.autograd.Variable(torch.ones(1, batch_size).long()), self.cfg)  # GO token
        text_length = gt_y.size(0)
        text_dec_proba = []
        text_dec_outs = []
        if mode == 'train':
            for t in range(text_length):
                teacher_forcing = toss_(self.teacher_force)
                proba, last_hidden, dec_out = self.decoder(slot_enc_out, x_np, \
                                                           personality_enc_out, personality_np, \
                                                             text_tm1, last_hidden)
                if teacher_forcing:
                    text_tm1 = gt_y[t].view(1, -1)
                else:
                    _, text_tm1 = torch.topk(proba, 1)
                    text_tm1 = text_tm1.view(1, -1)
                text_dec_proba.append(proba)
                text_dec_outs.append(dec_out)
            text_dec_proba = torch.stack(text_dec_proba, dim=0)  # [T,B,V]
            if self.cfg.VAE:
                return text_dec_proba, KLD
            else:
                return text_dec_proba, None
        else:
            if mode == 'test':
                if not self.beam_search:
                    text_dec_idx = self.greedy_decode(slot_enc_out, x_np, \
                                                       personality_enc_out, personality_np, \
                                                             text_tm1,  last_hidden)

                else:
                    text_dec_idx = self.beam_search_decode(slot_enc_out, x_np, \
                                                       personality_enc_out, personality_np, \
                                                             text_tm1,  last_hidden)

                return text_dec_idx

    def greedy_decode(self, slot_enc_out, slot_np, personality_enc_out, personality_np, text_tm1,  last_hidden):
        decoded = []
        for t in range(self.max_ts):
            proba, last_hidden, _ = self.decoder(slot_enc_out, slot_np, \
                                                       personality_enc_out, personality_np, \
                                                             text_tm1,  last_hidden)
            mt_proba, mt_index = torch.topk(proba, 1)  # [B,1]
            mt_index = mt_index.data.view(-1)
            decoded.append(mt_index.clone())
            for i in range(mt_index.size(0)):
                if mt_index[i] >= len(self.vocab):
                    mt_index[i] = 2  # unk
            text_tm1 = cuda_(torch.autograd.Variable(mt_index).view(1, -1), self.cfg)
        decoded = torch.stack(decoded, dim=0).transpose(0, 1)
        decoded = list(decoded)
        return [list(_) for _ in decoded]

    def beam_search_decode_single(self, slot_enc_out, slot_np, personality_enc_out, personality_np, text_tm1, last_hidden):
        eos_token_id = self.eos_token_idx
        batch_size = last_hidden.size(1)
        if batch_size != 1:
            raise ValueError('"Beam search single" requires batch size to be 1')

        class BeamState:
            def __init__(self, score, last_hidden, decoded, length):
                """
                Beam state in beam decoding
                :param score: sum of log-probabilities
                :param last_hidden: last hidden
                :param decoded: list of *Variable[1*1]* of all decoded words
                :param length: current decoded sentence length
                """
                self.score = score
                self.last_hidden = last_hidden
                self.decoded = decoded
                self.length = length

            def update_clone(self, score_incre, last_hidden, decoded_t):
                decoded = copy.copy(self.decoded)
                decoded.append(decoded_t)
                clone = BeamState(self.score + score_incre, last_hidden, decoded, self.length + 1)
                return clone

        def score_bonus(state, decoded):
            bonus = self.cfg.beam_len_bonus
            return bonus

        def soft_score_incre(score, turn):
            return score

        finished, failed = [], []
        states = []  # sorted by score decreasingly
        dead_k = 0
        states.append(BeamState(0, last_hidden, [text_tm1], 0))
        for t in range(self.max_ts):
            new_states = []
            k = 0
            while k < len(states) and k < self.beam_size - dead_k:
                state = states[k]
                last_hidden, text_tm1 = state.last_hidden, state.decoded[-1]
                proba, last_hidden, _ = self.decoder(slot_enc_out, slot_np, \
                                                       personality_enc_out, personality_np, \
                                                             text_tm1,  last_hidden)

                proba = torch.log(proba)
                mt_proba, mt_index = torch.topk(proba, self.beam_size - dead_k)  # [1,K]
                for new_k in range(self.beam_size - dead_k):
                    score_incre = soft_score_incre(mt_proba[0][new_k].item(), t) + score_bonus(state,
                                                                                                mt_index[0][new_k].item())
                    if len(new_states) >= self.beam_size - dead_k and state.score + score_incre < new_states[-1].score:
                        break
                    decoded_t = mt_index[0][new_k]
                    if decoded_t.item() >= len(self.vocab):
                        decoded_t.item()== 2  # unk
                    if decoded_t.item() == self.eos_token_idx:
                        finished.append(state)
                    else:
                        decoded_t = decoded_t.view(1, -1)
                        new_state = state.update_clone(score_incre, last_hidden, decoded_t)
                        new_states.append(new_state)

                k += 1
            if self.beam_size - dead_k < 0:
                break
            new_states = new_states[:self.beam_size - dead_k]
            new_states.sort(key=lambda x: -x.score)
            states = new_states

            if t == self.max_ts - 1 and not finished:
                finished = failed
                print('FAIL')
                if not finished:
                    finished.append(states[0])

        finished.sort(key=lambda x: -x.score)
        decoded_t = finished[0].decoded
        decoded_t = [_.view(-1).item() for _ in decoded_t]
        #decoded_sentence = self.vocab.sentence_decode(decoded_t, self.eos_m_token)
        #print(decoded_sentence)
        generated = torch.cat(finished[0].decoded, dim=1).data  # [B=1, T]
        return generated


    def beam_search_decode(self, slot_enc_out, slot_np, personality_enc_out, personality_np, text_tm1, last_hidden):

        vars = torch.split(slot_enc_out, 1, dim=1), torch.split(personality_enc_out, 1, dim=1), \
                torch.split(text_tm1, 1, dim=1), torch.split(last_hidden, 1, dim=1)
        decoded = []
        for i, (slot_enc_out_s, personality_enc_out_s, text_tm1_s, last_hidden_s) \
            in enumerate(zip(*vars)):
            decoded_s = self.beam_search_decode_single(slot_enc_out_s, slot_np[:, i].reshape((-1, 1)),\
                                                           personality_enc_out_s,personality_np[:, i].reshape((-1, 1)),\
                                                           text_tm1_s, last_hidden_s)
            decoded.append(decoded_s)
        return [list(_.view(-1)) for _ in decoded]

    def supervised_loss(self, pred_y, gt_y, KLD):
        network_loss = self.dec_loss(\
            torch.log(pred_y.view(-1, pred_y.size(2))),\
            gt_y.view(-1))
        if KLD is not None:
            loss = network_loss+KLD
        else:
            loss = network_loss
        return loss, network_loss


class ClassificationNetwork(torch.nn.Module):
    def __init__(self, cfg, vocab):
        super(ClassificationNetwork, self).__init__()
        self.cfg = cfg
        self.encoder = LSTMDynamicEncoder(len(vocab), cfg.emb_size, cfg.hidden_size, cfg.layer_num,
                                            cfg.dropout_rate, cfg)
        self.linear1 = torch.nn.Linear(in_features=2*cfg.hidden_size, out_features=cfg.hidden_size, bias=True)
        self.linear2 = torch.nn.Linear(in_features=cfg.hidden_size, out_features=cfg.output_size, bias=True)
        self.dropout_rate = cfg.dropout_rate

    def forward(self, x, gt_y, mode, **kwargs):
        text_seq = x#seqlen, batchsize
        text_seq_len = kwargs['delex_text_len']#batchsize
        _enc_out, (h, _c), _emb = self.encoder(text_seq, text_seq_len)
        z = torch.cat([h[0], h[1]], dim=-1)
        linear1_h = torch.nn.functional.relu(self.linear1(z))#x (N, *, in_features) linear1_x (N, *, out_features)
        linear1_h = torch.nn.functional.dropout(linear1_h, self.dropout_rate)
        linear2_h = self.linear2(linear1_h)
        if mode == 'test':
            output = torch.softmax(linear2_h, dim=-1)
            output = output.data.cpu().numpy()
        elif mode == 'train':
            output = torch.nn.functional.cross_entropy(linear2_h, gt_y)
        else:
            assert()
        return output


def get_network(cfg, vocab):
    if cfg.network == 'classification':
        return ClassificationNetwork(cfg, vocab)
    elif 'seq2seq' in cfg.network:
        if 'simple' in cfg.network:
            return Seq2Seq(cfg, vocab, mode='simple')
        else:
            return Seq2Seq(cfg, vocab, mode='copy')
    elif 'VQVAE' in cfg.network:
        if 'simple' in cfg.network:
            return VQVAE(cfg, vocab)
        elif 'controlled' in cfg.network:
            return Controlled_VQVAE(cfg, vocab)
        elif 'focused' in cfg.network:
            return Focused_VQVAE(cfg, vocab)
