# -*- coding: utf-8 -*-
# @Author: Xiaoyuan Yi
# @Last Modified by:   Xiaoyuan Yi
# @Last Modified time: 2020-06-11 20:16:17
# @Email: yi-xy16@mails.tsinghua.edu.cn
# @Description:
'''
Copyright 2020 THUNLP Lab. All Rights Reserved.
This code is part of the online Chinese poetry generation system, Jiuge.
System URL: https://jiuge.thunlp.cn/ and https://jiuge.thunlp.org/.
Github: https://github.com/THUNLP-AIPoet.
'''
import random
from itertools import chain
import torch
from torch import nn
import torch.nn.functional as F

from layers import BidirEncoder, Decoder, MLP, ContextLayer, AttentionReader, AttentionWriter

def get_non_pad_mask(seq, pad_idx, device):
    # seq: [B, L]
    assert seq.dim() == 2
    # [B, L]
    mask = seq.ne(pad_idx).type(torch.float)
    return mask.to(device)


def get_seq_length(seq, pad_idx, device):
    mask = get_non_pad_mask(seq, pad_idx, device)
    # mask: [B, T]
    lengths = mask.sum(dim=-1).long()
    return lengths


class WorkingMemoryModel(nn.Module):
    def __init__(self, hps, device):
        super(WorkingMemoryModel, self).__init__()
        self.hps = hps
        self.device = device

        self.global_trace_size = hps.global_trace_size
        self.topic_trace_size = hps.topic_trace_size
        self.topic_slots = hps.topic_slots
        self.his_mem_slots = hps.his_mem_slots

        self.vocab_size = hps.vocab_size
        self.mem_size = hps.mem_size

        self.sens_num = hps.sens_num

        self.pad_idx = hps.pad_idx
        self.bos_tensor = torch.tensor(hps.bos_idx, dtype=torch.long, device=device)

        # ----------------------------
        # build componets
        self.layers = nn.ModuleDict()
        self.layers['word_embed'] = nn.Embedding(hps.vocab_size,
            hps.word_emb_size, padding_idx=hps.pad_idx)

        # NOTE: We set fixed 33 phonology categories: 0~32
        #   please refer to preprocess.py for more details
        self.layers['ph_embed'] = nn.Embedding(33, hps.ph_emb_size)

        self.layers['len_embed'] = nn.Embedding(hps.sen_len, hps.len_emb_size)


        self.layers['encoder'] = BidirEncoder(hps.word_emb_size, hps.hidden_size, drop_ratio=hps.drop_ratio)
        self.layers['decoder'] = Decoder(hps.hidden_size, hps.hidden_size, drop_ratio=hps.drop_ratio)

        # project the decoder hidden state to a vocanbulary-size output logit
        self.layers['out_proj'] = nn.Linear(hps.hidden_size, hps.vocab_size)

        # update the context vector
        self.layers['global_trace_updater'] = ContextLayer(hps.hidden_size, hps.global_trace_size)
        self.layers['topic_trace_updater'] = MLP(self.mem_size+self.topic_trace_size,
            layer_sizes=[self.topic_trace_size], activs=['tanh'], drop_ratio=hps.drop_ratio)


        # MLP for calculate initial decoder state
        self.layers['dec_init'] = MLP(hps.hidden_size*2, layer_sizes=[hps.hidden_size],
            activs=['tanh'], drop_ratio=hps.drop_ratio)
        self.layers['key_init'] = MLP(hps.hidden_size*2, layer_sizes=[hps.hidden_size],
            activs=['tanh'], drop_ratio=hps.drop_ratio)

        # history memory reading and writing layers
        # query: concatenation of hidden state, global_trace and topic_trace
        self.layers['memory_read'] = AttentionReader(
            d_q=hps.hidden_size+self.global_trace_size+self.topic_trace_size+self.topic_slots,
            d_v=hps.mem_size, drop_ratio=hps.attn_drop_ratio)

        self.layers['memory_write'] = AttentionWriter(hps.mem_size+self.global_trace_size, hps.mem_size)

        # NOTE: a layer to compress the encoder hidden states to a smaller size for larger number of slots
        self.layers['mem_compress'] = MLP(hps.hidden_size*2, layer_sizes=[hps.mem_size],
            activs=['tanh'], drop_ratio=hps.drop_ratio)

        # [inp, attns, ph_inp, len_inp, global_trace]
        self.layers['merge_x'] = MLP(
            hps.word_emb_size+hps.ph_emb_size+hps.len_emb_size+hps.global_trace_size+hps.mem_size,
            layer_sizes=[hps.hidden_size],
            activs=['tanh'], drop_ratio=hps.drop_ratio)


        # two annealing parameters
        self._tau = 1.0
        self._teach_ratio = 0.8


        # ---------------------------------------------------------
        # only used for for pre-training
        self.layers['dec_init_pre'] = MLP(hps.hidden_size*2,
            layer_sizes=[hps.hidden_size],
            activs=['tanh'], drop_ratio=hps.drop_ratio)

        self.layers['merge_x_pre'] = MLP(
            hps.word_emb_size+hps.ph_emb_size+hps.len_emb_size,
            layer_sizes=[hps.hidden_size],
            activs=['tanh'], drop_ratio=hps.drop_ratio)



    #---------------------------------
    def set_tau(self, tau):
        if 0.0 < tau <= 1.0:
            self.layers['memory_write'].set_tau(tau)

    def get_tau(self):
        return self.layers['memory_write'].get_tau()

    def set_teach_ratio(self, teach_ratio):
        if 0.0 < teach_ratio <= 1.0:
            self._teach_ratio = teach_ratio

    def get_teach_ratio(self):
        return self._teach_ratio


    def set_null_idxes(self, null_idxes):
        self.null_idxes = null_idxes.to(self.device).unsqueeze(0)


    #---------------------------------
    def compute_null_mem(self, batch_size):
        # we initialize the null memory slot with an average of stop words
        #    by supposing that the model could learn to ignore these words
        emb_null = self.layers['word_embed'](self.null_idxes)

        # (1, L, 2*H)
        enc_outs, _ = self.layers['encoder'](emb_null)

        # (1, L, 2 * H) -> (1, L, D)
        null_mem = self.layers['mem_compress'](enc_outs)

        # (1, L, D)->(1, 1, D)->(B, 1, D)
        self.null_mem = null_mem.mean(dim=1, keepdim=True).repeat(batch_size, 1, 1)


    def computer_topic_memory(self, keys):
        # (B, key_len)
        emb_keys = [self.layers['word_embed'](key) for key in keys]
        key_lens = [get_seq_length(key, self.pad_idx, self.device) for key in keys]

        batch_size = emb_keys[0].size(0)

        # length == 0 means this is am empty topic slot
        topic_mask = torch.zeros(batch_size, self.topic_slots,
            dtype=torch.float, device=self.device).bool() # (B, topic_slots)
        for step in range(0, self.topic_slots):
            topic_mask[:, step] = torch.eq(key_lens[step], 0)


        key_states_vec, topic_slots = [], []
        for step, (emb_key, length) in enumerate(zip(emb_keys, key_lens)):

            # we set the length of empty keys to 1 for parallel processing,
            #   which will be masked then for memory reading
            length.masked_fill_(length.eq(0), 1)

            _, state = self.layers['encoder'](emb_key, length)
            # (2, B, H) -> (B, 2, H) -> (B, 2*H)
            key_state = state.transpose(0, 1).contiguous().view(batch_size, -1)
            mask = (1 - topic_mask[:, step].float()).unsqueeze(1) # (B, 1)

            key_states_vec.append((key_state*mask).unsqueeze(1))

            topic = self.layers['mem_compress'](key_state)
            topic_slots.append((topic*mask).unsqueeze(1))

        # (B, topic_slots, mem_size)
        topic_mem = torch.cat(topic_slots, dim=1)

        # (B, H)
        key_init_state = self.layers['key_init'](
            torch.cat(key_states_vec, dim=1).sum(1))

        return topic_mem, topic_mask, key_init_state


    def computer_local_memory(self, inps, with_length):
        batch_size = inps.size(0)
        if with_length:
            length = get_seq_length(inps, self.pad_idx, self.device)
        else:
            length = None

        emb_inps = self.layers['word_embed'](inps)

        # outs: (B, L, 2 * H)
        # states: (2, B, H)
        enc_outs, enc_states = self.layers['encoder'](emb_inps, length)

        init_state = self.layers['dec_init'](enc_states.transpose(0, 1).
            contiguous().view(batch_size, -1))

        # (B, L, 2 * H) -> (B, L, D)
        local_mem = self.layers['mem_compress'](enc_outs)

        local_mask = torch.eq(inps, self.pad_idx)

        return local_mem, local_mask, init_state


    def update_global_trace(self, old_global_trace, dec_states, dec_mask):
        states = torch.cat(dec_states, dim=2) # (B, H, L)
        global_trace = self.layers['global_trace_updater'](
            old_global_trace, states*(dec_mask.unsqueeze(1)))
        return global_trace


    def update_topic_trace(self, topic_trace, topic_mem, concat_aligns):
        # topic_trace: (B, topic_trace_size+topic_slots)
        # concat_aligns: (B, L_gen, mem_slots)

        # 1: topic memory, 2: history memory 3: local memory
        topic_align = concat_aligns[:, :, 0:self.topic_slots].mean(dim=1) # (B, topic_slots)

        # (B, topic_slots, mem_size) * (B, topic_slots, 1) -> (B, topic_slots, mem_size)
        #   -> (B, mem_size)
        topic_used = torch.mul(topic_mem, topic_align.unsqueeze(2)).mean(dim=1)


        new_topic_trace = self.layers['topic_trace_updater'](
            torch.cat([topic_trace[:, 0:self.topic_trace_size], topic_used], dim=1))

        read_log = topic_trace[:, self.topic_trace_size:] + topic_align

        fin_topic_trace = torch.cat([new_topic_trace, read_log], dim=1)

        return fin_topic_trace


    def dec_step(self, inp, state, ph, length, total_mem, total_mask,
        global_trace, topic_trace):

        emb_inp = self.layers['word_embed'](inp)
        emb_ph = self.layers['ph_embed'](ph)
        emb_len = self.layers['len_embed'](length)

        # query for reading read memory
        # (B, 1, H]
        query = torch.cat([state, global_trace, topic_trace], dim=1).unsqueeze(1)

        # attns: (B, 1, mem_size), align: (B, 1, L)
        attns, align = self.layers['memory_read'](query, total_mem, total_mem, total_mask)


        x = torch.cat([emb_inp, emb_ph, emb_len, attns, global_trace], dim=1).unsqueeze(1)
        x = self.layers['merge_x'](x)

        cell_out, new_state = self.layers['decoder'](x, state)
        out = self.layers['out_proj'](cell_out)
        return out, new_state, align


    def run_decoder(self, inps, trgs, phs, lens, key_init_state,
        history_mem, history_mask, topic_mem, topic_mask, global_trace, topic_trace,
        specified_teach_ratio):

        local_mem, local_mask, init_state = \
            self.computer_local_memory(inps, key_init_state is None)

        if key_init_state is not None:
            init_state = key_init_state

        if specified_teach_ratio is None:
            teach_ratio = self._teach_ratio
        else:
            teach_ratio = specified_teach_ratio


        # Note this order: 1: topic memory, 2: history memory 3: local memory
        total_mask = torch.cat([topic_mask, history_mask, local_mask], dim=1)
        total_mem = torch.cat([topic_mem, history_mem, local_mem], dim=1)

        batch_size = inps.size(0)
        trg_len = trgs.size(1)

        outs = torch.zeros(batch_size, trg_len, self.vocab_size,
            dtype=torch.float, device=self.device)

        state = init_state
        inp = self.bos_tensor.repeat(batch_size)
        dec_states, attn_weights = [], []

        # generate each line
        for t in range(0, trg_len):
            out, state, align = self.dec_step(inp, state, phs[:, t],
                lens[:, t], total_mem, total_mask, global_trace, topic_trace)
            outs[:, t, :] = out

            attn_weights.append(align)

            # teach force with a probability
            is_teach = random.random() < teach_ratio
            if is_teach or (not self.training):
                inp = trgs[:, t]
            else:
                normed_out = F.softmax(out, dim=-1)
                inp = normed_out.data.max(1)[1]

            dec_states.append(state.unsqueeze(2)) # (B, H, 1)
            attn_weights.append(align)



        # write the history memory
        if key_init_state is None:
            new_history_mem, _ = self.layers['memory_write'](history_mem, local_mem,
                1.0-local_mask.float(), global_trace, self.null_mem)
        else:
            new_history_mem = history_mem

        # (B, L)
        dec_mask = get_non_pad_mask(trgs, self.pad_idx, self.device)

        # update global trace vector
        new_global_trace = self.update_global_trace(global_trace, dec_states, dec_mask)


        # update topic trace vector
        # attn_weights: (B, 1, all_mem_slots) * L_gen
        concat_aligns = torch.cat(attn_weights, dim=1)
        new_topic_trace = self.update_topic_trace(topic_trace, topic_mem, concat_aligns)


        return outs, new_history_mem, new_global_trace, new_topic_trace



    def initialize_mems(self, keys):
        batch_size = keys[0].size(0)
        topic_mem, topic_mask, key_init_state = self.computer_topic_memory(keys)

        history_mem = torch.zeros(batch_size, self.his_mem_slots, self.mem_size,
            dtype=torch.float, device=self.device)

        # default: True, masked
        history_mask = torch.ones(batch_size, self.his_mem_slots,
            dtype=torch.float, device=self.device).bool()

        global_trace = torch.zeros(batch_size, self.global_trace_size,
            dtype=torch.float, device=self.device)
        topic_trace = torch.zeros(batch_size, self.topic_trace_size+self.topic_slots,
            dtype=torch.float, device=self.device)

        self.compute_null_mem(batch_size)

        return topic_mem, topic_mask, history_mem, history_mask,\
            global_trace, topic_trace, key_init_state


    def rebuild_inps(self, ori_inps, last_outs, teach_ratio):
        # ori_inps: (B, L)
        # last_outs: (B, L, V)
        inp_len = ori_inps.size(1)
        new_inps = torch.ones_like(ori_inps) * self.pad_idx

        mask = get_non_pad_mask(ori_inps, self.pad_idx, self.device).long()

        if teach_ratio is None:
            teach_ratio = self._teach_ratio

        for t in range(0, inp_len):
            is_teach = random.random() < teach_ratio
            if is_teach or (not self.training):
                new_inps[:, t] = ori_inps[:, t]
            else:
                normed_out = F.softmax(last_outs[:, t], dim=-1)
                new_inps[:, t] = normed_out.data.max(1)[1]

        new_inps = new_inps * mask

        return new_inps


    def forward(self, all_inps, all_trgs, all_ph_inps, all_len_inps, keys, teach_ratio=None,
        flexible_inps=False):
        '''
        all_inps: (B, L) * sens_num
        all_trgs: (B, L) * sens_num
        all_ph_inps: (B, L) * sens_num
        all_len_inps: (B, L) * sens_num
        keys: (B, L) * topic_slots
        flexible_inps: if apply partial teaching force to local memory.
            False: the ground-truth src line is stored into the local memory
            True: for local memory, ground-truth characters will be replaced with generated characters with
                the probability of 1- teach_ratio.
            NOTE: this trick is *not* adopted in our original paper, which could lead to
                better BLEU and topic relevance, but worse diversity of generated poems.
        '''
        all_outs = []

        topic_mem, topic_mask, history_mem, history_mask,\
            global_trace, topic_trace, key_init_state = self.initialize_mems(keys)

        for step in range(0, self.sens_num):
            if step > 0:
                key_init_state = None

            if step >= 1 and flexible_inps:
                inps = self.rebuild_inps(all_inps[step], all_outs[-1], teach_ratio)
            else:
                inps = all_inps[step]

            outs, history_mem, global_trace, topic_trace \
                = self.run_decoder(inps, all_trgs[step],
                    all_ph_inps[step], all_len_inps[step], key_init_state,
                    history_mem, history_mask, topic_mem, topic_mask,
                    global_trace, topic_trace, teach_ratio)

            if step >= 1:
                history_mask = history_mem.abs().sum(-1).eq(0) # (B, mem_slots)


            all_outs.append(outs)


        return all_outs



    # --------------------------
    # graphs for pre-training
    def dseq_graph(self, inps, trgs, ph_inps, len_inps, teach_ratio=None):
        # pre-train the encoder and decoder as a denoising Seq2Seq model
        batch_size, trg_len = trgs.size(0), trgs.size(1)
        length = get_seq_length(inps, self.pad_idx, self.device)


        emb_inps = self.layers['word_embed'](inps)
        emb_phs = self.layers['ph_embed'](ph_inps)
        emb_lens = self.layers['len_embed'](len_inps)


        # outs: (B, L, 2 * H)
        # states: (2, B, H)
        _, enc_states = self.layers['encoder'](emb_inps, length)


        init_state = self.layers['dec_init_pre'](enc_states.transpose(0, 1).
            contiguous().view(batch_size, -1))


        outs = torch.zeros(batch_size, trg_len, self.vocab_size,
            dtype=torch.float, device=self.device)

        if teach_ratio is None:
            teach_ratio = self._teach_ratio

        state = init_state
        inp = self.bos_tensor.repeat(batch_size, 1)

        # generate each line
        for t in range(0, trg_len):
            emb_inp = self.layers['word_embed'](inp)
            x = self.layers['merge_x_pre'](torch.cat(
                [emb_inp, emb_phs[:, t].unsqueeze(1), emb_lens[:, t].unsqueeze(1)],
                dim=-1))

            cell_out, state, = self.layers['decoder'](x, state)
            out = self.layers['out_proj'](cell_out)

            outs[:, t, :] = out

            # teach force with a probability
            is_teach = random.random() < teach_ratio
            if is_teach or (not self.training):
                inp = trgs[:, t].unsqueeze(1)
            else:
                normed_out = F.softmax(out, dim=-1)
                top1 = normed_out.data.max(1)[1]
                inp  = top1.unsqueeze(1)


        return outs


    # ----------------------------------------------
    def dseq_parameter_names(self):
        required_names = ['word_embed', 'ph_embed', 'len_embed',
            'encoder', 'decoder', 'out_proj',
            'dec_init_pre', 'merge_x_pre']
        return required_names

    def dseq_parameters(self):
        names = self.dseq_parameter_names()

        required_params = [self.layers[name].parameters() for name in names]

        return chain.from_iterable(required_params)

    # ------------------------------------------------