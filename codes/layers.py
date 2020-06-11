# -*- coding: utf-8 -*-
# @Author: Xiaoyuan Yi
# @Last Modified by:   Xiaoyuan Yi
# @Last Modified time: 2020-06-11 20:08:55
# @Email: yi-xy16@mails.tsinghua.edu.cn
# @Description:
'''
Copyright 2020 THUNLP Lab. All Rights Reserved.
This code is part of the online Chinese poetry generation system, Jiuge.
System URL: https://jiuge.thunlp.cn/ and https://jiuge.thunlp.org/.
Github: https://github.com/THUNLP-AIPoet.
'''
import math
from collections import OrderedDict

import torch
from torch import nn
import torch.nn.functional as F


class BidirEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, cell='GRU', n_layers=1, drop_ratio=0.1):
        super(BidirEncoder, self).__init__()

        if cell == 'GRU':
            self.rnn = nn.GRU(input_size, hidden_size, n_layers,
                bidirectional=True, batch_first=True)
        elif cell == 'Elman':
            self.rnn = nn.RNN(input_size, hidden_size, n_layers,
                bidirectional=True, batch_first=True)
        elif cell == 'LSTM':
            self.rnn = nn.LSTM(input_size, hidden_size, n_layers,
                bidirectional=True, batch_first=True)

        self.dropout_layer = nn.Dropout(drop_ratio)


    def forward(self, embed_seq, input_lens=None):
        # embed_seq: (B, L, emb_dim)
        # input_lens: (B)
        embed_inps = self.dropout_layer(embed_seq)

        if input_lens is None:
            outputs, state = self.rnn(embed_inps, None)
        else:
            # Dynamic RNN
            total_len = embed_inps.size(1)
            packed = torch.nn.utils.rnn.pack_padded_sequence(embed_inps,
                input_lens, batch_first=True, enforce_sorted=False)
            outputs, state = self.rnn(packed, None)
            # outputs: (B, L, num_directions*H)
            # state: (num_layers*num_directions, B, H)
            outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs,
                batch_first=True, total_length=total_len)

        return outputs, state


class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, cell='GRU', n_layers=1, drop_ratio=0.1):
        super(Decoder, self).__init__()

        self.dropout_layer = nn.Dropout(drop_ratio)

        if cell == 'GRU':
            self.rnn = nn.GRU(input_size, hidden_size, n_layers, batch_first=True)
        elif cell == 'Elman':
            self.rnn = nn.RNN(input_size, hidden_size, n_layers, batch_first=True)
        elif cell == 'LSTM':
            self.rnn = nn.LSTM(input_size, hidden_size, n_layers, batch_first=True)


    def forward(self, embed_seq, last_state):
        # embed_seq: (B, L, H)
        # last_state: (B, H)
        embed_inps = self.dropout_layer(embed_seq)
        output, state = self.rnn(embed_inps, last_state.unsqueeze(0))
        output = output.squeeze(1)  # (B, 1, N) -> (B,N)
        return output, state.squeeze(0) # (B, H)


class AttentionReader(nn.Module):
    def __init__(self, d_q, d_v, drop_ratio=0.0):
        super(AttentionReader, self).__init__()
        self.attn = nn.Linear(d_q+d_v, d_v)
        self.v = nn.Parameter(torch.rand(d_v))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.normal_(mean=0, std=stdv)
        self.dropout = nn.Dropout(drop_ratio)


    def forward(self, Q, K, V, attn_mask):
        # Q: (B, 1, d_q)
        # K: (B, L, d_v)
        # V: (B, L, d_v)
        # attn_mask: (B, L), True means mask
        k_len = K.size(1)
        q_state = Q.repeat(1, k_len, 1) # (B, L, d_q)

        attn_energies = self.score(q_state, K) # (B, L)

        attn_energies.masked_fill_(attn_mask, -1e12)

        attn_weights = F.softmax(attn_energies, dim=1).unsqueeze(1)
        attn_weights = self.dropout(attn_weights)

        # (B, 1, L) * (B, L, d_v)  -> (B, 1, d_v)
        context = attn_weights.bmm(V)

        return context.squeeze(1), attn_weights


    def score(self, query, memory):
        # query (B, L, d_q)
        # memory (B, L, d_v)

        # (B, L, d_q+d_v)->(B, L, d_v)
        energy = torch.tanh(self.attn(torch.cat([query, memory], 2)))
        energy = energy.transpose(1, 2)  # (B, d_v, L)

        v = self.v.repeat(memory.size(0), 1).unsqueeze(1)  # (B, 1, d_v)
        energy = torch.bmm(v, energy)  # (B, 1, d_v) * (B, d_v, L) -> [B, 1, L]
        return energy.squeeze(1)  # (B, L)



class AttentionWriter(nn.Module):
    def __init__(self, d_q, mem_size):
        super(AttentionWriter, self).__init__()
        self.attn = nn.Linear(d_q+mem_size, mem_size)
        self.v = nn.Parameter(torch.rand(mem_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.normal_(mean=0, std=stdv)

        self._tau = 1.0 # Gumbel temperature


    def set_tau(self, tau):
        self._tau = tau

    def get_tau(self):
        return self._tau


    def forward(self, his_mem, states, states_mask, global_trace, null_mem):
        # mem: (B, mem_slots, mem_size)
        # states: (B, L, mem_size)
        # states_mask: (B, L), 0 means pad_idx and not to be written
        # global_trace: (B, D)
        # null_mem: (B, 1, mem_size)
        n = states.size(1)
        mem_slots = his_mem.size(1) + 1 # including the null slot

        write_log = []

        for i in range(0, n):
            mem = torch.cat([his_mem, null_mem], dim=1)
            state = states[:, i, :] # (B, mem_size)


            query = torch.cat([state, global_trace], dim=1).unsqueeze(1).repeat(1, mem_slots, 1)
            attn_energies = self.score(query, mem)

            attn_weights = F.softmax(attn_energies, dim=-1) # (B, mem_slots)

            # manually give the empty slots higher weights
            empty_mask = mem.abs().sum(-1).eq(0).float() # (B, mem_slots)
            attn_weights = attn_weights + empty_mask * 10.0

            # one-hot (B, mem_slots)
            slot_select = F.gumbel_softmax(attn_weights, tau=self._tau, hard=True)

            write_mask = slot_select[:, 0:mem_slots-1] * \
                (states_mask[:, i].unsqueeze(1).repeat(1, mem_slots-1))
            write_mask = write_mask.unsqueeze(2) # (B, mem_slots-1, 1)


            write_state = state.unsqueeze(1).repeat(1, mem_slots-1, 1)

            his_mem = (1.0 - write_mask) * his_mem + write_mask * write_state

            write_log.append(slot_select.unsqueeze(1))

        write_log = torch.cat(write_log, dim=1)
        return his_mem, write_log


    def score(self, query, memory):
        energy = torch.tanh(self.attn(torch.cat([query, memory], 2)))
        energy = energy.transpose(1, 2)

        v = self.v.repeat(memory.size(0), 1).unsqueeze(1)

        energy = torch.bmm(v, energy)
        return energy.squeeze(1)


class MLP(nn.Module):
    def __init__(self, ori_input_size, layer_sizes, activs=None,
        drop_ratio=0.0, no_drop=False):
        super(MLP, self).__init__()

        layer_num = len(layer_sizes)

        orderedDic = OrderedDict()
        input_size = ori_input_size
        for i, (layer_size, activ) in enumerate(zip(layer_sizes, activs)):
            linear_name = 'linear_' + str(i)
            orderedDic[linear_name] = nn.Linear(input_size, layer_size)
            input_size = layer_size

            if activ is not None:
                assert activ in ['tanh', 'relu', 'leakyrelu']

            active_name = 'activ_' + str(i)
            if activ == 'tanh':
                orderedDic[active_name] = nn.Tanh()
            elif activ == 'relu':
                orderedDic[active_name] = nn.ReLU()
            elif activ == 'leakyrelu':
                orderedDic[active_name] = nn.LeakyReLU(0.2)


            if (drop_ratio > 0) and (i < layer_num-1) and (not no_drop):
                orderedDic["drop_" + str(i)] = nn.Dropout(drop_ratio)

        self.mlp = nn.Sequential(orderedDic)


    def forward(self, inps):
        return self.mlp(inps)


class ContextLayer(nn.Module):
    def __init__(self, inp_size, out_size, kernel_size=3):
        super(ContextLayer, self).__init__()
        # (B, L, H)
        self.conv = nn.Conv1d(inp_size, out_size, kernel_size)
        self.linear = nn.Linear(out_size+inp_size, out_size)

    def forward(self, last_context, dec_states):
        # last_context: (B, context_size)
        # dec_states: (B, H, L)
        hidden_feature = self.conv(dec_states) # (B, out_size, L_out)
        feature = torch.tanh(hidden_feature).mean(dim=2) # (B, out_size)
        new_context = torch.tanh(self.linear(torch.cat([last_context, feature], dim=1)))
        return new_context