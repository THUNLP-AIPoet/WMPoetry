# -*- coding: utf-8 -*-
# @Author: Xiaoyuan Yi
# @Last Modified by:   Xiaoyuan Yi
# @Last Modified time: 2020-06-11 20:38:33
# @Email: yi-xy16@mails.tsinghua.edu.cn
# @Description:
'''
Copyright 2020 THUNLP Lab. All Rights Reserved.
This code is part of the online Chinese poetry generation system, Jiuge.
System URL: https://jiuge.thunlp.cn/ and https://jiuge.thunlp.org/.
Github: https://github.com/THUNLP-AIPoet.
'''
import pickle
import numpy as np
import random
import copy
import torch


def readPickle(data_path):
    corpus_file = open(data_path, 'rb')
    corpus = pickle.load(corpus_file)
    corpus_file.close()

    return corpus

#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
class Tool(object):
    '''
    a tool to hold training data and the vocabulary
    '''
    def __init__(self, sens_num, sen_len, key_len, key_slots,
        corrupt_ratio=0):
        # corrupt ratio for dae
        self._sens_num = sens_num
        self._sen_len = sen_len
        self._key_len = key_len
        self._key_slots = key_slots

        self._corrupt_ratio = corrupt_ratio

        self._vocab = None
        self._ivocab = None

        self._PAD_ID = None
        self._B_ID = None
        self._E_ID = None
        self._UNK_ID = None

    # -----------------------------------
    # map functions
    def idxes2line(self, idxes, truncate=True):
        if truncate and self._E_ID in idxes:
            idxes = idxes[:idxes.index(self._E_ID)]

        tokens = self.idxes2tokens(idxes, truncate)
        line = self.tokens2line(tokens)
        return line

    def line2idxes(self, line):
        tokens = self.line2tokens(line)
        return self.tokens2idxes(tokens)

    def line2tokens(self, line):
        '''
        in this work, we treat each Chinese character as a token.
        '''
        line = line.strip()
        tokens = [c for c in line]
        return tokens


    def tokens2line(self, tokens):
        return "".join(tokens)


    def tokens2idxes(self, tokens):
        ''' Characters to idx list '''
        idxes = []
        for w in tokens:
            if w in self._vocab:
                idxes.append(self._vocab[w])
            else:
                idxes.append(self._UNK_ID)
        return idxes


    def idxes2tokens(self, idxes, omit_special=True):
        tokens = []
        for idx in idxes:
            if  (idx == self._PAD_ID or idx == self._B_ID
                or idx == self._E_ID) and omit_special:
                continue
            tokens.append(self._ivocab[idx])

        return tokens

    # -------------------------------------------------
    def greedy_search(self, probs):
        # probs: (V)
        out_idxes = [int(np.argmax(prob, axis=-1)) for prob in probs]

        return self.idxes2line(out_idxes)

    # ----------------------------
    def get_vocab(self):
        return copy.deepcopy(self._vocab)

    def get_ivocab(self):
        return copy.deepcopy(self._ivocab)

    def get_vocab_size(self):
        if self._vocab is not None:
            return len(self._vocab)
        else:
            return -1

    def get_PAD_ID(self):
        assert self._PAD_ID is not None
        return self._PAD_ID

    def get_B_ID(self):
        assert self._B_ID is not None
        return self._B_ID

    def get_E_ID(self):
        assert self._E_ID is not None
        return self._E_ID

    def get_UNK_ID(self):
        assert self._UNK_ID is not None
        return self._UNK_ID


    # ----------------------------------------------------------------
    def load_dic(self, vocab_path, ivocab_path):
        dic = readPickle(vocab_path)
        idic = readPickle(ivocab_path)

        assert len(dic) == len(idic)


        self._vocab = dic
        self._ivocab = idic

        self._PAD_ID = dic['PAD']
        self._UNK_ID = dic['UNK']
        self._E_ID = dic['<E>']
        self._B_ID = dic['<B>']


    def load_function_tokens(self, file_dir):
        # please run load_dict before using loading function tokens !
        with open(file_dir, 'r') as fin:
            lines = fin.readlines()

        tokens = [line.strip() for line in lines]

        f_idxes = []
        for token in tokens:
            if token in self._vocab:
                f_idxes.append(self._vocab[token])

        f_idxes = torch.tensor(f_idxes, dtype=torch.long)
        return f_idxes


    def build_data(self, train_data_path, valid_data_path, batch_size, mode):
        '''
        Build data as batches.
        NOTE: please run load_dic() at first.
        mode:
            dae: pre-train the encoder and decoder as a denoising Seq2Seq model
            wm: train the working memory model
        '''
        assert mode in ['dseq', 'wm']
        train_data = readPickle(train_data_path)
        valid_data = readPickle(valid_data_path)


        # data limit for debug
        self.train_batches = self._build_data_core(train_data, batch_size, mode, None)
        self.valid_batches = self._build_data_core(valid_data, batch_size, mode, None)

        self.train_batch_num = len(self.train_batches)
        self.valid_batch_num = len(self.valid_batches)


    def _build_data_core(self, data, batch_size, mode, data_limit=None):
        # data: [keywords, sens, key_num, pattern] * data_num
        if data_limit is not None:
            data = data[0:data_limit]

        if mode == 'dseq':
            return self.build_dseq_batches(data, batch_size)
        elif mode == 'wm':
            return self.build_wm_batches(data, batch_size)


    def build_dseq_batches(self, data, batch_size):
        # data: [keywords, sens, key_num, pattern] * data_num
        batched_data = []
        batch_num = int(np.ceil(len(data) / float(batch_size)))
        for bi in range(0, batch_num):
            instances = data[bi*batch_size : (bi+1)*batch_size]
            if len(instances) < batch_size:
                instances = instances + random.sample(data, batch_size-len(instances))

            # build poetry batch
            poems = [instance[1] for instance in instances] # all poems
            genre_patterns = [instance[3] for instance in instances]
            for i in range(0, self._sens_num-1):
                line0 = [poem[i] for poem in poems]
                line1 = [poem[i+1] for poem in poems]
                phs = [pattern[i+1] for pattern in genre_patterns]

                inps, trgs, ph_inps, len_inps = \
                    self._build_batch_seqs(line0, line1, phs, corrupt=True)

                batched_data.append((inps, trgs, ph_inps, len_inps))

        random.shuffle(batched_data)
        return batched_data


    def build_wm_batches(self, data, batch_size):
         # data: [keywords, sens, key_num, pattern] * data_num
        batched_data = []
        batch_num = int(np.ceil(len(data) / float(batch_size)))
        for bi in range(0, batch_num):
            instances = data[bi*batch_size : (bi+1)*batch_size]
            if len(instances) < batch_size:
                instances = instances + random.sample(data, batch_size-len(instances))

            # build poetry batch
            poems = [instance[1] for instance in instances] # all poems
            genre_patterns = [instance[3] for instance in instances]


            all_inps, all_trgs = [], []
            all_ph_inps, all_len_inps = [], []

            for i in range(-1, self._sens_num-1):

                if i < 0:
                    line0 = [[] for poem in poems]
                else:
                    line0 = [poem[i] for poem in poems]

                line1 = [poem[i+1] for poem in poems]
                phs = [pattern[i+1] for pattern in genre_patterns]


                inps, trgs, ph_inps, len_inps = \
                    self._build_batch_seqs(line0, line1, phs, corrupt=False)


                all_inps.append(inps)
                all_trgs.append(trgs)
                all_ph_inps.append(ph_inps)
                all_len_inps.append(len_inps)


            # build keys
            keywords = [instance[0] for instance in instances]
            keys = self._build_batch_keys(keywords)

            batched_data.append((all_inps, all_trgs, all_ph_inps, all_len_inps, keys))


        random.shuffle(batched_data)
        return batched_data


    def _build_batch_keys(self, keywords):
        # build key batch
        batch_size = len(keywords)
        key_inps = [[] for _ in range(self._key_slots)]

        for i in range(0, batch_size):
            keys = keywords[i] # batch_size * at most 4
            for step in range(0, len(keys)):
                key = keys[step]
                assert len(key) <= self._key_len
                key_inps[step].append(key + [self._PAD_ID] * (self._key_len-len(key)))

            for step in range(0, self._key_slots-len(keys)):
                key_inps[len(keys)+step].append([self._PAD_ID] * self._key_len)


        key_tensor = [self._sens2tensor(key) for key in key_inps]
        return key_tensor



    def _build_batch_seqs(self, inputs, targets, pattern, corrupt=False):
        # pack sequences as a tensor
        inps, _, _ = self._get_batch_seq(inputs, pattern,  False, corrupt=corrupt)
        trgs, phs, lens = self._get_batch_seq(targets, pattern, True, corrupt=False)

        inps_tensor = self._sens2tensor(inps)
        trgs_tensor = self._sens2tensor(trgs)

        phs_tensor = self._sens2tensor(phs)
        lens_tensor = self._sens2tensor(lens)


        return inps_tensor, trgs_tensor, phs_tensor, lens_tensor


    def _get_batch_seq(self, seqs, phs, with_E, corrupt):
        batch_size = len(seqs)
        max_len = max([len(seq) for seq in seqs])
        max_len = max_len + int(with_E)

        if max_len == 0:
            max_len = self._sen_len

        batched_seqs = []
        batched_lens, batched_phs = [], []
        for i in range(0, batch_size):
            # max length for each sequence
            ori_seq = copy.deepcopy(seqs[i])

            if corrupt:
                seq = self._do_corruption(ori_seq)
            else:
                seq = ori_seq
            # ----------------------------------

            pad_size = max_len - len(seq) - int(with_E)
            pads = [self._PAD_ID] * pad_size

            new_seq = seq + [self._E_ID] * int(with_E) + pads

            #---------------------------------
            # 0 means either
            ph = phs[i]
            ph_inp = ph + [0] * (max_len - len(ph))

            assert len(ph_inp) == len(new_seq)
            batched_phs.append(ph_inp)

            len_inp = list(range(1, len(seq)+1+int(with_E)))
            len_inp.reverse()
            len_inp = len_inp + [0] * pad_size

            assert len(len_inp) == len(new_seq)

            batched_lens.append(len_inp)
            #---------------------------------
            batched_seqs.append(new_seq)


        return batched_seqs, batched_phs, batched_lens



    def _sens2tensor(self, sens):
        batch_size = len(sens)
        sen_len = max([len(sen) for sen in sens])
        tensor = torch.zeros(batch_size, sen_len, dtype=torch.long)
        for i, sen in enumerate(sens):
            for j, token in enumerate(sen):
                tensor[i][j] = token
        return tensor


    def _do_corruption(self, inp):
        # corrupt the sequence by setting some tokens as UNK
        m = int(np.ceil(len(inp) * self._corrupt_ratio))
        m = min(m, len(inp))
        m = max(1, m)

        unk_id = self.get_UNK_ID()

        corrupted_inp = copy.deepcopy(inp)
        pos = random.sample(list(range(0, len(inp))), m)
        for p in pos:
            corrupted_inp[p] = unk_id

        return corrupted_inp



    def shuffle_train_data(self):
        random.shuffle(self.train_batches)



    # -----------------------------------------------------------
    # -----------------------------------------------------------
    # Tools for beam search
    def keywords2tensor(self, keywords):
        # input: keywords: list of string

        # string to idxes
        key_idxes = []
        for keyword in keywords:
            keys = [self.line2idxes(key_str) for key_str in keyword]
            if len(keys) < self._key_slots:
                add_num = self._key_slots - len(keys)
                add_keys = [[self._PAD_ID]*self._key_len]*add_num
                keys = keys + add_keys

            key_idxes.append(keys)

        keys_tensor = self._build_batch_keys(key_idxes)
        return keys_tensor



    def patterns2tensor(self, patterns):
        batch_size = len(patterns)
        # assume all poems share the same sens_num
        all_seqs = []
        all_lengths = []
        all_ph_inps, all_len_inps = [], []
        for step in range(0, self._sens_num):
            # each line
            phs = [pattern[step] for pattern in patterns]
            pseudo_seqs = [ [0] * len(ph) for ph in phs ]
            all_lengths.append(max([len(seq) for seq in pseudo_seqs]))

            batched_seqs, batched_phs, batched_lens = \
                self._get_batch_seq(pseudo_seqs, phs, True, False)

            phs_tensor = self._sens2tensor(batched_phs)
            lens_tensor = self._sens2tensor(batched_lens)
            inps_tensor = self._sens2tensor(batched_seqs)

            all_ph_inps.append(phs_tensor)
            all_len_inps.append(lens_tensor)
            all_seqs.append(inps_tensor)


        return all_seqs[0], all_ph_inps, all_len_inps, all_lengths


    def line2tensor(self, line, beam_size):
        idxes = self.line2idxes(line.strip())
        return self._sens2tensor([idxes]*beam_size)
