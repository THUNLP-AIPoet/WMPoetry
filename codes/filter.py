# -*- coding: utf-8 -*-
# @Author: Xiaoyuan Yi and Jiannan Liang
# @Last Modified by:   Xiaoyuan Yi
# @Last Modified time: 2020-06-11 18:22:16
# @Email: yi-xy16@mails.tsinghua.edu.cn
# @Description:
'''
Copyright 2020 THUNLP Lab. All Rights Reserved.
This code is part of the online Chinese poetry generation system, Jiuge.
System URL: https://jiuge.thunlp.cn/ and https://jiuge.thunlp.org/.
Github: https://github.com/THUNLP-AIPoet.
'''
import pickle
import copy
import os

class PoetryFilter(object):


    def __init__(self, vocab, ivocab, data_dir):
        self._vocab = vocab
        self._ivocab = ivocab

        self._load_rhythm_dic(data_dir+"pingsheng.txt", data_dir+"zesheng.txt")
        self._load_rhyme_dic(data_dir+"pingshui.txt", data_dir+"pingshui_amb.pkl")
        self._load_line_lib(data_dir+"training_lines.txt")


    def _load_line_lib(self, data_path):
        self._line_lib = {}

        with open(data_path, 'r') as fin:
            lines = fin.readlines()

        for line in lines:
            line = line.strip()
            self._line_lib[line] = 1

        print ("  line lib loaded, %d lines: " % (len(self._line_lib)))



    def _load_rhythm_dic(self, level_path, oblique_path):
        with open(level_path, 'r') as fin:
            level_chars = fin.read()

        with open(oblique_path, 'r') as fin:
            oblique_chars = fin.read()

        self._level_list = []
        self._oblique_list = []
        # convert char to id
        for char, idx in self._vocab.items():
            if char in level_chars:
                self._level_list.append(idx)

            if char in oblique_chars:
                self._oblique_list.append(idx)

        print ("  rhythm dic loaded, level tone chars: %d, oblique tone chars: %d" %\
            (len(self._level_list), len(self._oblique_list)))


    #------------------------------------------
    def _load_rhyme_dic(self, rhyme_dic_path, rhyme_disamb_path):

        self._rhyme_dic = {} # char id to rhyme category ids
        self._rhyme_idic = {} # rhyme category id to char ids

        with open(rhyme_dic_path, 'r') as fin:
            lines = fin.readlines()


        amb_count = 0
        for line in lines:
            (char, rhyme_id) = line.strip().split(' ')
            if char not in self._vocab:
                continue
            char_id = self._vocab[char]
            rhyme_id = int(rhyme_id)

            if not char_id in self._rhyme_dic:
                self._rhyme_dic.update({char_id:[rhyme_id]})
            elif not rhyme_id in self._rhyme_dic[char_id]:
                self._rhyme_dic[char_id].append(rhyme_id)


            if not rhyme_id in self._rhyme_idic:
                self._rhyme_idic.update({rhyme_id:[char_id]})
            else:
                self._rhyme_idic[rhyme_id].append(char_id)


        # load data for rhyme disambiguation
        self._ngram_rhyme_map = {} # rhyme id list of each bigram or trigram
        self._char_rhyme_map = {} # the most likely rhyme id for each char
        # load the calculated data, if there is any
        #print (rhyme_disamb_path)
        assert rhyme_disamb_path is not None and os.path.exists(rhyme_disamb_path)

        with open(rhyme_disamb_path, 'rb') as fin:
            self._char_rhyme_map = pickle.load(fin)
            self._ngram_rhyme_map = pickle.load(fin)

            print ("  rhyme disamb data loaded, cached chars: %d, ngrams: %d"
                % (len(self._char_rhyme_map), len(self._ngram_rhyme_map)))



    def get_line_rhyme(self, line):
        """ we use statistics of ngram to disambiguate the rhyme category,
        but there is still a risk of mismatching and ambiguity
        """
        tail_char = line[-1]

        if tail_char in self._char_rhyme_map:
            rhyme_candis = self._char_rhyme_map[tail_char]
            if len(rhyme_candis) == 1:
                return rhyme_candis[0]

            bigram = line[-2] + line[-1]
            if bigram in self._ngram_rhyme_map:
                return self._ngram_rhyme_map[bigram][0]

            trigram = line[-3] + line[-2] + line[-1]
            if trigram in self._ngram_rhyme_map:
                return self._ngram_rhyme_map[trigram][0]

            return self._char_rhyme_map[tail_char][0]


        if not tail_char in self._vocab:
            return -1
        else:
            tail_id = self._vocab[tail_char]


        if tail_id in self._rhyme_dic:
            return self._rhyme_dic[tail_id][0]

        return -1

    # ------------------------------
    def reset(self, length, verbose):
        assert length == 5 or length == 7
        self._length = length
        self._repetitive_ids = []
        self._verbose = verbose


    def add_repetitive(self, ids):
        self._repetitive_ids = list(set(ids+self._repetitive_ids))


    # -------------------------------
    def get_level_cids(self):
        return copy.deepcopy(self._level_list)

    def get_oblique_cids(self):
        return copy.deepcopy(self._oblique_list)

    def get_rhyme_cids(self, rhyme_id):
        if rhyme_id not in self._rhyme_idic:
            return []
        else:
            return copy.deepcopy(self._rhyme_idic[rhyme_id])

    def get_repetitive_ids(self):
        return copy.deepcopy(self._repetitive_ids)


    def filter_illformed(self, lines, costs, states, aligns, rhyme_id):
        if len(lines) == 0:
            return [], [], [], []

        new_lines, new_costs = [], []
        new_states, new_aligns = [], []

        len_error, lib_error, rhyme_error = 0, 0, 0

        for i in range(len(lines)):
            #print (lines[i])
            if len(lines[i]) < self._length:
                len_error += 1
                continue

            line = lines[i][0:self._length]

            # we filter out the lines that already exist in the
            #   training set, to guarantee the novelty of generated poems
            if line in self._line_lib:
                lib_error += 1
                continue

            if 1 <= rhyme_id <= 30:
                if self.get_line_rhyme(line) != rhyme_id:
                    rhyme_error != 1
                    continue

            new_lines.append(line)
            new_costs.append(costs[i])
            new_states.append(states[i])
            new_aligns.append(aligns[i])


        if self._verbose >= 3:
            print ("input lines: %d, ilter out %d illformed lines, %d remain"
                % (len(lines), len(lines)-len(new_lines), len(new_lines)))
            print ("%d len error, %d exist in lib, %d rhyme error"
                % (len_error, lib_error, rhyme_error))

        return new_lines, new_costs, new_states, new_aligns