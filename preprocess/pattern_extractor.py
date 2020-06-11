# -*- coding: utf-8 -*-
# @Author: Xiaoyuan Yi and Jiannan Liang
# @Last Modified by:   Xiaoyuan Yi
# @Last Modified time: 2020-06-11 20:24:22
# @Email: yi-xy16@mails.tsinghua.edu.cn
# @Description:
'''
Copyright 2020 THUNLP Lab. All Rights Reserved.
This code is part of the online Chinese poetry generation system, Jiuge.
System URL: https://jiuge.thunlp.cn/ and https://jiuge.thunlp.org/.
Github: https://github.com/THUNLP-AIPoet.
'''

import pickle
import os
import copy

import numpy as np

from rhythm_tool import RhythmRecognizer


class PatternExtractor(object):

    def __init__(self, data_dir):
        '''
        rhythm patterns.
            for Chinese quatrains, we generalize four main poem-level patterns.

        NOTE: We use pingshuiyun (The Pingshui Rhyme Category)
        We only consider level-tone rhyme in terms of the requirements of
           Chinese classical quatrains.
        0: either level or oblique tone; 1~30 rhyme categorizes,
            31: level tone, 32: oblique tone

        '''
        self._RHYTHM_PATTERNS = {7: [[0, 32, 0, 31, 31, 32, 32], [0, 31, 0, 32, 32, 31, 31],
            [0, 32, 31, 31, 32, 32, 31], [0, 31, 0, 32, 31, 31, 32]],
            5: [[0, 31, 31, 32, 32], [0, 32, 32, 31, 31], [31, 31, 32, 32, 31], [0, 32, 0, 31, 32]]}


        '''
        rhythm patterns.
        for Chinese quatrains, we generalize four main poem-level patterns
        '''
        self._RHYTHM_TYPES = [[0, 1, 3, 2], [1, 2, 0, 1], [2, 1, 3, 2], [3, 2, 0, 1]]


        self._rhythm_tool = RhythmRecognizer(data_dir+"pingsheng.txt", data_dir+"zesheng.txt")

        self._load_rhythm_dic(data_dir+"pingsheng.txt", data_dir+"zesheng.txt")
        self._load_rhyme_dic(data_dir+"pingshui.txt", data_dir+"pingshui_amb.pkl")



    def _load_rhythm_dic(self, level_path, oblique_path):
        with open(level_path, 'r') as fin:
            level_chars = fin.read()

        with open(oblique_path, 'r') as fin:
            oblique_chars = fin.read()

        self._level_list = list(level_chars)
        self._oblique_list = list(oblique_chars)


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

            rhyme_id = int(rhyme_id)

            if not char in self._rhyme_dic:
                self._rhyme_dic.update({char:[rhyme_id]})
            elif not rhyme_id in self._rhyme_dic[char]:
                self._rhyme_dic[char].append(rhyme_id)
                amb_count += 1

            if not rhyme_id in self._rhyme_idic:
                self._rhyme_idic.update({rhyme_id:[char]})
            else:
                self._rhyme_idic[rhyme_id].append(char)

        print ("  rhyme dic loaded, ambiguous rhyme chars: %d" % (amb_count))

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
        but there is still risk of mismatching and ambiguity
        """
        tail_char = line[-1]

        if tail_char in self._rhyme_dic:
            rhyme_candis = self._rhyme_dic[tail_char]
            if len(rhyme_candis) == 1:
                return rhyme_candis[0]

        if tail_char in self._char_rhyme_map:
            bigram = line[-2] + line[-1]
            if bigram in self._ngram_rhyme_map:
                return int(self._ngram_rhyme_map[bigram][0])

            trigram = line[-3] + line[-2] + line[-1]
            if trigram in self._ngram_rhyme_map:
                return int(self._ngram_rhyme_map[trigram][0])

            return int(self._char_rhyme_map[tail_char][0])


        return -1


    def get_poem_rhyme(self, sens):
        assert len(sens) == 4
        rhymes = [self.get_line_rhyme(sen) for sen in sens]

        #print (rhymes)
        #input(">")

        if rhymes[1] == -1 and rhymes[3] != -1:
            rhymes[1] = rhymes[3]
        elif rhymes[1] != -1 and rhymes[3] == -1:
            rhymes[3] = rhymes[1]
        elif rhymes[1] == -1 and rhymes[3] == -1:
            return []


        if (rhymes[0] != -1) and (rhymes[0]!= rhymes[1]):
            rhymes[0] = rhymes[1]

        rhymes[2] = -1

        return rhymes


    def pattern_complete(self, rhythm_ids):
        if rhythm_ids.count(-1) == 0:
            return rhythm_ids

        if rhythm_ids.count(-1) > 1:
            return []

        #print (rhythm_ids)
        for poem_pattern in self._RHYTHM_TYPES:
            eq = (np.array(poem_pattern) \
                == np.array(rhythm_ids)).astype(np.float)

            if np.sum(eq) != 3:
                continue

            pos = list(eq).index(0)
            rhythm_ids[pos] = poem_pattern[pos]
            #print (rhythm_ids)
            #input(">")
            return rhythm_ids

        return []


    def get_poem_rhythm(self, sens, length):
        assert len(sens) == 4

        rhythm_ids = []
        for sen in sens:
            #print (sen)
            rhythm_id = self._rhythm_tool.get_rhythm(sen)
            rhythm_ids.append(rhythm_id)


        rhythm_ids = self.pattern_complete(rhythm_ids)

        if len(rhythm_ids) == 0:
            return []

        rhythm_pattern = [copy.deepcopy(self._RHYTHM_PATTERNS[length][id])
            for id in rhythm_ids]

        return rhythm_pattern