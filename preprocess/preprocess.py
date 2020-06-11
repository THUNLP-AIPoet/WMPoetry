# -*- coding: utf-8 -*-
# @Author: Xiaoyuan Yi
# @Last Modified by:   Xiaoyuan Yi
# @Last Modified time: 2020-06-11 20:25:32
# @Email: yi-xy16@mails.tsinghua.edu.cn
# @Description:
'''
Copyright 2020 THUNLP Lab. All Rights Reserved.
This code is part of the online Chinese poetry generation system, Jiuge.
System URL: https://jiuge.thunlp.cn/ and https://jiuge.thunlp.org/.
Github: https://github.com/THUNLP-AIPoet.
'''
import pickle
import json
import random

from pattern_extractor import PatternExtractor

def outFile(data, file_name):
    print ("output data to %s, num: %d" % (file_name, len(data)))
    with open(file_name, 'w') as fout:
        for d in data:
            fout.write(d+"\n")


class PreProcess(object):
    """A Tool for data preprocess.
    Please note that this tool is only for Chinese quatrains.
    """
    def __init__(self):
        super(PreProcess, self).__init__()
        self.min_freq = 1
        self.sens_num = 4  # sens_num must be 4
        self.key_num = 4 # max number of keywords

        self.extractor = PatternExtractor("../data/")


    def create_dic(self, poems):
        print ("creating the word dictionary...")
        print ("input poems: %d" % (len(poems)))
        count_dic = {}
        for p in poems:
            poem = p.strip().replace("|", "")

            for c in poem:
                if c in count_dic:
                    count_dic[c] += 1
                else:
                    count_dic[c] = 1

        vec = sorted(count_dic.items(), key=lambda d:d[1], reverse=True)
        print ("original word num:%d" % (len(vec)))

        # add special symbols
        # --------------------------------------
        dic = {}
        idic = {}
        dic['PAD'] = 0
        idic[0] = 'PAD'

        dic['UNK'] = 1
        idic[1] = 'UNK'

        dic['<E>'] = 2
        idic[2] = '<E>'

        dic['<B>'] = 3
        idic[3] = '<B>'


        idx = 4
        print ("min freq:%d" % (self.min_freq))

        for c, v in vec:
            if v < self.min_freq:
                continue
            if not c in dic:
                dic[c] = idx
                idic[idx] = c
                idx += 1

        print ("total word num: %s" % (len(dic)))

        return dic, idic


    def build_dic(self, infile):
        with open(infile, 'r') as fin:
            lines = fin.readlines()

        poems = []
        training_lines = []
        for line in lines:
            dic = json.loads(line.strip())
            poem = dic['content']
            poems.append(poem)
            training_lines.extend(poem.split("|"))

        dic, idic = self.create_dic(poems)
        self.dic = dic
        self.idic = idic

        # output dic file
        # read
        dic_file = "vocab.pickle"
        idic_file = "ivocab.pickle"

        print ("saving dictionary to %s" % (dic_file))
        with open(dic_file, 'wb') as fout:
            pickle.dump(dic, fout, -1)


        print ("saving inverting dictionary to %s" % (idic_file))
        with open(idic_file, 'wb') as fout:
            pickle.dump(idic, fout, -1)


        # building training lines
        outFile(training_lines, "training_lines.txt")



    def line2idxes(self, line):
        chars = [c for c in line]
        idxes = []
        for c in chars:
            if c in self.dic:
                idx = self.dic[c]
            else:
                idx = self.dic['UNK']
            idxes.append(idx)

        return idxes



    def read_corpus(self, infile):
        with open(infile, 'r') as fin:
            lines = fin.readlines()

        corpus = []
        for line in lines:
            dic = json.loads(line.strip())
            poem = dic['content'].strip()
            keywords = dic['keywords'].strip().split(" ")

            corpus.append((keywords, poem))

        return corpus


    def build_pattern(self, sens):
        length = len(sens[0])
        assert length == 5 or length == 7

        rhymes = self.extractor.get_poem_rhyme(sens)
        if len(rhymes) == 0:
            return ""


        rhythm_pattern = self.extractor.get_poem_rhythm(sens, length)
        if len(rhythm_pattern) == 0:
            return ""

        for i in range(0, len(sens)):
            if rhymes[i] >= 1:
                rhythm_pattern[i][-1] = rhymes[i]

        return rhythm_pattern


    def build_data(self, corpus, convert_to_indices=True):
        skip_count = 0

        data = []
        for keywords, poem in corpus:
            lines = poem.strip().split("|")

            if len(keywords) == 0:
                skip_count += 1
                continue

            keywords = keywords[0:self.key_num]

            if len(lines) != 4:
                skip_count += 1
                continue


            lens = [len(line) for line in lines]

            if not (lens[0] == lens[1] == lens[2] == lens[3]):
                skip_count += 1
                continue


            length = lens[0]
            # only for Chinese quatrains
            if length != 5 and length != 7:
                skip_count += 1
                continue


            pattern = self.build_pattern(lines)
            if len(pattern) == 0:
                skip_count += 1
                continue

            if not convert_to_indices:
                for keynum in range(1, len(keywords)+1):
                    tup = (random.sample(keywords, keynum),
                        lines, keynum, pattern)
                    data.append(tup)
                continue

            # poem to indices
            line_idxes_vec = []
            for line in lines:
                idxes = self.line2idxes(line)
                assert len(idxes) == 5 or len(idxes) == 7
                line_idxes_vec.append(idxes)

            assert len(line_idxes_vec) == 4

            # keywords to indices
            key_idxes_vec = [self.line2idxes(keyword) for keyword in keywords]


            for keynum in range(1, len(key_idxes_vec)+1):
                tup = (random.sample(key_idxes_vec, keynum),
                    line_idxes_vec, keynum, pattern)
                data.append(tup)

        print ("data num: %d, skip_count: %d" %\
            (len(data), skip_count))

        return data


    def build_test_data(self, infile, out_inp_file, out_trg_file):
        with open(infile, 'r') as fin:
            lines = fin.readlines()

        test = self.read_corpus(infile)
        test_data = self.build_data(test, False)

        inps, trgs = [], []
        for tup in test_data:
            keywords = " ".join(tup[0])
            poem = "|".join(tup[1])
            pattern = "|".join([" ".join(map(str, p)) for p in tup[3]])

            inps.append(keywords+"#"+pattern)
            trgs.append(poem)


        outFile(inps, out_inp_file)
        outFile(trgs, out_trg_file)



    def process(self):
        # build the word dictionary
        self.build_dic("ccpc_train_v1.0.json")

        # build training and validation datasets
        train = self.read_corpus("ccpc_train_v1.0.json")
        valid = self.read_corpus("ccpc_valid_v1.0.json")

        train_data = self.build_data(train)
        valid_data = self.build_data(valid)

        random.shuffle(train_data)
        random.shuffle(valid_data)

        print ("training data: %d" % (len(train_data)))
        print ("validation data: %d" % (len(valid_data)))

        train_file = "train_data.pickle"
        print ("saving training data to %s" % (train_file))
        with open(train_file, 'wb') as fout:
            pickle.dump(train_data, fout, -1)


        valid_file = "valid_data.pickle"
        print ("saving validation data to %s" % (valid_file))
        with open(valid_file, 'wb') as fout:
            pickle.dump(valid_data, fout, -1)

        # build testing inputs and trgs
        self.build_test_data("ccpc_test_v1.0.json", "test_inps.txt", "test_trgs.txt")



def main():
    processor = PreProcess()
    processor.process()



if __name__ == "__main__":
    main()