# -*- coding: utf-8 -*-
# @Author: Xiaoyuan Yi
# @Last Modified by:   Xiaoyuan Yi
# @Last Modified time: 2020-06-11 18:17:27
# @Email: yi-xy16@mails.tsinghua.edu.cn
# @Description:
'''
Copyright 2020 THUNLP Lab. All Rights Reserved.
This code is part of the online Chinese poetry generation system, Jiuge.
System URL: https://jiuge.thunlp.cn/ and https://jiuge.thunlp.org/.
Github: https://github.com/THUNLP-AIPoet.
'''
from generator import Generator
from config import hparams, device
import copy
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="The parametrs for the generator.")
    parser.add_argument("-m", "--mode", type=str, choices=['interact', 'file'], default='interact',
        help='The mode of generation. interact: generate in a interactive mode.\
        file: take an input file and generate poems for each input in the file.')
    parser.add_argument("-b", "--bsize",  type=int, default=20, help="beam size, 20 by default.")
    parser.add_argument("-v", "--verbose", type=int, default=0, choices=[0, 1, 2, 3],
        help="Show other information during the generation, False by default.")
    parser.add_argument("-d", "--draw", type=int, default=0, choices=[0, 1, 2],
        help="Show the visualization of memory reading and writing. It only works in the interact mode.\
        0: not work, 1: save the visualization as pictures, 2: show the visualization at each step.")
    parser.add_argument("-s", "--select", type=int, default=0,
        help="If manually select each generated line from beam candidates? False by default.\
        It works only in the interact mode.")
    parser.add_argument("-i", "--inp", type=str,
        help="input file path. it works only in the file mode.")
    parser.add_argument("-o", "--out", type=str,
        help="output file path. it works only in the file mode")
    return parser.parse_args()


class GenerateTool(object):
    """docstring for GenerateTool"""
    def __init__(self):
        super(GenerateTool, self).__init__()
        self.generator = Generator(hparams, device)
        self._load_patterns(hparams.data_dir+"/GenrePatterns.txt")


    def _load_patterns(self, path):
        with open(path, 'r') as fin:
            lines = fin.readlines()

        self._patterns = []
        '''
        each line contains:
            pattern id, pattern name, the number of lines,
            pattern: 0 either, 31 pingze, 32 ze, 33 rhyme position
        '''
        for line in lines:
            line = line.strip()
            para = line.split("#")
            pas = para[3].split("|")
            newpas = []
            for pa in pas:
                pa = pa.split(" ")
                newpas.append([int(p) for p in pa])

            self._patterns.append((para[1], newpas))

        self.p_num = len(self._patterns)
        print ("load %d patterns." % (self.p_num))


    def build_pattern(self, pstr):
        pstr_vec = pstr.split("|")
        patterns = []
        for pstr in pstr_vec:
            pas = pstr.split(" ")
            pas = [int(pa) for pa in pas]
            patterns.append(pas)

        return patterns


    def generate_file(self, args):
        beam_size = args.bsize
        verbose = args.verbose
        manu = True if args.select ==1 else False

        assert args.inp is not None
        assert args.out is not None

        with open(args.inp, 'r') as fin:
            inps = fin.readlines()


        fout = open(args.out, 'w')

        poems = []
        N = len(inps)
        log_step = max(int(N/100), 2)
        for i, inp in enumerate(inps):
            para = inp.strip().split("#")
            keywords = para[0].split(" ")
            pattern = self.build_pattern(para[1])

            lines, info = self.generator.generate_one(keywords, pattern,
                beam_size, verbose, manu=manu)

            if len(lines) != 4:
                ans = info
            else:
                ans = "|".join(lines)

            fout.write(ans+"\n")

            if i % log_step == 0:
                print ("generating, %d/%d" % (i, N))
                fout.flush()


        fout.close()


    def _set_rhyme_into_pattern(self, ori_pattern, rhyme):
        pattern = copy.deepcopy(ori_pattern)
        for i in range(0, len(pattern)):
            if pattern[i][-1] == 33:
                pattern[i][-1] = rhyme
        return pattern


    def generate_manu(self, args):
        beam_size = args.bsize
        verbose = args.verbose
        manu = True if args.select ==1 else False
        visual_mode = args.draw

        while True:
            keys = input("please input keywords (with whitespace split), 4 at most > ")
            pattern_id = int(input("please select genre pattern 0~{} > ".format(self.p_num-1)))
            rhyme = int(input("please input rhyme id, 1~30> "))

            ori_pattern = self._patterns[pattern_id]
            name = ori_pattern[0]
            pattern = ori_pattern[1]
            pattern = self._set_rhyme_into_pattern(pattern, rhyme)
            print ("select pattern: %s" % (name))

            keywords = keys.strip().split(" ")
            lines, info = self.generator.generate_one(keywords, pattern,
                beam_size, verbose, manu=manu, visual=visual_mode)

            if len(lines) != 4:
                print("generation failed!")
                continue
            else:
                print("\n".join(lines))


def main():
    args = parse_args()
    generate_tool = GenerateTool()
    if args.mode == 'interact':
        generate_tool.generate_manu(args)
    else:
        generate_tool.generate_file(args)


if __name__ == "__main__":
    main()
