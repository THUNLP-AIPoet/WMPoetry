# -*- coding: utf-8 -*-
# @Author: Xiaoyuan Yi
# @Last Modified by:   Xiaoyuan Yi
# @Last Modified time: 2020-06-11 22:04:36
# @Email: yi-xy16@mails.tsinghua.edu.cn
# @Description:
'''
Copyright 2020 THUNLP Lab. All Rights Reserved.
This code is part of the online Chinese poetry generation system, Jiuge.
System URL: https://jiuge.thunlp.cn/ and https://jiuge.thunlp.org/.
Github: https://github.com/THUNLP-AIPoet.
'''
from matplotlib import pyplot as plt
plt.rcParams['font.family'] = ['simhei']

from matplotlib.colors import from_levels_and_colors

import numpy as np
import copy

import torch

class Visualization(object):
    """docstring for LogInfo"""
    def __init__(self, topic_slots, history_slots, log_path):
        super(Visualization).__init__()

        self._topic_slots = topic_slots
        self._history_slots = history_slots
        self._log_path = log_path


    def reset(self, keywords):
        self._keywords = keywords
        self._history_mem = [' ']*self._history_slots
        self._gen_lines = []


    def add_gen_line(self, line):
        self._gen_lines.append(line.strip())

    def normalization(self, ori_matrix):
        new_matrix = ori_matrix / ori_matrix.sum(axis=1, keepdims=True)
        return new_matrix


    def draw(self, read_log, write_log, step, visual_mode):
        assert visual_mode in [0, 1, 2]
        # read_log: (1, 1, mem_slots) * L_gen
        # write_log: (B, L_gen, mem_slots)
        current_gen_chars = [c for c in self._gen_lines[-1]]
        gen_len = len(current_gen_chars)

        if len(self._gen_lines) >= 2:
            last_gen_chars = [c for c in self._gen_lines[-2]]
            last_gen_len = len(last_gen_chars)
        else:
            last_gen_chars = [''] * gen_len
            last_gen_len = gen_len

        # (L_gen, mem_slots)
        mem_slots = self._topic_slots+self._history_slots+last_gen_len
        read_matrix = torch.cat(read_log, dim=1)[0, 0:gen_len, 0:mem_slots].detach().cpu().numpy()
        read_matrix = self.normalization(read_matrix)

        plt.figure(figsize=(11, 5))

        # visualization of reading attention weights
        num_levels = 100
        vmin, vmax = read_matrix.min(), read_matrix.max()
        midpoint = 0
        levels = np.linspace(vmin, vmax, num_levels)
        midp = np.mean(np.c_[levels[:-1], levels[1:]], axis=1)
        vals = np.interp(midp, [vmin, midpoint, vmax], [0, 0.5, 1])
        colors = plt.cm.seismic(vals)
        cmap, norm = from_levels_and_colors(levels, colors)


        plt.imshow(read_matrix, cmap=cmap,  interpolation='none')

        # print generated chars and chars in the memory
        fontsize = 14

        plt.text(0.2, gen_len+0.5, "Topic Memory", fontsize=fontsize)
        plt.text(self._topic_slots, gen_len+0.5, "History Memory", fontsize=fontsize)
        if last_gen_len == 5:
            shift = 5
        else:
            shift = 6
        plt.text(self._topic_slots+shift, gen_len+0.5, "Local Memory", fontsize=fontsize)

        # topic memory
        for i in range(0, len(self._keywords)):
            key = self._keywords[i]
            if len(key) == 1:
                key = " " + key + " "
            key = key + "|"
            plt.text(i-0.4,-0.7, key, fontsize=fontsize)

        start_pos = self._topic_slots
        end_pos = self._topic_slots + self._history_slots

        # history memory
        for i in range(start_pos, end_pos):
            c = self._history_mem[i - start_pos]
            if i == end_pos - 1:
                c = c + " |"

            plt.text(i-0.2,-0.7, c, fontsize=fontsize)

        start_pos = end_pos
        end_pos = start_pos + last_gen_len

        # local memory
        for i in range(start_pos, end_pos):
            idx = i - start_pos
            plt.text(i-0.2,-0.7, last_gen_chars[idx], fontsize=fontsize)

        # generated line
        for i in range(0, len(current_gen_chars)):
            plt.text(-1.2, i+0.15, current_gen_chars[i], fontsize=fontsize)

        plt.colorbar()
        plt.tick_params(labelbottom=False, labelleft=False)


        x_major_locator = plt.MultipleLocator(1)
        y_major_locator = plt.MultipleLocator(1)
        ax = plt.gca()
        ax.xaxis.set_major_locator(x_major_locator)
        ax.yaxis.set_major_locator(y_major_locator)
        #plt.tight_layout()


        if visual_mode == 1:
            fig = plt.gcf()
            fig.savefig(self._log_path + 'visual_step_{}.png'.format(step), dpi=300, quality=100, bbox_inches="tight")
        elif visual_mode == 2:
            plt.show()


        # update history memory
        if write_log is not None:
            if len(last_gen_chars) == 0:
                print ("last generated line is empty!")

            write_log = write_log[0, :, :].detach().cpu().numpy()
            history_mem = copy.deepcopy(self._history_mem)
            for i, c in enumerate(last_gen_chars):
                selected_slot = np.argmax(write_log[i, :])
                if selected_slot >= self._history_slots:
                    continue
                history_mem[selected_slot] = c

            self._history_mem = history_mem
