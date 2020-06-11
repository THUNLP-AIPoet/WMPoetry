# -*- coding: utf-8 -*-
# @Author: Xiaoyuan Yi
# @Last Modified by:   Xiaoyuan Yi
# @Last Modified time: 2020-06-11 18:09:32
# @Email: yi-xy16@mails.tsinghua.edu.cn
# @Description:
'''
Copyright 2020 THUNLP Lab. All Rights Reserved.
This code is part of the online Chinese poetry generation system, Jiuge.
System URL: https://jiuge.thunlp.cn/ and https://jiuge.thunlp.org/.
Github: https://github.com/THUNLP-AIPoet.
'''
import numpy as np
import time


class InfoLogger(object):
    """docstring for LogInfo"""
    def __init__(self, mode):
        super(InfoLogger).__init__()
        self._mode = mode # string, 'train' or 'valid'
        self._total_steps = 0
        self._batch_num = 0
        self._log_steps = 0
        self._cur_step = 0
        self._cur_epoch = 1

        self._start_time = 0
        self._end_time = 0

        #--------------------------
        self._log_path = "" # path to save the log file

        # -------------------------
        self._decay_rates = {'learning_rate':1.0,
            'teach_ratio':1.0, 'temperature':1.0}


    def set_batch_num(self, batch_num):
        self._batch_num = batch_num
    def set_log_steps(self, log_steps):
        self._log_steps = log_steps
    def set_log_path(self, log_path):
        self._log_path = log_path

    def set_rate(self, name, value):
        self._decay_rates[name] = value


    def set_start_time(self):
        self._start_time = time.time()

    def set_end_time(self):
        self._end_time = time.time()

    def add_step(self):
        self._total_steps += 1
        self._cur_step += 1

    def add_epoch(self):
        self._cur_step = 0
        self._cur_epoch += 1


    # ------------------------------
    @property
    def cur_process(self):
        ratio = float(self._cur_step) / self._batch_num * 100
        process_str = "%d/%d %.1f%%" % (self._cur_step, self._batch_num, ratio)
        return process_str

    @property
    def time_cost(self):
        return (self._end_time-self._start_time) / self._log_steps

    @property
    def total_steps(self):
        return self._total_steps

    @property
    def epoch(self):
        return self._cur_epoch

    @property
    def mode(self):
        return self._mode

    @property
    def log_path(self):
        return self._log_path


    @property
    def learning_rate(self):
        return self._decay_rates['learning_rate']

    @property
    def teach_ratio(self):
        return self._decay_rates['teach_ratio']

    @property
    def temperature(self):
        return self._decay_rates['temperature']


#------------------------------------
class SimpleLogger(InfoLogger):
    def __init__(self, mode):
        super(SimpleLogger, self).__init__(mode)
        self._gen_loss = 0.0

    def add_losses(self, gen_loss):
        self.add_step()
        self._gen_loss += gen_loss


    def print_log(self, epoch=None):

        gen_loss = self._gen_loss / self.total_steps
        ppl = np.exp(gen_loss)

        if self.mode == 'train':
            process_info = "epoch: %d, %s, %.2fs per iter, lr: %.4f, tr: %.2f, tau: %.3f" % (self.epoch,
                self.cur_process, self.time_cost, self.learning_rate, self.teach_ratio, self.temperature)
        else:
            process_info = "epoch: %d, lr: %.4f, tr: %.2f, tau: %.3f" % (
                epoch, self.learning_rate, self.teach_ratio, self.temperature)

        train_info = "  gen loss: %.3f  ppl:%.2f" % (gen_loss, ppl)


        print (process_info)
        print (train_info)
        print ("______________________")

        info = process_info + "\n" + train_info
        fout = open(self.log_path, 'a')
        fout.write(info + "\n\n")
        fout.close()