# -*- coding: utf-8 -*-
# @Author: Xiaoyuan Yi
# @Last Modified by:   Xiaoyuan Yi
# @Last Modified time: 2020-06-11 18:09:56
# @Email: yi-xy16@mails.tsinghua.edu.cn
# @Description:
'''
Copyright 2020 THUNLP Lab. All Rights Reserved.
This code is part of the online Chinese poetry generation system, Jiuge.
System URL: https://jiuge.thunlp.cn/ and https://jiuge.thunlp.org/.
Github: https://github.com/THUNLP-AIPoet.
'''
class ISRScheduler(object):
    '''Inverse Square Root Schedule
    '''
    def __init__(self, optimizer, warmup_steps, max_lr=5e-4, min_lr=3e-5, init_lr=1e-5, beta=0.55):
        self._optimizer = optimizer

        self._step = 0
        self._rate = init_lr

        self._warmup_steps = warmup_steps
        self._max_lr = max_lr
        self._min_lr = min_lr
        self._init_lr = init_lr

        self._alpha = (max_lr-init_lr) / warmup_steps
        self._beta = beta
        self._gama = max_lr * warmup_steps ** (beta)


    def step(self):
        self._step += 1
        rate = self.rate()
        for p in self._optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self._optimizer.step()

    def rate(self):
        step = self._step
        if step < self._warmup_steps:
            lr = self._init_lr + self._alpha * step
        else:
            lr = self._gama * step ** (-self._beta)

        if step > self._warmup_steps:
            lr = max(lr, self._min_lr)
        return lr

    def zero_grad(self):
        self._optimizer.zero_grad()

    def state_dict(self):
        return self._optimizer.state_dict()

    def load_state_dict(self, dic):
        self._optimizer.load_state_dict(dic)