# -*- coding: utf-8 -*-
# @Author: Xiaoyuan Yi
# @Last Modified by:   Xiaoyuan Yi
# @Last Modified time: 2020-06-11 18:06:09
# @Email: yi-xy16@mails.tsinghua.edu.cn
# @Description:
'''
Copyright 2020 THUNLP Lab. All Rights Reserved.
This code is part of the online Chinese poetry generation system, Jiuge.
System URL: https://jiuge.thunlp.cn/ and https://jiuge.thunlp.org/.
Github: https://github.com/THUNLP-AIPoet.
'''
import torch
from torch import nn

class Criterion(nn.Module):
    def __init__(self, pad_idx):
        super().__init__()
        self._criterion = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=pad_idx)
        self._pad_idx = pad_idx


    def forward(self, outputs, targets):
        # outputs: (B, L, V)
        # targets: (B, L)

        vocab_size = outputs.size(-1)
        outs = outputs.contiguous().view(-1, vocab_size) # outs: (N, V)
        tgts = targets.contiguous().view(-1) # tgts: (N)

        non_pad_mask = tgts.ne(self._pad_idx)

        loss = self._criterion(outs, tgts) # [N]
        loss = loss.masked_select(non_pad_mask).mean()

        return loss