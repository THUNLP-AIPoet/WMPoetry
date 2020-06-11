# -*- coding: utf-8 -*-
# @Author: Xiaoyuan Yi
# @Last Modified by:   Xiaoyuan Yi
# @Last Modified time: 2020-06-11 21:05:31
# @Email: yi-xy16@mails.tsinghua.edu.cn
# @Description:
'''
Copyright 2020 THUNLP Lab. All Rights Reserved.
This code is part of the online Chinese poetry generation system, Jiuge.
System URL: https://jiuge.thunlp.cn/ and https://jiuge.thunlp.org/.
Github: https://github.com/THUNLP-AIPoet.
'''

from collections import namedtuple
import torch
HParams = namedtuple('HParams',
    'vocab_size, pad_idx, bos_idx,'
    'word_emb_size, ph_emb_size, len_emb_size,'
    'hidden_size, mem_size, global_trace_size, topic_trace_size,'
    'his_mem_slots, topic_slots, sens_num, sen_len, key_len,'

    'batch_size, drop_ratio, attn_drop_ratio, weight_decay, clip_grad_norm,'
    'max_lr, min_lr, init_lr, warmup_steps,'
    'min_tr, burn_down_tr, decay_tr,'
    'tau_annealing_steps, min_tau,'

    'log_steps, sample_num, max_epoches,'
    'save_epoches, validate_epoches,'
    'vocab_path, ivocab_path, train_data, valid_data,'
    'model_dir, data_dir, train_log_path, valid_log_path,'

    'corrupt_ratio, dseq_epoches, dseq_batch_size,'
    'dseq_max_lr, dseq_min_lr, dseq_init_lr dseq_warmup_steps,'
    'dseq_min_tr, dseq_burn_down_tr, dseq_decay_tr,'

    'dseq_log_steps, dseq_validate_epoches, dseq_save_epoches,'
    'dseq_train_log_path, dseq_valid_log_path,'
)


hparams = HParams(
    # --------------------
    # general settings
    vocab_size=-1, pad_idx=-1, bos_idx=-1, # to be replaced by true size after loading dictionary
    word_emb_size=256, ph_emb_size=64, len_emb_size=32,
    hidden_size=512, mem_size=512, global_trace_size=512, topic_trace_size=20,
    his_mem_slots=4, topic_slots=4, sens_num=4, sen_len=10, key_len=2,


    batch_size=128, drop_ratio=0.25, attn_drop_ratio=0.1,
    weight_decay=2.5e-4, clip_grad_norm=2.0,
    max_lr=1e-3, min_lr=5e-8, init_lr=1e-4, warmup_steps=6000, # learning rate decay
    min_tr=0.80, burn_down_tr=3, decay_tr=10, # epoches for teach forcing ratio decay
    tau_annealing_steps=30000, min_tau=0.01,# Gumbel temperature, from 1 to min_tau

    log_steps=100, sample_num=1, max_epoches=14,
    save_epoches=2, validate_epoches=1,

    vocab_path="../corpus/vocab.pickle",
    ivocab_path="../corpus/ivocab.pickle",
    train_data="../corpus/train_data.pickle",
    valid_data="../corpus/valid_data.pickle",
    model_dir="../checkpoint/",
    data_dir="../data/",
    train_log_path="../log/wm_train_log.txt",
    valid_log_path="../log/wm_valid_log.txt",

    #--------------------------
    # for pre-training
    corrupt_ratio=0.1, dseq_epoches=10, dseq_batch_size=256,
    dseq_max_lr=1e-3, dseq_min_lr=5e-5, dseq_init_lr=1e-4, dseq_warmup_steps=6000,
    dseq_min_tr=0.80, dseq_burn_down_tr=3, dseq_decay_tr=7,

    dseq_log_steps=200, dseq_validate_epoches=1, dseq_save_epoches=2,
    dseq_train_log_path="../log/dae_train_log.txt",
    dseq_valid_log_path="../log/dae_valid_log.txt"

)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
