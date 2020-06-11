# -*- coding: utf-8 -*-
# @Author: Xiaoyuan Yi
# @Last Modified by:   Xiaoyuan Yi
# @Last Modified time: 2020-06-11 20:39:50
# @Email: yi-xy16@mails.tsinghua.edu.cn
# @Description:
'''
Copyright 2020 THUNLP Lab. All Rights Reserved.
This code is part of the online Chinese poetry generation system, Jiuge.
System URL: https://jiuge.thunlp.cn/ and https://jiuge.thunlp.org/.
Github: https://github.com/THUNLP-AIPoet.
'''
import torch

from scheduler import ISRScheduler
from criterion import Criterion
from decay import ExponentialDecay
from logger import SimpleLogger
import utils

class DSeqTrainer(object):

    def __init__(self, hps, device):
        self.hps = hps
        self.device = device


    def run_validation(self, epoch, wm_model, criterion, tool, lr):
        logger = SimpleLogger('valid')
        logger.set_batch_num(tool.valid_batch_num)
        logger.set_log_path(self.hps.dseq_valid_log_path)
        logger.set_rate('learning_rate', lr)
        logger.set_rate('teach_ratio', wm_model.get_teach_ratio())

        for step in range(0, tool.valid_batch_num):

            batch = tool.valid_batches[step]

            inps = batch[0].to(self.device)
            trgs = batch[1].to(self.device)
            ph_inps = batch[2].to(self.device)
            len_inps = batch[3].to(self.device)

            with torch.no_grad():
                gen_loss, _ = self.run_step(wm_model, None, criterion,
                    inps, trgs, ph_inps, len_inps, True)
            logger.add_losses(gen_loss)

        logger.print_log(epoch)


    def run_step(self, wm_model, optimizer, criterion,
        inps, trgs, ph_inps, len_inps, valid=False):
        if not valid:
            optimizer.zero_grad()

        outs = wm_model.dseq_graph(inps, trgs, ph_inps, len_inps)

        loss = criterion(outs, trgs)

        if not valid:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(wm_model.dseq_parameters(),
                self.hps.clip_grad_norm)
            optimizer.step()

        return loss.item(), outs


    def run_train(self, wm_model, tool, optimizer, criterion, logger):
        logger.set_start_time()

        for step in range(0, tool.train_batch_num):

            batch = tool.train_batches[step]

            inps = batch[0].to(self.device)
            trgs = batch[1].to(self.device)
            ph_inps = batch[2].to(self.device)
            len_inps = batch[3].to(self.device)

            gen_loss, outs = self.run_step(wm_model, optimizer, criterion,
                inps, trgs, ph_inps, len_inps)

            logger.add_losses(gen_loss)
            logger.set_rate("learning_rate", optimizer.rate())
            if step % self.hps.dseq_log_steps == 0:
                logger.set_end_time()
                utils.sample_dseq(inps, trgs, outs, self.hps.sample_num, tool)
                logger.print_log()
                logger.set_start_time()



    def train(self, wm_model, tool):
        #utils.print_parameter_list(wm_model, wm_model.dseq_parameter_names())

        # load data for pre-training
        print ("building data for dseq...")
        tool.build_data(self.hps.train_data, self.hps.valid_data,
            self.hps.dseq_batch_size, mode='dseq')

        print ("train batch num: %d" % (tool.train_batch_num))
        print ("valid batch num: %d" % (tool.valid_batch_num))

        #input("please check the parameters, and then press any key to continue >")


        # training logger
        logger = SimpleLogger('train')
        logger.set_batch_num(tool.train_batch_num)
        logger.set_log_steps(self.hps.dseq_log_steps)
        logger.set_log_path(self.hps.dseq_train_log_path)
        logger.set_rate('learning_rate', 0.0)
        logger.set_rate('teach_ratio', 1.0)


        # build optimizer
        opt = torch.optim.AdamW(wm_model.dseq_parameters(),
            lr=1e-3, betas=(0.9, 0.99), weight_decay=self.hps.weight_decay)
        optimizer = ISRScheduler(optimizer=opt, warmup_steps=self.hps.dseq_warmup_steps,
            max_lr=self.hps.dseq_max_lr, min_lr=self.hps.dseq_min_lr,
            init_lr=self.hps.dseq_init_lr, beta=0.6)

        wm_model.train()

        criterion = Criterion(self.hps.pad_idx)

        # tech forcing ratio decay
        tr_decay_tool = ExponentialDecay(self.hps.dseq_burn_down_tr, self.hps.dseq_decay_tr,
            self.hps.dseq_min_tr)

        # train
        for epoch in range(1, self.hps.dseq_epoches+1):

            self.run_train(wm_model, tool, optimizer, criterion, logger)

            if epoch % self.hps.dseq_validate_epoches == 0:
                print("run validation...")
                wm_model.eval()
                print ("in training mode: %d" % (wm_model.training))
                self.run_validation(epoch, wm_model, criterion, tool, optimizer.rate())
                wm_model.train()
                print ("validation Done: %d" % (wm_model.training))


            if (self.hps.dseq_save_epoches >= 1) and \
                (epoch % self.hps.dseq_save_epoches) == 0:
                # save checkpoint
                print("saving model...")
                utils.save_checkpoint(self.hps.model_dir, epoch, wm_model, prefix="dseq")


            logger.add_epoch()

            print ("teach forcing ratio decay...")
            wm_model.set_teach_ratio(tr_decay_tool.do_step())
            logger.set_rate('teach_ratio', tr_decay_tool.get_rate())

            print("shuffle data...")
            tool.shuffle_train_data()