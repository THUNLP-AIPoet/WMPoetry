# -*- coding: utf-8 -*-
# @Author: Xiaoyuan Yi
# @Last Modified by:   Xiaoyuan Yi
# @Last Modified time: 2020-06-11 20:39:43
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


class WMTrainer(object):

    def __init__(self, hps, device):
        self.hps = hps
        self.device = device

    def run_validation(self, epoch, wm_model, criterion, tool, lr):
        logger = SimpleLogger('valid')
        logger.set_batch_num(tool.valid_batch_num)
        logger.set_log_path(self.hps.valid_log_path)
        logger.set_rate('learning_rate', lr)
        logger.set_rate('teach_ratio', wm_model.get_teach_ratio())
        logger.set_rate('temperature', wm_model.get_tau())

        for step in range(0, tool.valid_batch_num):

            batch = tool.valid_batches[step]

            all_inps = [inps.to(self.device) for inps in batch[0]]
            all_trgs = [trgs.to(self.device) for trgs in batch[1]]
            all_ph_inps = [ph_inps.to(self.device) for ph_inps in batch[2]]
            all_len_inps = [len_inps.to(self.device) for len_inps in batch[3]]
            keys = [key.to(self.device) for key in batch[4]]

            with torch.no_grad():
                gen_loss, _ = self.run_step(wm_model, None, criterion,
                    all_inps, all_trgs, all_ph_inps, all_len_inps, keys, True)

            logger.add_losses(gen_loss)

        logger.print_log(epoch)


    def run_step(self, wm_model, optimizer, criterion,
        all_inps, all_trgs, all_ph_inps, all_len_inps, keys, valid=False):

        if not valid:
            optimizer.zero_grad()

        all_outs = wm_model(all_inps, all_trgs, all_ph_inps, all_len_inps, keys)

        loss_vec = []
        assert len(all_outs) == len(all_trgs)
        for out, trg in zip(all_outs, all_trgs):
            loss = criterion(out, trg)
            loss_vec.append(loss.unsqueeze(0))

        loss = torch.cat(loss_vec, dim=0).mean()

        if not valid:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(wm_model.parameters(),
                self.hps.clip_grad_norm)
            optimizer.step()

        return loss.item(), all_outs

    # -------------------------------------------------------------------------
    def run_train(self, wm_model, tool, optimizer, criterion, logger):

        logger.set_start_time()

        for step in range(0, tool.train_batch_num):

            batch = tool.train_batches[step]
            all_inps = [inps.to(self.device) for inps in batch[0]]
            all_trgs = [trgs.to(self.device) for trgs in batch[1]]
            all_ph_inps = [ph_inps.to(self.device) for ph_inps in batch[2]]
            all_len_inps = [len_inps.to(self.device) for len_inps in batch[3]]
            keys = [key.to(self.device) for key in batch[4]]

            # train the classifier, recognition network and decoder
            gen_loss, all_outs = self.run_step(wm_model, optimizer, criterion,
                    all_inps, all_trgs, all_ph_inps, all_len_inps, keys)

            logger.add_losses(gen_loss)
            logger.set_rate("learning_rate", optimizer.rate())

            # temperature annealing
            wm_model.set_tau(self.tau_decay_tool.do_step())
            logger.set_rate('temperature', self.tau_decay_tool.get_rate())

            if step % self.hps.log_steps == 0:
                logger.set_end_time()
                utils.sample_wm(keys, all_trgs, all_outs, self.hps.sample_num, tool)
                logger.print_log()
                logger.set_start_time()



    def train(self, wm_model, tool):
        #utils.print_parameter_list(wm_model)
        # load data for pre-training
        print ("building data for wm...")
        tool.build_data(self.hps.train_data, self.hps.valid_data,
            self.hps.batch_size, mode='wm')

        print ("train batch num: %d" % (tool.train_batch_num))
        print ("valid batch num: %d" % (tool.valid_batch_num))


        #input("please check the parameters, and then press any key to continue >")

        # training logger
        logger = SimpleLogger('train')
        logger.set_batch_num(tool.train_batch_num)
        logger.set_log_steps(self.hps.log_steps)
        logger.set_log_path(self.hps.train_log_path)
        logger.set_rate('learning_rate', 0.0)
        logger.set_rate('teach_ratio', 1.0)
        logger.set_rate('temperature', 1.0)


        # build optimizer
        opt = torch.optim.AdamW(wm_model.parameters(),
            lr=1e-3, betas=(0.9, 0.99), weight_decay=self.hps.weight_decay)
        optimizer = ISRScheduler(optimizer=opt, warmup_steps=self.hps.warmup_steps,
            max_lr=self.hps.max_lr, min_lr=self.hps.min_lr,
            init_lr=self.hps.init_lr, beta=0.6)

        wm_model.train()

        null_idxes = tool.load_function_tokens(self.hps.data_dir + "fchars.txt").to(self.device)
        wm_model.set_null_idxes(null_idxes)

        criterion = Criterion(self.hps.pad_idx)

        # change each epoch
        tr_decay_tool = ExponentialDecay(self.hps.burn_down_tr, self.hps.decay_tr, self.hps.min_tr)
        # change each iteration
        self.tau_decay_tool = ExponentialDecay(0, self.hps.tau_annealing_steps, self.hps.min_tau)


        # -----------------------------------------------------------
        # train with all data
        for epoch in range(1, self.hps.max_epoches+1):

            self.run_train(wm_model, tool, optimizer, criterion, logger)

            if epoch % self.hps.validate_epoches == 0:
                print("run validation...")
                wm_model.eval()
                print ("in training mode: %d" % (wm_model.training))
                self.run_validation(epoch, wm_model, criterion, tool, optimizer.rate())
                wm_model.train()
                print ("validation Done: %d" % (wm_model.training))


            if (self.hps.save_epoches >= 1) and \
                (epoch % self.hps.save_epoches) == 0:
                # save checkpoint
                print("saving model...")
                utils.save_checkpoint(self.hps.model_dir, epoch, wm_model, prefix="wm")


            logger.add_epoch()

            print ("teach forcing ratio decay...")
            wm_model.set_teach_ratio(tr_decay_tool.do_step())
            logger.set_rate('teach_ratio', tr_decay_tool.get_rate())

            print("shuffle data...")
            tool.shuffle_train_data()