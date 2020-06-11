# -*- coding: utf-8 -*-
# @Author: Xiaoyuan Yi
# @Last Modified by:   Xiaoyuan Yi
# @Last Modified time: 2020-06-11 18:15:55
# @Email: yi-xy16@mails.tsinghua.edu.cn
# @Description:
'''
Copyright 2020 THUNLP Lab. All Rights Reserved.
This code is part of the online Chinese poetry generation system, Jiuge.
System URL: https://jiuge.thunlp.cn/ and https://jiuge.thunlp.org/.
Github: https://github.com/THUNLP-AIPoet.
'''
from dseq_trainer import DSeqTrainer
from wm_trainer import WMTrainer

from graphs import WorkingMemoryModel
from tool import Tool
from config import device, hparams
import utils


def pretrain(wm_model, tool, hps, specified_device):
    dseq_trainer = DSeqTrainer(hps, specified_device)

    print ("dseq pretraining...")
    dseq_trainer.train(wm_model, tool)
    print ("dseq pretraining done!")



def train(wm_model, tool, hps, specified_device):
    last_epoch = utils.restore_checkpoint(
        hps.model_dir, specified_device, wm_model)

    if last_epoch is not None:
         print ("checkpoint exsits! directly recover!")
    else:
         print ("checkpoint not exsits! train from scratch!")

    wm_trainer = WMTrainer(hps, specified_device)
    wm_trainer.train(wm_model, tool)


def main():
    hps = hparams
    tool = Tool(hps.sens_num, hps.sen_len,
        hps.key_len, hps.topic_slots, hps.corrupt_ratio)
    tool.load_dic(hps.vocab_path, hps.ivocab_path)
    vocab_size = tool.get_vocab_size()
    PAD_ID = tool.get_PAD_ID()
    B_ID = tool.get_B_ID()
    assert vocab_size > 0 and PAD_ID >=0 and B_ID >= 0
    hps = hps._replace(vocab_size=vocab_size, pad_idx=PAD_ID, bos_idx=B_ID)

    print ("hyper-patameters:")
    print (hps)
    input("please check the hyper-parameters, and then press any key to continue >")

    wm_model = WorkingMemoryModel(hps, device)
    wm_model = wm_model.to(device)

    pretrain(wm_model, tool, hps, device)
    train(wm_model, tool, hps, device)


if __name__ == "__main__":
    main()
