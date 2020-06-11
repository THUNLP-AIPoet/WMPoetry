# -*- coding: utf-8 -*-
# @Author: Xiaoyuan Yi
# @Last Modified by:   Xiaoyuan Yi
# @Last Modified time: 2020-06-11 20:07:53
# @Email: yi-xy16@mails.tsinghua.edu.cn
# @Description:
'''
Copyright 2020 THUNLP Lab. All Rights Reserved.
This code is part of the online Chinese poetry generation system, Jiuge.
System URL: https://jiuge.thunlp.cn/ and https://jiuge.thunlp.org/.
Github: https://github.com/THUNLP-AIPoet.
'''
import os
import torch
import torch.nn.functional as F

import random

def save_checkpoint(model_dir, epoch, model, prefix='', optimizer=None):
    # save model state dict
    checkpoint_name = "model_ckpt_{}_{}e.tar".format(prefix, epoch)
    model_state_path = os.path.join(model_dir, checkpoint_name)

    saved_dic = {
        'epoch': epoch,
        'model_state_dict': model.state_dict()
    }

    if optimizer is not None:
        saved_dic['optimizer'] = optimizer.state_dict()


    torch.save(saved_dic, model_state_path)

    # write checkpoint information
    log_path = os.path.join(model_dir, "ckpt_list.txt")
    fout = open(log_path, 'a')
    fout.write(checkpoint_name+"\n")
    fout.close()


def restore_checkpoint(model_dir, device, model, optimizer=None):
    ckpt_list_path = os.path.join(model_dir, "ckpt_list.txt")
    if not os.path.exists(ckpt_list_path):
        print ("checkpoint list not exists, creat new one!")
        return None

    # get latest ckpt name
    fin = open(ckpt_list_path, 'r')
    latest_ckpt_path = fin.readlines()[-1].strip()
    fin.close()

    latest_ckpt_path = os.path.join(model_dir, latest_ckpt_path)
    if not os.path.exists(latest_ckpt_path):
        print ("latest checkpoint not exists!")
        return None


    print ("restore checkpoint from %s" % (latest_ckpt_path))
    print ("loading...")
    checkpoint = torch.load(latest_ckpt_path, map_location=device)
    #checkpoint = torch.load(latest_ckpt_path)
    print ("load state dic, params: %d..." % (len(checkpoint['model_state_dict'])))
    model.load_state_dict(checkpoint['model_state_dict'])


    if optimizer is not None:
        print ("load optimizer dic...")
        optimizer.load_state_dict(checkpoint['optimizer'])


    epoch = checkpoint['epoch']


    return epoch


def sample_dseq(inputs, targets, logits, sample_num, tool):
    # inps, trgs [batch size, sen len]
    # logits [batch size, trg len, vocab size]
    batch_size = inputs.size(0)
    inp_len = inputs.size(1)
    trg_len = targets.size(1)
    out_len = logits.size(1)


    sample_num = min(sample_num, batch_size)

    # randomly select some examples
    sample_ids = random.sample(list(range(0, batch_size)), sample_num)

    for sid in sample_ids:
        # Build lines
        inps = [inputs[sid, t].item() for t in range(0, inp_len)]
        sline = tool.idxes2line(inps)

        # -------------------------------------------
        trgs = [targets[sid, t].item() for t in range(0, trg_len)]
        tline = tool.idxes2line(trgs)


        # ------------------------------------------
        probs = F.softmax(logits, dim=-1)
        outs = [probs[sid, t, :].cpu().data.numpy() for t in range(0, out_len)]
        oline = tool.greedy_search(outs)


        print("inp: " + sline)
        print("trg: " + tline)
        print("out: " + oline)
        print ("")


#------------------------------
def sample_wm(keys, all_trgs, all_outs, sample_num, tool):
    batch_size = all_trgs[0].size(0)
    sample_num = min(sample_num, batch_size)

    # random select some examples
    sample_ids = random.sample(list(range(0, batch_size)), sample_num)

    for sid in sample_ids:
        key_lines = []
        for key in keys:
            key_idxes = [key[sid, t].item() for t in range(0, key.size(1))]
            key_str = tool.idxes2line(key_idxes)
            if len(key_str) == 0:
                key_str = "PAD"
            key_lines.append(key_str)

        trg_lines = []
        for trg in all_trgs:
            trg_idxes = [trg[sid, t].item() for t in range(0, trg.size(1))]
            trg_lines.append(tool.idxes2line(trg_idxes))

        out_lines = []
        for out in all_outs:
            probs = F.softmax(out, dim=-1)
            out_probs = [probs[sid, t, :].cpu().data.numpy() for t in range(0, probs.size(1))]
            out_lines.append(tool.greedy_search(out_probs))

        #--------------------------------------------
        print("keywords: " + "|".join(key_lines))
        print("target: " + "|".join(trg_lines))
        print("output: " + "|".join(out_lines))
        print ("")


def print_parameter_list(model, prefix=None):
    params = model.named_parameters()

    param_num = 0
    for name, param in params:
        if prefix is not None:
            seg = name.split(".")[1]
            if seg in prefix:
                print(name, param.size())
                param_num += 1
        else:
            print(name, param.size())
            param_num += 1

    print ("params num: %d" % (param_num))
#------------------------------