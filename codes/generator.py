# -*- coding: utf-8 -*-
# @Author: Xiaoyuan Yi
# @Last Modified by:   Xiaoyuan Yi
# @Last Modified time: 2020-06-11 20:19:25
# @Email: yi-xy16@mails.tsinghua.edu.cn
# @Description:
'''
Copyright 2020 THUNLP Lab. All Rights Reserved.
This code is part of the online Chinese poetry generation system, Jiuge.
System URL: https://jiuge.thunlp.cn/ and https://jiuge.thunlp.org/.
Github: https://github.com/THUNLP-AIPoet.
'''
import torch
import torch.nn.functional as F

from graphs import WorkingMemoryModel

from tool import Tool
from beam import PoetryBeam
from filter import PoetryFilter
from visualization import Visualization
import utils

class Generator(object):
    '''
    generator for testing
    '''

    def __init__(self, hps, device):
        self.tool = Tool(hps.sens_num, hps.sen_len,
            hps.key_len, hps.topic_slots, 0.0)
        self.tool.load_dic(hps.vocab_path, hps.ivocab_path)
        vocab_size = self.tool.get_vocab_size()
        print ("vocabulary size: %d" % (vocab_size))
        PAD_ID = self.tool.get_PAD_ID()
        B_ID = self.tool.get_B_ID()
        assert vocab_size > 0 and PAD_ID >=0 and B_ID >= 0
        self.hps = hps._replace(vocab_size=vocab_size, pad_idx=PAD_ID, bos_idx=B_ID)
        self.device = device

        # load model
        model = WorkingMemoryModel(self.hps, device)

        # load trained model
        utils.restore_checkpoint(self.hps.model_dir, device, model)
        self.model = model.to(device)
        self.model.eval()

        null_idxes = self.tool.load_function_tokens(self.hps.data_dir + "fchars.txt").to(self.device)
        self.model.set_null_idxes(null_idxes)

        self.model.set_tau(hps.min_tau)

        # load poetry filter
        print ("loading poetry filter...")
        self.filter = PoetryFilter(self.tool.get_vocab(),
            self.tool.get_ivocab(), self.hps.data_dir)

        self.visual_tool = Visualization(hps.topic_slots, hps.his_mem_slots,
            "../log/")
        print("--------------------------")



    def generate_one(self, keywords, pattern, beam_size=20, verbose=1, manu=False, visual=0):
        '''
        generate one poem according to the inputs:
            keyword: a list of topic words, at most key_slots
            pattern: genre pattern of the poem to be generated
            verbose: 0, 1, 2, 3
            visual: 0, 1, 2
        '''
        key_inps = self.tool.keywords2tensor([keywords]*beam_size)
        key_inps = [key.to(self.device) for key in key_inps]

        if visual > 0:
            self.visual_tool.reset(keywords)


        # inps is a pseudo tensor with pad symbols for generating the first line
        inps, all_ph_inps, all_len_inps, all_lengths = self.tool.patterns2tensor([pattern]*beam_size)

        inps = inps.to(self.device)
        all_ph_inps = [ph_inps.to(self.device) for ph_inps in all_ph_inps]
        all_len_inps = [len_inps.to(self.device) for len_inps in all_len_inps]

        # for quatrains, all lines in a poem share the same length
        length = all_lengths[0]

        # initialize beam pool
        beam_pool = PoetryBeam(self.device, beam_size, length,
            self.tool.get_B_ID(), self.tool.get_E_ID(), self.tool.get_UNK_ID(),
            self.filter.get_level_cids(), self.filter.get_oblique_cids())

        self.filter.reset(length, verbose)

        with torch.no_grad():
            topic_mem, topic_mask, history_mem, history_mask,\
                global_trace, topic_trace, key_init_state = self.model.initialize_mems(key_inps)

        # beam search
        poem = []
        for step in range(0, self.hps.sens_num):
            # generate each line
            if verbose >= 1:
                print ("\ngenerating step: %d" % (step))

            if step > 0:
                key_init_state = None

            candidates, costs, states, read_aligns, local_mem, local_mask = self.beam_search(beam_pool,
                length, inps,
                all_ph_inps[step], pattern[step], all_len_inps[step],
                key_init_state, history_mem, history_mask,
                topic_mem, topic_mask, global_trace, topic_trace)

            lines = [self.tool.idxes2line(idxes) for idxes in candidates]

            lines, costs, states, read_aligns = self.filter.filter_illformed(lines, costs,
                states, read_aligns, pattern[step][-1])

            if len(lines) == 0:
                return [], "line {} generation failed!".format(step)

            which = 0
            if manu:
                for i, (line, cost) in enumerate(zip(lines, costs)):
                    print ("%d, %s, %.2f" % (i, line, cost))
                which = int(input("select sentence>"))

            line = lines[which]
            poem.append(line)

            # set repetitive chars
            self.filter.add_repetitive(self.tool.line2idxes(line))

            # ---------------------------------------
            # write into history memory
            write_log = None
            if step >= 1:
                with torch.no_grad():
                    history_mem, write_log = self.update_history_mem(history_mem,
                        local_mem, local_mask, global_trace)

                history_mask = history_mem.abs().sum(-1).eq(0) # (B, mem_slots)


            with torch.no_grad():
                # update global trace
                global_trace = self.update_glocal_trace(global_trace, states[which], length)

                # update topic trace
                topic_trace = self.update_topic_trace(topic_trace, topic_mem, read_aligns[which])


            # build inps
            inps = self.tool.line2tensor(line, beam_size).to(self.device)


            if visual > 0:
                # show visualization of memory reading
                self.visual_tool.add_gen_line(line)
                self.visual_tool.draw(read_aligns[which], write_log, step, visual)


        return poem, "ok"


    # ------------------------------------
    def beam_search(self, beam_pool, trg_len, inputs, phs, ph_labels, lens, key_init_state,
        history_mem, history_mask, topic_mem, topic_mask, global_trace, topic_trace):

        local_mem, local_mask, init_state = \
            self.model.computer_local_memory(inputs, key_init_state is None)


        if key_init_state is not None:
            init_state = key_init_state

        # reset beam pool
        if 1 <= ph_labels[-1] <= 30:
            rhyme = ph_labels[-1]
        else:
            rhyme = -1

        beam_pool.reset(init_state[0, :].unsqueeze(0), ph_labels+[0]*10,
            self.filter.get_rhyme_cids(rhyme), self.filter.get_repetitive_ids())

        # current size of beam candidates in the beam pool
        n_samples = beam_pool.uncompleted_num()

        total_mask = torch.cat([topic_mask, history_mask, local_mask], dim=1)
        total_mem = torch.cat([topic_mem, history_mem, local_mem], dim=1)


        for k in range(0, trg_len+5):
            inp, state = beam_pool.get_beam_tails()

            if k <= trg_len:
                ph_inp = phs[0:n_samples, k]
                len_inp = lens[0:n_samples, k]
            else:
                ph_inp = torch.zeros(n_samples, dtype=torch.long, device=self.device)
                len_inp = torch.zeros(n_samples, dtype=torch.long, device=self.device)


            with torch.no_grad():
                logit, new_state, read_align = self.model.dec_step(inp, state,
                    ph_inp, len_inp,
                    total_mem[0:n_samples, :, :], total_mask[0:n_samples, :],
                    global_trace[0:n_samples, :], topic_trace[0:n_samples, :])


            beam_pool.advance(logit, new_state, read_align, k)

            n_samples = beam_pool.uncompleted_num()

            if n_samples == 0:
                break

        candidates, costs, dec_states, read_aligns = beam_pool.get_search_results()
        return candidates, costs, dec_states, read_aligns, local_mem, local_mask


    # ---------------
    def update_history_mem(self, history_mem, local_mem, local_mask, global_trace):
        new_history_mem, write_log = self.model.layers['memory_write'](history_mem, local_mem,
            1.0-local_mask.float(), global_trace, self.model.null_mem)


        return new_history_mem, write_log


    def update_glocal_trace(self, global_trace, dec_states, length):
        # dec_states: (1, H) * L_gen

        batch_size = global_trace.size(0)

        # (1, H) -> (B, H, 1)
        states = [state.unsqueeze(2).repeat(batch_size, 1, 1) for state in dec_states]

        l = len(states)

        mask = torch.zeros(batch_size, l, dtype=torch.float, device=self.device)
        mask[:, 0:length+1] = 1.0

        # update global trace vector
        new_global_trace = self.model.update_global_trace(global_trace, states, mask)


        return new_global_trace


    def update_topic_trace(self, topic_trace, topic_mem, read_align):
        # read_align: (1, 1, mem_slots) * L_gen

        batch_size = topic_trace.size(0)
        # concat_aligns: (B, L_gen, mem_slots)
        concat_aligns = torch.cat(read_align, dim=1).repeat(batch_size, 1, 1)
        new_topic_trace = self.model.update_topic_trace(topic_trace, topic_mem, concat_aligns)

        return new_topic_trace


    # --------------------------------------