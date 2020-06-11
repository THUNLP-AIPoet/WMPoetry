# -*- coding: utf-8 -*-
# @Author: Xiaoyuan Yi
# @Last Modified by:   Xiaoyuan Yi
# @Last Modified time: 2020-06-11 18:04:40
# @Email: yi-xy16@mails.tsinghua.edu.cn
# @Description:
'''
Copyright 2020 THUNLP Lab. All Rights Reserved.
This code is part of the online Chinese poetry generation system, Jiuge.
System URL: https://jiuge.thunlp.cn/ and https://jiuge.thunlp.org/.
Github: https://github.com/THUNLP-AIPoet.
'''
import numpy as np
import torch
import copy


class Hypothesis(object):
    '''
    a hypothesis which holds the generated tokens,
        current state, beam score and memory reading weights
    '''
    def __init__(self, tokens, states, score, read_aligns):
        self.score = score
        self.states = states
        self.candidate = copy.deepcopy(tokens)
        self.read_aligns = read_aligns # (1, L_i, mem_slots)


class PoetryBeam(object):
    def __init__(self, device, beam_size, length, B_ID, E_ID, UNK_ID,
         level_char_ids, oblique_char_ids):
        """Initialize params."""
        self.device = device

        self._length = length
        self._beam_size = beam_size

        self._B_ID = B_ID
        self._E_ID = E_ID
        self._UNK_ID = UNK_ID

        self._level_cids = level_char_ids
        self._oblique_cids = oblique_char_ids


    def reset(self, init_state, rhythms, rhyme_char_ids, repetitive_ids):
        # reset before generating each line
        self._hypotheses \
            = [Hypothesis([self._B_ID], [init_state.clone().detach()], 0.0, [])
            for _ in range(0, self._beam_size)]

        self._completed_hypotheses = []

        self._rhythms = rhythms # rhythm pattern of each chars in a line
        self._rhyme_cids = rhyme_char_ids # char ids in the required rhyme category
        self._repetitive_ids = repetitive_ids


    def get_candidates(self, completed=False, with_states=False):
        if completed:
            hypotheses = self._completed_hypotheses
        else:
            hypotheses = self._hypotheses

        candidates = [hypo.candidate for hypo in hypotheses]
        scores = [hypo.score for hypo in hypotheses]

        read_aligns = [hypo.read_aligns for hypo in hypotheses]

        if with_states:
            # (L, H) * B
            all_states = [hypo.states for hypo in hypotheses]
            return candidates, scores, read_aligns, all_states

        else:
            return candidates, scores, read_aligns


    def get_search_results(self, only_completed=True, sort=True):
        candidates, scores, aligns, states = self.get_candidates(True, True)

        if not only_completed:
            add_candis, add_scores, add_aligns, add_states = self.get_candidates(False, True)
            candidates = candidates + add_candis
            scores = scores + add_scores
            states = states + add_states
            aligns = aligns + add_aligns


        scores = [score/(len(candi)-1) for score, candi in zip(scores, candidates)]

        # sort with costs
        if sort:
            sort_indices = list(np.argsort(scores))
            candidates = [candidates[i] for i in sort_indices]
            scores = [scores[i] for i in sort_indices]
            states = [states[i] for i in sort_indices]
            aligns = [aligns[i] for i in sort_indices]

        # ignore the bos symbol and initial state
        candidates = [candi[1: ] for candi in candidates]
        states = [ state[1:] for state in states ]

        return candidates, scores, states, aligns


    def get_beam_tails(self):
        # get the last token and state of each hypothesis
        tokens = [hypo.candidate[-1] for hypo in self._hypotheses]
        # (B)
        tail_tokens = torch.tensor(tokens, dtype=torch.long, device=self.device)

        tail_states = [hypo.states[-1] for hypo in self._hypotheses]
        # [1, H] * B -> [B, H]
        tail_states = torch.cat(tail_states, dim=0)

        return tail_tokens, tail_states


    def uncompleted_num(self):
        return len(self._hypotheses)


    def advance(self, logit, state, read_align, position):
        # logit: (B, V)
        # state: (B, H)
        # read_align: (B, 1, mem_slots)
        log_prob = torch.nn.functional.log_softmax(logit, dim=-1).cpu().data.numpy()

        beam_ids, word_ids, scores = self._beam_select(log_prob, position)

        # update beams
        updated_hypotheses = []
        for beam_id, word_id, score in zip(beam_ids, word_ids, scores):
            # (1, H)
            new_states = self._hypotheses[beam_id].states + [state[beam_id, :].unsqueeze(0)]

            new_candidate = self._hypotheses[beam_id].candidate + [word_id]

            new_aligns = self._hypotheses[beam_id].read_aligns + \
                [read_align[beam_id, :, :].unsqueeze(0)]

            hypo = Hypothesis(new_candidate, new_states, score, new_aligns)

            if word_id == self._E_ID:
                self._completed_hypotheses.append(hypo)
            else:
                updated_hypotheses.append(hypo)

        self._hypotheses = updated_hypotheses


    def _beam_select(self, log_probs, position):
        # log_probs: (B, V)
        B, V = log_probs.shape[0], log_probs.shape[1]


        if position == 0:
            costs = - log_probs[0, :].reshape(1, V) # (1, V)
        else:
            current_scores = [hypo.score for hypo in self._hypotheses]
            costs = np.reshape(current_scores, (B, 1)) - log_probs # (B, V)

        # filter with rhythm, rhyme and length
        #   candidates that don't meet requirements are assigned a large cost
        filter_v = 1e5

        costs[:, self._UNK_ID] = filter_v

        # filter eos symbol
        if position < self._length:
            costs[:, self._E_ID] = filter_v

        # restrain the model from generating chars
        #   that already generated in previous lines
        costs[:, self._repetitive_ids] = filter_v

        # restrain in-line repetitive chars
        inline_filter_ids = self.inline_filter(position)
        for i in range(0, costs.shape[0]):
            costs[i, inline_filter_ids[i]] = filter_v


        # for the tail char, filter out non-rhyme chars
        if (position == self._length-1) and (1 <= self._rhythms[-1] <= 30):
            filter_ids = list(set(range(0, V)) - set(self._rhyme_cids))
            costs[:, filter_ids] = filter_v

        '''
        filter out chars of undesired tones
        NOTE: since some Chinese characters may belong to both tones,
            here we only consider the non-overlap ones
        TODO: disambiguation
        '''
        pos_rhythm = self._rhythms[position]
        if position < self._length and pos_rhythm != 0:
            if pos_rhythm == 31:  # level tone
                costs[:, self._oblique_cids] = filter_v
            elif pos_rhythm == 32:  # oblique
                costs[:, self._level_cids] = filter_v

        flat_costs = costs.flatten() # (B*V)

        # idx of the smallest B elements
        best_indices = np.argpartition(
            flat_costs, B)[0:B].copy()

        scores = flat_costs[best_indices]

        # get beam id and word id
        beam_ids = [int(idx //  V) for idx in best_indices]
        word_ids = [int(idx % V) for idx in best_indices]

        if position == 0:
            beam_ids = list(range(0, B))

        return beam_ids, word_ids, scores


    def inline_filter(self, pos):
        candidates, _, _ = self.get_candidates()
        # candidates: (L_i) * B
        B = len(candidates)
        forbidden_list = [[] for _ in range(0, B)]

        limit_pos = pos - 1 if pos % 2 != 0 else pos
        preidx = range(0, limit_pos)

        for i in range(0, B):  # iter ever batch
            forbidden_list[i] = [candidates[i][c] for c in preidx]

        return forbidden_list