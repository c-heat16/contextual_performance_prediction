__author__ = 'Connor Heaton'

import os
import json
import itertools

import numpy as np

from .base_vocab import BaseVocab
from .vocab_utils import calc_entropy, ab_ending_token


class GeneralGamestateDeltaVocabByBen(BaseVocab):

    def __init__(self, bos_inning_no=True, max_inning_no=10, bos_score_diff=True,
                 bos_max_score_diff=6, bos_base_occ=True,
                 balls_delta=True, strikes_delta=True, outs_delta=True, score_delta=True, base_occ_delta=True,
                 sep_char='|', secondary_sep_char='!', bos_token='[BOS]', eos_token='[EOS]', boi_token='[BOI]',
                 boab_token='[BOAB]', eoab_token='[EOAB]', mask_token='[MASK]', bpe_encoder=None,
                 swing_status=False
                 ):
        BaseVocab.__init__(self)
        self.key_component_names = []
        self.component_attributes = []
        self.bos_component_names = []
        self.bos_component_attributes = []

        self.bos_inning_no = bos_inning_no
        self.bos_score_diff = bos_score_diff
        self.bos_max_score_diff = bos_max_score_diff
        self.bos_base_occ = bos_base_occ
        self.balls_delta = balls_delta
        self.strikes_delta = strikes_delta
        self.outs_delta = outs_delta
        self.score_delta = score_delta
        self.base_occ_delta = base_occ_delta
        self.max_inning_no = max_inning_no
        self.bpe_encoder = bpe_encoder

        self.sep_char = sep_char
        self.secondary_sep_char = secondary_sep_char
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.boi_token = boi_token
        self.boab_token = boab_token
        self.eoab_token = eoab_token
        self.mask_token = mask_token

        self.bos_id = None
        self.eos_id = None
        self.boi_id = None
        self.boab_id = None
        self.eoab_id = None
        self.mask_id = None

        self.vocab = {}
        self.reverse_vocab = {}

        self.bos_vocab = {}
        self.reverse_bos_vocab = {}

        self.component_value_ids = {}

        if bos_inning_no:
            self.bos_component_names.append('inning')
            inning_no_attributes = [i for i in range(1, max_inning_no + 1)]  # innings start at 1
            self.bos_component_attributes.append(inning_no_attributes)

            value_ids = {v: i for i, v in enumerate(inning_no_attributes)}
            self.component_value_ids['inning'] = value_ids

        if bos_score_diff:
            self.bos_component_names.append('score_diff')
            score_diff_attributes = [i for i in range(-1 * bos_max_score_diff, bos_max_score_diff + 1)]
            self.bos_component_attributes.append(score_diff_attributes)

            value_ids = {v: i for i, v in enumerate(score_diff_attributes)}
            self.component_value_ids['score_diff'] = value_ids

        if bos_base_occ:
            self.bos_component_names.append('on_1b')
            self.bos_component_names.append('on_2b')
            self.bos_component_names.append('on_3b')
            base_occ_components = ['0', '1']
            self.bos_component_attributes.append(base_occ_components)
            self.bos_component_attributes.append(base_occ_components)
            self.bos_component_attributes.append(base_occ_components)

            # all_components = [self.boi_token, self.boab_token, self.eoab_token, self.eos_token]
            all_components = base_occ_components
            value_ids = {v: i for i, v in enumerate(all_components)}
            self.component_value_ids['bos_on_1b'] = value_ids
            self.component_value_ids['bos_on_2b'] = value_ids
            self.component_value_ids['bos_on_3b'] = value_ids

        if balls_delta:
            self.key_component_names.append('balls')
            # balls_delta_components = ['+1', '0', '-1', '-2', '-3']
            balls_delta_components = ['+1', '0', '--']
            self.component_attributes.append(balls_delta_components)

            all_components = [self.boi_token, self.boab_token, self.eoab_token, self.eos_token]
            all_components.extend(balls_delta_components)
            value_ids = {v: i for i, v in enumerate(all_components)}
            self.component_value_ids['balls'] = value_ids

        if strikes_delta:
            self.key_component_names.append('strikes')
            # strikes_delta_components = ['+1', '0', '-1', '-2']
            if swing_status:
                strikes_delta_components = ['+1', '+1\'', '0', '--']
            else:
                strikes_delta_components = ['+1', '0', '--']
            self.component_attributes.append(strikes_delta_components)

            all_components = [self.boi_token, self.boab_token, self.eoab_token, self.eos_token]
            all_components.extend(strikes_delta_components)
            value_ids = {v: i for i, v in enumerate(all_components)}
            self.component_value_ids['strikes'] = value_ids

        if outs_delta:
            self.key_component_names.append('outs')
            outs_components = ['0', '+1', '+2', '+3']
            self.component_attributes.append(outs_components)

            all_components = [self.boi_token, self.boab_token, self.eoab_token, self.eos_token]
            all_components.extend(outs_components)
            value_ids = {v: i for i, v in enumerate(all_components)}
            self.component_value_ids['outs'] = value_ids

        if score_delta:
            self.key_component_names.append('score')
            score_components = ['0', '+1', '+2', '+3', '+4']
            self.component_attributes.append(score_components)

            all_components = [self.boi_token, self.boab_token, self.eoab_token, self.eos_token]
            all_components.extend(score_components)
            value_ids = {v: i for i, v in enumerate(all_components)}
            self.component_value_ids['score'] = value_ids

        if base_occ_delta:
            self.key_component_names.append('on_1b')
            self.key_component_names.append('on_2b')
            self.key_component_names.append('on_3b')
            base_occ_components = ['+1', '0', '-1']
            self.component_attributes.append(base_occ_components)
            self.component_attributes.append(base_occ_components)
            self.component_attributes.append(base_occ_components)

            all_components = [self.boi_token, self.boab_token, self.eoab_token, self.eos_token]
            all_components.extend(base_occ_components)
            value_ids = {v: i for i, v in enumerate(all_components)}
            self.component_value_ids['on_1b'] = value_ids
            self.component_value_ids['on_2b'] = value_ids
            self.component_value_ids['on_3b'] = value_ids

        self.create_vocab()

    def __len__(self):
        return len(self.vocab)

    def calc_component_entropies(self):
        token_ids = [i for i in range(len(self))]
        entropies = {}

        for component_name in self.key_component_names:
            component_values = [self.get_item_components(i)[component_name] for i in token_ids]
            _, count_per_value = np.unique(component_values, return_counts=True)
            component_entropy = calc_entropy(count_per_value)
            entropies[component_name] = component_entropy

        return entropies

    def valid_component_combo(self, component):
        valid = True

        if self.score_delta and self.outs_delta and self.base_occ_delta:
            on_first_idx = self.key_component_names.index('on_1b')
            on_second_idx = self.key_component_names.index('on_2b')
            on_third_idx = self.key_component_names.index('on_3b')
            score_idx = self.key_component_names.index('score')
            outs_idx = self.key_component_names.index('outs')
            balls_idx = self.key_component_names.index('balls')
            strikes_idx = self.key_component_names.index('strikes')

            on_first_val = component[on_first_idx]
            on_second_val = component[on_second_idx]
            on_third_val = component[on_third_idx]
            score_val = component[score_idx]
            outs_val = component[outs_idx]
            balls_val = component[balls_idx]
            strikes_val = component[strikes_idx]
            # lead base runner cannot be removed unless an out or run is added
            if on_third_val[0] == '-' and outs_val[0] == '0' and score_val[0] == '0':
                valid = False
            elif on_third_val[0] == '0' and on_second_val[0] == '-' and outs_val[0] == '0' and score_val[0] == '0':
                valid = False
            elif on_third_val[0] == '0' and on_second_val[0] == '0' and on_first_val[0] == '-' and outs_val[0] == '0' and score_val[0] == '0':
                valid = False
            # Count cannot reset and result in no change to outs, base occupancy, or score
            elif on_third_val[0] == '0' and on_second_val[0] == '0' and on_first_val[0] == '0' and outs_val[0] == '0' and score_val[0] == '0' and balls_val[0] == '-' and strikes_val[0] == '-':
                valid = False

        if self.base_occ_delta and self.outs_delta and self.score_delta:
            on_first_idx = self.key_component_names.index('on_1b')
            on_second_idx = self.key_component_names.index('on_2b')
            on_third_idx = self.key_component_names.index('on_3b')
            score_idx = self.key_component_names.index('score')
            outs_idx = self.key_component_names.index('outs')

            on_first_val = component[on_first_idx]
            on_second_val = component[on_second_idx]
            on_third_val = component[on_third_idx]
            score_val = component[score_idx]
            outs_val = component[outs_idx]
            netBaserunnerChange = 0
            strVal = str(outs_val + "|" + score_val + "|" + on_first_val + "|" + on_second_val + "|" + on_third_val)

            for base_val in [on_first_val, on_second_val, on_third_val]:
                if base_val[0] == '+':
                    netBaserunnerChange += 1
                elif base_val[0] == '-':
                    netBaserunnerChange -= 1

            # net base runner change of 0 or less with no outs and no runs added cannot happen
            if netBaserunnerChange < 1 and outs_val[0] == '0' and score_val[0] == '0':
                valid = False
                #print(outs_val + '|' + score_val + '|' + on_first_val + '|' + on_second_val + '|' + '|' + on_third_val + ' Eliminated at Pos 1')
            # For a run to score, net base runner change must be 0 or negative
            elif netBaserunnerChange > 0 and score_val[0] == '+':
                valid = False
            # For an out to be added, net base runner change must be 0 or negative
            if netBaserunnerChange > 0 and outs_val[0] == '+':
                valid = False
            # You can't lose two lead runners without an out and only score 1 run
            if on_third_val[0] == '-' and on_second_val[0] == '-' and outs_val[0] == '0' and score_val[0] == '+':
                if score_val[1] == '1':
                    valid = False
            # 3 person base change (net -1) without an out being recorded and only one run cannot happen
            if on_third_val[0] == '-' and (on_second_val[0] == '-' or on_first_val[0] == '-') and score_val[0] == '+' and outs_val[0] == '0':
                if score_val[1] == '1':
                    valid = False
            # if one run scores without an out being recorded the base occupancy change has to be net 0
            if netBaserunnerChange != 0 and score_val[0] == '+' and outs_val[0] == '0':
                if score_val[1] == '1':
                    valid = False
            # if two runs score without an out being recorded the net base occupancy change has to -1
            if netBaserunnerChange != -1 and score_val[0] == '+' and outs_val[0] == '0':
                if score_val[1] == '2':
                    valid = False
            # if three runs score without an out being recorded the net base occupancy change has to -2
            if netBaserunnerChange != -2 and score_val[0] == '+' and outs_val[0] == '0':
                if score_val[1] == '3':
                    valid = False
            # if 4 runs score base occupancy must be -3 and outs must be 0
            if netBaserunnerChange != -3 and score_val[0] == '+':
                if score_val[1] == '4':
                    valid = False
            if outs_val[0] != '0' and score_val[0] == '+':
                if score_val[1] == '4':
                    valid = False
            #Same thing as previous 4 but with outs and runs swapped (CHECK FOR LAST OUT OF INNING)
            if (netBaserunnerChange != 0 and score_val[0] == '0' and outs_val[0] == '+') \
                    and (on_first_val[0] == '+' or on_second_val[0] == '+' or on_third_val[0] == '+'):
                if outs_val[1] == '1':
                    valid = False

            if netBaserunnerChange != -1 and score_val[0] == '0' and outs_val[0] == '+' \
                    and (on_first_val[0] == '+' or on_second_val[0] == '+' or on_third_val[0] == '+'):
                if outs_val[1] == '2':
                    valid = False

            if netBaserunnerChange != -2 and score_val[0] == '0' and outs_val[0] == '+':
                if on_first_val[0] == '-' and on_second_val[0] == '-' and on_third_val[0] == '-':
                    valid = True
                elif outs_val[1] == '3':
                    valid = False
            # If 2/3 bases have a (-1) with no runs scored, there
            # needs to be more than 1 out added in order to be valid.
            if outs_val[0] == "+" and score_val[0] == '0' \
                    and ((on_first_val[0] == '-' and on_second_val[0] == '-' and on_third_val[0] == '+') \
                         or (on_first_val[0] == '-' and on_third_val[0] == '-' and on_second_val[0] == '+') \
                         or (on_second_val[0] == '-' and on_third_val[0] == '-' and on_first_val[0] == '+')):
                if outs_val[1] == '1':
                    valid = False
            # Minimum base occupancy change when run and out occur = Runs+outs - 1
            if outs_val[0] == "+" and score_val[0] == '+':
                actual_outs_val = int(outs_val[1])
                actual_score_val = int(score_val[1])
                neededBaseChange = actual_score_val + actual_outs_val
                neededBaseChange = neededBaseChange - 1
                neededBaseChange = neededBaseChange * (-1)
                if neededBaseChange < netBaserunnerChange:
                    valid = False
            # LAST ONE BABYYYYYYYY
            # TWO OUTS AND NO CHANGE IN BASE OCCUPANCY CANNOT OCCUR
            if outs_val[0] == '+' and score_val[0] == '0' and on_first_val[0] == '0' and on_second_val[0] == '0' and \
                    on_third_val == '0':
                if outs_val[1] == '2':
                    valid = False

        if self.balls_delta and self.strikes_delta:
            balls_idx = self.key_component_names.index('balls')
            strikes_idx = self.key_component_names.index('strikes')

            balls_val = component[balls_idx]
            strikes_val = component[strikes_idx]
            # balls and strikes cannot BOTH go up
            if balls_val[0] == '+' and strikes_val[0] == '+':
                valid = False
            # balls and strikes CANNOT move in diff directions
            elif balls_val[0] == '-' and strikes_val[0] != '-' or strikes_val[0] == '-' and balls_val[0] != '-':
                valid = False
            else:
                non_bs_components = [v for v_idx, v in enumerate(component) if v_idx not in [balls_idx, strikes_idx]]
                n_non_zero = sum([0 if nbc == '0' else 1 for nbc in non_bs_components])
                if n_non_zero != 0 and balls_val[0] != '-' and strikes_val[0] != '-':
                    valid = False

        if self.balls_delta and self.strikes_delta and self.outs_delta:
            balls_idx = self.key_component_names.index('balls')
            strikes_idx = self.key_component_names.index('strikes')
            outs_idx = self.key_component_names.index('outs')

            balls_val = component[balls_idx]
            strikes_val = component[strikes_idx]
            outs_val = component[outs_idx]

            # for outs to increase, balls and strikes must go to zero (go down)
            if (balls_val[0] == '+' or strikes_val[0] == '+') and outs_val[0] == '+':
                valid = False

        if self.balls_delta and self.strikes_delta and self.score_delta:
            balls_idx = self.key_component_names.index('balls')
            strikes_idx = self.key_component_names.index('strikes')
            score_idx = self.key_component_names.index('score')

            balls_val = component[balls_idx]
            strikes_val = component[strikes_idx]
            score_val = component[score_idx]

            # balls and strikes must go to zero for run to score (AB is over)
            if (balls_val[0] == '+' or strikes_val[0] == '+') and score_val[0] == '+':
                valid = False

        if self.balls_delta and self.strikes_delta and self.base_occ_delta:
            balls_idx = self.key_component_names.index('balls')
            strikes_idx = self.key_component_names.index('strikes')
            on_1b_idx = self.key_component_names.index('on_1b')
            on_2b_idx = self.key_component_names.index('on_2b')
            on_3b_idx = self.key_component_names.index('on_3b')

            balls_val = component[balls_idx]
            strikes_val = component[strikes_idx]
            on_1b_val = component[on_1b_idx]
            on_2b_val = component[on_2b_idx]
            on_3b_val = component[on_3b_idx]

            # bases cannot change mid-AB (not accounting for steals ATM)
            if (balls_val[0] == '+' or strikes_val[0] == '+') and (on_1b_val[0] != '0' or on_2b_val[0] != '0'
                                                                   or on_3b_val[0] != '0'):
                valid = False

        if self.base_occ_delta:
            on_1b_idx = self.key_component_names.index('on_1b')
            on_2b_idx = self.key_component_names.index('on_2b')
            on_3b_idx = self.key_component_names.index('on_3b')

            on_1b_val = component[on_1b_idx]
            on_2b_val = component[on_2b_idx]
            on_3b_val = component[on_3b_idx]
            n_runner_delta = 0

            # CANNOT have more than two NEW base runner
            for base_val in [on_1b_val, on_2b_val, on_3b_val]:
                if base_val[0] == '+':
                    n_runner_delta += 1
                elif base_val[0] == '-':
                    n_runner_delta -= 1
            if n_runner_delta > 1:
                valid = False

        if self.score_delta and self.outs_delta:
            score_idx = self.key_component_names.index('score')
            outs_idx = self.key_component_names.index('outs')

            score_val = int(component[score_idx])
            outs_val = int(component[outs_idx])
            if score_val + outs_val > 4:
                valid = False

        if self.strikes_delta:
            on_first_idx = self.key_component_names.index('on_1b')
            on_second_idx = self.key_component_names.index('on_2b')
            on_third_idx = self.key_component_names.index('on_3b')
            score_idx = self.key_component_names.index('score')
            outs_idx = self.key_component_names.index('outs')
            balls_idx = self.key_component_names.index('balls')
            strikes_idx = self.key_component_names.index('strikes')

            on_first_val = component[on_first_idx]
            on_second_val = component[on_second_idx]
            on_third_val = component[on_third_idx]
            score_val = component[score_idx]
            outs_val = component[outs_idx]
            balls_val = component[balls_idx]
            strikes_val = component[strikes_idx]

            if on_first_val[0] == '0' and on_second_val[0] == '0' and on_third_val[0] == '0' \
                    and score_val[0] == '0' and outs_val[0] == '0' \
                    and strikes_val[0] == '0' and balls_val[0] == '+':
                valid = True
            elif on_first_val[0] == '0' and on_second_val[0] == '0' and on_third_val[0] == '0' \
                    and score_val[0] == '0' and outs_val[0] == '0' \
                    and strikes_val[0] == '0' and balls_val[0] == '0':
                valid = True
            elif on_first_val[0] == '0' and on_second_val[0] == '0' and on_third_val[0] == '0' \
                    and score_val[0] == '0' and outs_val[0] == '0' \
                    and strikes_val[0] == '+' and balls_val[0] == '0':
                valid = True

        return valid

    def create_vocab(self):
        self.vocab = {}
        self.reverse_vocab = {}
        self.bos_vocab = {}
        self.reverse_bos_vocab = {}

        bos_component_combinations = itertools.product(*self.bos_component_attributes)
        bos_vocab_tokens = [self.sep_char.join([str(c) for c in cc]) for cc in bos_component_combinations]
        self.bos_vocab = {token: idx for idx, token in enumerate(bos_vocab_tokens)}
        self.reverse_bos_vocab = {idx: token for idx, token in enumerate(bos_vocab_tokens)}

        component_combinations = itertools.product(*self.component_attributes)
        vocab_tokens = [self.sep_char.join([str(c) for c in cc]) for cc in component_combinations if self.valid_component_combo(cc)]
        self.vocab = {token: idx for idx, token in enumerate(vocab_tokens)}
        self.reverse_vocab = {idx: token for idx, token in enumerate(vocab_tokens)}

        boab_key = self.sep_char.join([self.boab_token for _ in self.key_component_names])
        boab_id = len(self.vocab)
        self.vocab[boab_key] = boab_id
        self.reverse_vocab[boab_id] = boab_key
        self.boab_id = boab_id

        eoab_key = self.sep_char.join([self.eoab_token for _ in self.key_component_names])
        eoab_id = len(self.vocab)
        self.vocab[eoab_key] = eoab_id
        self.reverse_vocab[eoab_id] = eoab_key
        self.eoab_id = eoab_id

        boi_key = self.sep_char.join([self.boi_token for _ in self.key_component_names])
        boi_id = len(self.vocab)
        self.vocab[boi_key] = boi_id
        self.reverse_vocab[boi_id] = boi_key
        self.boi_id = boi_id

        # bos_key = self.sep_char.join([self.bos_token for _ in self.key_component_names])
        # bos_id = len(self.vocab)
        # self.vocab[bos_key] = bos_id
        # self.reverse_vocab[bos_id] = bos_key
        # self.bos_id = bos_id

        eos_key = self.sep_char.join([self.eos_token for _ in self.key_component_names])
        eos_id = len(self.vocab)
        self.vocab[eos_key] = eos_id
        self.reverse_vocab[eos_id] = eos_key
        self.eos_id = eos_id

        mask_key = self.sep_char.join([self.mask_token for _ in self.key_component_names])
        mask_id = len(self.vocab)
        self.vocab[mask_key] = mask_id
        self.reverse_vocab[mask_id] = mask_key
        self.mask_id = mask_id

    def get_id(self, gamestate, bos=False):
        if bos:
            component_names = self.bos_component_names
        else:
            component_names = self.key_component_names

        if ab_ending_token(gamestate):
            gamestate['balls'] = '--'
            gamestate['strikes'] = '--'

        gamestate_components = []
        for component in component_names:
            component_val = gamestate.get(component, None)

            if component == 'inning':
                if type(component_val) == int:
                    if component_val > self.max_inning_no - 1:
                        component_val = self.max_inning_no - 1
            elif component == 'score_diff':
                if type(component_val) == int:
                    if component_val < (-1 * self.bos_max_score_diff):
                        component_val = -1 * self.bos_max_score_diff
                    elif component_val > self.bos_max_score_diff:
                        component_val = self.bos_max_score_diff
            elif component in ['balls', 'strikes']:
                if component_val[0] == '-':
                    component_val = '--'
            gamestate_components.append(component_val)

        gamestate_key = self.sep_char.join([str(gc) for gc in gamestate_components])
        if bos:
            gamestate_id = self.bos_vocab.get(gamestate_key, -1)
        else:
            gamestate_id = self.vocab.get(gamestate_key, -1)

        # if gamestate_id in [-1]:
        #     print('*' * 25)
        #     print('gamestate: {}'.format(gamestate))
        #     print('gamestate_id: {}'.format(gamestate_id))
        #     print('*' * 25)

        return gamestate_id

    def save_vocab(self, out_dir):
        out_dir = os.path.join(out_dir, 'general_gamestate_delta_vocab')
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        bos_out_fp = os.path.join(out_dir, 'bos_vocab.json')
        bos_reverse_out_fp = os.path.join(out_dir, 'bos_reverse_vocab.json')
        bos_meta_fp = os.path.join(out_dir, 'bos_meta.txt')

        out_fp = os.path.join(out_dir, 'vocab.json')
        reverse_out_fp = os.path.join(out_dir, 'reverse_vocab.json')
        meta_fp = os.path.join(out_dir, 'meta.txt')

        with open(bos_meta_fp, 'w+') as f:
            f.write('Key components: {}\n'.format(','.join(self.bos_component_names)))
            f.write('Example key: \'{}\'\n'.format(self.sep_char.join(self.bos_component_names)))
            f.write('# keys: {}\n\n'.format(len(self.bos_vocab)))
            for component_name, component_attributes in zip(self.bos_component_names, self.bos_component_attributes):
                f.write('{} possible values:\n'.format(component_name))
                for attr_idx, attr_name in enumerate(component_attributes):
                    f.write('\t{}: {}\n'.format(attr_idx, attr_name))

        with open(bos_out_fp, 'w+') as f:
            f.write('{}'.format(json.dumps(self.bos_vocab, indent=2)))

        with open(bos_reverse_out_fp, 'w+') as f:
            f.write('{}'.format(json.dumps(self.reverse_bos_vocab, indent=2)))

        with open(meta_fp, 'w+') as f:
            f.write('Key components: {}\n'.format(','.join(self.key_component_names)))
            f.write('Example key: \'{}\'\n'.format(self.sep_char.join(self.key_component_names)))
            f.write('# keys: {}\n\n'.format(len(self.vocab)))
            for component_name, component_attributes in zip(self.key_component_names, self.component_attributes):
                f.write('{} possible values:\n'.format(component_name))
                for attr_idx, attr_name in enumerate(component_attributes):
                    f.write('\t{}: {}\n'.format(attr_idx, attr_name))

        with open(out_fp, 'w+') as f:
            f.write('{}'.format(json.dumps(self.vocab, indent=2)))

        with open(reverse_out_fp, 'w+') as f:
            f.write('{}'.format(json.dumps(self.reverse_vocab, indent=2)))

