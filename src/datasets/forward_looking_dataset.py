__author__ = 'Connor Heaton'

import os
import json
import torch
import einops
import random

import numpy as np

from torch.utils.data import Dataset

from .dataset_utils import parse_json, find_state_deltas_v2, parse_pitches_updated, read_file_lines, \
    group_ab_files_by_game

from ..utils import jsonKeys2int, calc_ptb, is_barrel, is_quab
from ..vocabularies import GeneralGamestateDeltaVocabByBen


class FLDataset(Dataset):
    def __init__(self, args, mode, gamestate_vocab=None, finetune=False, ab_data_cache=None):
        self.args = args
        self.mode = mode if mode in ['train', 'test', 'apply', 'identify'] else 'dev'
        self.gamestate_vocab = gamestate_vocab
        self.finetune = finetune
        self.ab_data_cache = ab_data_cache
        self.type = getattr(self.args, 'type', 'batter')

        if self.finetune:
            self.finetune_target_map = {}
            self.targets = getattr(self.args, 'pitcher_targets', ['k', 'h']) if self.type == 'pitcher' \
                else getattr(self.args, 'batter_targets', ['k', 'h'])
            # batter_targets = getattr(self.args, 'pitcher_targets', ['k', 'h'])
            print('FLDataset.targets: {}'.format(self.targets))
            for target in self.targets:
                if target in ['k', 'h']:
                    finetune_target_map_fp = os.path.join(self.args.config_dir,
                                                          'pitch_event_to_{}_status.json'.format(target))
                    self.finetune_target_map[target] = json.load(open(finetune_target_map_fp))
        else:
            self.finetune_target_map = {}

        self.prepend_entity_type_id = getattr(self.args, 'prepend_entity_type_id', False)
        self.use_ball_data = getattr(self.args, 'use_ball_data', True)
        self.mask_ball_data = getattr(self.args, 'mask_ball_data', False)
        self.drop_ball_data = getattr(self.args, 'drop_ball_data', False)
        self.pitcher_career_n_ball_data = getattr(self.args, 'pitcher_career_n_ball_data', 47)
        self.batter_career_n_ball_data = getattr(self.args, 'batter_career_n_ball_data', 47)
        self.else_n_ball_data = getattr(self.args, 'else_n_ball_data', 47)
        self.context_only = getattr(self.args, 'context_only', False)
        self.player_cls = getattr(self.args, 'player_cls', False)
        print('FLDataset.player_cls: {}'.format(self.player_cls))

        self.batter_data_scope_sizes = getattr(self.args, 'batter_data_scope_sizes', [167, 74, 74])
        self.pitcher_data_scope_sizes = getattr(self.args, 'pitcher_data_scope_sizes', [141, 74, 74])
        self.matchup_data_scope_sizes = getattr(self.args, 'matchup_data_scope_sizes', [74, 74, 74])

        self.batter_data_scope_sizes = [int(v) for v in self.batter_data_scope_sizes]
        self.pitcher_data_scope_sizes = [int(v) for v in self.pitcher_data_scope_sizes]
        self.matchup_data_scope_sizes = [int(v) for v in self.matchup_data_scope_sizes]

        self.batter_data_n_ball_data = [self.batter_career_n_ball_data]
        while len(self.batter_data_n_ball_data) < len(self.batter_data_scope_sizes):
            self.batter_data_n_ball_data.append(self.else_n_ball_data)

        self.pitcher_data_n_ball_data = [self.pitcher_career_n_ball_data]
        while len(self.pitcher_data_n_ball_data) < len(self.pitcher_data_scope_sizes):
            self.pitcher_data_n_ball_data.append(self.else_n_ball_data)

        self.matchup_data_n_ball_data = [self.else_n_ball_data]
        while len(self.matchup_data_n_ball_data) < len(self.matchup_data_scope_sizes):
            self.matchup_data_n_ball_data.append(self.else_n_ball_data)

        self.n_games_context = getattr(self.args, 'n_games_context', 3)
        self.context_max_len = getattr(self.args, 'context_max_len', 378)
        self.chop_context_p = getattr(self.args, 'chop_context_p', 0.15)
        self.n_ab_peak_ahead = getattr(self.args, 'n_ab_peak_ahead', 0)
        self.n_games_completion = getattr(self.args, 'n_games_completion', 1)
        self.completion_max_len = getattr(self.args, 'completion_max_len', 128)

        self.v2_player_attn = getattr(self.args, 'v2_player_attn', False)
        self.v2_attn_max_n_batter = 9 * self.n_games_context
        self.v2_attn_max_n_pitcher = self.n_games_context
        self.v2_attn_offset = self.v2_attn_max_n_batter + self.v2_attn_max_n_pitcher

        self.dataset_size_train = getattr(self.args, 'dataset_size_train', 50000)
        self.dataset_size_else = getattr(self.args, 'dataset_size_else', 5000)
        self.batter_min_game_to_be_included_in_dataset = getattr(self.args, 'batter_min_game_to_be_included_in_dataset', 6)
        self.pitcher_min_game_to_be_included_in_dataset = getattr(self.args, 'pitcher_min_game_to_be_included_in_dataset', 6)
        self.ab_data_dir = getattr(self.args, 'ab_data', '/home/czh/sata1/SportsAnalytics/ab_seqs/ab_seqs_v10')
        self.bad_data_fps = getattr(self.args, 'bad_data_fps', [])
        self.distribution_based_player_sampling_prob = getattr(self.args,
                                                               'distribution_based_player_sampling_prob', 0.75)
        self.use_intermediate_data = getattr(self.args, 'use_intermediate_data', False)
        self.use_fast_data = getattr(self.args, 'use_fast_data', False)
        self.use_boab_tokens = getattr(self.args, 'use_boab_tokens', False)
        self.only_sos_statistics = getattr(self.args, 'only_sos_statistics', False)
        self.reduced_event_map = getattr(self.args, 'reduced_event_map', False)
        self.vocab_use_swing_status = getattr(self.args, 'vocab_use_swing_status', False)
        self.description_to_swing_status = {
            'swinging_strike': 1, 'foul': 1, 'foul_tip': 1, 'foul_bunt': 1, 'swinging_strike_blocked': 1,
            'missed_bunt': 1, 'hit_into_play_score': 1, 'hit_into_play_no_out': 1, 'hit_into_play': 1,
            'foul_pitchout': 1, 'swinging_pitchout': 1, 'bunt_foul_tip': 1
        }

        self.batter_ab_threshold_custom_id = getattr(self.args, 'batter_ab_threshold_custom_id', 40)
        self.pitcher_ab_threshold_custom_id = getattr(self.args, 'pitcher_ab_threshold_custom_id', 200)
        batter_pre2021_n_ab_d_fp = getattr(self.args, 'batter_pre2021_n_ab_d_fp',
                                           '/home/czh/nvme1/SportsAnalytics/data/batter_n_ab_pre_2021.json')
        self.batter_pre2021_n_ab_d = json.load(open(batter_pre2021_n_ab_d_fp), object_hook=jsonKeys2int)
        pitcher_pre2021_n_ab_d_fp = getattr(self.args, 'pitcher_pre2021_n_ab_d_fp',
                                            '/home/czh/nvme1/SportsAnalytics/data/pitcher_n_ab_pre_2021.json')
        self.pitcher_pre2021_n_ab_d = json.load(open(pitcher_pre2021_n_ab_d_fp), object_hook=jsonKeys2int)

        # self.game_tensor_dir = getattr(self.args, 'game_tensor_dir', '/home/czh/md0/game_tensors')
        self.game_tensor_dir = os.path.join(self.ab_data_dir, 'game_tensors')
        if not os.path.exists(self.game_tensor_dir):
            os.makedirs(self.game_tensor_dir)

        self.intermediate_data_dir = os.path.join(self.ab_data_dir, 'intermediate_data')
        self.batter_data_scopes_to_use = getattr(self.args, 'batter_data_scopes_to_use',
                                                 ['career', 'season', 'last15'])
        self.pitcher_data_scopes_to_use = getattr(self.args, 'pitcher_data_scopes_to_use',
                                                  ['career', 'season', 'last15'])
        self.matchup_data_scopes_to_use = getattr(self.args, 'matchup_data_scopes_to_use',
                                                  ['career', 'season'])
        print('dataset indermediate_data_dir: {}'.format(self.intermediate_data_dir))

        self.raw_pitcher_data_dim = getattr(self.args, 'raw_pitcher_data_dim', 216)
        self.raw_batter_data_dim = getattr(self.args, 'raw_batter_data_dim', 216)
        self.raw_matchup_data_dim = getattr(self.args, 'raw_matchup_data_dim', 243)

        self.gsd_n_attn = getattr(self.args, 'gsd_n_attn', 8)
        self.event_n_attn = getattr(self.args, 'event_n_attn', 8)
        self.pitch_type_n_attn = getattr(self.args, 'pitch_type_n_attn', 4)
        self.n_attn = getattr(self.args, 'n_attn', 4)

        self.career_data_dir = getattr(self.args, 'career_data',
                                       '/home/czh/sata1/SportsAnalytics/player_career_data_08182022')
        self.whole_game_record_dir = getattr(self.args, 'whole_game_record_dir',
                                             '/home/czh/sata1/SportsAnalytics/whole_game_records')
        pitch_type_config_fp = getattr(self.args, 'pitch_type_config_fp',
                                       '/home/czh/nvme1/SportsAnalytics/config/pitch_type_id_mapping.json')
        player_bio_info_fp = getattr(self.args, 'player_bio_info_fp',
                                     '/home/czh/nvme1/SportsAnalytics/config/statcast_id_to_bio_info.json')
        player_id_map_fp = getattr(self.args, 'player_id_map_fp',
                                   '/home/czh/nvme1/SportsAnalytics/config/all_player_id_mapping.json')
        team_stadiums_fp = getattr(self.args, 'team_stadiums_fp',
                                   '/home/czh/nvme1/SportsAnalytics/config/team_stadiums.json')
        # record_norm_values_fp = getattr(self.args, 'record_norm_values_fp',
        #                                 '/home/czh/nvme1/SportsAnalytics/config/max_values.json')
        record_norm_values_fp = os.path.join(self.ab_data_dir, 'max_vals.json')
        # record_norm_values_fp = os.path.join(self.whole_game_record_dir,
        #                                      'game_event_splits', 'stats', 'max_values.json')
        player_pos_source = getattr(self.args, 'player_pos_source', 'mlb')
        if self.reduced_event_map:
            pitch_event_map_fp = getattr(self.args, 'reduced_pitch_event_map_fp',
                                         '../config/reduced_pitch_event_id_mapping.json')
            raw_pitch_event_map_fp = getattr(self.args, 'pitch_event_map_fp',
                                             '/home/czh/nvme1/SportsAnalytics/config/pitch_event_id_mapping.json')
            intermediate_pitch_event_map_fp = getattr(self.args, 'intermediate_pitch_event_map_fp',
                                                      '/home/czh/nvme1/SportsAnalytics/config/pitch_event_intermediate_id_mapping.json')
            raw_pitch_event_map = json.load(open(raw_pitch_event_map_fp))

            self.reverse_raw_pitch_event_map = {v: k for k, v in raw_pitch_event_map.items()}
            self.intermediate_pitch_event_map = json.load(open(intermediate_pitch_event_map_fp))
        else:
            pitch_event_map_fp = getattr(self.args, 'pitch_event_map_fp',
                                         '/home/czh/nvme1/SportsAnalytics/config/pitch_event_id_mapping.json')

        self.pitch_event_map = json.load(open(pitch_event_map_fp))
        print('pitch_event_map: {}'.format(self.pitch_event_map))

        raw_pitch_event_map_fp = getattr(self.args, 'pitch_event_map_fp',
                                         '/home/czh/nvme1/SportsAnalytics/config/pitch_event_id_mapping.json')
        self.raw_pitch_event_map = json.load(open(raw_pitch_event_map_fp))
        self.player_pos_key = '{}_pos'.format(player_pos_source)
        self.player_pos_id_map = json.load(open(getattr(self.args, '{}_id_map_fp'.format(self.player_pos_key),
                                                        '/home/czh/nvme1/SportsAnalytics/config/{}_mapping.json'.format(
                                                            self.player_pos_key))))

        self.record_norm_values = json.load(open(record_norm_values_fp))
        self.pitch_type_mapping = json.load(open(pitch_type_config_fp))
        self.player_id_map = json.load(open(player_id_map_fp), object_hook=jsonKeys2int)
        self.player_bio_info_mapping = json.load(open(player_bio_info_fp), object_hook=jsonKeys2int)
        self.team_stadium_map = json.load(open(team_stadiums_fp))
        self.handedness_id_map = {'L': 0, 'R': 1}
        self.generic_batter_id = 0
        self.generic_pitcher_id = 0

        pitcher_avg_inning_entry_fp = getattr(self.args, 'pitcher_avg_inning_entry_fp',
                                              '/home/czh/nvme1/SportsAnalytics/data/pitcher_avg_inning_entry.json')
        self.pitcher_avg_inning_entry = json.load(open(pitcher_avg_inning_entry_fp), object_hook=jsonKeys2int)
        # starting_pitcher_only
        self.starting_pitcher_only = getattr(self.args, 'starting_pitcher_only', False)
        print('starting_pitcher_only: {}'.format(self.starting_pitcher_only))

        self.items_by_entity = []
        if self.mode == 'test' and self.mode != 'identify':
            self.test_set_items = {}
            curr_test_idx = 0
            print('* FLDataset Loading EXPLICIT TEST DATA *')
            explicit_test_set_dir = getattr(self.args, 'explicit_test_set_dir',
                                            '/home/czh/sata1/SportsAnalytics/whole_game_records/fl_test_sets')

            for entity_type in self.type:
                print('\tLoading {} test data...'.format(entity_type))
                if self.starting_pitcher_only and entity_type in ['pitcher']:
                    this_test_set_fp = os.path.join(
                        explicit_test_set_dir,
                        '{}_test_set_{}-context_{}-completion_starter_only.json'.format(
                            entity_type, self.n_games_context, self.n_games_completion
                        )
                    )
                else:
                    this_test_set_fp = os.path.join(
                        explicit_test_set_dir,
                        '{}_test_set_{}-context_{}-completion.json'.format(entity_type,
                                                                           self.n_games_context,
                                                                           self.n_games_completion)
                    )
                these_test_set_items = json.load(open(this_test_set_fp), object_hook=jsonKeys2int)
                print('\t\tlen(these_test_set_items): {}'.format(len(these_test_set_items)))
                print('\t\tkeys: {}'.format(list(these_test_set_items.keys())[:6]))
                these_test_set_items = {k: v for k, v in these_test_set_items.items() if
                                        k < (len(these_test_set_items) // 5)}
                for k, v in these_test_set_items.items():
                    self.test_set_items[curr_test_idx] = [entity_type, v]
                    curr_test_idx += 1

                # if self.type == 'batter':
                #     self.test_set_items = {k: v for k, v in self.test_set_items.items() if k < (len(self.test_set_items) // 2)}
                #     # self.test_set_items = {k: v for k, v in self.test_set_items.items()}
                # else:
                #     self.test_set_items = {k: v for k, v in self.test_set_items.items() if k < (len(self.test_set_items) // 4)}
                #     # self.test_set_items = {k: v for k, v in self.test_set_items.items()}

        elif self.mode == 'identify':
            print('* FLDataset loading data for player identification *')
            self.identification_data = []
            for entity_type in self.type:
                type_career_data_dir = os.path.join(self.career_data_dir, entity_type)
                this_identification_data = self.load_identification_data(type_career_data_dir, entity_type=entity_type)
                self.identification_data.extend(this_identification_data)

        elif not self.finetune:
            print('FLDataset Loading initial data...')
            curr_item_count = 0
            for entity_type in self.type:
                print('\tLoading {} career data...'.format(entity_type))
                type_career_data_dir = os.path.join(self.career_data_dir, entity_type)
                self.load_initial_data(type_career_data_dir, entity_type=entity_type)
                _, _, entity_item_counts = map(list, zip(*self.items_by_entity))
                n_entity_items = sum(entity_item_counts) - curr_item_count
                print('\t\tFound {} {} items...'.format(n_entity_items, entity_type))
                curr_item_count = sum(entity_item_counts)

            self.entity_types, self.entity_items, self.entity_item_counts = map(list, zip(*self.items_by_entity))

            self.n_entity_items = sum(self.entity_item_counts)
            self.n_entity = len(self.entity_item_counts)
            self.entity_idxs = [i for i in range(self.n_entity)]

            print('# entities: {}'.format(self.n_entity))
            print('# entity records: {}'.format(self.n_entity_items))

        if self.gamestate_vocab is None:
            gamestate_vocab_bos_inning_no = getattr(self.args, 'gamestate_vocab_bos_inning_no', False)
            gamestate_vocab_bos_score_diff = getattr(self.args, 'gamestate_vocab_bos_score_diff', False)
            gamestate_vocab_bos_base_occ = getattr(self.args, 'gamestate_vocab_bos_base_occ', True)
            gamestate_vocab_use_balls_strikes = getattr(self.args, 'gamestate_vocab_use_balls_strikes', True)
            gamestate_vocab_use_base_occupancy = getattr(self.args, 'gamestate_vocab_use_base_occupancy', True)
            gamestate_vocab_use_inning_no = getattr(self.args, 'gamestate_vocab_use_inning_no', False)
            gamestate_vocab_use_inning_topbot = getattr(self.args, 'gamestate_vocab_use_inning_topbot', False)
            gamestate_vocab_use_score_diff = getattr(self.args, 'gamestate_vocab_use_score_diff', True)
            gamestate_vocab_use_outs = getattr(self.args, 'gamestate_vocab_use_outs', True)
            gamestate_n_innings = getattr(self.args, 'gamestate_n_innings', 10)
            gamestate_max_score_diff = getattr(self.args, 'gamestate_max_score_diff', 6)
            print('Creating Vocab')
            self.gamestate_vocab = GeneralGamestateDeltaVocabByBen(
                bos_inning_no=gamestate_vocab_bos_inning_no, max_inning_no=gamestate_n_innings,
                bos_score_diff=gamestate_vocab_bos_score_diff, bos_max_score_diff=gamestate_max_score_diff,
                bos_base_occ=gamestate_vocab_bos_base_occ,
                balls_delta=gamestate_vocab_use_balls_strikes, strikes_delta=gamestate_vocab_use_balls_strikes,
                outs_delta=gamestate_vocab_use_outs, score_delta=gamestate_vocab_use_score_diff,
                base_occ_delta=gamestate_vocab_use_base_occupancy,
                swing_status=self.vocab_use_swing_status
            )

        self.gsd_vocab_size = len(self.gamestate_vocab.vocab)
        self.bos_vocab_size = len(self.gamestate_vocab.bos_vocab)
        self.boab_id = self.gamestate_vocab.boab_id
        self.eos_id = self.gamestate_vocab.eos_id
        print('self.boab_id: {}'.format(self.boab_id))
        print('self.eos_id: {}'.format(self.eos_id))

    def load_initial_data(self, career_data_dir, entity_type):
        n_not_processed = 0
        for entity_file in os.listdir(career_data_dir):
            process_entity = True
            if self.starting_pitcher_only and entity_type in ['pitcher']:
                player_id = int(entity_file[:-4])
                avg_inning_entry = self.pitcher_avg_inning_entry[player_id]
                if avg_inning_entry > 3.5:
                    # print('Not processing {} because they are pitcher w/ avg inning of entry of {} (> 2.5)'.format(
                    #     player_id, avg_inning_entry
                    # ))
                    process_entity = False
                    n_not_processed += 1

            if process_entity:
                entity_data_fp = os.path.join(career_data_dir, entity_file)
                entity_ab_files = read_file_lines(entity_data_fp, self.bad_data_fps)

                if self.mode == 'train':
                    if self.use_ball_data:
                        # print('use_ball_data=True, only getting records from 2015-2020, inclusive')
                        entity_ab_files = [eabf for eabf in entity_ab_files if int(eabf[:4]) >= 2015]
                    else:
                        # print('use_ball_data=False, getting all records before 2021')
                        entity_ab_files = [eabf for eabf in entity_ab_files
                                           if not any(eabf.startswith(year_) for year_ in ['2021', '2022'])]

                else:
                    entity_ab_files = [eabf for eabf in entity_ab_files
                                       if any(eabf.startswith(year_) for year_ in ['2021', '2022'])]

                entity_files_by_game = group_ab_files_by_game(entity_ab_files)

                if entity_type in ['pitcher'] and len(entity_files_by_game) >= self.pitcher_min_game_to_be_included_in_dataset:
                    n_possible_items = max(1,
                                           len(entity_files_by_game) - self.n_games_context - self.n_games_completion + 1)
                    self.items_by_entity.append([entity_type, entity_files_by_game, n_possible_items])
                elif entity_type not in ['pitcher'] and len(entity_files_by_game) >= self.batter_min_game_to_be_included_in_dataset:
                    n_possible_items = max(1,
                                           len(entity_files_by_game) - self.n_games_context - self.n_games_completion + 1)
                    self.items_by_entity.append([entity_type, entity_files_by_game, n_possible_items])

        if n_not_processed > 0:
            print('\tDid not process {} {} records...'.format(n_not_processed, entity_type))

    def load_identification_data(self, career_data_dir, entity_type):
        n_not_processed = 0
        identification_data = []
        for entity_file in os.listdir(career_data_dir):
            process_entity = True
            if self.starting_pitcher_only and entity_type in ['pitcher']:
                player_id = int(entity_file[:-4])
                avg_inning_entry = self.pitcher_avg_inning_entry[player_id]
                if avg_inning_entry > 3.5:
                    # print('Not processing {} because they are pitcher w/ avg inning of entry of {} (> 2.5)'.format(
                    #     player_id, avg_inning_entry
                    # ))
                    process_entity = False
                    n_not_processed += 1
            if process_entity:
                entity_data_fp = os.path.join(career_data_dir, entity_file)
                entity_ab_files = read_file_lines(entity_data_fp, self.bad_data_fps)

                entity_ab_files = [eabf for eabf in entity_ab_files if eabf.startswith('2021')]
                entity_files_by_game = group_ab_files_by_game(entity_ab_files)

                for game_idx in range(1, len(entity_files_by_game) - 1):
                    context_start_index = max(0, game_idx - self.n_games_context)
                    context_files = entity_files_by_game[context_start_index:game_idx]
                    completion_files = entity_files_by_game[game_idx:game_idx + self.n_games_completion]

                    identification_data.append([entity_type, context_files, completion_files])

        if n_not_processed > 0:
            print('\tDid not process {} {} records...'.format(n_not_processed, entity_type))

        return identification_data

    def __len__(self):
        if self.mode == 'test':
            sz = len(self.test_set_items)
        elif self.mode == 'identify':
            sz = len(self.identification_data)
        else:
            sz = self.dataset_size_train if self.mode == 'train' else self.dataset_size_else
        return sz

    def __getitem__(self, idx):
        context_masked_indices = None
        completion_masked_indices = None
        entity_type = 'unk'
        if self.mode == 'test':
            entity_type, item_files = self.test_set_items[idx]
            context_files = item_files['context_files']
            completion_files = item_files['completion_files']
            # context_masked_indices = item_files['context_masked_indices']
            # completion_masked_indices = item_files['completion_masked_indices']
            context_masked_indices = None
            completion_masked_indices = None

            # if len(context_masked_indices) > self.context_max_len:
            #     context_masked_indices = context_masked_indices[:self.context_max_len]
            # elif len(context_masked_indices) < self.context_max_len:
            #     n_pad = self.context_max_len - len(context_masked_indices)
            #     context_masked_indices.extend([False for _ in range(n_pad)])
            #
            # if len(completion_masked_indices) > self.completion_max_len:
            #     completion_masked_indices = completion_masked_indices[:self.completion_max_len]
            # elif len(completion_masked_indices) < self.completion_max_len:
            #     n_pad = self.completion_max_len - len(completion_masked_indices)
            #     completion_masked_indices.extend([False for _ in range(n_pad)])

            if context_masked_indices is not None:
                context_masked_indices = torch.tensor(context_masked_indices)
            if completion_masked_indices is not None:
                completion_masked_indices = torch.tensor(completion_masked_indices)
        elif self.mode == 'identify':
            entity_type, context_files, completion_files = self.identification_data[idx]
        else:
            # entity_files = random.choices(self.entity_items, weights=self.entity_item_counts, k=1)[0]
            selected_idx = random.choices(self.entity_idxs, weights=self.entity_item_counts, k=1)[0]
            entity_type = self.entity_types[selected_idx]
            entity_files = self.entity_items[selected_idx]
            n_entity_files = len(entity_files)
            record_start_idx = random.randint(0,
                                              max(n_entity_files - self.n_games_context - self.n_games_completion, 1))
            context_files = entity_files[record_start_idx:(record_start_idx + self.n_games_context)]
            completion_files = entity_files[(record_start_idx + self.n_games_context):(
                    record_start_idx + self.n_games_context + self.n_games_completion)]

            if len(completion_files) == 0:
                completion_files.append(context_files[-1])
                context_files = context_files[:-1]
        # os.path.join(self.ab_data_dir, item_fp)
        # first_context_j = json.load(open(os.path.join(self.ab_data_dir, context_files[0][0])))
        # raw_entity_id = first_context_j[self.type]['__id__']
        # custom_entity_id = self.player_id_map[raw_entity_id]
        raw_entity_id = -1
        custom_entity_id = -1
        # print('context_files: {}'.format(context_files))
        # print('completion_files: {}'.format(completion_files))
        first_completion_j = json.load(open(os.path.join(self.ab_data_dir, completion_files[0][0])))
        completion_first_game_pk = first_completion_j['game']['game_pk']
        completion_first_game_year = first_completion_j['game']['game_year']

        if not self.finetune and random.random() <= self.chop_context_p and not self.mode == 'test':
            og_len = len(context_files)
            if self.v2_player_attn:
                len_choices = [i for i in range(1, len(context_files) + 1)]
            else:
                len_choices = [i for i in range(len(context_files) + 1)]
            # len_choices.insert(0, None)
            if self.type == 'batter':
                if self.v2_player_attn:
                    len_weights = [(self.n_games_context // 10) + i for i in range(1, len(context_files) + 1)]
                else:
                    len_weights = [(self.n_games_context // 10) + i for i in range(len(context_files) + 1)]
            else:
                if self.v2_player_attn:
                    len_weights = [(self.n_games_context // 3) + i for i in range(1, len(context_files) + 1)]
                else:
                    len_weights = [(self.n_games_context // 3) + i for i in range(len(context_files) + 1)]

            # print('len_choices: {}'.format(len_choices))
            # print('len_weights: {}'.format(len_weights))

            chosen_len = random.choices(len_choices, weights=len_weights, k=1)[0]
            # chosen_len = 0
            if chosen_len == 0:
                context_files = [[]]
            else:
                context_files = context_files[-chosen_len:]

        context_data = self.process_file_set(context_files, mode='context', entity_type=entity_type)

        # if not self.context_only:
        completion_data = self.process_file_set(completion_files, mode='completion', entity_type=entity_type)
        if len(completion_files) == 0:
            print_items = [
                '!!! len(completion_files) == 0 !!!',
                'record_start_idx: {}'.format(record_start_idx),
                'n_entity_files: {}'.format(n_entity_files)
            ]
            print('\n'.join(print_items))
        # else:
        #     completion_data = None
        del context_data['game_pk_t']
        del completion_data['game_pk_t']
        del context_data['game_year_t']
        del completion_data['game_year_t']

        record_data = {
            'custom_entity_id': torch.tensor(custom_entity_id),
            'raw_entity_id': torch.tensor(raw_entity_id),
            'completion_first_game_pk': torch.tensor(completion_first_game_pk),
            'completion_first_game_year': torch.tensor(completion_first_game_year),
        }
        if context_masked_indices is not None and completion_masked_indices is not None:
            record_data['context_masked_indices'] = context_masked_indices
            record_data['completion_masked_indices'] = completion_masked_indices

        if self.only_sos_statistics:
            # print('*' * 40)
            # print('Adjusting stats to only reflect sos')
            # print('raw context stats: {}'.format(context_data['{}_supp_inputs'.format(self.type)]))
            # print('raw completion stats: {}'.format(completion_data['{}_supp_inputs'.format(self.type)]))
            # print('context pad mask: {}'.format(context_data['my_src_pad_mask']))

            if len(context_files) > 0 and len(context_files[0]):
                first_context_j = json.load(open(os.path.join(self.ab_data_dir, context_files[0][0])))
                context_player_supp_inputs = self.make_supp_inputs_from_j(first_context_j)

                context_player_supp_inputs = einops.repeat(context_player_supp_inputs,
                                                           'l w -> (repeat l) w',
                                                           repeat=self.context_max_len)
                tmp_context_pad_mask = einops.repeat(
                    context_data['my_src_pad_mask'].unsqueeze(-1),
                    'l w -> l (repeat w)',
                    repeat=context_player_supp_inputs.shape[-1]
                )

                context_player_supp_inputs = context_player_supp_inputs * tmp_context_pad_mask

                context_data['{}_supp_inputs'.format(self.type)] = context_player_supp_inputs

            first_completion_j = json.load(open(os.path.join(self.ab_data_dir, completion_files[0][0])))
            completion_player_supp_inputs = self.make_supp_inputs_from_j(first_completion_j)

            completion_player_supp_inputs = einops.repeat(completion_player_supp_inputs,
                                                          'l w -> (repeat l) w',
                                                          repeat=self.completion_max_len)

            tmp_completion_pad_mask = einops.repeat(
                completion_data['my_src_pad_mask'].unsqueeze(-1),
                'l w -> l (repeat w)',
                repeat=completion_player_supp_inputs.shape[-1]
            )

            completion_player_supp_inputs = completion_player_supp_inputs * tmp_completion_pad_mask
            completion_data['{}_supp_inputs'.format(self.type)] = completion_player_supp_inputs
            # print('new context stats: {}'.format(context_data['{}_supp_inputs'.format(self.type)]))
            # print('new completion stats: {}'.format(completion_data['{}_supp_inputs'.format(self.type)]))
            # print('*' * 40)

        if not self.use_ball_data:
            context_data['pitcher_supp_inputs'] = self.strip_ball_data(context_data['pitcher_supp_inputs'],
                                                                       data_type='pitcher')
            context_data['batter_supp_inputs'] = self.strip_ball_data(context_data['batter_supp_inputs'],
                                                                      data_type='batter')
            context_data['matchup_supp_inputs'] = self.strip_ball_data(context_data['matchup_supp_inputs'],
                                                                       data_type='matchup')

            context_data['rv_thrown_pitch_data'] = torch.rand_like(context_data['rv_thrown_pitch_data'])
            context_data['rv_batted_ball_data'] = torch.rand_like(context_data['rv_batted_ball_data'])
            context_data['pitch_types'] = torch.zeros_like(context_data['pitch_types'])

            if not self.context_only:
                completion_data['pitcher_supp_inputs'] = self.strip_ball_data(completion_data['pitcher_supp_inputs'],
                                                                              data_type='pitcher')
                completion_data['batter_supp_inputs'] = self.strip_ball_data(completion_data['batter_supp_inputs'],
                                                                             data_type='batter')
                completion_data['matchup_supp_inputs'] = self.strip_ball_data(completion_data['matchup_supp_inputs'],
                                                                              data_type='matchup')

                completion_data['rv_thrown_pitch_data'] = torch.rand_like(completion_data['rv_thrown_pitch_data'])
                completion_data['rv_batted_ball_data'] = torch.rand_like(completion_data['rv_batted_ball_data'])
                completion_data['pitch_types'] = torch.zeros_like(completion_data['pitch_types'])

            # del context_data['rv_thrown_pitch_data']
            # del context_data['rv_batted_ball_data']
            # del context_data['pitch_types']
            #
            # del completion_data['rv_thrown_pitch_data']
            # del completion_data['rv_batted_ball_data']
            # del completion_data['pitch_types']

        if self.v2_player_attn:
            player_attention_data = self.construct_player_attn_data_v2(context_data, completion_data,
                                                                       game_year=completion_first_game_year)
            custom_player_ids, real_player_ids, context_player_attn_mask, completion_player_attn_mask = player_attention_data
        else:
            player_attention_data = self.construct_player_attn_data(context_data, completion_data, game_year=completion_first_game_year)
            custom_player_ids, real_player_ids, context_player_attn_mask, completion_player_attn_mask = player_attention_data
        # context_data['pitcher_attn_mask'] = pitcher_context_attn_mask
        # context_data['batter_attn_mask'] = batter_context_attn_masks
        # completion_data['pitcher_attn_mask'] = pitcher_completion_attn_mask
        # completion_data['batter_attn_mask'] = batter_completion_attn_masks
        context_data['player_attn_mask'] = context_player_attn_mask
        if completion_data is not None:
            completion_data['player_attn_mask'] = completion_player_attn_mask

        # record_data['custom_pitcher_id'] = pitcher_custom_id
        # record_data['custom_batter_ids'] = batter_custom_ids
        record_data['custom_player_ids'] = custom_player_ids
        record_data['real_player_ids'] = real_player_ids

        entity_type_id = self.args.entity_type_d[entity_type]
        record_data['entity_type_id'] = torch.tensor([entity_type_id])

        for k, v in context_data.items():
            record_data['context_{}'.format(k)] = v
        if not self.context_only:
            for k, v in completion_data.items():
                record_data['completion_{}'.format(k)] = v

        return record_data

    def construct_player_attn_data_v2(self, context_data, completion_data=None,
                                      given_pitcher_id=None, given_batter_ids=None, game_year=0):
        # ASSUMES CONTEXT ONLY
        real_player_ids = []
        custom_player_ids = []

        """
            FIND PLAYER IDS
        """

        context_pitcher_ids = context_data['pitcher_id']
        context_batter_ids = context_data['batter_id']

        # Find pitcher IDs
        u_pitcher_ids, pitcher_id_counts = np.unique(context_pitcher_ids[context_pitcher_ids != 0], return_counts=True)
        sorted_p_count_data = list(sorted(zip(u_pitcher_ids, pitcher_id_counts), key=lambda x: x[-1], reverse=True))
        sorted_p_count_data = sorted_p_count_data[:self.v2_attn_max_n_pitcher]
        if len(sorted_p_count_data) > 0:
            pitcher_ids, _ = map(list, zip(*sorted_p_count_data))
        else:
            pitcher_ids = []
        if given_pitcher_id is not None:
            if given_pitcher_id in pitcher_ids:
                given_pitcher_idx = pitcher_ids.index(given_pitcher_id)
                del pitcher_ids[given_pitcher_idx]
            else:
                pitcher_ids = pitcher_ids[:-1]
            pitcher_ids.insert(0, given_pitcher_id)

        for pitcher_id in pitcher_ids:
            pitcher_pre2021_ab = self.pitcher_pre2021_n_ab_d[pitcher_id]
            if game_year < 2021 or pitcher_pre2021_ab >= self.pitcher_ab_threshold_custom_id:
                pitcher_custom_id = self.player_id_map[pitcher_id]
                custom_player_ids.append(pitcher_custom_id)
            else:
                custom_player_ids.append(self.generic_pitcher_id)
            real_player_ids.append(pitcher_id)

        while len(custom_player_ids) < self.v2_attn_max_n_pitcher:
            custom_player_ids.append(self.generic_batter_id)
            real_player_ids.append(-1)

        # Now find batter IDs
        u_batter_ids, batter_id_counts = np.unique(context_batter_ids[context_batter_ids != 0], return_counts=True)
        sorted_b_count_data = list(sorted(zip(u_batter_ids, batter_id_counts), key=lambda x: x[-1], reverse=True))
        sorted_b_count_data = sorted_b_count_data[:self.v2_attn_max_n_batter]
        if len(sorted_b_count_data) > 0:
            batter_ids, _ = map(list, zip(*sorted_b_count_data))
        else:
            batter_ids = []
        if given_batter_ids is not None:
            for given_batter_id in given_batter_ids:
                if given_batter_id in batter_ids:
                    given_batter_idx = batter_ids.index(given_batter_id)
                    del batter_ids[given_batter_idx]
                else:
                    batter_ids = batter_ids[:-1]
                batter_ids.insert(0, given_batter_id)

        for batter_id in batter_ids:
            batter_pre2021_ab = self.batter_pre2021_n_ab_d[batter_id]
            if game_year < 2021 or batter_pre2021_ab >= self.batter_ab_threshold_custom_id:
                custom_player_ids.append(self.player_id_map[batter_id])
            else:
                custom_player_ids.append(self.generic_batter_id)
            real_player_ids.append(batter_id)

        while len(custom_player_ids) < self.v2_attn_max_n_pitcher + self.v2_attn_max_n_batter:
            custom_player_ids.append(self.generic_batter_id)
            real_player_ids.append(-1)

        """
            MAKE THE MASK
        """
        context_player_attn_mask = torch.zeros(self.context_max_len + self.v2_attn_offset + 1,
                                               self.context_max_len + self.v2_attn_offset + 1,
                                               dtype=torch.float)
        context_player_attn_mask[:, 1:self.v2_attn_offset + 1] = float('-inf')
        context_player_attn_mask[1:self.v2_attn_offset + 1, :] = float('-inf')
        for i in range(self.v2_attn_offset + 1):
            context_player_attn_mask[i, i] = 0
        # context_player_attn_mask[:self.v2_attn_offset + 1, 0] = 0

        for player_idx, player_id in enumerate(real_player_ids):
            if player_idx < self.v2_attn_max_n_pitcher:
                player_context_presence = (context_pitcher_ids == player_id).nonzero()
            else:
                player_context_presence = (context_batter_ids == player_id).nonzero()
            # print('player_idx: {} player_id: {} player_context_presence: {} self.v2_attn_offset: {}'.format(
            #     player_idx, player_id, player_context_presence, self.v2_attn_offset
            # ))
            context_player_attn_mask[player_idx, player_context_presence + self.v2_attn_offset + 1] = 0
            context_player_attn_mask[player_context_presence + self.v2_attn_offset + 1, player_idx] = 0

        context_player_attn_mask = context_player_attn_mask.unsqueeze(0).expand(self.n_attn, -1, -1)
        custom_player_ids = torch.tensor(custom_player_ids)
        real_player_ids = torch.tensor(real_player_ids)

        return custom_player_ids, real_player_ids, context_player_attn_mask, None

    def construct_player_attn_data(self, context_data, completion_data=None,
                                   given_pitcher_id=None, given_batter_ids=None, game_year=0):

        real_player_ids = []
        custom_player_ids = []

        if given_pitcher_id is not None and given_batter_ids is not None:
            pitcher_id = given_pitcher_id
            pitcher_pre2021_ab = self.pitcher_pre2021_n_ab_d[pitcher_id]
            if game_year < 2021 or pitcher_pre2021_ab >= self.pitcher_ab_threshold_custom_id:
                # print('Adding real custom ID b/b {} had {} AB pre 2021 and year is {}'.format(
                #     pitcher_id, pitcher_pre2021_ab, game_year
                # ))
                pitcher_custom_id = self.player_id_map[pitcher_id]
                custom_player_ids.append(pitcher_custom_id)
            else:
                # print('Adding generic ID b/c {} had {} AB pre 2021 (< {}) and year is {}'.format(
                #     pitcher_id, pitcher_pre2021_ab, self.batter_ab_threshold_custom_id, game_year
                # ))
                custom_player_ids.append(self.generic_pitcher_id)
            real_player_ids.append(pitcher_id)

            for batter_id in given_batter_ids:
                real_player_ids.append(batter_id)
                batter_pre2021_ab = self.batter_pre2021_n_ab_d[batter_id]
                if game_year < 2021 or batter_pre2021_ab >= self.batter_ab_threshold_custom_id:
                    custom_player_ids.append(self.player_id_map[batter_id])
                else:
                    custom_player_ids.append(self.generic_batter_id)
        else:
            completion_pitcher_ids = completion_data['pitcher_id']
            pitcher_id = completion_pitcher_ids[0].item()
            pitcher_pre2021_ab = self.pitcher_pre2021_n_ab_d[pitcher_id]
            if game_year < 2021 or pitcher_pre2021_ab >= self.pitcher_ab_threshold_custom_id:
                # print('Adding real custom ID b/b {} had {} AB pre 2021 and year is {}'.format(
                #     pitcher_id, pitcher_pre2021_ab, game_year
                # ))
                pitcher_custom_id = self.player_id_map[pitcher_id]
                custom_player_ids.append(pitcher_custom_id)
            else:
                # print('Adding generic ID b/c {} had {} AB pre 2021 (< {}) and year is {}'.format(
                #     pitcher_id, pitcher_pre2021_ab, self.batter_ab_threshold_custom_id, game_year
                # ))
                custom_player_ids.append(self.generic_pitcher_id)
            real_player_ids.append(pitcher_id)

            completion_batter_ids = completion_data['batter_id']
            i = 0
            while len(real_player_ids) < 10 and i < completion_batter_ids.shape[0] and \
                    completion_data['my_src_pad_mask'][i] > 0:
                if completion_batter_ids[i].item() not in real_player_ids:
                    batter_pre2021_ab = self.batter_pre2021_n_ab_d[completion_batter_ids[i].item()]
                    if game_year < 2021 or batter_pre2021_ab >= self.batter_ab_threshold_custom_id:
                        # print('Adding real custom ID b/b {} had {} AB pre 2021 and year is {}'.format(
                        #     completion_batter_ids[i].item(), batter_pre2021_ab, game_year
                        # ))
                        real_player_ids.append(completion_batter_ids[i].item())
                        custom_player_ids.append(self.player_id_map[completion_batter_ids[i].item()])
                    else:
                        # print('Adding generic ID b/c {} had {} AB pre 2021 (< {}) and year is {}'.format(
                        #     completion_batter_ids[i].item(), batter_pre2021_ab, self.batter_ab_threshold_custom_id,
                        #     game_year
                        # ))
                        custom_player_ids.append(self.generic_batter_id)
                        real_player_ids.append(completion_batter_ids[i].item())
                i += 1

        while len(custom_player_ids) < 10:
            custom_player_ids.append(self.generic_batter_id)
            real_player_ids.append(-1)

        context_pitcher_ids = context_data['pitcher_id']
        context_batter_ids = context_data['batter_id']

        context_player_attn_mask = torch.zeros(self.context_max_len + 10 + 1, self.context_max_len + 10 + 1,
                                               dtype=torch.float)
        context_player_attn_mask[:, :10 + 1] = float('-inf')
        context_player_attn_mask[:10 + 1, :] = float('-inf')
        for i in range(11):
            context_player_attn_mask[i, i] = 0
        context_player_attn_mask[:, 0] = 0
        # context_player_attn_mask[0, 1:] = float('-inf')
        context_player_attn_mask[0, :] = 0

        if not self.context_only:
            if self.prepend_entity_type_id:
                completion_offset = 11
            else:
                completion_offset = 10

            if completion_data is not None:
                completion_player_attn_mask = torch.zeros(self.completion_max_len + completion_offset,
                                                          self.completion_max_len + completion_offset,
                                                          dtype=torch.float)
                completion_player_attn_mask[:, :completion_offset] = float('-inf')
                completion_player_attn_mask[:completion_offset, :] = float('-inf')
                for i in range(completion_offset):
                    completion_player_attn_mask[i, i] = 0

                if self.prepend_entity_type_id:
                    completion_player_attn_mask[:, 0] = 0
                    completion_player_attn_mask[0, :] = 0
            else:
                completion_player_attn_mask = None
        else:
            completion_player_attn_mask = None

        for player_idx, player_id in enumerate(real_player_ids):
            if player_idx == 0:
                player_context_presence = (context_pitcher_ids == player_id).nonzero()
                # print('pitcher_id: {}'.format(player_id))
                # print('pitcher_context_presence:\n{}\n\t{}'.format(player_context_presence.T, player_context_presence.shape))
                if not self.context_only:
                    player_completion_presence = (completion_pitcher_ids == player_id).nonzero()
            else:
                player_context_presence = (context_batter_ids == player_id).nonzero()
                if not self.context_only:
                    player_completion_presence = (completion_batter_ids == player_id).nonzero()

            context_player_attn_mask[player_idx + 1, player_context_presence + 10 + 1] = 0
            context_player_attn_mask[player_context_presence + 10 + 1, player_idx + 1] = 0

            if not self.context_only:
                completion_player_attn_mask[player_idx, player_completion_presence + 10] = 0
                completion_player_attn_mask[player_completion_presence + 10, player_idx] = 0

        # print('context_player_attn_mask:\n{}\n\t{}'.format(context_player_attn_mask[:, 11:],
        #                                                    context_player_attn_mask.shape))

        context_player_attn_mask = context_player_attn_mask.unsqueeze(0).expand(self.n_attn, -1, -1)
        if not self.context_only:
            completion_player_attn_mask = completion_player_attn_mask.unsqueeze(0).expand(self.n_attn, -1, -1)
        custom_player_ids = torch.tensor(custom_player_ids)
        real_player_ids = torch.tensor(real_player_ids)

        return custom_player_ids, real_player_ids, context_player_attn_mask, completion_player_attn_mask

    def construct_player_attn_data_DEP(self, context_data, completion_data=None,
                                   given_pitcher_id=None, given_batter_ids=None, game_year=0):
        context_pitcher_ids = context_data['pitcher_id']
        context_batter_ids = context_data['batter_id']

        context_player_attn_mask = torch.zeros(self.context_max_len + 10 + 1, self.context_max_len + 10 + 1,
                                               dtype=torch.float)
        context_player_attn_mask[:, :10 + 1] = float('-inf')
        context_player_attn_mask[:10 + 1, :] = float('-inf')
        for i in range(11):
            context_player_attn_mask[i, i] = 0
        context_player_attn_mask[:, 0] = 0
        # context_player_attn_mask[0, 1:] = float('-inf')
        context_player_attn_mask[0, :] = 0

        if self.prepend_entity_type_id:
            completion_offset = 11
        else:
            completion_offset = 10

        if completion_data is not None:
            completion_player_attn_mask = torch.zeros(self.completion_max_len + completion_offset, self.completion_max_len + completion_offset,
                                                      dtype=torch.float)
            completion_player_attn_mask[:, :completion_offset] = float('-inf')
            completion_player_attn_mask[:completion_offset, :] = float('-inf')
            for i in range(completion_offset):
                completion_player_attn_mask[i, i] = 0

            if self.prepend_entity_type_id:
                completion_player_attn_mask[:, 0] = 0
                completion_player_attn_mask[0, :] = 0
        else:
            completion_player_attn_mask = None

        real_player_ids = []
        custom_player_ids = []

        if given_pitcher_id is not None:
            pitcher_id = given_pitcher_id
        else:
            completion_pitcher_ids = completion_data['pitcher_id']
            pitcher_id = completion_pitcher_ids[0].item()

        pitcher_pre2021_ab = self.pitcher_pre2021_n_ab_d[pitcher_id]
        if game_year < 2021 or pitcher_pre2021_ab >= self.pitcher_ab_threshold_custom_id:
            # print('Adding real custom ID b/b {} had {} AB pre 2021 and year is {}'.format(
            #     pitcher_id, pitcher_pre2021_ab, game_year
            # ))
            pitcher_custom_id = self.player_id_map[pitcher_id]
            custom_player_ids.append(pitcher_custom_id)
        else:
            # print('Adding generic ID b/c {} had {} AB pre 2021 (< {}) and year is {}'.format(
            #     pitcher_id, pitcher_pre2021_ab, self.batter_ab_threshold_custom_id, game_year
            # ))
            custom_player_ids.append(self.generic_pitcher_id)
        real_player_ids.append(pitcher_id)

        if given_batter_ids is not None:
            for batter_id in given_batter_ids:
                real_player_ids.append(batter_id)
                batter_pre2021_ab = self.batter_pre2021_n_ab_d[batter_id]
                if game_year < 2021 or batter_pre2021_ab >= self.batter_ab_threshold_custom_id:
                    custom_player_ids.append(self.player_id_map[batter_id])
                    custom_player_ids.append(self.generic_batter_id)
        else:
            completion_batter_ids = completion_data['batter_id']
            # batter_ids = []
            i = 0
            while len(real_player_ids) < 10 and i < completion_batter_ids.shape[0] and completion_data['my_src_pad_mask'][i] > 0:
                if completion_batter_ids[i].item() not in real_player_ids:
                    batter_pre2021_ab = self.batter_pre2021_n_ab_d[completion_batter_ids[i].item()]
                    if game_year < 2021 or batter_pre2021_ab >= self.batter_ab_threshold_custom_id:
                        # print('Adding real custom ID b/b {} had {} AB pre 2021 and year is {}'.format(
                        #     completion_batter_ids[i].item(), batter_pre2021_ab, game_year
                        # ))
                        real_player_ids.append(completion_batter_ids[i].item())
                        custom_player_ids.append(self.player_id_map[completion_batter_ids[i].item()])
                    else:
                        # print('Adding generic ID b/c {} had {} AB pre 2021 (< {}) and year is {}'.format(
                        #     completion_batter_ids[i].item(), batter_pre2021_ab, self.batter_ab_threshold_custom_id,
                        #     game_year
                        # ))
                        custom_player_ids.append(self.generic_batter_id)
                        real_player_ids.append(completion_batter_ids[i].item())
                i += 1

        while len(custom_player_ids) < 10:
            custom_player_ids.append(self.generic_batter_id)
            real_player_ids.append(-1)
            # batter_context_attn_masks.append(torch.zeros_like(context_batter_ids).unsqueeze(0) + float('-inf'))
            # batter_completion_attn_masks.append(torch.zeros_like(completion_batter_ids).unsqueeze(0) + float('-inf'))

        # batter_context_attn_masks = torch.concat(batter_context_attn_masks, dim=0)
        # batter_completion_attn_masks = torch.concat(batter_completion_attn_masks, dim=0)
        context_only = completion_data is None

        for player_idx, player_id in enumerate(real_player_ids):
            if player_idx == 0:
                player_context_presence = (context_pitcher_ids == player_id).nonzero()
                if not context_only:
                    player_completion_presence = (completion_pitcher_ids == player_id).nonzero()
            else:
                player_context_presence = (context_batter_ids == player_id).nonzero()
                if not context_only:
                    player_completion_presence = (completion_batter_ids == player_id).nonzero()

            context_player_attn_mask[player_idx, player_context_presence + 10 + 1] = 0
            context_player_attn_mask[player_context_presence + 10 + 1, player_idx] = 0

            if not context_only:
                completion_player_attn_mask[player_idx, player_completion_presence + 10] = 0
                completion_player_attn_mask[player_completion_presence + 10, player_idx] = 0

        context_player_attn_mask = context_player_attn_mask.unsqueeze(0).expand(self.n_attn, -1, -1)
        if not context_only:
            completion_player_attn_mask = completion_player_attn_mask.unsqueeze(0).expand(self.n_attn, -1, -1)
        custom_player_ids = torch.tensor(custom_player_ids)
        real_player_ids = torch.tensor(real_player_ids)

        return custom_player_ids, real_player_ids, context_player_attn_mask, completion_player_attn_mask

    def strip_ball_data(self, x, data_type):
        adjust_shape = False
        if len(x.shape) > 2:
            dim1, dim2, dim3 = x.shape
            adjust_shape = True
            x = x.reshape(dim1 * dim2, dim3)

        # print('*' * 30)
        # print('\t{} x before stripping ball data: {}'.format(data_type, x.shape))
        if data_type == 'batter':
            scope_sizes = self.batter_data_scope_sizes
            ball_data_sizes = self.batter_data_n_ball_data
        elif data_type == 'pitcher':
            scope_sizes = self.pitcher_data_scope_sizes
            ball_data_sizes = self.pitcher_data_n_ball_data
        else:
            scope_sizes = self.matchup_data_scope_sizes
            ball_data_sizes = self.matchup_data_n_ball_data

        # print('\tscope_sizes: {}'.format(scope_sizes))
        # print('\tball_data_sizes: {}'.format(ball_data_sizes))

        x = torch.split(x, scope_sizes, dim=-1)
        new_x = []
        for idx, scope_data in enumerate(x):
            # print('\t\traw scope_data {}: {}'.format(idx, scope_data.shape))
            n_ball_data = ball_data_sizes[idx]
            if self.drop_ball_data:
                scope_data = scope_data[:, :-n_ball_data]
            elif self.mask_ball_data:
                scope_data[:, -n_ball_data:] = 0.0
            # print('\t\tstripped scope_data {}: {}'.format(idx, scope_data.shape))
            new_x.append(scope_data)

        x = torch.cat(new_x, dim=-1)
        # print('\t{} x after strip: {}'.format(data_type, x.shape))
        # print('*' * 30)

        if adjust_shape:
            x = x.reshape(dim1, dim2, -1)

        return x

    def make_supp_inputs_from_j(self, j):
        player_j = j[self.type]
        raw_player_inputs = parse_json(player_j, j_type=self.type, max_value_data=self.record_norm_values)
        player_supp_inputs = []
        if self.type == 'batter':
            data_scopes_to_use = self.batter_data_scopes_to_use
        else:
            data_scopes_to_use = self.pitcher_data_scopes_to_use
        for data_scope in data_scopes_to_use:
            player_supp_inputs.extend(raw_player_inputs[data_scope])

        player_supp_inputs = torch.tensor(player_supp_inputs).unsqueeze(0)
        return player_supp_inputs

    def process_file_set(self, file_set, mode, entity_type):
        tensor_values = ['inning', 'ab_number', 'state_delta_ids',
                         'pitch_types', 'plate_x', 'plate_z',
                         'batter_supp_inputs', 'pitcher_supp_inputs', 'matchup_supp_inputs', 'pitcher_id',
                         'batter_id', 'pitcher_pos_ids', 'batter_pos_ids',  # 'rv_pitch_data',
                         'rv_thrown_pitch_data', 'rv_batted_ball_data',
                         'game_year', 'game_pk', 'stadium_ids', 'pitcher_handedness_ids', 'batter_handedness_ids',
                         'game_year_t', 'game_pk_t', 'state_delta_labels', 'pitch_event_ids',
                         'pitch_event_labels', 'pitch_type_labels']
        entity_data = {'relative_game_no': [], 'relative_ab_no': [], 'relative_pitch_no': [],
                       'ab_lengths': [], 'game_pk': [], 'game_year': []}

        if mode == 'context':
            # supp_init_dim = self.context_supp_init_dim
            max_seq_len = self.context_max_len
            window_size = self.n_games_context
        else:
            # supp_init_dim = self.completion_supp_init_dim
            max_seq_len = self.completion_max_len
            window_size = self.n_games_completion

        max_game_idx = len(file_set) - 1
        for game_idx, game_ab_fps in enumerate(file_set):
            # print('game_ab_fps: {}'.format(game_ab_fps))
            game_ab_len, game_pk, game_year = 0, 0, 0
            if len(game_ab_fps) > 0:
                game_data, game_ab_len, game_pk, game_year = self.process_game_fps(game_ab_fps, mode,
                                                                                   max_game_idx, game_idx,
                                                                                   entity_type=entity_type)
                game_data['relative_game_no'].extend([max_game_idx - game_idx
                                                      for _ in range(game_data['state_delta_ids'].shape[0])])
                for k, v in game_data.items():
                    if k not in ['ab_lengths', 'game_pk', 'game_year']:
                        old_v = entity_data.get(k, None)
                        if old_v is None:
                            if k in tensor_values:
                                new_v = v
                            else:
                                new_v = [v]
                        else:
                            if k in tensor_values:
                                new_v = torch.cat([old_v, v], dim=0)
                            else:
                                new_v = old_v[:]
                                new_v.extend(v)

                        entity_data[k] = new_v

                # game_pk = game_pk if game_pk is not None else 0
                # game_year = game_year if game_year is not None else 0

                entity_data['ab_lengths'].append(game_ab_len)
                entity_data['game_pk'].append(game_pk)
                entity_data['game_year'].append(game_year)

        entity_data['relative_pitch_no'] = torch.tensor(entity_data['relative_pitch_no'], dtype=torch.long)
        entity_data['relative_ab_no'] = torch.tensor(entity_data['relative_ab_no'], dtype=torch.long)
        entity_data['relative_game_no'] = torch.tensor(entity_data['relative_game_no'], dtype=torch.long)
        entity_data['ab_lengths'] = torch.tensor(entity_data.get('ab_lengths', []), dtype=torch.long)
        entity_data['game_pk'] = torch.tensor(entity_data.get('game_pk', []), dtype=torch.long)
        entity_data['game_year'] = torch.tensor(entity_data.get('game_year', []), dtype=torch.long)

        entity_data['game_pk'] = torch.tensor([], dtype=torch.long) if entity_data.get('game_pk',
                                                                                       None) is None else entity_data.get(
            'game_pk', [])
        entity_data['game_pk_t'] = torch.tensor([], dtype=torch.long) if entity_data.get('game_pk_t',
                                                                                         None) is None else entity_data.get(
            'game_pk_t', [])
        entity_data['game_year'] = torch.tensor([], dtype=torch.long) if entity_data.get('game_year',
                                                                                         None) is None else entity_data.get(
            'game_year', [])
        entity_data['game_year_t'] = torch.tensor([], dtype=torch.long) if entity_data.get('game_year_t',
                                                                                           None) is None else entity_data.get(
            'game_year_t', [])
        # entity_data['inning'] = torch.tensor([], dtype=torch.long) if entity_data.get('inning',
        #                                                                               None) is None else entity_data.get(
        #     'inning', [])
        entity_data['stadium_ids'] = torch.tensor([], dtype=torch.long) if entity_data.get('stadium_ids',
                                                                                           None) is None else entity_data.get(
            'stadium_ids', [])
        entity_data['ab_number'] = torch.tensor([], dtype=torch.long) if entity_data.get('ab_number',
                                                                                         None) is None else entity_data.get(
            'ab_number', [])
        entity_data['state_delta_ids'] = torch.tensor([], dtype=torch.long) if entity_data.get('state_delta_ids',
                                                                                               None) is None else entity_data.get(
            'state_delta_ids', [])
        entity_data['state_delta_labels'] = torch.tensor([], dtype=torch.long) if entity_data.get('state_delta_labels',
                                                                                                  None) is None else entity_data.get(
            'state_delta_labels', [])
        entity_data['pitch_type_labels'] = torch.tensor([], dtype=torch.long) if entity_data.get('pitch_type_labels',
                                                                                                 None) is None else entity_data.get(
            'pitch_type_labels', [])
        entity_data['pitch_types'] = torch.tensor([], dtype=torch.long) if entity_data.get('pitch_types',
                                                                                           None) is None else entity_data.get(
            'pitch_types', [])
        # pitch_event_labels
        entity_data['pitch_event_labels'] = torch.tensor([], dtype=torch.long) if entity_data.get('pitch_event_labels',
                                                                                                  None) is None else entity_data.get(
            'pitch_event_labels', [])
        entity_data['pitch_event_ids'] = torch.tensor([], dtype=torch.long) if entity_data.get('pitch_event_ids',
                                                                                               None) is None else entity_data.get(
            'pitch_event_ids', [])
        plate_pos_dtype = torch.float
        # entity_data['plate_x'] = torch.tensor([], dtype=plate_pos_dtype) if entity_data.get('plate_x',
        #                                                                                     None) is None else entity_data.get(
        #     'plate_x', [])
        # entity_data['plate_z'] = torch.tensor([], dtype=plate_pos_dtype) if entity_data.get('plate_z',
        #                                                                                     None) is None else entity_data.get(
        #     'plate_z', [])
        entity_data['batter_supp_inputs'] = torch.tensor(
            [[0.0 for _ in range(self.raw_batter_data_dim)]]) if entity_data.get('batter_supp_inputs',
                                                                                 None) is None else entity_data.get(
            'batter_supp_inputs', [])
        entity_data['pitcher_supp_inputs'] = torch.tensor(
            [[0.0 for _ in range(self.raw_pitcher_data_dim)]]) if entity_data.get('pitcher_supp_inputs',
                                                                                  None) is None else entity_data.get(
            'pitcher_supp_inputs', [])
        entity_data['matchup_supp_inputs'] = torch.tensor(
            [[0.0 for _ in range(self.raw_matchup_data_dim)]]) if entity_data.get('matchup_supp_inputs',
                                                                                  None) is None else entity_data.get(
            'matchup_supp_inputs', [])
        entity_data['pitcher_id'] = torch.tensor([], dtype=torch.long) if entity_data.get('pitcher_id',
                                                                                          None) is None else entity_data.get(
            'pitcher_id', [])
        entity_data['batter_id'] = torch.tensor([], dtype=torch.long) if entity_data.get('batter_id',
                                                                                         None) is None else entity_data.get(
            'batter_id', [])
        entity_data['pitcher_pos_ids'] = torch.tensor([], dtype=torch.long) if entity_data.get('pitcher_pos_ids',
                                                                                               None) is None else entity_data.get(
            'pitcher_pos_ids', [])
        entity_data['batter_pos_ids'] = torch.tensor([], dtype=torch.long) if entity_data.get('batter_pos_ids',
                                                                                              None) is None else entity_data.get(
            'batter_pos_ids', [])
        # entity_data['rv_pitch_data'] = torch.tensor([[0.0 for _ in range(19)]]) if entity_data.get('rv_pitch_data',
        #                                                                                            None) is None else entity_data.get(
        #     'rv_pitch_data', [])
        entity_data['rv_thrown_pitch_data'] = torch.tensor([[0.0 for _ in range(14)]]) if entity_data.get(
            'rv_thrown_pitch_data',
            None) is None else entity_data.get(
            'rv_thrown_pitch_data', [])
        entity_data['rv_batted_ball_data'] = torch.tensor([[0.0 for _ in range(5)]]) if entity_data.get(
            'rv_batted_ball_data',
            None) is None else entity_data.get(
            'rv_batted_ball_data', [])
        entity_data['pitcher_handedness_ids'] = torch.tensor([], dtype=torch.long) if entity_data.get(
            'pitcher_handedness_ids',
            None) is None else entity_data.get(
            'pitcher_handedness_ids', [])
        entity_data['batter_handedness_ids'] = torch.tensor([], dtype=torch.long) if entity_data.get(
            'batter_handedness_ids',
            None) is None else entity_data.get(
            'batter_handedness_ids', [])

        entity_data = self.finalize_entity_data(entity_data, mode)

        """
            Make custom bd attention mask, blocking CLS access to BOAB tokens, if any
        """

        if not self.player_cls:
            state_delta_labels = entity_data['state_delta_labels']
            boab_mask = torch.zeros_like(state_delta_labels, dtype=torch.float)
            boab_mask[state_delta_labels == self.boab_id] = float('-inf')
            premade_attn_masks = torch.zeros((max_seq_len, max_seq_len), dtype=torch.float)

            premade_attn_masks[0] += boab_mask
            premade_attn_masks = premade_attn_masks.unsqueeze(0)
            gsd_attn_masks = einops.repeat(premade_attn_masks,
                                           'b h w -> (repeat b) h w',
                                           repeat=self.gsd_n_attn)
            event_attn_masks = einops.repeat(premade_attn_masks,
                                             'b h w -> (repeat b) h w',
                                             repeat=self.event_n_attn)
            pitch_type_attn_masks = einops.repeat(premade_attn_masks,
                                                  'b h w -> (repeat b) h w',
                                                  repeat=self.pitch_type_n_attn)
            agg_attn_masks = einops.repeat(premade_attn_masks,
                                                  'b h w -> (repeat b) h w',
                                                  repeat=self.n_attn)

            entity_data['gsd_attn_masks'] = gsd_attn_masks
            entity_data['event_attn_masks'] = event_attn_masks
            entity_data['pitch_type_attn_masks'] = pitch_type_attn_masks
            entity_data['agg_attn_masks'] = agg_attn_masks

        return entity_data

    def process_game_fps(self, game_ab_fps, mode, max_game_idx, game_idx, entity_type):
        game_season = os.path.split(game_ab_fps[0])[0]
        game_pk = os.path.split(game_ab_fps[0])[-1].split('-')[0]
        game_ab_nos = [os.path.split(ab_fp)[-1].split('-')[1][:-5] for ab_fp in game_ab_fps]
        # print('game_ab_fps: {}'.format(game_ab_fps))
        # print('game_ab_nos: {}'.format(game_ab_nos))
        if entity_type in 'team_batting':
            data_meta_str = 'pitcher'
        else:
            data_meta_str = entity_type

        game_tensor_fp = os.path.join(self.game_tensor_dir, game_season, '{}_{}_{}.pt'.format(
            game_pk, data_meta_str, '-'.join([str(v) for v in game_ab_nos])
        ))
        if os.path.exists(game_tensor_fp):
            # print('\t\t$ loading {} from disk $'.format(game_tensor_fp))
            try:
                all_game_data = torch.load(game_tensor_fp)

                game_data = all_game_data['game_data']
                game_ab_len = all_game_data['game_ab_len']
                game_pk = all_game_data['game_pk']
                game_year = all_game_data['game_year']
            except Exception as ex:
                all_game_data = self.process_game_fps_(game_ab_fps, mode)
                torch.save(all_game_data, game_tensor_fp)
                game_data = all_game_data['game_data']
                game_ab_len = all_game_data['game_ab_len']
                game_pk = all_game_data['game_pk']
                game_year = all_game_data['game_year']
        else:
            all_game_data = self.process_game_fps_(game_ab_fps, mode)
            torch.save(all_game_data, game_tensor_fp)
            game_data = all_game_data['game_data']
            game_ab_len = all_game_data['game_ab_len']
            game_pk = all_game_data['game_pk']
            game_year = all_game_data['game_year']
            # print('\t\t* saved {} to disk *'.format(game_tensor_fp))

        return game_data, game_ab_len, game_pk, game_year

    def process_game_fps_(self, game_ab_fps, mode):
        tensor_values = ['inning', 'ab_number', 'state_delta_ids',
                         'pitch_types', 'plate_x', 'plate_z',
                         'batter_supp_inputs', 'pitcher_supp_inputs', 'matchup_supp_inputs', 'pitcher_id',
                         'batter_id', 'pitcher_pos_ids', 'batter_pos_ids',  # 'rv_pitch_data',
                         'rv_thrown_pitch_data', 'rv_batted_ball_data',
                         'game_year', 'game_pk', 'stadium_ids', 'pitcher_handedness_ids', 'batter_handedness_ids',
                         'game_year_t', 'game_pk_t', 'state_delta_labels', 'pitch_event_ids',
                         'pitch_event_labels', 'pitch_type_labels']
        game_data = {'relative_game_no': [], 'relative_ab_no': [], 'relative_pitch_no': [],
                     'ab_lengths': [], 'game_pk': [], 'game_year': []}
        game_ab_len = 0
        game_pk = None
        game_year = None

        for ab_idx, ab_fp in enumerate(game_ab_fps):
            ab_data, bos_id = self.parse_item_fp(ab_fp, mode, add_eos=False)

            if self.use_boab_tokens:
                ab_data = self.add_boab_to_ab_data(ab_data, bos_id)

            if game_pk is None:
                game_pk = ab_data['game_pk']
            if game_year is None:
                game_year = ab_data['game_year']

            game_data['relative_pitch_no'].extend([i for i in range(ab_data['ab_lengths'])])
            game_data['relative_ab_no'].extend([ab_idx for _ in range(ab_data['ab_lengths'])])
            # game_data['relative_game_no'].extend([max_game_idx - game_idx for _ in range(ab_data['ab_lengths'])])
            game_ab_len += ab_data['ab_lengths']

            for k, v in ab_data.items():
                if k not in ['ab_lengths', 'game_pk', 'game_year']:
                    old_v = game_data.get(k, None)
                    if old_v is None:
                        if k in tensor_values:
                            new_v = v
                        else:
                            new_v = [v]
                    else:
                        if k in tensor_values:
                            new_v = torch.cat([old_v, v], dim=0)
                        else:
                            new_v = old_v[:]
                            new_v.append(v)

                    game_data[k] = new_v

        all_game_data = {
            'game_data': game_data,
            'game_ab_len': game_ab_len,
            'game_pk': game_pk,
            'game_year': game_year,
        }
        return all_game_data

    def finalize_entity_data(self, entity_data, mode):
        pad_to_window_size = ['ab_lengths', 'game_pk', 'game_year']
        if mode == 'context':
            max_seq_len = self.context_max_len
            window_size = self.n_games_context
        else:
            max_seq_len = self.completion_max_len
            window_size = self.n_games_completion
        seq_n_pad = None

        for k in entity_data.keys():
            v = entity_data[k]
            if type(v) == torch.Tensor:
                if k in pad_to_window_size:
                    n_pad = window_size - v.shape[0]
                    v_pad = torch.zeros(n_pad, dtype=v.dtype)
                    v = torch.cat([v, v_pad], dim=0)
                else:
                    if v.shape[0] > max_seq_len:
                        v = v[-max_seq_len:]
                    elif v.shape[0] < max_seq_len:
                        n_pad = max_seq_len - v.shape[0]

                        if seq_n_pad is None:
                            seq_n_pad = n_pad

                        if len(v.shape) == 1:
                            v_pad = torch.zeros(n_pad, dtype=v.dtype)
                        else:
                            v_pad = torch.zeros(n_pad, v.shape[1], dtype=v.dtype)

                        v = torch.cat([v, v_pad], dim=0)
                entity_data[k] = v

        if seq_n_pad is None:
            my_pad_mask = torch.ones(max_seq_len, dtype=torch.float)
            model_pad_mask = torch.zeros(max_seq_len, dtype=torch.bool)
        else:
            n_present = max_seq_len - seq_n_pad
            my_pad_mask = torch.cat([torch.ones(n_present, dtype=torch.float),
                                     torch.zeros(seq_n_pad, dtype=torch.float)], dim=0)
            model_pad_mask = torch.cat([torch.zeros(n_present, dtype=torch.bool),
                                        torch.ones(seq_n_pad, dtype=torch.bool)], dim=0)

        entity_data['my_src_pad_mask'] = my_pad_mask
        entity_data['model_src_pad_mask'] = model_pad_mask

        return entity_data

    def add_boab_to_ab_data(self, ab_data, bos_id):
        keys_to_adjust = [
            'state_delta_ids', 'pitch_types', 'pitch_type_labels', 'pitch_event_ids', 'pitch_event_labels',
            'batter_supp_inputs', 'pitcher_supp_inputs', 'matchup_supp_inputs', 'pitcher_id', 'batter_id',
            # 'pitcher_pos_ids', 'batter_pos_ids', 'inning',
            'stadium_ids',
            'pitcher_handedness_ids', 'batter_handedness_ids',  # 'game_pk_t',
            # 'game_year_t',  'rv_pitch_data',
            'rv_thrown_pitch_data',
            'rv_batted_ball_data', 'state_delta_labels'
        ]

        for kta in keys_to_adjust:
            ab_data[kta] = torch.cat(
                (ab_data[kta][0].unsqueeze(0), ab_data[kta])
            )

        ab_data['state_delta_ids'][0] = bos_id + self.gsd_vocab_size
        ab_data['state_delta_labels'][0] = self.boab_id
        ab_data['pitch_types'][0] = 0
        ab_data['pitch_type_labels'][0] = 0
        ab_data['pitch_event_ids'][0] = 0
        ab_data['pitch_event_labels'][0] = 0
        # ab_data['rv_pitch_data'][0, :] = 0.0
        ab_data['rv_thrown_pitch_data'][0, :] = 0.0
        ab_data['rv_batted_ball_data'][0, :] = 0.0

        return ab_data

    def parse_item_fp(self, item_fp, mode, add_eos=False, cast_tensor=True):
        if self.ab_data_cache is not None:
            item_intermediate_data = self.ab_data_cache[item_fp[:-5]]
            print('GOT INTERMEDIATE DATA FROM CACHE')
        elif self.use_intermediate_data:
            intermediate_data_fp = os.path.join(self.intermediate_data_dir, '{}.pt'.format(item_fp[:-5]))
            if os.path.exists(intermediate_data_fp):
                try:
                    item_intermediate_data = torch.load(intermediate_data_fp)
                except Exception as ex:
                    item_intermediate_data = self.parse_item_intermediate(item_fp)
                    torch.save(item_intermediate_data, intermediate_data_fp)
            else:
                item_intermediate_data = self.parse_item_intermediate(item_fp)
                torch.save(item_intermediate_data, intermediate_data_fp)
        else:
            item_intermediate_data = self.parse_item_intermediate(item_fp)

        game_pk = item_intermediate_data['game_pk']
        game_year = item_intermediate_data['game_year']
        ab_number = item_intermediate_data['ab_number']
        inning_no = item_intermediate_data['inning_no']
        pitcher_id = item_intermediate_data['pitcher_id']
        batter_id = item_intermediate_data['batter_id']

        bos_id = item_intermediate_data['bos_id']
        pitch_event_ids = item_intermediate_data['pitch_event_ids']
        if self.reduced_event_map:
            potential_print_strs = ['pitch_event_ids: {}'.format(pitch_event_ids)]
            raw_pitch_event_names = [self.reverse_raw_pitch_event_map[pe] for pe in pitch_event_ids]
            potential_print_strs.append('raw_pitch_event_names: {}'.format(raw_pitch_event_names))
            new_event_names = [self.intermediate_pitch_event_map.get(rpen, rpen) for rpen in raw_pitch_event_names]
            potential_print_strs.append('new_event_names: {}'.format(new_event_names))
            pitch_event_ids = [self.pitch_event_map[nen] for nen in new_event_names]
            potential_print_strs.append('new pitch_event_ids: {}'.format(pitch_event_ids))
            if max(pitch_event_ids) > 22:
                print('\n'.join(potential_print_strs))
            # print_str = '{}\nraw names: {}\nnew names: {}\nnew ids: {}\n{}'.format(
            #     '*' * 40,
            #     raw_pitch_event_names,
            #     new_event_names,
            #     pitch_event_ids,
            #     '*' * 40
            # )
            # print(print_str)

        pitch_event_ids = [pei + 2 for pei in pitch_event_ids]
        pitch_event_labels = [pei for pei in pitch_event_ids]

        stadium_id = item_intermediate_data['stadium_id']
        # state_delta_ids = item_intermediate_data['state_delta_ids']
        state_deltas = item_intermediate_data['state_deltas']
        pitch_types = item_intermediate_data['pitch_types']
        pitcher_pos_id = item_intermediate_data['pitcher_pos_id']
        batter_pos_id = item_intermediate_data['batter_pos_id']
        # rv_pitch_data = item_intermediate_data['rv_pitch_data']
        rv_thrown_pitch_data = item_intermediate_data['rv_thrown_pitch_data']
        rv_batted_ball_data = item_intermediate_data['rv_batted_ball_data']
        # print('state_delta_ids: {}'.format(state_delta_ids))

        intermediate_pitcher_inputs = item_intermediate_data['raw_pitcher_inputs']
        pitcher_handedness_str = item_intermediate_data['raw_pitcher_inputs']['handedness'].upper()
        pitcher_handedness_id = self.handedness_id_map[pitcher_handedness_str] + 1
        intermediate_batter_inputs = item_intermediate_data['raw_batter_inputs']
        batter_handedness_str = item_intermediate_data['raw_batter_inputs']['handedness'].upper()
        batter_handedness_id = self.handedness_id_map[batter_handedness_str] + 1
        intermediate_matchup_inputs = item_intermediate_data['raw_matchup_inputs']

        batter_supp_inputs = []
        pitcher_supp_inputs = []
        matchup_supp_inputs = []
        for data_scope in self.batter_data_scopes_to_use:
            batter_supp_inputs.extend(intermediate_batter_inputs[data_scope])
        for data_scope in self.pitcher_data_scopes_to_use:
            pitcher_supp_inputs.extend(intermediate_pitcher_inputs[data_scope])
        for data_scope in self.matchup_data_scopes_to_use:
            matchup_supp_inputs.extend(intermediate_matchup_inputs[data_scope])
        # print('len(pitcher_supp_inputs): {}'.format(len(pitcher_supp_inputs)))
        if not self.vocab_use_swing_status:
            for idx in range(len(state_deltas)):
                if state_deltas[idx]['strikes'] == '+1\'':
                    state_deltas[idx]['strikes'] = '+1'
                    # print('!! Reverting swinging strike !!')
        state_delta_ids = [self.gamestate_vocab.get_id(sd) for sd in state_deltas]

        pitch_types = [pt if pt is not None else 'NONE' for pt in pitch_types]
        pitch_types = [self.pitch_type_mapping[pt] + 2 for pt in pitch_types]
        pitch_type_labels = [lbl for lbl in pitch_types]
        batter_supp_inputs = [batter_supp_inputs for _ in state_delta_ids]
        pitcher_supp_inputs = [pitcher_supp_inputs for _ in state_delta_ids]
        matchup_supp_inputs = [matchup_supp_inputs for _ in state_delta_ids]
        pitcher_id = [pitcher_id for _ in state_delta_ids]
        batter_id = [batter_id for _ in state_delta_ids]
        pitcher_pos_ids = [pitcher_pos_id for _ in state_delta_ids]
        batter_pos_ids = [batter_pos_id for _ in state_delta_ids]
        ab_number_ids = [ab_number for _ in state_delta_ids]
        inning = [inning_no for _ in state_delta_ids]
        stadium_id = [stadium_id for _ in state_delta_ids]
        pitcher_handedness_id = [pitcher_handedness_id for _ in state_delta_ids]
        batter_handedness_id = [batter_handedness_id for _ in state_delta_ids]
        game_pk_t = [game_pk for _ in state_delta_ids]
        game_year_t = [game_year for _ in state_delta_ids]

        state_delta_labels = [sdid if sdid < self.gsd_vocab_size else self.boab_id for sdid in state_delta_ids]

        if cast_tensor:
            pitch_types = torch.tensor(pitch_types)
            pitch_type_labels = torch.tensor(pitch_type_labels)
            pitch_event_ids = torch.tensor(pitch_event_ids)
            pitch_event_labels = torch.tensor(pitch_event_labels)
            batter_supp_inputs = torch.tensor(batter_supp_inputs)
            pitcher_supp_inputs = torch.tensor(pitcher_supp_inputs)
            matchup_supp_inputs = torch.tensor(matchup_supp_inputs)
            pitcher_id = torch.tensor(pitcher_id)
            batter_id = torch.tensor(batter_id)
            pitcher_pos_ids = torch.tensor(pitcher_pos_ids)
            batter_pos_ids = torch.tensor(batter_pos_ids)
            ab_number = torch.tensor(ab_number_ids)

            inning = torch.tensor(inning)
            stadium_id = torch.tensor(stadium_id)
            pitcher_handedness_id = torch.tensor(pitcher_handedness_id)
            batter_handedness_id = torch.tensor(batter_handedness_id)
            # rv_pitch_data = torch.tensor(rv_pitch_data, dtype=torch.float)
            rv_thrown_pitch_data = torch.tensor(rv_thrown_pitch_data, dtype=torch.float)
            rv_batted_ball_data = torch.tensor(rv_batted_ball_data, dtype=torch.float)
            state_delta_ids = torch.tensor(state_delta_ids)

            ab_lengths = state_delta_ids.shape[0]
            game_pk = torch.tensor([game_pk])
            game_year = torch.tensor([game_year])
            game_pk_t = torch.tensor(game_pk_t)
            game_year_t = torch.tensor(game_year_t)
            state_delta_labels = torch.tensor(state_delta_labels)
        else:
            ab_lengths = len(state_delta_ids)

        # print('pitcher_supp_inputs: {}'.format(pitcher_supp_inputs.shape))

        if -1 in state_delta_ids:
            print('*' * 50)
            print(os.path.join(self.ab_data_dir, item_fp))
            print('state_delta_ids: {}'.format(state_delta_ids))
            print('state_deltas: {}'.format(state_deltas))

            print('*' * 50)

        item_data = {
            'game_pk': game_pk,
            'game_pk_t': game_pk_t,
            'game_year': game_year,
            'game_year_t': game_year_t,
            # 'inning': inning,
            'stadium_ids': stadium_id,
            'pitcher_handedness_ids': pitcher_handedness_id,
            'batter_handedness_ids': batter_handedness_id,
            'ab_number': ab_number,
            'state_delta_ids': state_delta_ids,
            'ab_lengths': ab_lengths,
            'pitch_types': pitch_types,
            'pitch_type_labels': pitch_type_labels,
            'pitch_event_ids': pitch_event_ids,
            'pitch_event_labels': pitch_event_labels,
            'batter_supp_inputs': batter_supp_inputs,
            'pitcher_supp_inputs': pitcher_supp_inputs,
            'matchup_supp_inputs': matchup_supp_inputs,
            'pitcher_id': pitcher_id,
            'batter_id': batter_id,
            # 'pitcher_pos_ids': pitcher_pos_ids,
            # 'batter_pos_ids': batter_pos_ids,
            'rv_thrown_pitch_data': rv_thrown_pitch_data,
            'rv_batted_ball_data': rv_batted_ball_data,
            'state_delta_labels': state_delta_labels
        }
        return item_data, bos_id

    def parse_item_intermediate(self, item_fp):
        try:
            item_j = json.load(open(os.path.join(self.ab_data_dir, item_fp)))
        except Exception as ex:
            print('!!! Error parsing {} !!!'.format(os.path.join(self.ab_data_dir, item_fp)))
            print('ex: {}'.format(ex))

        game_pk = item_j['game']['game_pk']
        game_year = item_j['game']['game_year']
        ab_number = item_j['game']['at_bat_number']
        inning_no = item_j['game']['inning']
        inning_topbot = item_j['game']['inning_topbot']
        outs_when_up = item_j['game']['outs_when_up']
        next_game_state = item_j['game']['next_game_state']
        pitcher_j = item_j['pitcher']
        pitcher_id = item_j['pitcher']['__id__']
        batter_j = item_j['batter']
        batter_id = item_j['batter']['__id__']
        matchup_j = item_j['matchup']

        home_team_str = item_j['game']['home_team']
        # stadium_id = int(self.team_stadium_map[home_team_str][str(game_year)]) - 1
        stadium_id = -1

        # pitcher_pos_str = self.player_bio_info_mapping[item_j['pitcher']['__id__']][self.player_pos_key]
        # batter_pos_str = self.player_bio_info_mapping[item_j['batter']['__id__']][self.player_pos_key]
        # if str(pitcher_pos_str) == 'nan':
        #     pitcher_pos_str = 'UNK'
        # if str(batter_pos_str) == 'nan':
        #     batter_pos_str = 'UNK'

        pitcher_pos_str = 'UNK'
        batter_pos_str = 'UNK'

        on_1b = 0 if item_j['game']['on_1b'] in [None, 0] else 1
        on_2b = 0 if item_j['game']['on_2b'] in [None, 0] else 1
        on_3b = 0 if item_j['game']['on_3b'] in [None, 0] else 1
        next_game_state['on_1b'] = 0 if next_game_state['on_1b'] is None else 1
        next_game_state['on_2b'] = 0 if next_game_state['on_2b'] is None else 1
        next_game_state['on_3b'] = 0 if next_game_state['on_3b'] is None else 1
        pitcher_pos_id = self.player_pos_id_map[pitcher_pos_str]
        batter_pos_id = self.player_pos_id_map[batter_pos_str]
        pitcher_score = item_j['game']['fld_score']
        batter_score = item_j['game']['bat_score']

        n_balls, n_strikes, pitch_types, rv_pitch_data, \
        pitch_events, pitch_descriptions = parse_pitches_updated(item_j['pitches'],
                                                                 norm_vals=self.record_norm_values['pitches'])
        last_gamestate = {'balls': 0, 'strikes': 0, 'outs': outs_when_up, 'score': batter_score,
                          'on_1b': on_1b, 'on_2b': on_2b, 'on_3b': on_3b}

        state_deltas, _ = find_state_deltas_v2(n_balls, n_strikes, outs_when_up, int(inning_no), inning_topbot,
                                               next_game_state, batter_score, on_1b, on_2b, on_3b,
                                               last_state=last_gamestate,
                                               pitch_description=pitch_descriptions,
                                               use_swing_status=self.vocab_use_swing_status)
        # print('state_deltas:\n{}\n\t{}'.format(state_deltas, len(state_deltas)))
        # print('len(pitch_descriptions): {}'.format(len(pitch_descriptions)))
        state_deltas = state_deltas[1:]
        if self.vocab_use_swing_status:
            for idx in range(len(state_deltas)):
                if state_deltas[idx]['strikes'] == '+1' and self.description_to_swing_status.get(
                        pitch_descriptions[idx], 0) == 1:
                    state_deltas[idx]['strikes'] = '+1\''
                    # print('!! Found swinging strike !!')

        bos_id = self.gamestate_vocab.get_id({'on_1b': on_1b, 'on_2b': on_2b, 'on_3b': on_3b}, bos=True)
        state_delta_ids = [self.gamestate_vocab.get_id(sd) for sd in state_deltas]
        pitch_event_ids = [self.raw_pitch_event_map[pe] for pe in pitch_events]
        # print('state_delta_ids: {}'.format(state_delta_ids))

        raw_pitcher_inputs = parse_json(pitcher_j, j_type='pitcher', max_value_data=self.record_norm_values)
        raw_batter_inputs = parse_json(batter_j, j_type='batter', max_value_data=self.record_norm_values)
        raw_matchup_inputs = parse_json(matchup_j, j_type='matchup', max_value_data=self.record_norm_values)

        # rv_inputs = [pitch_speed, release_x, release_y, release_z, spin_rate, extension, hc_x, hc_y, vx0, vy0, vz0,
        #                  ax, ay, az, hit_dist, launch_speed, launch_angle, plate_x, plate_z]
        rv_thrown_pitch_data = [
            [rvpd[0], rvpd[1], rvpd[2], rvpd[3], rvpd[4], rvpd[5], rvpd[8], rvpd[9], rvpd[10],
             rvpd[11], rvpd[12], rvpd[13], rvpd[17], rvpd[18]] for rvpd in rv_pitch_data
        ]
        rv_batted_ball_data = [
            [rvpd[6], rvpd[7], rvpd[14], rvpd[15], rvpd[16]] for rvpd in rv_pitch_data
        ]

        intermediate_data = {
            'game_pk': game_pk,
            'game_year': game_year,
            'ab_number': ab_number,
            'inning_no': inning_no,
            'pitcher_id': pitcher_id,
            'batter_id': batter_id,
            'stadium_id': stadium_id,
            'pitcher_pos_id': pitcher_pos_id,
            'batter_pos_id': batter_pos_id,
            'state_delta_ids': state_delta_ids,
            'state_deltas': state_deltas,
            'raw_pitcher_inputs': raw_pitcher_inputs,
            'raw_batter_inputs': raw_batter_inputs,
            'raw_matchup_inputs': raw_matchup_inputs,
            'pitch_types': pitch_types,
            'rv_thrown_pitch_data': rv_thrown_pitch_data,
            'rv_batted_ball_data': rv_batted_ball_data,
            'bos_id': bos_id,
            'pitch_event_ids': pitch_event_ids,
            'pitch_descriptions': pitch_descriptions,
        }

        return intermediate_data

    def process_file_set_for_finetune(self, file_set, reqd_pitcher_id=None, return_matchup_stats=False,
                                      return_hit_label=False):
        # output = {self.finetune_target: 0}
        targets = {t: 0 for t in self.targets if t != 'ptb'}
        ptb_d = {'h': 0, 'hr': 0, 'bb': 0, 'hbp': 0, 'ibb': 0}
        # print('targets: {}'.format(targets))
        has_hit = 0
        # print('file_set for finetune: {}'.format(file_set))
        for game_idx, game_ab_fps in enumerate(file_set):
            for ab_idx, ab_fp in enumerate(game_ab_fps):
                # open(os.path.join(self.ab_data_dir, ab_fp))
                # j = json.load(open(ab_fp))
                j = json.load(open(os.path.join(self.ab_data_dir, ab_fp)))
                if reqd_pitcher_id is None or j['pitcher']['__id__'] == reqd_pitcher_id:
                    # print('$$ Found valid file where reqd_pitcher_id = {} $$'.format(reqd_pitcher_id))
                    last_event = j['pitches'][-1]['events']
                    last_description = j['pitches'][-1]['description']

                    for target, target_mapping in self.finetune_target_map.items():
                        last_event_target_val = target_mapping.get(last_event, 0)
                        targets[target] += last_event_target_val

                    if last_event == 'home_run':
                        ptb_d['hr'] += 1
                        ptb_d['h'] += 1
                        has_hit = 1
                    elif last_event in ['single', 'double', 'triple']:
                        ptb_d['h'] += 1
                        has_hit = 1
                    elif last_event == 'intent_walk':
                        ptb_d['ibb'] += 1
                    elif last_event == 'walk' and last_description == 'hit_by_pitch':
                        ptb_d['hbp'] += 1
                    elif last_event == 'walk':
                        ptb_d['bb'] += 1

                    if last_event == 'intent_walk':
                        last_event = 'walk'

                    # if last_event in ['home_run', 'single', 'double', 'triple', 'walk']:
                    if targets.get(last_event, None) is not None:
                        targets[last_event] = targets.get(last_event, 0) + 1

                    if targets.get('r', None) is not None:
                        outs_when_up = j['game']['outs_when_up']
                        inning_topbot = j['game']['inning_topbot']
                        inning_no = j['game']['inning']
                        next_game_state = j['game']['next_game_state']
                        on_1b = 0 if j['game']['on_1b'] in [None, 0] else 1
                        on_2b = 0 if j['game']['on_2b'] in [None, 0] else 1
                        on_3b = 0 if j['game']['on_3b'] in [None, 0] else 1
                        next_game_state['on_1b'] = 0 if next_game_state['on_1b'] is None else 1
                        next_game_state['on_2b'] = 0 if next_game_state['on_2b'] is None else 1
                        next_game_state['on_3b'] = 0 if next_game_state['on_3b'] is None else 1
                        batter_score = j['game']['bat_score']

                        n_balls, n_strikes, pitch_types, rv_pitch_data, pitch_events, pitch_descriptions = parse_pitches_updated(
                            j['pitches'],
                            norm_vals=self.record_norm_values['pitches'])
                        last_gamestate = {'balls': 0, 'strikes': 0, 'outs': outs_when_up, 'score': batter_score,
                                          'on_1b': on_1b, 'on_2b': on_2b, 'on_3b': on_3b}
                        state_deltas, _ = find_state_deltas_v2(n_balls, n_strikes, outs_when_up, int(inning_no),
                                                               inning_topbot,
                                                               next_game_state, batter_score, on_1b, on_2b, on_3b,
                                                               last_state=last_gamestate,
                                                               pitch_description=pitch_descriptions,
                                                               use_swing_status=self.vocab_use_swing_status)
                        state_deltas = state_deltas[1:]
                        if self.vocab_use_swing_status:
                            for idx in range(len(state_deltas)):
                                if state_deltas[idx]['strikes'] == '+1' and self.description_to_swing_status.get(
                                        pitch_descriptions[idx], 0) == 1:
                                    state_deltas[idx]['strikes'] = '+1\''
                        # print('state_deltas: {}'.format(state_deltas))
                        runs_scored = sum(float(sd['score']) for sd in state_deltas)
                        targets['r'] += runs_scored

                    if targets.get('barrel', None) is not None:
                        last_exit_vel = j['pitches'][-1]['launch_speed']
                        last_launch_angle = j['pitches'][-1]['launch_angle']

                        last_exit_vel = last_exit_vel if last_exit_vel is not None else 0.0
                        last_launch_angle = last_launch_angle if last_launch_angle is not None else 0.0
                        if is_barrel(last_exit_vel, last_launch_angle):
                            targets['barrel'] += 1

                    if targets.get('quab', None) is not None:
                        if is_quab(j['pitches']):
                            targets['quab'] += 1
                else:
                    break

        matchup_data = None
        if return_matchup_stats:
            item_fp = file_set[0][0]
            if self.use_intermediate_data:
                intermediate_data_fp = os.path.join(self.intermediate_data_dir, '{}.pt'.format(item_fp[:-5]))
                # print('intermediate_data_fp: {}'.format(intermediate_data_fp))
                if os.path.exists(intermediate_data_fp):
                    # print('** Loading intermediate data from disk **')
                    try:
                        item_intermediate_data = torch.load(intermediate_data_fp)
                        # print(' \tSuccess')
                    except Exception as ex:
                        # print(' \tNo Success... making data...')
                        item_intermediate_data = self.parse_item_intermediate(item_fp)
                        torch.save(item_intermediate_data, intermediate_data_fp)
                        # print(' \tSaved it')
                else:
                    # print('** Creating intermediate data and saving **')
                    item_intermediate_data = self.parse_item_intermediate(item_fp)
                    torch.save(item_intermediate_data, intermediate_data_fp)
                # input('okty!')
            else:

                # item_j = json.load(open(os.path.join(self.ab_data_dir, item_fp)))
                item_intermediate_data = self.parse_item_intermediate(item_fp)

            matchup_data = []
            for data_scope in ['career', 'season']:
                matchup_data.extend(item_intermediate_data['raw_matchup_inputs'][data_scope])
            matchup_data = torch.tensor(matchup_data)
            # print('matchup_data.shape: {}'.format(matchup_data.shape))
        ptb = calc_ptb(ptb_d)
        # print('ptb_d: {} ptb: {}'.format(ptb_d, ptb))
        output = {'ptb': torch.tensor(ptb, dtype=torch.float)}

        if matchup_data is not None:
            output['matchup_data'] = matchup_data
        # output = {self.finetune_target: torch.tensor([target], dtype=torch.float)}
        for k, v in targets.items():
            output[k] = torch.tensor(v)
            # print('k: {} v: {}'.format(k, output[k]))

        # print('output[ptb].dtype: {}'.format(output['ptb'].dtype))
        # print('output: {}'.format(output))

        if return_hit_label:
            return output, torch.tensor([has_hit])
        else:
            return output
