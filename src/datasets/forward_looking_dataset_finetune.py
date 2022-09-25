__author__ = 'Connor Heaton'

import os
import json
import torch

from torch.utils.data import Dataset
from .dataset_utils import read_file_lines


class FLDatasetFinetune(Dataset):
    def __init__(self, args, mode, pitcher_dataset, batter_dataset, pitcher_args, batter_args,
                 game_pk_to_date_d=None, team_str_to_id_d=None,
                 projection_mode=False):
        self.args = args
        self.mode = mode if mode in ['train', 'test', 'apply', 'identify'] else 'dev'
        self.pitcher_dataset = pitcher_dataset
        self.batter_dataset = batter_dataset
        self.pitcher_args = pitcher_args
        self.batter_args = batter_args
        self.game_pk_to_date_d = game_pk_to_date_d
        self.team_str_to_id_d = team_str_to_id_d
        self.projection_mode = projection_mode

        self.use_ball_data = getattr(self.args, 'use_ball_data', True)
        self.use_matchup_data = getattr(self.args, 'use_matchup_data', False)
        self.pitcher_targets = getattr(self.args, 'pitcher_targets', ['k', 'h'])
        self.batter_targets = getattr(self.args, 'batter_targets', ['k', 'h'])
        self.whole_game_record_dir = getattr(self.args, 'whole_game_record_dir',
                                             '/home/czh/sata1/SportsAnalytics/whole_game_records')
        self.whole_game_record_dir = os.path.join(self.whole_game_record_dir, 'by_season')
        self.single_model = getattr(self.args, 'single_model', False)
        self.entity_models = getattr(self.args, 'entity_models', False)
        team_stadiums_fp = getattr(self.args, 'team_stadiums_fp',
                                   '/home/czh/nvme1/SportsAnalytics/config/team_stadiums.json')
        self.team_stadium_map = json.load(open(team_stadiums_fp))
        self.pitcher_n_context = self.args.pitcher_n_context
        self.pitcher_n_completion = self.args.pitcher_n_completion
        self.batter_n_context = self.args.batter_n_context
        self.batter_n_completion = self.args.batter_n_completion
        self.single_pitcher_batter_completion = getattr(self.args, 'single_pitcher_batter_completion', False)
        self.only_starting_pitchers = getattr(self.args, 'only_starting_pitchers', False)
        self.career_data_dir = getattr(self.args, 'career_data',
                                       '/home/czh/sata1/SportsAnalytics/player_career_data')
        self.bad_data_fps = getattr(self.args, 'bad_data_fps', [])
        pitcher_avg_entry_fp = getattr(self.args, 'pitcher_avg_entry_fp',
                                       '/home/czh/nvme1/SportsAnalytics/data/pitcher_avg_inning_entry.json')
        self.pitcher_avg_entry_j = json.load(open(pitcher_avg_entry_fp))

        self.use_swing_status = getattr(self.args, 'use_swing_status', False)
        self.use_intermediate_whole_game_data = getattr(self.args, 'use_intermediate_whole_game_data', False)
        self.intermediate_whole_game_data_dir = getattr(self.args, 'intermediate_whole_game_data_dir',
                                                        '/home/czh/md0/intermediate_whole_game_data/')
        print('FLDatasetFinetune - use_intermediate_data: {}\n\tintermediate_data_dir: {}'.format(
            self.use_intermediate_whole_game_data, self.intermediate_whole_game_data_dir
        ))
        if self.use_intermediate_whole_game_data and not os.path.exists(self.intermediate_whole_game_data_dir):
            os.makedirs(self.intermediate_whole_game_data_dir)

        splits_dir = getattr(self.args, 'splits_dir',
                             '/home/czh/sata1/SportsAnalytics/whole_game_records/ar_game_splits')
        if self.mode == 'train':
            splits_fp = os.path.join(splits_dir, 'train.txt')
        elif self.mode == 'test':
            splits_fp = os.path.join(splits_dir, 'test.txt')
        else:
            splits_fp = os.path.join(splits_dir, 'dev.txt')

        self.split_items = read_file_lines(splits_fp)
        if self.use_ball_data:
            self.split_items = [si for si in self.split_items if int(si[:4]) >= 2015]

        print('self.mode: {} self.split_items[:25]: {}'.format(self.mode, self.split_items[:25]))
        self.items = self.load_data()
        print('Loading pitcher data...')
        self.pitcher_data = self.load_player_data(player_type='pitcher')
        print('Loading batter data...')
        self.batter_data = self.load_player_data(player_type='batter')
        print('Loading team_batting data...')
        self.team_batting_data = self.load_player_data(player_type='team_batting')

        print('FLDatasetFinetune.only_starting_pitchers: {}'.format(self.only_starting_pitchers))

    def load_data(self):
        items = []
        for split_item in self.split_items:
            items.append(split_item)
            items.append(split_item)

        return items

    def load_player_data(self, player_type):
        player_data = {}

        player_data_dir = os.path.join(self.career_data_dir, player_type)
        for player_fp in os.listdir(player_data_dir):
            if player_type == 'team_batting':
                player_id = player_fp[:-4]
            else:
                player_id = int(player_fp[:-4])
            full_player_fp = os.path.join(player_data_dir, player_fp)
            player_game_pks = []
            fps_by_game = []
            last_game_pk = None

            valid_player = True
            if self.only_starting_pitchers and self.pitcher_avg_entry_j.get(str(player_id), 9) > 1.5:
                valid_player = False

            if valid_player:
                with open(full_player_fp, 'r') as f:
                    for line in f:
                        line = line.strip()
                        # if (self.mode == 'train' and not line.startswith('2021')) or (self.mode != 'train' \
                        #         and line.startswith('2021')) and line not in self.bad_data_fps:
                        if line not in self.bad_data_fps:  # and not line.startswith('2021')
                            game_pk = line.split('/')[-1].split('-')[0]

                            if last_game_pk != game_pk:
                                player_game_pks.append(game_pk)
                                fps_by_game.append([])

                            fps_by_game[-1].append(line)
                            last_game_pk = game_pk

                    # player_game_pks.append(-1)
                    # fps_by_game.append([])

                player_data[player_id] = {'game_pks': player_game_pks, 'fps_by_game': fps_by_game}

        return player_data

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        game_fp = self.items[idx]

        if idx % 2 == 0:
            topbot = 'bot'
        else:
            topbot = 'top'

        comb_data = self.load_game_data(game_fp, topbot)
        return comb_data

    def find_player_files(self, game_pk, player_meta_d, player_type):
        if player_type == 'pitcher':
            n_context = self.pitcher_n_context
            n_completion = self.pitcher_n_completion
        else:
            n_context = self.batter_n_context
            n_completion = self.batter_n_completion

        if game_pk == -1:
            game_idx = len(player_meta_d.get('game_pks', []))
        else:
            game_idx = player_meta_d['game_pks'].index(game_pk)

        player_files = player_meta_d.get('fps_by_game', [])

        context_files = player_files[(game_idx - n_context):game_idx]
        completion_files = player_files[game_idx:(game_idx + n_completion)]

        return context_files, completion_files

    def load_game_data(self, game_fp, topbot):
        game_pk = game_fp.split('/')[-1][:-5]
        intermediate_data_fp = os.path.join(
            self.intermediate_whole_game_data_dir, '{}-{}_{}.pt'.format(
                game_pk, topbot,
                'w_swing' if self.use_swing_status else 'wo_swing'
            )
        )
        if os.path.exists(intermediate_data_fp) and self.use_intermediate_whole_game_data:
            intermediate_data = torch.load(intermediate_data_fp)
        else:
            intermediate_data = self.parse_whole_game_intermedate(game_fp, topbot)
            if self.use_intermediate_whole_game_data:
                torch.save(intermediate_data, intermediate_data_fp)

        pitcher_inputs = intermediate_data['pitcher_inputs']
        pitcher_completion_files = intermediate_data['pitcher_completion_files']
        batter_inputs = intermediate_data['batter_inputs']
        batter_completion_files = intermediate_data['batter_completion_files']
        pitcher_id = intermediate_data['pitcher_id']
        batter_ids = intermediate_data['batter_ids']
        game_pk = intermediate_data['game_pk']
        # print('pitcher_inputs.keys(): {}'.format(pitcher_inputs.keys()))
        # print('batter_inputs.keys(): {}'.format(batter_inputs.keys()))
        if not self.pitcher_dataset.use_ball_data:
            # print("raw pitcher_inputs['context_pitcher_supp_inputs']: {}".format(
            #     pitcher_inputs['context_pitcher_supp_inputs'].shape))
            pitcher_inputs['context_pitcher_supp_inputs'] = self.pitcher_dataset.strip_ball_data(
                pitcher_inputs['context_pitcher_supp_inputs'], data_type='pitcher'
            )
            pitcher_inputs['context_batter_supp_inputs'] = self.pitcher_dataset.strip_ball_data(
                pitcher_inputs['context_batter_supp_inputs'], data_type='batter'
            )
            pitcher_inputs['context_matchup_supp_inputs'] = self.pitcher_dataset.strip_ball_data(
                pitcher_inputs['context_matchup_supp_inputs'], data_type='matchup'
            )

            pitcher_inputs['context_rv_thrown_pitch_data'] = torch.rand_like(
                pitcher_inputs['context_rv_thrown_pitch_data'])
            pitcher_inputs['context_rv_batted_ball_data'] = torch.rand_like(
                pitcher_inputs['context_rv_batted_ball_data'])
            pitcher_inputs['context_pitch_types'] = torch.zeros_like(pitcher_inputs['context_pitch_types'])

            # del pitcher_inputs['context_rv_thrown_pitch_data']
            # del pitcher_inputs['context_rv_batted_ball_data']
            # del pitcher_inputs['context_pitch_types']
            # print("new pitcher_inputs['context_pitcher_supp_inputs']: {}".format(
            #     pitcher_inputs['context_pitcher_supp_inputs'].shape))
        if not self.batter_dataset.use_ball_data:
            # print("raw batter_inputs['context_pitcher_supp_inputs']: {}".format(
            #     batter_inputs['context_pitcher_supp_inputs'].shape))
            batter_inputs['context_pitcher_supp_inputs'] = self.batter_dataset.strip_ball_data(
                batter_inputs['context_pitcher_supp_inputs'], data_type='pitcher'
            )
            batter_inputs['context_batter_supp_inputs'] = self.batter_dataset.strip_ball_data(
                batter_inputs['context_batter_supp_inputs'], data_type='batter'
            )
            batter_inputs['context_matchup_supp_inputs'] = self.batter_dataset.strip_ball_data(
                batter_inputs['context_matchup_supp_inputs'], data_type='matchup'
            )

            batter_inputs['context_rv_thrown_pitch_data'] = torch.rand_like(
                batter_inputs['context_rv_thrown_pitch_data'])
            batter_inputs['context_rv_batted_ball_data'] = torch.rand_like(batter_inputs['context_rv_batted_ball_data'])
            batter_inputs['context_pitch_types'] = torch.zeros_like(batter_inputs['context_pitch_types'])

            # del batter_inputs['context_rv_thrown_pitch_data']
            # del batter_inputs['context_rv_batted_ball_data']
            # del batter_inputs['context_pitch_types']
            # print("raw batter_inputs['context_pitcher_supp_inputs']: {}".format(
            #     batter_inputs['context_pitcher_supp_inputs'].shape))

        batter_hit_labels = []

        if len(self.pitcher_targets) > 0 and not self.projection_mode:
            pitcher_completion_inputs = self.pitcher_dataset.process_file_set_for_finetune(pitcher_completion_files)
            for k, v in pitcher_completion_inputs.items():
                pitcher_inputs['completion_{}'.format(k)] = v

        all_batter_completion_inputs = {}
        if len(self.batter_targets) > 0 and not self.projection_mode:
            for completion_file_set in batter_completion_files:
                batter_completion_inputs, this_batter_hit_label = self.batter_dataset.process_file_set_for_finetune(
                    completion_file_set,
                    reqd_pitcher_id=int(pitcher_id) if self.single_pitcher_batter_completion else None,
                    return_matchup_stats=self.use_matchup_data,
                    return_hit_label=True
                )
                batter_hit_labels.append(this_batter_hit_label)
                for k, v in batter_completion_inputs.items():
                    k = 'completion_{}'.format(k)
                    v = v.unsqueeze(0)
                    curr_vs = all_batter_completion_inputs.get(k, [])
                    curr_vs.append(v)
                    all_batter_completion_inputs[k] = curr_vs

        for k in all_batter_completion_inputs.keys():
            batter_inputs[k] = torch.cat(all_batter_completion_inputs[k], dim=0)

        game_date = 'N/A' if self.game_pk_to_date_d is None else self.game_pk_to_date_d[game_pk]
        batter_hit_labels = torch.cat(batter_hit_labels, dim=0)

        comb_data = {
            'game_pk': game_pk,
            'game_date': game_date,
            'pitcher_id': pitcher_id,
            'batter_ids': torch.tensor(batter_ids),
            'pitcher_team': intermediate_data['pitcher_team'],
            'pitcher_team_id': self.team_str_to_id_d[intermediate_data['pitcher_team']],
            'batter_team': intermediate_data['batter_team'],
            'batter_team_id': self.team_str_to_id_d[intermediate_data['batter_team']],
            'stadium_id': -1,
            'batter_hit_labels': batter_hit_labels,
        }
        for k, v in pitcher_inputs.items():
            comb_data['pitcher_{}'.format(k)] = v

        # print('comb_data[pitcher_completion_ptb]: {}'.format(comb_data['pitcher_completion_ptb'].dtype))

        for k, v in batter_inputs.items():
            comb_data['batter_{}'.format(k)] = v

        return comb_data

    def load_game_data_(self, game_fp, topbot):
        game_pk = game_fp.split('/')[-1][:-5]
        full_game_fp = os.path.join(self.whole_game_record_dir, game_fp)
        game_j = json.load(open(full_game_fp))

        game_date = self.game_pk_to_date_d[int(game_pk)]
        game_year = game_date.split('-')[0]

        batter_ids = []
        batter_meta = []
        home_team = game_j['home_team']
        stadium_id = - 1
        if topbot == 'bot':
            pitcher_id = game_j['away_starter']['__id__']
            pitcher_team = game_j['away_team']
            batter_team = game_j['home_team']
            batter_ids.extend(game_j['home_batting_order'])
        else:
            pitcher_id = game_j['home_starter']['__id__']
            pitcher_team = game_j['home_team']
            batter_team = game_j['away_team']
            batter_ids.extend(game_j['away_batting_order'])

        pitcher_inputs = {}
        pitcher_meta_d = self.pitcher_data.get(pitcher_id, None)
        if pitcher_meta_d is None:
            print('pitcher_meta_d is None for pitcher {}\n\tgame_fp: {}'.format(pitcher_id, game_fp))
            pitcher_meta_d = {}

        pitcher_context_files, pitcher_completion_files = self.find_player_files(game_pk, pitcher_meta_d,
                                                                                 player_type='pitcher')
        pitcher_context_inputs = self.parse_player_context_file_set(pitcher_id, pitcher_context_files, 'pitcher')
        # pitcher_context_inputs = self.pitcher_dataset.process_file_set(pitcher_context_files, mode='context')
        for k, v in pitcher_context_inputs.items():
            pitcher_inputs['context_{}'.format(k)] = v

        if len(self.pitcher_targets) > 0 and not self.projection_mode:
            pitcher_completion_inputs = self.pitcher_dataset.process_file_set_for_finetune(pitcher_completion_files)
            for k, v in pitcher_completion_inputs.items():
                pitcher_inputs['completion_{}'.format(k)] = v

        batter_inputs = {}
        batter_hit_labels = []
        for batter_id in batter_ids:
            batter_meta_d = self.batter_data.get(batter_id, None)
            if batter_meta_d is None:
                print('batter_meta_d is None for batter {}\n\tgame_fp: {}'.format(batter_id, game_fp))
                batter_meta_d = {}
            batter_context_files, batter_completion_files = self.find_player_files(game_pk, batter_meta_d,
                                                                                   player_type='batter')
            batter_meta.append([batter_context_files, batter_completion_files, batter_id])

        for these_batter_context_files, these_batter_completion_files, this_batter_id in batter_meta:
            these_batter_inputs = self.parse_player_context_file_set(this_batter_id,
                                                                     these_batter_context_files, 'batter')
            # these_batter_inputs = self.batter_dataset.process_file_set(these_batter_context_files, mode='context')
            for k, v in these_batter_inputs.items():
                # print('k: {} v: {}'.format(k, v.shape))
                k = 'context_{}'.format(k)
                v = v.unsqueeze(0)
                curr_vs = batter_inputs.get(k, [])
                curr_vs.append(v)
                batter_inputs[k] = curr_vs

            if len(self.batter_targets) > 0 and not self.projection_mode:
                batter_completion_inputs, this_batter_hit_label = self.batter_dataset.process_file_set_for_finetune(
                    these_batter_completion_files,
                    reqd_pitcher_id=int(pitcher_id) if self.single_pitcher_batter_completion else None,
                    return_matchup_stats=True,
                    return_hit_label=True
                )
                batter_hit_labels.append(this_batter_hit_label)
                for k, v in batter_completion_inputs.items():
                    k = 'completion_{}'.format(k)
                    v = v.unsqueeze(0)
                    curr_vs = batter_inputs.get(k, [])
                    curr_vs.append(v)
                    batter_inputs[k] = curr_vs

        for k in batter_inputs.keys():
            batter_inputs[k] = torch.cat(batter_inputs[k], dim=0)

        game_pk = game_j['game_pk']
        game_date = 'N/A' if self.game_pk_to_date_d is None else self.game_pk_to_date_d[game_pk]
        game_season = int(game_date.split('-')[0])
        batter_hit_labels = torch.cat(batter_hit_labels, dim=0)

        comb_data = {
            'game_pk': game_pk,
            'game_date': game_date,
            'pitcher_id': pitcher_id,
            'batter_ids': torch.tensor(batter_ids),
            'pitcher_team': pitcher_team,
            'pitcher_team_id': self.team_str_to_id_d[pitcher_team],
            'batter_team': batter_team,
            'batter_team_id': self.team_str_to_id_d[batter_team],
            'stadium_id': stadium_id,
            'batter_hit_labels': batter_hit_labels,
        }
        for k, v in pitcher_inputs.items():
            comb_data['pitcher_{}'.format(k)] = v

        for k, v in batter_inputs.items():
            comb_data['batter_{}'.format(k)] = v

        return comb_data

    def parse_whole_game_intermedate(self, game_fp, topbot):
        intermediate_data = {}
        game_pk = game_fp.split('/')[-1][:-5]
        full_game_fp = os.path.join(self.whole_game_record_dir, game_fp)
        game_j = json.load(open(full_game_fp))

        game_date = self.game_pk_to_date_d[int(game_pk)]
        game_year = int(game_date.split('-')[0])

        batter_ids = []
        batter_meta = []
        home_team = game_j['home_team']
        stadium_id = -1

        if topbot == 'bot':
            pitcher_id = game_j['away_starter']['__id__']
            pitcher_team = game_j['away_team']
            batter_team = game_j['home_team']
            batter_ids.extend(game_j['home_batting_order'])
        else:
            pitcher_id = game_j['home_starter']['__id__']
            pitcher_team = game_j['home_team']
            batter_team = game_j['away_team']
            batter_ids.extend(game_j['away_batting_order'])

        intermediate_data['game_pk'] = game_j['game_pk']
        intermediate_data['stadium_id'] = stadium_id
        intermediate_data['home_team'] = home_team
        intermediate_data['pitcher_team'] = pitcher_team
        intermediate_data['batter_team'] = batter_team
        intermediate_data['pitcher_id'] = pitcher_id
        intermediate_data['batter_ids'] = batter_ids

        pitcher_inputs = {}
        pitcher_meta_d = self.pitcher_data.get(pitcher_id, None)
        if pitcher_meta_d is None:
            print('pitcher_meta_d is None for pitcher {}\n\tgame_fp: {}'.format(pitcher_id, game_fp))
            pitcher_meta_d = {}

        pitcher_context_files, pitcher_completion_files = self.find_player_files(game_pk, pitcher_meta_d,
                                                                                 player_type='pitcher')
        pitcher_context_inputs = self.pitcher_dataset.process_file_set(pitcher_context_files, mode='context',
                                                                       entity_type='pitcher')

        if self.pitcher_args.v2_player_attn:
            pitcher_attn_data = self.pitcher_dataset.construct_player_attn_data_v2(pitcher_context_inputs, None,
                                                                                   given_pitcher_id=pitcher_id,
                                                                                   given_batter_ids=batter_ids,
                                                                                   game_year=game_year)
        else:
            pitcher_attn_data = self.pitcher_dataset.construct_player_attn_data(pitcher_context_inputs, None,
                                                                                given_pitcher_id=pitcher_id,
                                                                                given_batter_ids=batter_ids,
                                                                                game_year=game_year)
        pitcher_custom_player_ids, pitcher_real_player_ids, pitcher_context_player_attn_mask, _ = pitcher_attn_data
        pitcher_context_inputs['player_attn_mask'] = pitcher_context_player_attn_mask
        for k, v in pitcher_context_inputs.items():
            pitcher_inputs['context_{}'.format(k)] = v
        pitcher_inputs['custom_player_ids'] = pitcher_custom_player_ids
        # pitcher_inputs['custom_player_ids'] = torch.zeros_like(pitcher_custom_player_ids)
        pitcher_inputs['real_player_ids'] = pitcher_real_player_ids

        if self.single_model or self.entity_models:
            entity_type_id = self.args.entity_type_d['pitcher']
            pitcher_inputs['entity_type_id'] = torch.tensor([entity_type_id])

        intermediate_data['pitcher_inputs'] = pitcher_inputs
        intermediate_data['pitcher_completion_files'] = pitcher_completion_files

        if self.single_model or self.entity_models:
            batter_inputs = {}
            team_batting_meta_d = self.team_batting_data.get(home_team, {})
            batting_context_files, batting_completion_files = self.find_player_files(game_pk, team_batting_meta_d,
                                                                                     player_type='team_batting')
            batter_context_inputs = self.batter_dataset.process_file_set(batting_context_files, mode='context',
                                                                         entity_type='team_batting')
            if self.batter_args.v2_player_attn:
                batter_attn_data = self.batter_dataset.construct_player_attn_data_v2(batter_context_inputs, None,
                                                                                     given_pitcher_id=pitcher_id,
                                                                                     given_batter_ids=batter_ids,
                                                                                     game_year=game_year)
            else:
                batter_attn_data = self.batter_dataset.construct_player_attn_data(batter_context_inputs, None,
                                                                                  given_pitcher_id=pitcher_id,
                                                                                  given_batter_ids=batter_ids,
                                                                                  game_year=game_year)
            batter_custom_player_ids, batter_real_player_ids, batter_context_player_attn_mask, _ = batter_attn_data
            batter_context_inputs['player_attn_mask'] = batter_context_player_attn_mask
            for k, v in batter_context_inputs.items():
                batter_inputs['context_{}'.format(k)] = v
            batter_inputs['custom_player_ids'] = batter_custom_player_ids
            # batter_inputs['custom_player_ids'] = torch.zeros_like(batter_custom_player_ids)
            batter_inputs['real_player_ids'] = batter_real_player_ids
            entity_type_id = self.args.entity_type_d['team_batting']
            batter_inputs['entity_type_id'] = torch.tensor([entity_type_id])

            all_batter_completion_files = []
            for batter_id in batter_ids:
                batter_meta_d = self.batter_data.get(batter_id, {})
                _, batter_completion_files = self.find_player_files(game_pk, batter_meta_d, player_type='batter')
                all_batter_completion_files.append(batter_completion_files)
        else:
            batter_inputs = {}
            all_batter_completion_files = []
            for batter_id in batter_ids:
                batter_meta_d = self.batter_data.get(batter_id, None)
                if batter_meta_d is None:
                    print('batter_meta_d is None for batter {}\n\tgame_fp: {}'.format(batter_id, game_fp))
                    batter_meta_d = {}
                batter_context_files, batter_completion_files = self.find_player_files(game_pk, batter_meta_d,
                                                                                       player_type='batter')
                all_batter_completion_files.append(batter_completion_files)
                batter_meta.append([batter_context_files, batter_completion_files, batter_id])

            for these_batter_context_files, these_batter_completion_files, this_batter_id in batter_meta:
                these_batter_inputs = self.batter_dataset.process_file_set(these_batter_context_files, mode='context')
                # these_batter_inputs = self.batter_dataset.process_file_set(these_batter_context_files, mode='context')
                for k, v in these_batter_inputs.items():
                    # print('k: {} v: {}'.format(k, v.shape))
                    k = 'context_{}'.format(k)
                    v = v.unsqueeze(0)
                    curr_vs = batter_inputs.get(k, [])
                    curr_vs.append(v)
                    batter_inputs[k] = curr_vs

            for k in batter_inputs.keys():
                batter_inputs[k] = torch.cat(batter_inputs[k], dim=0)

        intermediate_data['batter_inputs'] = batter_inputs
        intermediate_data['batter_completion_files'] = all_batter_completion_files

        return intermediate_data

    def parse_player_context_file_set(self, player_id, file_set, player_type):
        if len(file_set) == 0 or (len(file_set) == 1 and len(file_set[0]) == 0):
            first_game_pk = -1
            last_game_pk = -1
        else:
            first_ab_filename = os.path.split(file_set[0][0])[-1]
            first_game_pk = first_ab_filename.split('-')[0]
            last_ab_filename = os.path.split(file_set[-1][-1])[-1]
            last_game_pk = last_ab_filename.split('-')[0]

        intermediate_data_fp = os.path.join(
            self.intermediate_data_dir, '{}_{}_{}-{}_{}.pt'.format(
                player_id, player_type, first_game_pk, last_game_pk,
                'w_swing' if self.use_swing_status else 'wo_swing'
            )
        )
        print('** intermediate_data_fp: {} **'.format(intermediate_data_fp))
        if os.path.exists(intermediate_data_fp):
            context_inputs = torch.load(intermediate_data_fp)
        else:
            if player_type == 'pitcher':
                context_inputs = self.pitcher_dataset.process_file_set(file_set, mode='context')
            else:
                context_inputs = self.batter_dataset.process_file_set(file_set, mode='context')

            torch.save(context_inputs, intermediate_data_fp)

        return context_inputs
