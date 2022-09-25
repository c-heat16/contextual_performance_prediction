__author__ = 'Connor Heaton'

import json
import torch

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import CrossEntropyLoss, MSELoss

from . import RawDataProjectionModule
from .model_utils import PositionalEmbedding, gen_nopeek_mask
from ..utils import jsonKeys2int
from .my_transformer_encoder_layer import MyTransformerEncoderLayer


class FLModel(nn.Module):
    def __init__(self, args, gamestate_vocab, make_pred_heads=True):
        super(FLModel, self).__init__()
        self.args = args
        self.gamestate_vocab = gamestate_vocab
        self.make_pred_heads = make_pred_heads

        self.gsd_mask_id = self.args.gsd_mask_id
        self.event_mask_id = 1
        self.ptype_mask_id = 1
        self.n_pitch_types = 18
        self.gamestate_embd_dim = getattr(self.args, 'gsd_embd_dim', 256)
        self.general_dropout_prob = getattr(self.args, 'general_dropout_prob', 0.15)
        self.n_transformer_layers = getattr(self.args, 'n_layers', 8)
        self.n_attn = getattr(self.args, 'n_attn', 8)
        self.n_raw_data_proj_layers = getattr(self.args, 'n_raw_data_proj_layers', 2)
        self.type = getattr(self.args, 'type', 'batter')
        self.attn_mask_type = getattr(self.args, 'attn_mask_type', 'bidirectional')
        self.token_mask_pct = getattr(self.args, 'token_mask_pct', 0.15)
        self.player_id_mask_pct = getattr(self.args, 'player_id_mask_pct', 0.15)
        self.mask_override_prob = getattr(self.args, 'mask_override_prob', 0.15)
        self.use_prefix_position_embd = getattr(self.args, 'use_prefix_position_embd', False)
        self.boab_can_be_masked = getattr(self.args, 'boab_can_be_masked', False)
        self.context_only = getattr(self.args, 'context_only', False)
        # self.predict_pitches = getattr(self.args, 'predict_pitches', False)
        self.predict_ball_data = getattr(self.args, 'predict_ball_data', False)
        self.entity_type_d = self.args.entity_type_d
        self.reverse_entity_type_d = self.args.reverse_entity_type_d

        self.only_mask_eoi_present = getattr(self.args, 'only_mask_eoi_present', False)
        print('FLModel.only_mask_eoi_present: {}'.format(self.only_mask_eoi_present))
        self.norm_first = getattr(self.args, 'norm_first', False)
        print('FLModel.norm_first: {}'.format(self.norm_first))

        self.n_games_context = getattr(self.args, 'n_games_context', 3)
        self.n_games_completion = getattr(self.args, 'n_games_completion', 1)
        self.v2_player_attn = getattr(self.args, 'v2_player_attn', False)
        self.v2_attn_max_n_batter = 9 * self.n_games_context
        self.v2_attn_max_n_pitcher = self.n_games_context

        if self.v2_player_attn:
            self.attn_offset = self.v2_attn_max_n_batter + self.v2_attn_max_n_pitcher
        else:
            self.attn_offset = 10

        self.pitcher_entity_id = self.entity_type_d.get('pitcher', -1)
        self.team_pitching_entity_id = self.entity_type_d.get('team_pitching', -1)
        self.batter_entity_id = self.entity_type_d.get('batter', -1)
        self.team_batting_entity_id = self.entity_type_d.get('team_batting', -1)
        print('pitcher_entity_id: {}'.format(self.pitcher_entity_id))
        print('team_pitching_entity_id: {}'.format(self.team_pitching_entity_id))
        print('batter_entity_id: {}'.format(self.batter_entity_id))
        print('team_batting_entity_id: {}'.format(self.team_batting_entity_id))

        print('!! FL MODEL CONTEXT ONLY = {} !!'.format(self.context_only))
        print('!! FLModel.attn_mask_type: {} !!'.format(self.attn_mask_type))

        self.prepend_entity_type_id = getattr(self.args, 'prepend_entity_type_id', False)
        self.use_ball_data = getattr(self.args, 'use_ball_data', True)
        self.mask_ball_data = getattr(self.args, 'mask_ball_data', True)
        self.drop_ball_data = getattr(self.args, 'drop_ball_data', True)
        self.ordinal_pos_embeddings = getattr(self.args, 'ordinal_pos_embeddings', False)
        self.player_cls = getattr(self.args, 'player_cls', False)
        print('FLModel player_cls: {}'.format(self.player_cls))

        player_id_map_fp = getattr(self.args, 'player_id_map_fp',
                                   '/home/czh/nvme1/SportsAnalytics/config/all_player_id_mapping.json')
        self.player_id_map = json.load(open(player_id_map_fp), object_hook=jsonKeys2int)

        self.raw_pitcher_data_dim = getattr(self.args, 'raw_pitcher_data_dim', 216)
        self.raw_batter_data_dim = getattr(self.args, 'raw_batter_data_dim', 216)
        self.raw_matchup_data_dim = getattr(self.args, 'raw_matchup_data_dim', 243)
        self.complete_embd_dim = getattr(self.args, 'complete_embd_dim', 256)
        self.pitch_type_embd_dim = getattr(self.args, 'pitch_type_embd_dim', 15)
        self.pitch_event_embd_dim = getattr(self.args, 'event_embd_dim', 15)
        self.n_stadium_ids = getattr(self.args, 'n_stadium_ids', 40)
        self.stadium_embd_dim = getattr(self.args, 'stadium_embd_dim', 10)
        self.handedness_embd_dim = getattr(self.args, 'handedness_embd_dim', 5)
        self.both_player_positions = getattr(self.args, 'both_player_positions', False)
        self.both_player_handedness = getattr(self.args, 'both_player_handedness', False)
        self.n_thrown_pitch_metrics = getattr(self.args, 'n_thrown_pitch_metrics', 14)
        self.n_batted_ball_metrics = getattr(self.args, 'n_batted_ball_metrics', 5)
        self.rv_pred_dim = getattr(self.args, 'rv_pred_dim', 5)
        self.n_supp_rv = getattr(self.args, 'n_supp_rv', 14)
        self.statcast_data_dim = self.rv_pred_dim + self.n_supp_rv

        print('complete_embd_dim: {}\npitch_event_embd_dim: {}\ngamestate_embd_dim: {}'.format(
            self.complete_embd_dim, self.pitch_event_embd_dim, self.gamestate_embd_dim
        ))

        self.n_games_context = getattr(self.args, 'n_games_context', 3)
        self.context_max_len = getattr(self.args, 'context_max_len', 378)
        self.n_games_completion = getattr(self.args, 'n_games_completion', 1)
        self.completion_max_len = getattr(self.args, 'completion_max_len', 128)
        self.max_seq_len = max(self.context_max_len, self.completion_max_len) + 2
        self.generic_player_id = 0

        if self.drop_ball_data:
            self.pitcher_career_n_ball_data = getattr(self.args, 'pitcher_career_n_ball_data', 47)
            self.batter_career_n_ball_data = getattr(self.args, 'batter_career_n_ball_data', 47)
            self.else_n_ball_data = getattr(self.args, 'else_n_ball_data', 47)

            self.batter_data_scope_sizes = getattr(self.args, 'batter_data_scope_sizes', [167, 74, 74])
            self.pitcher_data_scope_sizes = getattr(self.args, 'pitcher_data_scope_sizes', [141, 74, 74])
            self.matchup_data_scope_sizes = getattr(self.args, 'matchup_data_scope_sizes', [74, 74, 74])

            self.batter_data_n_ball_data = [self.batter_career_n_ball_data]
            while len(self.batter_data_n_ball_data) < len(self.batter_data_scope_sizes):
                self.batter_data_n_ball_data.append(self.else_n_ball_data)

            self.pitcher_data_n_ball_data = [self.pitcher_career_n_ball_data]
            while len(self.pitcher_data_n_ball_data) < len(self.pitcher_data_scope_sizes):
                self.pitcher_data_n_ball_data.append(self.else_n_ball_data)

            self.matchup_data_n_ball_data = [self.else_n_ball_data]
            while len(self.matchup_data_n_ball_data) < len(self.matchup_data_scope_sizes):
                self.matchup_data_n_ball_data.append(self.else_n_ball_data)

            self.raw_pitcher_data_dim -= sum(self.pitcher_data_n_ball_data)
            self.raw_batter_data_dim -= sum(self.batter_data_n_ball_data)
            self.raw_matchup_data_dim -= sum(self.matchup_data_n_ball_data)

            print('updated pitcher_dim: {}'.format(self.raw_pitcher_data_dim))
            print('updated batter_dim: {}'.format(self.raw_batter_data_dim))
            print('updated matchup_dim: {}'.format(self.raw_matchup_data_dim))

        self.context_supp_in_dim = 0
        self.completion_supp_in_dim = 0
        self.context_supp_proj_dim = self.complete_embd_dim - self.gamestate_embd_dim - self.pitch_event_embd_dim
        if not self.drop_ball_data:
            self.context_supp_proj_dim = self.context_supp_proj_dim - self.pitch_type_embd_dim - self.statcast_data_dim

        print('context_supp_proj_dim: {}'.format(self.context_supp_proj_dim))
        # if self.both_player_positions:
        #     self.context_supp_proj_dim -= self.player_pos_embd_dim

        # if self.both_player_handedness:
        #     self.context_supp_proj_dim -= self.handedness_embd_dim

        # if 'batter' in self.args.context_components:
        self.context_supp_in_dim += self.raw_batter_data_dim
        # if 'pitcher' in self.args.context_components:
        self.context_supp_in_dim += self.raw_pitcher_data_dim
        # if 'matchup' in self.args.context_components:
        self.context_supp_in_dim += self.raw_matchup_data_dim

        # pitch_event_map_fp = getattr(self.args, 'pitch_event_map_fp', '../config/pitch_event_id_mapping.json')
        # self.pitch_event_map = json.load(open(pitch_event_map_fp))
        self.n_pitch_events = 22

        self.n_gamestate_tokens = len(self.gamestate_vocab)
        self.n_gamestate_bos_tokens = len(self.gamestate_vocab.bos_vocab)
        self.mask_id = self.gamestate_vocab.mask_id
        self.boab_id = self.gamestate_vocab.boab_id

        if self.attn_mask_type == 'autoregressive':
            self.sos_embds = nn.Embedding(1, self.complete_embd_dim, max_norm=None)
            nn.init.xavier_uniform_(self.sos_embds.weight)

        self.dropout = nn.Dropout(self.general_dropout_prob)
        # create embeddings
        if self.player_cls:
            self.player_embds = nn.Embedding(len(self.player_id_map), self.complete_embd_dim)
            nn.init.xavier_uniform_(self.player_embds.weight)

            if self.prepend_entity_type_id:
                self.entity_type_embedding = nn.Embedding(len(self.entity_type_d), self.complete_embd_dim)
                nn.init.xavier_uniform_(self.entity_type_embedding.weight)

        self.cls_embd = nn.Embedding(1, self.complete_embd_dim, max_norm=None)
        if not self.v2_player_attn:
            self.player_type_embd = nn.Embedding(2, self.complete_embd_dim, max_norm=None)

        self.gamestate_embds = nn.Embedding(self.n_gamestate_tokens + self.n_gamestate_bos_tokens,
                                            self.gamestate_embd_dim, max_norm=None)
        # self.stadium_embds = nn.Embedding(self.n_stadium_ids, self.stadium_embd_dim, max_norm=None)
        self.game_no_embds = nn.Embedding(self.n_games_context + 1, self.complete_embd_dim, max_norm=None)
        nn.init.xavier_uniform_(self.game_no_embds.weight)
        if self.ordinal_pos_embeddings:
            print('** MAKING ORDINAL POS EMBDS **')
            self.pos_embds = PositionalEmbedding(self.max_seq_len + self.attn_offset + 1, self.complete_embd_dim)
        else:
            self.ab_no_embds = nn.Embedding(150, self.complete_embd_dim, max_norm=None)
            self.pitch_no_embds = nn.Embedding(25, self.complete_embd_dim, max_norm=None)

            nn.init.xavier_uniform_(self.ab_no_embds.weight)
            nn.init.xavier_uniform_(self.pitch_no_embds.weight)

        self.pitch_event_embds = nn.Embedding(self.n_pitch_events + 2, self.pitch_event_embd_dim, max_norm=None)
        nn.init.xavier_uniform_(self.cls_embd.weight)
        nn.init.xavier_uniform_(self.gamestate_embds.weight)
        # nn.init.xavier_uniform_(self.stadium_embds.weight)
        nn.init.xavier_uniform_(self.pitch_event_embds.weight)

        if not self.drop_ball_data:
            self.pitch_type_embds = nn.Embedding(self.n_pitch_types + 2, self.pitch_type_embd_dim, max_norm=None)
            nn.init.xavier_uniform_(self.pitch_type_embds.weight)

        if make_pred_heads:
            if self.use_prefix_position_embd and not self.context_only:
                print('** Using prefix positional embedding **')
                self.prefix_position_embd = nn.Embedding(1, self.complete_embd_dim, max_norm=None)
                nn.init.xavier_uniform_(self.prefix_position_embd.weight)

            self.gamestate_clf_head = nn.Linear(self.complete_embd_dim,
                                                self.n_gamestate_tokens + self.n_gamestate_bos_tokens)
            self.pitch_event_clf_head = nn.Linear(self.complete_embd_dim, self.n_pitch_events + 2)
            nn.init.xavier_uniform_(self.gamestate_clf_head.weight)
            nn.init.xavier_uniform_(self.pitch_event_clf_head.weight)
            self.gamestate_clf_head.bias.data.fill_(0)
            self.pitch_event_clf_head.bias.data.fill_(0)

            # if self.type == 'pitcher' and self.use_ball_data:
            if 'pitcher' in self.type and self.use_ball_data and self.predict_ball_data:
                self.pitch_type_clf_head = nn.Linear(self.complete_embd_dim, self.n_pitch_types + 2)
                nn.init.xavier_uniform_(self.pitch_type_clf_head.weight)
                self.pitch_type_clf_head.bias.data.fill_(0)

                self.thrown_pitch_pred_head = nn.Linear(self.complete_embd_dim, self.n_thrown_pitch_metrics)
                nn.init.xavier_uniform_(self.thrown_pitch_pred_head.weight)
                self.thrown_pitch_pred_head.bias.data.fill_(0)
            else:
                self.pitch_type_clf_head = None
                self.thrown_pitch_pred_head = None

            if 'batter' in self.type or 'team_batting' in self.type and self.use_ball_data and self.predict_ball_data:
                self.batted_ball_pred_head = nn.Linear(self.complete_embd_dim, self.n_batted_ball_metrics)
                nn.init.xavier_uniform_(self.batted_ball_pred_head.weight)
                self.batted_ball_pred_head.bias.data.fill_(0)
            else:
                self.batted_ball_pred_head = None

        if not self.context_only:
            self.cls_proj = nn.Linear(self.complete_embd_dim, self.complete_embd_dim)
            nn.init.xavier_uniform_(self.cls_proj.weight)
            self.cls_proj_layernorm = nn.LayerNorm([self.complete_embd_dim])

        transformer_layernorm = nn.LayerNorm([self.complete_embd_dim])
        if self.norm_first:
            encoder_layer = nn.TransformerEncoderLayer(self.complete_embd_dim,
                                                       self.n_attn,
                                                       4 * self.complete_embd_dim,
                                                       self.general_dropout_prob,
                                                       norm_first=self.norm_first)
        else:
            encoder_layer = MyTransformerEncoderLayer(d_model=self.complete_embd_dim,
                                                      nhead=self.n_attn,
                                                      dim_feedforward=4 * self.complete_embd_dim,
                                                      dropout=self.general_dropout_prob,
                                                      norm_first=self.norm_first,
                                                      n_total_layers=self.n_transformer_layers)
        self.transformer = nn.TransformerEncoder(encoder_layer, self.n_transformer_layers,
                                                 norm=transformer_layernorm)

        self.supp_proj_module = RawDataProjectionModule(
            in_dim=self.context_supp_in_dim, out_dim=self.context_supp_proj_dim, n_layers=self.n_raw_data_proj_layers,
            dropout_p=self.general_dropout_prob, do_layernorm=True
        )

    def forward(self, context_inputs, completion_inputs, do_masking=True, return_real_mask=False,
                return_embds=False, mask_seed=None, custom_player_ids=None, entity_type_ids=None):
        context_only = self.context_only
        if do_masking and self.attn_mask_type == 'bidirectional':
            player_id_mask_probs = torch.rand(custom_player_ids.shape)
            custom_player_ids[player_id_mask_probs < self.player_id_mask_pct] = 0

        # print("context_inputs['rv_thrown_pitch_data']: {}".format(context_inputs['rv_thrown_pitch_data']))
        # print("context_inputs['rv_thrown_pitch_data'][:, :10, :]: {}".format(context_inputs['rv_thrown_pitch_data'][:, :10, :]))

        context_outputs = self.process_data(
            gsd_ids=context_inputs['state_delta_ids'], event_ids=context_inputs['pitch_event_ids'],
            ptype_ids=context_inputs['pitch_types'],
            batter_data=context_inputs['batter_supp_inputs'],
            pitcher_data=context_inputs['pitcher_supp_inputs'], matchup_data=context_inputs['matchup_supp_inputs'],
            thrown_pitch_data=context_inputs['rv_thrown_pitch_data'],
            batted_ball_data=context_inputs['rv_batted_ball_data'],
            stadium_ids=None,
            src_mask=context_inputs.get('agg_attn_masks', None),
            src_key_padding_mask=context_inputs['model_src_pad_mask'],
            game_no_ids=context_inputs['relative_game_no'], ab_no_ids=context_inputs['relative_ab_no'],
            pitch_no_ids=context_inputs['relative_pitch_no'], do_masking=do_masking, mode='context',
            return_real_mask=return_real_mask, pitcher_handedness_ids=context_inputs['pitcher_handedness_ids'],
            batter_handedness_ids=context_inputs['batter_handedness_ids'],
            context_only=context_only, mask_seed=mask_seed,
            player_attn_mask=context_inputs['player_attn_mask'],
            custom_player_ids=custom_player_ids,
            entity_type_ids=entity_type_ids,
        )
        # gamestate_preds, pitch_type_preds, pitch_event_preds, cls_embds, processed_seq, mask_idxs
        context_gsd_preds, context_ptype_preds = context_outputs[0], context_outputs[1]
        context_event_preds, context_embds = context_outputs[2], context_outputs[3]
        context_output_seq, context_masked_idxs = context_outputs[4], context_outputs[5]
        context_thrown_pitch_data_preds, context_batted_ball_data_preds = context_outputs[6], context_outputs[7]
        context_masked_pitcher_idxs, context_masked_batter_idxs = context_outputs[8], context_outputs[9]

        # context_items_no_context = (context_inputs['game_pk_t'].sum(-1) == 0).nonzero()
        # print('context_items_no_context: {}'.format(context_items_no_context))
        # print('context_embds[context_items_no_context]: {}'.format(context_embds[context_items_no_context]))

        if self.attn_mask_type == 'bidirectional':
            losses = self.calc_loss(
                context_gsd_preds, context_event_preds, context_ptype_preds,
                context_inputs['state_delta_labels'][context_masked_idxs],
                context_inputs['pitch_event_labels'][context_masked_idxs],
                context_inputs['pitch_type_labels'][context_masked_idxs][
                    context_masked_pitcher_idxs] if 'pitcher' in self.type and self.use_ball_data and self.predict_ball_data else None,
                context_thrown_pitch_data_preds, context_batted_ball_data_preds,
                None if context_masked_pitcher_idxs is None
                else context_inputs['rv_thrown_pitch_data_labels'][context_masked_idxs][context_masked_pitcher_idxs],
                None if context_masked_batter_idxs is None
                else context_inputs['rv_batted_ball_data_labels'][context_masked_idxs][context_masked_batter_idxs],
            )

            context_gsd_loss = losses[0]
            context_event_loss = losses[1]
            context_ptype_loss = losses[2]
            context_thrown_pitch_data_loss = losses[3]
            context_batted_ball_data_loss = losses[4]

        else:
            indicator = context_inputs['model_src_pad_mask'] == 0
            # print('raw indicator:\n{}'.format(indicator))
            indicator2 = context_inputs['state_delta_labels'].detach() != self.boab_id
            indicator = indicator * indicator2
            # print('adj indicator:\n{}'.format(indicator))
            # input('indicator:\n{}'.format(indicator))
            context_gsd_preds = context_gsd_preds[indicator]
            context_event_preds = context_event_preds[indicator]
            if context_ptype_preds is not None:
                context_ptype_preds = context_ptype_preds[indicator]
            context_gsd_loss, context_event_loss, context_ptype_loss = self.calc_loss(
                context_gsd_preds, context_event_preds, context_ptype_preds,
                context_inputs['state_delta_labels'][indicator],
                context_inputs['pitch_event_labels'][indicator],
                context_inputs['pitch_type_labels'][
                    indicator] if 'pitcher' in self.type and self.use_ball_data else None
            )

        if not context_only:
            completion_outputs = self.process_data(
                prefix=context_embds,
                gsd_ids=completion_inputs['state_delta_ids'], event_ids=completion_inputs['pitch_event_ids'],
                ptype_ids=completion_inputs['pitch_types'],
                batter_data=completion_inputs['batter_supp_inputs'],
                pitcher_data=completion_inputs['pitcher_supp_inputs'],
                matchup_data=completion_inputs['matchup_supp_inputs'],
                thrown_pitch_data=completion_inputs['rv_thrown_pitch_data'],
                batted_ball_data=completion_inputs['rv_batted_ball_data'],
                stadium_ids=None,
                src_mask=completion_inputs.get('agg_attn_masks', None),
                src_key_padding_mask=completion_inputs['model_src_pad_mask'],
                game_no_ids=completion_inputs['relative_game_no'], ab_no_ids=completion_inputs['relative_ab_no'],
                pitch_no_ids=completion_inputs['relative_pitch_no'], do_masking=do_masking, mode='completion',
                return_real_mask=return_real_mask, pitcher_handedness_ids=completion_inputs['pitcher_handedness_ids'],
                batter_handedness_ids=completion_inputs['batter_handedness_ids'], mask_seed=mask_seed,
                player_attn_mask=completion_inputs['player_attn_mask'],
                custom_player_ids=custom_player_ids,
                entity_type_ids=entity_type_ids
            )
            completion_gsd_preds, completion_ptype_preds = completion_outputs[0], completion_outputs[1]
            completion_event_preds, completion_embds = completion_outputs[2], completion_outputs[3]
            completion_output_seq, completion_masked_idxs = completion_outputs[4], completion_outputs[5]
            completion_thrown_pitch_data_preds, completion_batted_ball_data_preds = completion_outputs[6], \
                                                                                    completion_outputs[7]
            completion_masked_pitcher_idxs, completion_masked_batter_idxs = completion_outputs[8], completion_outputs[9]

            if self.attn_mask_type == 'bidirectional':
                losses = self.calc_loss(
                    completion_gsd_preds, completion_event_preds, completion_ptype_preds,
                    completion_inputs['state_delta_labels'][completion_masked_idxs],
                    completion_inputs['pitch_event_labels'][completion_masked_idxs],
                    completion_inputs['pitch_type_labels'][completion_masked_idxs][
                        completion_masked_pitcher_idxs] if 'pitcher' in self.type and self.use_ball_data and self.predict_ball_data else None,
                    completion_thrown_pitch_data_preds, completion_batted_ball_data_preds,
                    None if completion_masked_pitcher_idxs is None
                    else completion_inputs['rv_thrown_pitch_data_labels'][completion_masked_idxs][
                        completion_masked_pitcher_idxs],
                    None if completion_masked_batter_idxs is None
                    else completion_inputs['rv_batted_ball_data_labels'][completion_masked_idxs][
                        completion_masked_batter_idxs],
                )

                completion_gsd_loss = losses[0]
                completion_event_loss = losses[1]
                completion_ptype_loss = losses[2]
                completion_thrown_pitch_data_loss = losses[3]
                completion_batted_ball_data_loss = losses[4]
            else:
                indicator = completion_inputs['model_src_pad_mask'] == 0
                # print('raw indicator:\n{}'.format(indicator))
                indicator2 = completion_inputs['state_delta_labels'].detach() != self.boab_id
                indicator = indicator * indicator2
                # print('adj indicator:\n{}'.format(indicator))

                completion_gsd_preds = completion_gsd_preds[indicator]
                completion_event_preds = completion_event_preds[indicator]
                if completion_ptype_preds is not None:
                    completion_ptype_preds = completion_ptype_preds[indicator]
                completion_gsd_loss, completion_event_loss, completion_ptype_loss = self.calc_loss(
                    completion_gsd_preds, completion_event_preds, completion_ptype_preds,
                    completion_inputs['state_delta_labels'][indicator],
                    completion_inputs['pitch_event_labels'][indicator],
                    completion_inputs['pitch_type_labels'][
                        indicator] if 'pitcher' in self.type and self.use_ball_data else None
                )
        else:
            completion_gsd_preds, completion_ptype_preds = None, None
            completion_event_preds, completion_embds = None, None
            completion_output_seq, completion_masked_idxs = None, None

            completion_gsd_loss, completion_event_loss, completion_ptype_loss = None, None, None
            completion_thrown_pitch_data_loss, completion_batted_ball_data_loss = None, None
            completion_thrown_pitch_data_preds, completion_batted_ball_data_preds = None, None

            completion_masked_pitcher_idxs = None
            completion_masked_batter_idxs = None

        outputs = [
            context_gsd_loss, context_gsd_preds, context_event_loss, context_event_preds,
            context_ptype_loss, context_ptype_preds, context_masked_idxs,
            completion_gsd_loss, completion_gsd_preds, completion_event_loss, completion_event_preds,
            completion_ptype_loss, completion_ptype_preds, completion_masked_idxs,
            context_thrown_pitch_data_loss, context_thrown_pitch_data_preds,
            context_batted_ball_data_loss, context_batted_ball_data_preds,
            completion_thrown_pitch_data_loss, completion_thrown_pitch_data_preds,
            completion_batted_ball_data_loss, completion_batted_ball_data_preds,
            context_masked_pitcher_idxs, context_masked_batter_idxs,
            completion_masked_pitcher_idxs, completion_masked_batter_idxs
        ]

        if return_embds:
            outputs.append(context_embds)

        return outputs

    def calc_loss(self, gsd_preds, event_preds, ptype_preds, gsd_labels, event_labels, ptype_labels=None,
                  thrown_pitch_data_preds=None, batted_ball_data_preds=None, thrown_pitch_data_labels=None,
                  batted_ball_data_labels=None):
        xent = CrossEntropyLoss()
        mse = MSELoss()
        if self.attn_mask_type != 'bidirectional':
            indicator = gsd_labels < self.n_gamestate_tokens
            # print('indicator:\n{}'.format(indicator))

            gsd_preds = gsd_preds[indicator]
            gsd_labels = gsd_labels[indicator]
            event_preds = event_preds[indicator]
            event_labels = event_labels[indicator]
            if ptype_labels is not None and ptype_preds is not None:
                ptype_preds = ptype_preds[indicator]
                ptype_labels = ptype_labels[indicator]

        if gsd_preds is not None and gsd_labels is not None:
            gsd_loss = xent(gsd_preds, gsd_labels)
        else:
            gsd_loss = None

        if event_preds is not None and event_labels is not None:
            event_loss = xent(event_preds, event_labels)
        else:
            event_loss = None

        if ptype_labels is not None and ptype_preds is not None:
            # print('ptype_labels: {}'.format(ptype_labels))
            ptype_loss = xent(ptype_preds, ptype_labels)
        else:
            ptype_loss = None

        if thrown_pitch_data_labels is not None and thrown_pitch_data_preds is not None:
            # print('thrown_pitch_data_labels: {}'.format(thrown_pitch_data_labels))
            thrown_pitch_data_loss = mse(thrown_pitch_data_preds, thrown_pitch_data_labels)
        else:
            thrown_pitch_data_loss = None

        if batted_ball_data_labels is not None and batted_ball_data_preds is not None:
            batted_ball_data_loss = mse(batted_ball_data_preds, batted_ball_data_labels)
        else:
            batted_ball_data_loss = None

        return gsd_loss, event_loss, ptype_loss, thrown_pitch_data_loss, batted_ball_data_loss

    def process_data(self, gsd_ids, event_ids, ptype_ids, batter_data, pitcher_data, matchup_data,
                     thrown_pitch_data, batted_ball_data, stadium_ids, src_mask,
                     src_key_padding_mask, game_no_ids, ab_no_ids, pitch_no_ids, pitcher_handedness_ids,
                     batter_handedness_ids, do_masking=True, mode='context', prefix=None, return_real_mask=False,
                     masked_indices=None, context_only=False, mask_seed=None,
                     player_attn_mask=None, custom_player_ids=None, entity_type_ids=None, id_masking=False):
        mask_idxs = None
        if do_masking and self.attn_mask_type == 'bidirectional':
            gsd_ids, event_ids, ptype_ids, batter_data, \
            pitcher_data, matchup_data, thrown_pitch_data, \
            batted_ball_data, stadium_ids, mask_idxs = self.mask_inputs(
                gsd_ids, event_ids, ptype_ids, src_key_padding_mask, batter_data, pitcher_data, matchup_data,
                thrown_pitch_data, batted_ball_data, stadium_ids, return_real_mask, pitcher_handedness_ids,
                batter_handedness_ids, masked_indices=masked_indices, mask_seed=mask_seed,
                player_attn_mask=player_attn_mask, only_mask_eoi_present=self.only_mask_eoi_present,
                entity_type_ids=entity_type_ids, mode=mode
            )
        elif id_masking and not do_masking:
            # print('Masking ID')
            player_id_mask_probs = torch.rand(custom_player_ids.shape)
            custom_player_ids[player_id_mask_probs < self.player_id_mask_pct] = 0

        # print('gsd_ids: {}'.format(gsd_ids.shape))
        # print('raw entity_type_ids: {}'.format(entity_type_ids.shape))
        # entity_type_ids = entity_type_ids.unsqueeze(-1).expand(-1, gsd_ids.shape[1])
        # print('new entity_type_ids: {}'.format(entity_type_ids.shape))
        # print('entity_type_ids: {}'.format(entity_type_ids))

        gamestate_embds = self.gamestate_embds(gsd_ids)
        pitch_event_embds = self.pitch_event_embds(event_ids)
        # stadium_embds = self.stadium_embds(stadium_ids)
        supp_inputs = torch.cat([batter_data, pitcher_data, matchup_data], dim=-1)

        supp_proj = self.supp_proj_module(supp_inputs)
        # print('gamestate_embds: {}'.format(gamestate_embds.shape))
        # print('supp_proj: {}'.format(supp_proj.shape))
        # print('pitch_event_embds: {}'.format(pitch_event_embds.shape))
        # print('thrown_pitch_data: {}'.format(thrown_pitch_data.shape))
        # print('batted_ball_data: {}'.format(batted_ball_data.shape))

        input_seq_components = [gamestate_embds, supp_proj, pitch_event_embds]
        if not self.drop_ball_data:
            input_seq_components.extend([thrown_pitch_data, batted_ball_data])
            pitch_type_embds = self.pitch_type_embds(ptype_ids)
            input_seq_components.append(pitch_type_embds)

        # if self.both_player_handedness:
        #     pitcher_handendess_embds = self.p_handedness_embds(pitcher_handedness_ids)
        #     batter_handendess_embds = self.b_handedness_embds(batter_handedness_ids)
        #     input_seq_components.extend([pitcher_handendess_embds, batter_handendess_embds])
        # elif self.type in ['batter', 'team_batting']:
        #     pitcher_handendess_embds = self.p_handedness_embds(pitcher_handedness_ids)
        #     input_seq_components.append(pitcher_handendess_embds)
        # else:
        #     batter_handendess_embds = self.b_handedness_embds(batter_handedness_ids)
        #     input_seq_components.append(batter_handendess_embds)

        input_seq = torch.cat(input_seq_components, dim=-1)

        if self.ordinal_pos_embeddings:
            if self.v2_player_attn:
                input_seq = self.pos_embds(input_seq)
            relative_game_no_embds = self.game_no_embds(game_no_ids)
            input_seq = input_seq + relative_game_no_embds
        else:
            relative_game_no_embds = self.game_no_embds(game_no_ids)
            relative_ab_no_embds = self.ab_no_embds(ab_no_ids)
            relative_pitch_no_embds = self.pitch_no_embds(pitch_no_ids)
            input_seq = input_seq + relative_game_no_embds + relative_ab_no_embds + relative_pitch_no_embds
        # print('input_seq: {}'.format(input_seq.shape))

        if mode == 'context':
            if self.player_cls:
                cls_ids = torch.zeros(
                    (input_seq.shape[0], self.attn_offset + 1 if self.v2_player_attn else self.attn_offset),
                    dtype=gsd_ids.dtype, device=input_seq.device)
                # all_player_ids = torch.concat([custom_pitcher_id, custom_batter_ids], dim=-1)

                cls_embds = self.cls_embd(cls_ids)
                player_embds = self.player_embds(custom_player_ids)
                effective_embds = cls_embds + player_embds
                if not self.v2_player_attn:
                    player_type_ids = torch.tensor([0, 1, 1, 1, 1, 1, 1, 1, 1, 1]).to(input_seq.device)
                    player_type_embds = self.player_type_embd(player_type_ids)
                    # print('player_type_embds: {}'.format(player_type_embds.shape))
                    effective_embds += player_type_embds

                # print('effective_embds: {}'.format(effective_embds.shape))
                if self.prepend_entity_type_id:
                    extra = self.entity_type_embedding(entity_type_ids).unsqueeze(1)
                else:
                    extra = torch.zeros((input_seq.shape[0], 1, self.complete_embd_dim), dtype=gsd_ids.dtype,
                                        device=input_seq.device)
                # print('entity_type_embds: {}'.format(entity_type_embds.shape))
                # print('effective_embds: {}'.format(effective_embds.shape))
                effective_embds = torch.concat(
                    [extra,
                     effective_embds], dim=1
                )
                # print('effective_embds: {}'.format(effective_embds.shape))
                # effective_embds = torch.cat(
                #     [(input_seq.shape[0], 11), dtype=gsd_ids.dtype, device=input_seq.device), effective_embds]
                # )
                src_key_padding_mask_adj = torch.zeros(src_key_padding_mask.shape[0], self.attn_offset + 1,
                                                       dtype=torch.bool).to(src_key_padding_mask.device)
                # print('src_key_padding_mask_adj:\n{}\n\t{}'.format(src_key_padding_mask_adj, src_key_padding_mask_adj.shape))
                # print('src_key_padding_mask:\n{}\n\t{}'.format(src_key_padding_mask, src_key_padding_mask.shape))

                if self.attn_mask_type == 'bidirectional':
                    input_seq = torch.cat([effective_embds, input_seq], dim=1)
                    src_key_padding_mask = torch.cat([src_key_padding_mask_adj, src_key_padding_mask], dim=1)
                else:
                    src_key_padding_mask_adj_sos = torch.zeros(src_key_padding_mask.shape[0], 1,
                                                               dtype=torch.bool).to(src_key_padding_mask.device)
                    sos_ids = torch.zeros((input_seq.shape[0], 1), dtype=gsd_ids.dtype, device=input_seq.device)
                    sos_embds = self.sos_embds(sos_ids)
                    input_seq = torch.cat([sos_embds, input_seq, effective_embds], dim=1)
                    src_key_padding_mask = torch.cat([src_key_padding_mask_adj_sos,
                                                      src_key_padding_mask, src_key_padding_mask_adj], dim=1)

                # print('src_key_padding_mask[0]: {}'.format(src_key_padding_mask[0]))
                # print('input_seq: {}'.format(input_seq.shape))
                # print('src_key_padding_mask: {}'.format(src_key_padding_mask.shape))
                # print('src_key_padding_mask:\n{}\n\t{}'.format(src_key_padding_mask, src_key_padding_mask.shape))
                # input('okty')

            else:
                cls_ids = torch.zeros((input_seq.shape[0], 1), dtype=gsd_ids.dtype, device=input_seq.device)
                cls_embds = self.cls_embd(cls_ids)
                src_key_padding_mask_adj = torch.zeros(src_key_padding_mask.shape[0], 1,
                                                       dtype=torch.bool).to(src_key_padding_mask.device)

                # PLACE [CLS] FIRST
                if self.attn_mask_type == 'bidirectional':
                    input_seq = torch.cat([cls_embds, input_seq], dim=1)
                    src_key_padding_mask = torch.cat([src_key_padding_mask_adj, src_key_padding_mask], dim=1)
                else:
                    sos_ids = torch.zeros((input_seq.shape[0], 1), dtype=gsd_ids.dtype, device=input_seq.device)
                    sos_embds = self.sos_embds(sos_ids)
                    input_seq = torch.cat([sos_embds, input_seq, cls_embds], dim=1)
                    src_key_padding_mask = torch.cat([src_key_padding_mask_adj,
                                                      src_key_padding_mask, src_key_padding_mask_adj], dim=1)

        elif prefix is not None:
            if self.use_prefix_position_embd:
                prefix_pos_ids = torch.zeros((input_seq.shape[0]), dtype=gsd_ids.dtype, device=input_seq.device)
                prefix_pos_embds = self.prefix_position_embd(prefix_pos_ids)
                prefix = prefix + prefix_pos_embds

            # print('input_seq: {}'.format(input_seq.shape))
            # print('prefix: {}'.format(prefix.shape))
            # prefix = prefix.unsqueeze(1)
            completion_offset = 11 if self.prepend_entity_type_id else 10
            src_key_padding_mask_adj = torch.zeros(src_key_padding_mask.shape[0],
                                                   completion_offset if self.player_cls else 1,
                                                   dtype=torch.bool).to(src_key_padding_mask.device)
            # print('src_key_padding_mask: {}'.format(src_key_padding_mask.shape))
            # print('src_key_padding_mask_adj: {}'.format(src_key_padding_mask_adj.shape))
            if not self.player_cls:
                prefix = prefix.unsqueeze(1)
            else:
                if self.prepend_entity_type_id:
                    entity_type_embds = self.entity_type_embedding(entity_type_ids).unsqueeze(1)
                    # print('entity_type_embds: {}'.format(entity_type_embds.shape))
                    # print('prefix: {}'.format(prefix.shape))
                    prefix = torch.cat([entity_type_embds, prefix], dim=1)
                    # print('prefix: {}'.format(prefix.shape))

            if self.attn_mask_type == 'bidirectional':
                input_seq = torch.cat([prefix, input_seq], dim=1)
                src_key_padding_mask = torch.cat([src_key_padding_mask_adj, src_key_padding_mask], dim=1)
            else:
                sos_ids = torch.zeros((input_seq.shape[0], 1), dtype=gsd_ids.dtype, device=input_seq.device)
                sos_embds = self.sos_embds(sos_ids)
                input_seq = torch.cat([prefix, sos_embds, input_seq], dim=1)
                src_key_padding_mask = torch.cat([src_key_padding_mask_adj, src_key_padding_mask_adj,
                                                  src_key_padding_mask], dim=1)

            # print('new input_seq: {}'.format(input_seq.shape))
            # print('new src_key_padding_mask: {}'.format(src_key_padding_mask.shape))

        if self.ordinal_pos_embeddings and not self.v2_player_attn:
            input_seq = self.pos_embds(input_seq)

        if self.attn_mask_type == 'bidirectional':
            if self.player_cls:
                # src_mask = player_attn_mask
                bz = player_attn_mask.shape[0]
                n_attn = player_attn_mask.shape[1]
                src_mask = player_attn_mask.reshape(bz * n_attn, *player_attn_mask.shape[2:])
                # src_mask = src_mask.reshape(bz * n_attn, *src_mask.shape[2:])
                # # print('src_mask: {}'.format(src_mask.shape))
                # # print('pitcher_attn_mask:\n{}\n\t{}'.format(pitcher_attn_mask, pitcher_attn_mask.shape))
                # # print('batter_attn_mask:\n{}\n\t{}'.format(batter_attn_mask, batter_attn_mask.shape))
                # pitcher_attn_mask = pitcher_attn_mask.reshape(bz * n_attn, *pitcher_attn_mask.shape[2:])
                # batter_attn_mask = batter_attn_mask.reshape(bz * n_attn, *batter_attn_mask.shape[2:])
                # player_attn_mask = torch.cat([pitcher_attn_mask.unsqueeze(1), batter_attn_mask], dim=1)
                # # print('player_attn_mask:\n{}\n\t{}'.format(player_attn_mask, player_attn_mask.shape))
                # # player_mask_adj = torch.zeros(10, 10, dtype=src_mask.dtype, device=src_mask.device).unsqueeze(0).expand(src_mask.shape[0], -1, -1)
                # player_mask_adj = torch.zeros(10, 10, dtype=src_mask.dtype, device=src_mask.device) + float('-inf')
                # player_mask_adj = player_mask_adj.fill_diagonal_(0).unsqueeze(0).expand(src_mask.shape[0], -1, -1)
                # # print('player_mask_adj:\n{}\n\t{}'.format(player_mask_adj, player_mask_adj.shape))
                #
                # final_player_mask = torch.cat([player_mask_adj, player_attn_mask], dim=-1)
                # # print('final_player_mask:\n{}\n\t{}'.format(final_player_mask, final_player_mask.shape))
                #
                # # seq_mask_adj = torch.zeros(src_mask.shape[1], 10, dtype=src_mask.dtype, device=src_mask.device) + float('-inf')
                # seq_mask_adj = player_attn_mask.mT
                # # print('player_mask_adj:\n{}\n\t{}'.format(player_mask_adj, player_mask_adj.shape))
                # # print('seq_mask_adj:\n{}\n\t{}'.format(seq_mask_adj, seq_mask_adj.shape))
                # # print('src_mask:\n{}\n\t{}'.format(src_mask, src_mask.shape))
                # src_mask = torch.concat([seq_mask_adj, src_mask], dim=2)
                # # print('src_mask:\n{}\n\t{}'.format(src_mask, src_mask.shape))
                # # print('src_mask[0]: {}'.format(src_mask[0]))
                # # mask_adj = torch.concat([player_mask_adj, seq_mask_adj],
                # #                         dim=1)
                # # print('mask_adj: {}'.format(mask_adj.shape))
                # src_mask = torch.cat([final_player_mask, src_mask], dim=1)

                # print('src_mask:\n{}\n\t{}'.format(src_mask, src_mask.shape))
                # print('src_mask[0]: {}'.format(src_mask[0]))
                # for i in range(10):
                #     print('src_mask[0, {}]: {}'.format(i, src_mask[0, i]))
                # input('okty')

            else:
                bz = src_mask.shape[0]
                n_attn = src_mask.shape[1]
                src_mask = src_mask.reshape(bz * n_attn, *src_mask.shape[2:])

                mask_adj = torch.zeros(src_mask.shape[0], input_seq.shape[1] - src_mask.shape[1], src_mask.shape[2],
                                       dtype=src_mask.dtype, device=src_mask.device)
                src_mask = torch.concat([src_mask, mask_adj], dim=1)

                mask_adj = torch.zeros(src_mask.shape[0], src_mask.shape[1], input_seq.shape[1] - src_mask.shape[2],
                                       dtype=src_mask.dtype, device=src_mask.device)
                src_mask = torch.concat([mask_adj, src_mask], dim=2)
        else:
            src_mask = gen_nopeek_mask(input_seq.shape[1]).to(gamestate_embds.device)

        input_seq = input_seq.transpose(0, 1)
        output_seq = self.transformer(input_seq, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        output_seq = output_seq.transpose(0, 1)

        if self.attn_mask_type == 'bidirectional':
            if mode == 'context':
                if self.player_cls:
                    if self.v2_player_attn:
                        processed_seq = output_seq[:, self.v2_attn_offset + 1:]
                        cls_embds = output_seq[:, 1:self.v2_attn_offset + 1]
                    else:
                        processed_seq = output_seq[:, 11:]
                        cls_embds = output_seq[:, 1:11]
                else:
                    processed_seq = output_seq[:, 1:]
                    cls_embds = output_seq[:, 0]
                if not self.context_only:
                    cls_embds = self.cls_proj_layernorm(self.cls_proj(cls_embds))
            else:
                if self.player_cls:
                    if self.prepend_entity_type_id:
                        processed_seq = output_seq[:, 11:]
                    else:
                        processed_seq = output_seq[:, 10:]
                else:
                    processed_seq = output_seq[:, 1:]  # same as above, but here do not want prefix token
                cls_embds = None
            processed_seq = processed_seq[mask_idxs] if mask_idxs is not None else None
        else:
            if mode == 'context':
                processed_seq = output_seq[:, :-2]  # clip one short to slide everything over - preds are for NEXT idx
                cls_embds = output_seq[:, -1]
                if not self.context_only:
                    cls_embds = self.cls_proj_layernorm(self.cls_proj(cls_embds))
            else:
                processed_seq = output_seq[:, 1:-1]  # same as above, but here do not want prefix token
                cls_embds = None

        masked_pitcher_idxs = None
        masked_batter_idxs = None
        if processed_seq is not None and self.make_pred_heads:
            entity_type_ids = entity_type_ids.unsqueeze(-1).expand(-1, gsd_ids.shape[1])
            processed_seq = self.dropout(processed_seq)
            gamestate_preds = self.gamestate_clf_head(processed_seq)
            pitch_event_preds = self.pitch_event_clf_head(processed_seq)

            if self.make_pred_heads and self.predict_ball_data:
                masked_entity_type_ids = entity_type_ids[mask_idxs]
                # print('masked_entity_type_ids: {}'.format(masked_entity_type_ids))
                # print('masked_entity_type_ids: {}'.format(masked_entity_type_ids.shape))

                masked_single_pitcher_idxs = masked_entity_type_ids == self.pitcher_entity_id
                masked_team_pitching_idxs = masked_entity_type_ids == self.team_pitching_entity_id
                # print('masked_single_pitcher_idxs: {}'.format(masked_single_pitcher_idxs))
                # print('masked_team_pitching_idxs: {}'.format(masked_team_pitching_idxs))
                masked_pitcher_idxs = masked_single_pitcher_idxs + masked_team_pitching_idxs
                # print('masked_pitcher_idxs: {}'.format(masked_pitcher_idxs))

                masked_single_batter_idxs = masked_entity_type_ids == self.batter_entity_id
                masked_team_batting_idxs = masked_entity_type_ids == self.team_batting_entity_id
                # print('masked_single_batter_idxs: {}'.format(masked_single_batter_idxs))
                # print('masked_team_batting_idxs: {}'.format(masked_team_batting_idxs))
                masked_batter_idxs = masked_single_batter_idxs + masked_team_batting_idxs
                # print('masked_batter_idxs: {}'.format(masked_batter_idxs))

                processed_pitcher_seq = processed_seq[masked_pitcher_idxs]
                processed_batter_seq = processed_seq[masked_batter_idxs]

                if processed_pitcher_seq.shape[0] > 0:
                    pitch_type_preds = self.pitch_type_clf_head(processed_pitcher_seq)
                    thrown_pitch_metric_preds = self.thrown_pitch_pred_head(processed_pitcher_seq)
                else:
                    pitch_type_preds = None
                    thrown_pitch_metric_preds = None

                if processed_batter_seq.shape[0] > 0:
                    batted_ball_metric_preds = self.batted_ball_pred_head(processed_batter_seq)
                else:
                    batted_ball_metric_preds = None

            else:
                pitch_type_preds = None
                thrown_pitch_metric_preds = None
                batted_ball_metric_preds = None

            # if 'pitcher' in self.type and self.use_ball_data and self.predict_pitches:
            #     pitch_type_preds = self.pitch_type_clf_head(processed_seq)
            # else:
            #     pitch_type_preds = None
        else:
            gamestate_preds = None
            pitch_event_preds = None
            pitch_type_preds = None
            thrown_pitch_metric_preds = None
            batted_ball_metric_preds = None

        return gamestate_preds, pitch_type_preds, pitch_event_preds, cls_embds, processed_seq, mask_idxs, \
               thrown_pitch_metric_preds, batted_ball_metric_preds, masked_pitcher_idxs, masked_batter_idxs

    def mask_inputs(self, gsd_ids, event_ids, ptype_ids, pad_mask, batter_data, pitcher_data, matchup_data,
                    thrown_pitch_data, batted_ball_data, stadium_ids, return_real_mask, pitcher_handedness_ids,
                    batter_handedness_ids, mask_seed=None, masked_indices=None,
                    player_attn_mask=None, only_mask_eoi_present=None, entity_type_ids=None, mode='context'):
        if masked_indices is None:
            batch_mask_idxs = torch.zeros_like(gsd_ids, dtype=torch.bool)
            if mask_seed is not None:
                # print('* Setting torch seed *')
                torch.manual_seed(mask_seed)

            mask_probs = torch.rand(batch_mask_idxs.shape)
            mask_probs[pad_mask > 0] = 1.0
            if not self.boab_can_be_masked:
                mask_probs[gsd_ids >= self.n_gamestate_tokens] = 1.0

            batch_mask_idxs[mask_probs <= self.token_mask_pct] = True
            if not return_real_mask:
                # batch_mask_idxs[mask_probs <= self.token_mask_pct] = True
                mask_override_prob = torch.rand(batch_mask_idxs.shape)
                mask_probs[mask_override_prob <= self.mask_override_prob] = 1.0
        else:
            batch_mask_idxs = masked_indices
            mask_probs = torch.ones(batch_mask_idxs.shape)
            mask_probs[masked_indices] = 0.0

        gsd_ids[mask_probs <= self.token_mask_pct] = self.gsd_mask_id
        event_ids[mask_probs <= self.token_mask_pct] = self.event_mask_id
        pitcher_handedness_ids[mask_probs <= self.token_mask_pct] = 0
        batter_handedness_ids[mask_probs <= self.token_mask_pct] = 0
        # stadium_ids[mask_probs <= self.token_mask_pct] = 0

        batter_data[mask_probs <= self.token_mask_pct] = 0.0
        pitcher_data[mask_probs <= self.token_mask_pct] = 0.0
        matchup_data[mask_probs <= self.token_mask_pct] = 0.0
        if self.use_ball_data:
            thrown_pitch_data[mask_probs <= self.token_mask_pct] = 0.0
            batted_ball_data[mask_probs <= self.token_mask_pct] = 0.0
            ptype_ids[mask_probs <= self.token_mask_pct] = self.ptype_mask_id

        return gsd_ids, event_ids, ptype_ids, batter_data, pitcher_data, matchup_data, thrown_pitch_data, \
               batted_ball_data, stadium_ids, batch_mask_idxs
