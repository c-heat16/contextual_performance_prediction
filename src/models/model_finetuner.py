__author__ = 'Connor Heaton'

import torch
import torch.nn as nn

import torch.nn.functional as F
from torch.nn import MSELoss, L1Loss, CrossEntropyLoss


class ModelFinetuner(nn.Module):
    def __init__(self, args, pitcher_model, batter_model, pitcher_args, batter_args):
        super(ModelFinetuner, self).__init__()
        self.args = args
        self.pitcher_model = pitcher_model
        self.batter_model = batter_model if batter_model is not None else pitcher_model
        self.pitcher_args = pitcher_args
        self.batter_args = batter_args

        self.pitcher_output_dim = self.args.pitcher_output_dim
        self.batter_output_dim = self.args.batter_output_dim

        self.model_types = getattr(self.args, 'model_types', 'mt_data2vec')
        self.general_dropout_prob = getattr(self.args, 'general_dropout_prob', 0.15)
        self.use_matchup_data = getattr(self.args, 'use_matchup_data', False)
        self.injury_n_days = getattr(self.args, 'injury_n_days', [7, 15, 30])
        self.matchup_data_dim = getattr(self.args, 'matchup_data_dim', 274)
        self.use_pitcher = getattr(self.args, 'use_pitcher', False)
        self.use_batter = getattr(self.args, 'use_batter', False)
        self.single_model = getattr(self.args, 'single_model', False)
        self.entity_models = getattr(self.args, 'entity_models', False)
        self.loss_type = getattr(self.args, 'loss_type', 'mse')
        print('FLFinetuneModel.loss_type: {}'.format(self.loss_type))

        self.do_ln = getattr(self.args, 'do_ln', False)
        print('FLFinetuneModel.do_ln: {}'.format(self.do_ln))

        self.classify = getattr(self.args, 'classify', False)
        self.binary_hit_preds = getattr(self.args, 'binary_hit_preds', True)
        self.causal_classifier = getattr(self.args, 'causal_classifier', False)
        self.causal_classifier_alpha = getattr(self.args, 'causal_classifier_alpha', 3.0)

        self.v2_player_attn = getattr(self.args, 'v2_player_attn', False)
        self.v2_attn_max_n_batter = 9 * self.batter_args.n_games_context
        self.v2_attn_max_n_pitcher = self.pitcher_args.n_games_context
        self.v2_attn_offset = self.v2_attn_max_n_batter + self.v2_attn_max_n_pitcher

        # self.combined_output_dim = self.pitcher_output_dim + self.batter_output_dim
        self.combined_output_dim = 0
        if self.use_pitcher:
            self.combined_output_dim += self.pitcher_output_dim
        if self.use_batter:
            self.combined_output_dim += self.batter_output_dim
        self.dropout = nn.Dropout(self.general_dropout_prob)

        if self.use_matchup_data:
            self.combined_output_dim += self.matchup_data_dim
            self.pitcher_stat_proj = nn.Linear(9 * self.matchup_data_dim, self.matchup_data_dim)
            # self.pitcher_stat_proj_ln = nn.LayerNorm(self.matchup_data_dim)
            nn.init.xavier_uniform_(self.pitcher_stat_proj.weight)
            self.pitcher_stat_proj.bias.data.fill_(0)

        if self.do_ln:
            self.pitcher_embd_ln = nn.LayerNorm(self.pitcher_output_dim)
            if self.use_batter:
                self.batter_embd_ln = nn.LayerNorm(self.batter_output_dim)
                self.batter_proj_ln = nn.LayerNorm([self.batter_output_dim])

            if self.use_matchup_data:
                self.pitcher_stat_proj_ln = nn.LayerNorm(self.matchup_data_dim)

        self.pitcher_targets = getattr(self.args, 'pitcher_targets', ['k', 'h'])
        self.pitcher_scalars = getattr(self.args, 'pitcher_scalars', [21, 17, 12])
        self.n_pitcher_targets = len(self.pitcher_targets)

        self.pitcher_pred_head = nn.Linear(self.combined_output_dim, self.n_pitcher_targets)
        nn.init.xavier_uniform_(self.pitcher_pred_head.weight)
        self.pitcher_pred_head.bias.data.fill_(0)

        if self.use_batter:
            # self.batter_embd_ln = nn.LayerNorm(self.batter_output_dim)
            self.n_batter_targets = len(getattr(self.args, 'batter_targets', ['k', 'h']))
            self.batter_scalars = getattr(self.args, 'batter_scalars', [6, 7])

            self.batter_proj = nn.Linear(9 * self.batter_output_dim, self.batter_output_dim)

            nn.init.xavier_uniform_(self.batter_proj.weight)
            self.batter_proj.bias.data.fill_(0)

            if self.binary_hit_preds:
                self.batter_has_hit_pred_head = nn.Linear(self.combined_output_dim, 2)
                nn.init.xavier_uniform_(self.batter_has_hit_pred_head.weight)
                self.batter_has_hit_pred_head.bias.data.fill_(0)

            self.batter_pred_head = nn.Linear(self.combined_output_dim, self.n_batter_targets)
            nn.init.xavier_uniform_(self.batter_pred_head.weight)
            self.batter_pred_head.bias.data.fill_(0)

    def forward(self, pitcher_inputs, batter_inputs, pitcher_targets=None, batter_targets=None, matchup_inputs=None,
                batter_has_hit_labels=None, pitcher_custom_player_ids=None, pitcher_real_player_ids=None,
                batter_custom_player_ids=None, batter_real_player_ids=None, id_masking=False):
        pitcher_pred_inputs = []
        batter_pred_inputs = []
        batch_size = pitcher_inputs['state_delta_ids'].shape[0]
        n_batters = 9

        if matchup_inputs is not None:
            pitcher_stat_proj = self.pitcher_stat_proj(matchup_inputs.view(-1, 9 * self.matchup_data_dim))
            if self.do_ln:
                pitcher_stat_proj = self.pitcher_stat_proj_ln(pitcher_stat_proj)
            pitcher_pred_inputs.append(pitcher_stat_proj)
            # print('pitcher_stat_proj.shape: {}'.format(pitcher_stat_proj.shape))

            batter_pred_inputs.append(matchup_inputs.view(-1, self.matchup_data_dim))

        # process pitcher data first
        p_outputs = self.pitcher_model.process_data(
            gsd_ids=pitcher_inputs['state_delta_ids'], event_ids=pitcher_inputs['pitch_event_ids'],
            ptype_ids=pitcher_inputs['pitch_types'],
            batter_data=pitcher_inputs['batter_supp_inputs'],
            pitcher_data=pitcher_inputs['pitcher_supp_inputs'],
            matchup_data=pitcher_inputs['matchup_supp_inputs'],
            thrown_pitch_data=pitcher_inputs['rv_thrown_pitch_data'],
            batted_ball_data=pitcher_inputs['rv_batted_ball_data'],
            stadium_ids=None,
            src_mask=pitcher_inputs.get('agg_attn_masks', None),
            src_key_padding_mask=pitcher_inputs['model_src_pad_mask'],
            game_no_ids=pitcher_inputs['relative_game_no'], ab_no_ids=pitcher_inputs['relative_ab_no'],
            pitch_no_ids=pitcher_inputs['relative_pitch_no'], do_masking=False, mode='context',
            return_real_mask=pitcher_inputs, pitcher_handedness_ids=pitcher_inputs['pitcher_handedness_ids'],
            batter_handedness_ids=pitcher_inputs['batter_handedness_ids'],
            player_attn_mask=pitcher_inputs['player_attn_mask'],
            custom_player_ids=pitcher_custom_player_ids,
            entity_type_ids=pitcher_inputs['entity_type_id'],
            id_masking=id_masking
        )
        p_embds = p_outputs[3]
        if self.single_model or self.entity_models:
            p_embds = p_embds[:, 0, :]

        # print('p_embds: {}'.format(p_embds.shape))
        if self.do_ln:
            p_embds = self.pitcher_embd_ln(p_embds)
        pitcher_pred_inputs.append(p_embds)

        p_embds = torch.cat([p_embds.unsqueeze(1) for _ in range(n_batters)], dim=-1).view(-1, self.pitcher_output_dim)
        batter_pred_inputs.append(p_embds)

        if self.use_batter:
            # process batter data next
            if not self.single_model and not self.entity_models:
                batter_inputs = {k: v.view(batch_size * n_batters, *v.shape[2:]) for k, v in batter_inputs.items()}

            b_outputs = self.batter_model.process_data(
                gsd_ids=batter_inputs['state_delta_ids'], event_ids=batter_inputs['pitch_event_ids'],
                ptype_ids=batter_inputs['pitch_types'],
                batter_data=batter_inputs['batter_supp_inputs'],
                pitcher_data=batter_inputs['pitcher_supp_inputs'],
                matchup_data=batter_inputs['matchup_supp_inputs'],
                thrown_pitch_data=batter_inputs['rv_thrown_pitch_data'],
                batted_ball_data=batter_inputs['rv_batted_ball_data'],
                stadium_ids=None,
                src_mask=batter_inputs.get('agg_attn_masks', None),
                src_key_padding_mask=batter_inputs['model_src_pad_mask'],
                game_no_ids=batter_inputs['relative_game_no'], ab_no_ids=batter_inputs['relative_ab_no'],
                pitch_no_ids=batter_inputs['relative_pitch_no'], do_masking=False, mode='context',
                return_real_mask=batter_inputs, pitcher_handedness_ids=batter_inputs['pitcher_handedness_ids'],
                batter_handedness_ids=batter_inputs['batter_handedness_ids'],
                player_attn_mask=batter_inputs['player_attn_mask'],
                custom_player_ids=batter_custom_player_ids,
                entity_type_ids=batter_inputs['entity_type_id'],
                id_masking=id_masking
            )
            b_embds = b_outputs[3]
            if self.single_model or self.entity_models:
                # b_embds = b_embds[:, 1:, :]
                if self.batter_model.v2_player_attn:
                    b_embds = b_embds[:, self.v2_attn_max_n_pitcher:self.v2_attn_max_n_pitcher + 9, :]
                else:
                    b_embds = b_embds[:, 1:, :]
                # nan_b_embds = torch.isnan(p_embds).nonzero()
                # print('nan_b_embds: {}'.format(nan_b_embds))
                # print("batter_inputs['game_pk']: {}".format(batter_inputs['game_pk']))

            # print('b_embds: {}'.format(b_embds.shape))
            if self.do_ln:
                b_embds = self.batter_embd_ln(b_embds)
            if self.single_model or self.entity_models:
                batter_pred_inputs.append(b_embds.reshape(-1, b_embds.shape[-1]))
            else:
                batter_pred_inputs.append(b_embds)
            b_embds = b_embds.reshape(batch_size, -1)
            b_proj = self.batter_proj(self.dropout(b_embds))
            if self.do_ln:
                b_proj = self.batter_proj_ln(b_proj)
            pitcher_pred_inputs.append(b_proj)

        pitcher_target_preds = None
        batter_target_preds = None
        batter_has_hit_preds = None
        # aggregate prediction inputs and make predictions
        # print('pitcher_pred_inputs:')
        # for idx, v in enumerate(pitcher_pred_inputs):
        #     print('idx: {} shape: {}'.format(idx, v.shape))
        pitcher_pred_inputs = torch.cat(pitcher_pred_inputs, dim=-1)

        pitcher_target_preds = self.pitcher_pred_head(self.dropout(pitcher_pred_inputs))

        if self.use_batter:
            # print('batter_pred_inputs:')
            # for idx, v in enumerate(batter_pred_inputs):
            #     print('idx: {} shape: {}'.format(idx, v.shape))
            batter_pred_inputs = torch.cat(batter_pred_inputs, dim=-1)
            if self.binary_hit_preds:
                batter_has_hit_preds = self.batter_has_hit_pred_head(self.dropout(batter_pred_inputs))
            else:
                batter_has_hit_preds = None
            batter_target_preds = self.batter_pred_head(self.dropout(batter_pred_inputs))

        pitcher_loss = None

        if pitcher_targets is not None:
            if self.injury_mode:
                loss_fn = CrossEntropyLoss(reduction='none')
                pitcher_targets = pitcher_targets.view(-1)
                pitcher_target_preds = pitcher_target_preds.view(-1, 2)
            else:
                if self.loss_type == 'mae':
                    loss_fn = L1Loss()
                else:
                    loss_fn = MSELoss()

                # print('pitcher_target_preds: {} pitcher_targets: {}'.format(pitcher_target_preds.shape,
                #                                                             pitcher_targets.shape))
                pitcher_loss = loss_fn(pitcher_target_preds, pitcher_targets)
                # var_loss = variance_loss(pitcher_target_preds)
                # print('pitcher_loss: {}'.format(pitcher_loss.shape))
                if self.injury_mode:
                    pitcher_loss = pitcher_loss.view(-1, len(self.injury_n_days)).mean(-1).mean(-1)
                    # print('pitcher_loss: {}'.format(pitcher_loss))
                pitcher_k_loss, pitcher_h_loss, pitcher_r_loss = None, None, None
        else:
            pitcher_loss = None

        batter_loss = None

        batter_has_hit_loss = None
        if batter_targets is not None:
            batter_targets = batter_targets.view(-1, batter_targets.shape[-1])

            if self.loss_type == 'mae':
                loss_fn = L1Loss()
            else:
                loss_fn = MSELoss()
            batter_loss = loss_fn(batter_target_preds, batter_targets)
            batter_k_loss, batter_h_loss, batter_r_loss = None, None, None
        else:
            batter_loss = None
            batter_k_loss, batter_h_loss, batter_r_loss = None, None, None

        if batter_has_hit_labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            batter_has_hit_labels = batter_has_hit_labels.view(-1)
            if self.binary_hit_preds:
                batter_has_hit_loss = loss_fn(batter_has_hit_preds, batter_has_hit_labels)
            else:
                batter_has_hit_loss = None

        outputs = [
            pitcher_loss, batter_loss, batter_has_hit_loss,
            pitcher_target_preds, batter_target_preds, batter_has_hit_preds
        ]

        return outputs
