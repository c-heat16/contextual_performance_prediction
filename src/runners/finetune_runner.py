__author__ = 'Connor Heaton'

import os
import time
import math
import json
import torch

import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import get_constant_schedule_with_warmup
from torch.nn.parallel import DistributedDataParallel as DDP

from ..utils import read_model_args, jsonKeys2int
from ..datasets import FLDataset, FLDatasetFinetune
from ..vocabularies import GeneralGamestateDeltaVocabByBen
from ..models import ModelFinetuner, FLModel


class FinetuneRunner(object):
    def __init__(self, gpu, mode, args, game_pk_to_date_d, team_str_to_id_d, ab_data_cache):
        self.rank = gpu
        self.mode = mode
        self.args = args
        self.game_pk_to_date_d = game_pk_to_date_d
        self.team_str_to_id_d = team_str_to_id_d
        self.ab_data_cache = ab_data_cache

        print('Initializing FinetuneRunner on device {}...'.format(gpu))
        if self.args.on_cpu:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda:{}'.format(self.rank))
            torch.cuda.set_device(self.device)

        self.world_size = len(self.args.gpus)
        print('** FinetuneRunner.world_size: {} **'.format(self.world_size))
        print('\ttorch.cuda.device_count(): {}'.format(torch.cuda.device_count()))
        # torch.manual_seed(self.args.seed)
        dist.init_process_group('nccl',
                                world_size=self.world_size,
                                rank=self.rank)

        self.out = args.out
        # self.pred_dir = args.pred_dir
        self.pred_dir = getattr(self.args, 'pred_dir', '../preds')
        self.pred_fp_tmplt = os.path.join(self.pred_dir, '{}_{}_preds_e{}.csv')
        self.binary_hit_preds = getattr(self.args, 'binary_hit_preds', True)
        self.task_specific = getattr(self.args, 'task_specific', False)
        self.single_model = getattr(self.args, 'single_model', False)
        self.entity_models = getattr(self.args, 'entity_models', False)
        self.cyclic_lr = getattr(self.args, 'cyclic_lr', False)
        self.cosine_lr = getattr(self.args, 'cosine_lr', False)
        self.do_group_gathers = getattr(self.args, 'do_group_gathers', False)

        self.use_batter = getattr(self.args, 'use_batter', True)
        self.use_pitcher = getattr(self.args, 'use_pitcher', True)

        self.lr = getattr(self.args, 'lr', 1e-5)
        self.l2 = getattr(self.args, 'l2', 0.0001)
        self.save_model_every = getattr(self.args, 'save_model_every', -1)
        self.n_warmup_iters = getattr(self.args, 'n_warmup_iters', -1)
        self.torch_amp = getattr(self.args, 'torch_amp', True)
        self.batter_ckpt = getattr(self.args, 'batter_ckpt', '../batter/models/model.pt')
        self.pitcher_ckpt = getattr(self.args, 'pitcher_ckpt', '../pitcher/models/model.pt')
        self.model_ckpt = getattr(self.args, 'model_ckpt', '../model_dir/models/model.pt')
        self.log_every = getattr(self.args, 'log_every', 100)
        self.save_model_every = getattr(self.args, 'save_model_every', -1)
        self.save_preds_every = getattr(self.args, 'save_preds_every', 5)
        self.use_matchup_data = getattr(self.args, 'use_matchup_data', False)

        self.batter_weight = getattr(self.args, 'batter_weight', 10)
        self.pitcher_weight = getattr(self.args, 'pitcher_weight', 10)
        self.batter_has_hit_weight = getattr(self.args, 'batter_has_hit_weight', 10)
        print('batter_weight: {}'.format(self.batter_weight))
        print('pitcher_weight: {}'.format(self.pitcher_weight))
        print('batter_has_hit_weight: {}'.format(self.batter_has_hit_weight))
        print('binary_hit_preds: {}'.format(self.binary_hit_preds))

        self.total_weight = self.batter_weight + self.pitcher_weight
        if self.binary_hit_preds:
            self.total_weight += self.batter_has_hit_weight
        print('total_weight: {}'.format(self.total_weight))

        self.batter_weight_pct = self.batter_weight / self.total_weight
        self.pitcher_weight_pct = self.pitcher_weight / self.total_weight
        if self.binary_hit_preds:
            self.batter_has_hit_weight_pct = self.batter_has_hit_weight / self.total_weight
        else:
            self.batter_has_hit_weight_pct = 0.0

        print('batter_weight_pct: {}'.format(self.batter_weight_pct))
        print('pitcher_weight_pct: {}'.format(self.pitcher_weight_pct))
        print('batter_has_hit_weight_pct: {}'.format(self.batter_has_hit_weight_pct))

        self.pitcher_targets = getattr(self.args, 'pitcher_targets', ['k', 'h'])
        self.pitcher_scalars = getattr(self.args, 'pitcher_scalars', [20, 20])
        self.batter_targets = getattr(self.args, 'batter_targets', ['k', 'h'])
        self.batter_scalars = getattr(self.args, 'batter_scalars', [7, 7])

        self.pitcher_scalars_list = [v for v in self.pitcher_scalars]
        self.batter_scalars_list = [v for v in self.batter_scalars]

        self.pitcher_scalars = torch.tensor(self.pitcher_scalars).to(self.device)
        self.batter_scalars = torch.tensor(self.batter_scalars).to(self.device)
        self.n_batter_targets = len(self.batter_targets)
        self.n_pitcher_targets = len(self.pitcher_targets)

        self.batter_args, self.pitcher_args = self.read_model_args()
        self.use_swing_status = getattr(self.batter_args, 'vocab_use_swing_status', False)
        self.args.use_swing_status = self.use_swing_status
        batter_l2 = self.batter_args.l2
        pitcher_l2 = self.pitcher_args.l2
        self.args.pitcher_n_context = self.pitcher_args.n_games_context
        self.args.pitcher_n_completion = self.pitcher_args.n_games_completion
        self.args.batter_n_context = self.batter_args.n_games_context
        self.args.batter_n_completion = self.batter_args.n_games_completion
        gamestate_vocab_bos_inning_no = getattr(self.pitcher_args, 'gamestate_vocab_bos_inning_no', False)
        gamestate_vocab_bos_score_diff = getattr(self.pitcher_args, 'gamestate_vocab_bos_score_diff', False)
        gamestate_vocab_bos_base_occ = getattr(self.pitcher_args, 'gamestate_vocab_bos_base_occ', True)
        gamestate_vocab_use_balls_strikes = getattr(self.pitcher_args, 'gamestate_vocab_use_balls_strikes', True)
        gamestate_vocab_use_base_occupancy = getattr(self.pitcher_args, 'gamestate_vocab_use_base_occupancy', True)
        gamestate_vocab_use_score_diff = getattr(self.pitcher_args, 'gamestate_vocab_use_score_diff', True)
        gamestate_vocab_use_outs = getattr(self.pitcher_args, 'gamestate_vocab_use_outs', True)
        gamestate_n_innings = getattr(self.pitcher_args, 'gamestate_n_innings', 10)
        gamestate_max_score_diff = getattr(self.pitcher_args, 'gamestate_max_score_diff', 6)
        self.gamestate_vocab = GeneralGamestateDeltaVocabByBen(
            bos_inning_no=gamestate_vocab_bos_inning_no, max_inning_no=gamestate_n_innings,
            bos_score_diff=gamestate_vocab_bos_score_diff, bos_max_score_diff=gamestate_max_score_diff,
            bos_base_occ=gamestate_vocab_bos_base_occ, balls_delta=gamestate_vocab_use_balls_strikes,
            strikes_delta=gamestate_vocab_use_balls_strikes, outs_delta=gamestate_vocab_use_outs,
            score_delta=gamestate_vocab_use_score_diff, base_occ_delta=gamestate_vocab_use_base_occupancy,
            swing_status=self.use_swing_status,
        )

        self.args.n_gamestate_tokens = len(self.gamestate_vocab)
        self.args.n_gamestate_bos_tokens = len(self.gamestate_vocab.bos_vocab)
        self.args.gsd_mask_id = self.gamestate_vocab.mask_id
        self.args.pitcher_output_dim = self.pitcher_args.complete_embd_dim
        self.args.batter_output_dim = self.batter_args.complete_embd_dim

        self.batter_args.n_gamestate_tokens = len(self.gamestate_vocab)
        self.batter_args.n_gamestate_bos_tokens = len(self.gamestate_vocab.bos_vocab)
        self.batter_args.gsd_mask_id = self.gamestate_vocab.mask_id
        self.batter_args.batter_targets = self.batter_targets
        self.pitcher_args.n_gamestate_tokens = len(self.gamestate_vocab)
        self.pitcher_args.n_gamestate_bos_tokens = len(self.gamestate_vocab.bos_vocab)
        self.pitcher_args.gsd_mask_id = self.gamestate_vocab.mask_id
        self.pitcher_args.pitcher_targets = self.pitcher_targets
        self.args.use_ball_data = self.pitcher_args.use_ball_data and self.batter_args.use_ball_data
        print('self.args.use_ball_data: {}'.format(self.args.use_ball_data))

        self.dataset = self.make_dataset(mode=self.mode)
        if self.args.on_cpu:
            data_sampler = None
        else:
            data_sampler = torch.utils.data.distributed.DistributedSampler(self.dataset,
                                                                           num_replicas=args.world_size,
                                                                           rank=self.rank,
                                                                           shuffle=True if self.mode == 'train' else False, )
        self.data_loader = DataLoader(
            self.dataset, batch_size=self.args.batch_size, shuffle=False,
            num_workers=self.args.n_data_workers if self.mode == 'train' else self.args.n_data_workers_else,
            pin_memory=True, sampler=data_sampler, drop_last=False, persistent_workers=False,
            # pin_memory_device='cuda:{}'.format(self.rank),
            prefetch_factor=2
        )
        self.n_iters = int(math.ceil(len(self.dataset) / (self.args.batch_size * len(self.args.gpus))))
        self.aux_dataset = None
        self.aux_data_loader = None
        self.aux_n_iters = None
        if self.mode == 'train' and (self.args.dev or self.args.dev_every > 0):
            print('FLFinetuneRunner on device {} creating auxiliary dataset...'.format(self.rank))
            self.aux_dataset = self.make_dataset(mode='dev')

            if self.args.on_cpu:
                aux_data_sampler = None
            else:
                aux_data_sampler = torch.utils.data.distributed.DistributedSampler(self.aux_dataset,
                                                                                   num_replicas=args.world_size,
                                                                                   rank=self.rank,
                                                                                   shuffle=False)

            self.aux_data_loader = DataLoader(
                self.aux_dataset, batch_size=self.args.batch_size, shuffle=False,
                num_workers=int(self.args.n_data_workers_else), pin_memory=True,
                sampler=aux_data_sampler, drop_last=False, persistent_workers=False,
                # pin_memory_device='cuda:{}'.format(self.rank),
                prefetch_factor=2
            )
            self.aux_n_iters = int(math.ceil(len(self.aux_dataset) / (self.args.batch_size * len(self.args.gpus))))

        self.model = self.make_model()

        self.start_epoch = 0
        self.n_epochs = 1
        if self.mode != 'train':
            # ckpt_file = self.args.out_dir
            if self.args.train:
                ckpt_epoch_offset = 1
                ckpt_file = os.path.join(self.args.model_save_dir,
                                         self.args.ckpt_file_tmplt.format(self.args.epochs - ckpt_epoch_offset))
                while not os.path.exists(ckpt_file) and self.args.epochs - ckpt_epoch_offset >= 0:
                    ckpt_epoch_offset += 1
                    ckpt_file = os.path.join(self.args.model_save_dir,
                                             self.args.ckpt_file_tmplt.format(self.args.epochs - ckpt_epoch_offset))
            else:
                ckpt_file = self.args.ckpt_file
        else:
            ckpt_file = self.args.ckpt_file

        if self.rank == 0:
            print('*** ckpt_file: {} ***'.format(ckpt_file))
        if ckpt_file is not None:
            if self.rank == 0:
                print('Loading model from ckpt...')
                print('\tckpt_file: {}'.format(ckpt_file))
            map_location = {'cuda:{}'.format(0): 'cpu' for gpu_id in args.gpus}
            state_dict = torch.load(ckpt_file, map_location=map_location)
            self.model.load_state_dict(state_dict, strict=False)
            model_epoch = int(ckpt_file.split('_')[-1].split('e')[0])
            self.start_epoch = model_epoch
            self.n_epochs = model_epoch + 1

        self.model = self.model.to(self.device)

        if not self.args.on_cpu:
            self.model = DDP(self.model, device_ids=[self.rank],
                             find_unused_parameters=False)

        self.summary_writer = None
        if self.rank == 0 and self.mode == 'train':
            self.summary_writer = SummaryWriter(log_dir=self.args.tb_dir)

        self.scheduler = None
        if self.mode == 'train':
            self.n_epochs = self.args.epochs

            no_decay = ['layernorm', 'norm']
            param_optimizer = list(self.model.named_parameters())
            no_decay_parms = []
            reg_parms = []
            picher_model_parms = []
            batter_model_parms = []
            # if self.rank == 0:
            #     for idx, (n, p) in enumerate(self.model.named_parameters()):
            #         print('{}: {}'.format(idx, n))

            for n, p in param_optimizer:
                if any(nd in n for nd in no_decay):
                    no_decay_parms.append(p)
                elif 'pitcher_model' in n:
                    picher_model_parms.append(p)
                elif 'batter_model' in n:
                    batter_model_parms.append(p)
                else:
                    reg_parms.append(p)

            optimizer_grouped_parameters = [
                {'params': reg_parms, 'weight_decay': self.l2},
                {'params': picher_model_parms, 'weight_decay': pitcher_l2},
                {'params': batter_model_parms, 'weight_decay': batter_l2},
                {'params': no_decay_parms, 'weight_decay': 0.0},
            ]
            if self.rank == 0:
                print('n parms: {}'.format(len(param_optimizer)))
                print('len(optimizer_grouped_parameters[0]): {}'.format(len(optimizer_grouped_parameters[0]['params'])))
                print('len(optimizer_grouped_parameters[1]): {}'.format(len(optimizer_grouped_parameters[1]['params'])))
                print('len(optimizer_grouped_parameters[2]): {}'.format(len(optimizer_grouped_parameters[2]['params'])))
                print('len(optimizer_grouped_parameters[3]): {}'.format(len(optimizer_grouped_parameters[3]['params'])))

            if self.rank == 0:
                print('Making Adam optimizer...')
            # self.optimizer = optim.Adam(optimizer_grouped_parameters, lr=self.lr)  # , betas=(0.9, 0.95)
            self.optimizer = optim.AdamW(optimizer_grouped_parameters, lr=self.lr)  # , betas=(0.9, 0.95)

            if self.n_warmup_iters > 0:
                if self.cyclic_lr:
                    print('Making cyclic scheduler...')
                    self.scheduler = optim.lr_scheduler.CyclicLR(self.optimizer,
                                                                 base_lr=1e-9,
                                                                 max_lr=self.lr,
                                                                 step_size_up=self.n_warmup_iters,
                                                                 mode='triangular2',
                                                                 cycle_momentum=False)
                elif self.cosine_lr:
                    print('Making OneCycleLR (cosine) scheduler...')
                    # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=50)
                    self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=self.lr,
                                                                         steps_per_epoch=self.n_iters,
                                                                         epochs=self.n_epochs)
                else:
                    print('Making constantLR_w_warmup scheduler..')
                    self.scheduler = get_constant_schedule_with_warmup(self.optimizer,
                                                                       num_warmup_steps=self.n_warmup_iters)

        if self.torch_amp:
            print('** Using Torch AMP **')
            self.scaler = torch.cuda.amp.GradScaler()

        # [pitcher_ids.float(), pitcher_target_preds, pitcher_target_labels]
        self.pitcher_headers = ['player_id', 'pitcher_id', 'game_pk', 'pitcher_team', 'batter_team']
        for target in self.pitcher_targets:
            self.pitcher_headers.append('{}_pred'.format(target))
        for target in self.pitcher_targets:
            self.pitcher_headers.append('{}_label'.format(target))

        self.batter_headers = ['player_id', 'batter_id', 'game_pk', 'pitcher_team', 'batter_team']
        for target in self.batter_targets:
            self.batter_headers.append('{}_pred'.format(target))
        for target in self.pitcher_targets:
            self.batter_headers.append('{}_label'.format(target))

        self.batter_has_hit_headers = ['batter_id', 'pitcher_id', 'game_pk',
                                       'batter_no_hit_p', 'batter_hit_p',
                                       'batter_has_hit_pred', 'batter_has_hit_label']

        self.run()

    def run(self):
        for epoch in range(self.start_epoch, self.n_epochs):
            if self.rank == 0:
                print('Performing epoch {} of {}'.format(epoch, self.n_epochs))

            if self.mode == 'train':
                self.model.train()
            else:
                self.model.eval()

            if self.mode == 'train':
                self.run_one_epoch(epoch, self.mode)
            else:
                with torch.no_grad():
                    self.run_one_epoch(epoch, self.mode)

            if self.rank == 0:
                print('Done with epoch {} of {}'.format(epoch, self.n_epochs))

            if self.mode == 'train':
                if epoch % self.save_model_every == 0 or epoch == self.n_epochs - 1 and self.save_model_every > 0:
                    if self.rank == 0:
                        print('Saving model...')
                        if not self.args.on_cpu:
                            torch.save(self.model.module.state_dict(),
                                       os.path.join(self.args.model_save_dir, self.args.ckpt_file_tmplt.format(epoch)))
                        else:
                            torch.save(self.model.state_dict(),
                                       os.path.join(self.args.model_save_dir, self.args.ckpt_file_tmplt.format(epoch)))
                    # dist.barrier()

                if self.args.dev_every > 0 and (epoch % self.args.dev_every == 0 or epoch == self.n_epochs - 1):
                    if self.rank == 0:
                        print('Performing train-dev for epoch {} of {}'.format(epoch, self.n_epochs))
                    self.model.eval()
                    with torch.no_grad():
                        self.run_one_epoch(epoch, 'train-dev')

    def run_one_epoch(self, epoch, mode):
        if mode == self.mode:
            dataset = self.data_loader
            n_iters = self.n_iters
            n_samples = len(self.dataset)
        else:
            dataset = self.aux_data_loader
            n_iters = self.aux_n_iters
            n_samples = len(self.aux_dataset)

        last_batch_end_time = None
        iter_since_grad_accum = 1
        pitcher_write_data = []
        batter_write_data = []
        batter_has_hit_write_data = []
        id_masking = True if mode == 'train' else False

        for batch_idx, batch_data in enumerate(dataset):
            # if batch_idx > 3:
            #     print('break!')
            #     break
            global_item_idx = (epoch * n_iters) + batch_idx
            batch_start_time = time.time()
            pitcher_custom_player_ids = batch_data['pitcher_custom_player_ids'].to(self.device, non_blocking=True)
            pitcher_real_player_ids = batch_data['pitcher_real_player_ids'].to(self.device, non_blocking=True)
            batter_custom_player_ids = batch_data['batter_custom_player_ids'].to(self.device, non_blocking=True)
            batter_real_player_ids = batch_data['batter_real_player_ids'].to(self.device, non_blocking=True)

            pitcher_inputs = {
                k[len('pitcher_context_'):]: v.to(self.device, non_blocking=True) for k, v in batch_data.items()
                if k.startswith('pitcher_context_')
            }
            if self.single_model or self.entity_models:
                pitcher_inputs['entity_type_id'] = batch_data['pitcher_entity_type_id'].to(self.device,
                                                                                           non_blocking=True).view(-1)

            pitcher_target_labels = [
                batch_data['pitcher_completion_{}'.format(p_target)].to(self.device, non_blocking=True).unsqueeze(-1)
                for p_target in self.pitcher_targets
            ]

            if len(pitcher_target_labels) > 0:
                pitcher_target_labels = torch.cat(pitcher_target_labels, dim=-1)
                if self.injury_prediction_mode or self.classify:
                    pitcher_target_labels = pitcher_target_labels.long()
                else:
                    pitcher_target_labels = pitcher_target_labels.float()
                    pitcher_target_labels /= self.pitcher_scalars
            else:
                pitcher_target_labels = None

            if self.use_batter:
                batter_inputs = {
                    k[len('batter_context_'):]: v.to(self.device, non_blocking=True) for k, v in batch_data.items()
                    if k.startswith('batter_context_')
                }
                if self.single_model or self.entity_models:
                    batter_inputs['entity_type_id'] = batch_data['batter_entity_type_id'].to(self.device, non_blocking=True).view(-1)

                batter_target_labels = [
                    batch_data['batter_completion_{}'.format(b_target)].to(self.device, non_blocking=True).unsqueeze(-1)
                    for b_target in self.batter_targets
                ]

                if len(batter_target_labels) > 0:
                    batter_target_labels = torch.cat(batter_target_labels, dim=-1).long()

                    batter_target_labels = batter_target_labels.float()
                    batter_target_labels /= self.batter_scalars
                else:
                    batter_target_labels = None
            else:
                batter_inputs = None
                batter_target_labels = None

            if self.use_matchup_data:
                matchup_data = batch_data['batter_completion_matchup_data']
            else:
                matchup_data = None

            batter_hit_labels = batch_data['batter_hit_labels'].to(self.device, non_blocking=True)
            pitcher_ids = batch_data['pitcher_id'].to(self.device, non_blocking=True)
            game_pks = batch_data['game_pk'].to(self.device, non_blocking=True)
            # game_dates = batch_data['game_date']
            batter_ids = batch_data['batter_ids'].to(self.device, non_blocking=True)
            pitcher_team_ids = batch_data['pitcher_team_id'].to(self.device, non_blocking=True)
            batter_team_ids = batch_data['batter_team_id'].to(self.device, non_blocking=True)

            if self.rank == 0 and epoch == 0 and batch_idx == 0:  # batch_idx == 0
                print('pitcher_inputs:')
                for k, v in pitcher_inputs.items():
                    print('\tk: {} v: {} min: {} max: {}'.format(k, v.shape, v.min(), v.max()))
                if batter_inputs is not None:
                    print('batter_inputs:')
                    for k, v in batter_inputs.items():
                        print('\tk: {} v: {} min: {} max: {}'.format(k, v.shape, v.min(), v.max()))

                if batter_target_labels is not None:
                    print('batter_target_labels v: {} min: {} max: {}'.format(batter_target_labels.shape,
                                                                              batter_target_labels.min(),
                                                                              batter_target_labels.max()))
                if pitcher_target_labels is not None:
                    print('pitcher_target_labels v: {} min: {} max: {}'.format(pitcher_target_labels.shape,
                                                                               pitcher_target_labels.min(),
                                                                               pitcher_target_labels.max()))

                if matchup_data is not None:
                    print('matchup_data v: {} min: {} max: {}'.format(matchup_data.shape,
                                                                      matchup_data.min(),
                                                                      matchup_data.max()))

                print('pitcher_ids v: {} min: {} max: {}'.format(pitcher_ids.shape, pitcher_ids.min(), pitcher_ids.max()))
                print('batter_hit_labels v: {} min: {} max: {}'.format(batter_hit_labels.shape, batter_hit_labels.min(), batter_hit_labels.max()))

                print('game_pks v: {} min: {} max: {}'.format(game_pks.shape, game_pks.min(), game_pks.max()))
                print('batter_ids v: {} min: {} max: {}'.format(batter_ids.shape, batter_ids.min(), batter_ids.max()))
                print('pitcher_team_ids v: {} min: {} max: {}'.format(pitcher_team_ids.shape, pitcher_team_ids.min(), pitcher_team_ids.max()))
                print('batter_team_ids v: {} min: {} max: {}'.format(batter_team_ids.shape, batter_team_ids.min(), batter_team_ids.max()))
                # print('stadium_ids v: {} min: {} max: {}'.format(stadium_ids.shape, stadium_ids.min(), stadium_ids.max()))

                # pitcher_custom_player_ids = batch_data['pitcher_custom_player_ids']
                #             pitcher_real_player_ids = batch_data['pitcher_real_player_ids']
                #             batter_custom_player_ids = batch_data['batter_custom_player_ids']
                #             batter_real_player_ids = batch_data['batter_real_player_ids']
                print('pitcher_custom_player_ids: {}'.format(pitcher_custom_player_ids.shape))
                print('pitcher_real_player_ids: {}'.format(pitcher_real_player_ids.shape))
                print('batter_custom_player_ids: {}'.format(batter_custom_player_ids.shape))
                print('batter_real_player_ids: {}'.format(batter_real_player_ids.shape))

            if self.torch_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(
                        pitcher_inputs, batter_inputs, pitcher_target_labels, batter_target_labels,
                        matchup_inputs=matchup_data, batter_has_hit_labels=batter_hit_labels,
                        pitcher_custom_player_ids=pitcher_custom_player_ids,
                        pitcher_real_player_ids=pitcher_real_player_ids,
                        batter_custom_player_ids=batter_custom_player_ids,
                        batter_real_player_ids=batter_real_player_ids,
                        id_masking=id_masking
                    )
            else:
                outputs = self.model(
                    pitcher_inputs, batter_inputs, pitcher_target_labels, batter_target_labels,
                    matchup_inputs=matchup_data, batter_has_hit_labels=batter_hit_labels,
                    pitcher_custom_player_ids=pitcher_custom_player_ids,
                    pitcher_real_player_ids=pitcher_real_player_ids,
                    batter_custom_player_ids=batter_custom_player_ids,
                    batter_real_player_ids=batter_real_player_ids,
                    id_masking=id_masking
                )

            pitcher_loss = outputs[0]
            batter_loss = outputs[1]
            batter_has_hit_loss = outputs[2]
            pitcher_target_preds = outputs[3]
            batter_target_preds = outputs[4]
            batter_has_hit_preds = outputs[5]

            if self.use_pitcher and self.use_batter:
                loss = (self.pitcher_weight_pct * pitcher_loss) + (self.batter_weight_pct * batter_loss)
                if batter_has_hit_loss is not None:
                    loss += (self.batter_has_hit_weight_pct * batter_has_hit_loss)
            elif self.use_pitcher:
                loss = pitcher_loss
            else:
                loss = batter_loss
                if batter_has_hit_loss is not None:
                    loss += batter_has_hit_loss

            if mode == 'train':
                if self.torch_amp:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

            if global_item_idx % self.save_preds_every == 0 or mode == 'train-dev':
                pitcher_target_preds *= self.pitcher_scalars
                pitcher_target_labels *= self.pitcher_scalars

                if self.do_group_gathers:
                    pitcher_target_labels, pitcher_target_preds = self.group_gather([pitcher_target_labels, pitcher_target_preds])
                else:
                    pitcher_target_labels = self.gather(pitcher_target_labels)
                    pitcher_target_preds = self.gather(pitcher_target_preds)

                if self.use_batter:
                    batter_target_preds *= self.batter_scalars
                    batter_target_labels *= self.batter_scalars

                    if self.do_group_gathers:
                        if batter_has_hit_preds is not None:
                            batter_target_preds, batter_target_labels, batter_has_hit_preds, batter_hit_labels = self.group_gather(
                                [batter_target_preds, batter_target_labels, batter_has_hit_preds, batter_hit_labels])
                        else:
                            batter_target_preds, batter_target_labels = self.group_gather(
                                [batter_target_preds, batter_target_labels])
                    else:
                        batter_target_preds = self.gather(batter_target_preds)
                        batter_target_labels = self.gather(batter_target_labels)

                        batter_has_hit_preds = self.gather(batter_has_hit_preds) if batter_has_hit_preds is not None else None
                        batter_hit_labels = self.gather(batter_hit_labels) if batter_has_hit_preds is not None else None

                    game_pks_batter = torch.cat([game_pks.unsqueeze(-1) for _ in range(9)], dim=-1)
                    pitcher_team_ids_batter = torch.cat([pitcher_team_ids.unsqueeze(-1) for _ in range(9)], dim=-1)
                    batter_team_ids_batter = torch.cat([batter_team_ids.unsqueeze(-1) for _ in range(9)], dim=-1)

                    if self.do_group_gathers:
                        game_pks_batter, pitcher_team_ids_batter, batter_team_ids_batter = self.group_gather(
                            [game_pks_batter, pitcher_team_ids_batter, batter_team_ids_batter])
                    else:
                        game_pks_batter = self.gather(game_pks_batter)
                        pitcher_team_ids_batter = self.gather(pitcher_team_ids_batter)
                        batter_team_ids_batter = self.gather(batter_team_ids_batter)

                if self.do_group_gathers:
                    game_pks, pitcher_ids, pitcher_team_ids, batter_team_ids = self.group_gather(
                        [game_pks, pitcher_ids, pitcher_team_ids, batter_team_ids])
                else:
                    game_pks = self.gather(game_pks)
                    pitcher_ids = self.gather(pitcher_ids)
                    pitcher_team_ids = self.gather(pitcher_team_ids)
                    batter_team_ids = self.gather(batter_team_ids)

                this_pitcher_write_data = torch.cat(
                    [pitcher_ids.view(-1, 1).float(),
                     pitcher_ids.view(-1, 1).float(), game_pks.view(-1, 1).float(),
                     pitcher_team_ids.view(-1, 1).float(), batter_team_ids.view(-1, 1).float(),
                     pitcher_target_preds, pitcher_target_labels],
                    dim=-1).cpu()
                pitcher_write_data.append(this_pitcher_write_data)

                batter_ids = self.gather(batter_ids)
                this_batter_write_data = torch.cat([
                    batter_ids.view(-1, 1).float(),
                    batter_ids.view(-1, 1).float(), game_pks_batter.view(-1, 1).float(),
                    pitcher_team_ids_batter.view(-1, 1).float(), batter_team_ids_batter.view(-1, 1).float(),
                    batter_target_preds.view(-1, self.n_batter_targets),
                    batter_target_labels.view(-1, self.n_batter_targets)],
                    dim=-1).cpu()
                batter_write_data.append(this_batter_write_data)

                if batter_has_hit_preds is not None:
                    pitcher_ids = torch.cat([pitcher_ids.unsqueeze(-1) for _ in range(9)], dim=-1)
                    sm = nn.Softmax(dim=-1)
                    batter_has_hit_probs = sm(batter_has_hit_preds)
                    _, batter_has_hit_pred_ids = torch.topk(batter_has_hit_probs, 1)

                    this_has_hit_write_data = torch.cat([
                        batter_ids.view(-1, 1).float(),
                        pitcher_ids.view(-1, 1).float(),
                        game_pks_batter.view(-1, 1).float(),
                        batter_has_hit_probs.view(-1, 2),
                        batter_has_hit_pred_ids.view(-1, 1),
                        batter_hit_labels.view(-1, 1),
                    ], dim=-1).cpu()
                    batter_has_hit_write_data.append(this_has_hit_write_data)

            if global_item_idx % self.args.print_every == 0 and self.rank == 0:
                batch_elapsed_time = time.time() - batch_start_time
                if last_batch_end_time is not None:
                    time_btw_batches = batch_start_time - last_batch_end_time
                else:
                    time_btw_batches = 0.0

                print_str = '{0}- epoch:{1}/{2} iter:{3}/{4} loss:{5:2.4f} p loss: {6:.4f} b loss {7:.4f}/{8:.4f}'.format(
                    mode, epoch, self.n_epochs, batch_idx, n_iters, loss, pitcher_loss,
                    batter_loss if batter_loss is not None else -1.0,
                    batter_has_hit_loss if batter_has_hit_loss is not None else -1.0,
                )
                print_str = '{0} Time: {1:.2f}s ({2:.2f}s)'.format(print_str, batch_elapsed_time, time_btw_batches)
                print(print_str)

            if (global_item_idx % self.args.summary_every == 0 and self.summary_writer is not None) or (
                    mode == 'train-dev' and self.summary_writer is not None and global_item_idx % int(
                self.args.summary_every / 3) == 0):

                self.summary_writer.add_scalar('lr/{}'.format(mode), self.optimizer.param_groups[0]['lr'],
                                               global_item_idx)
                self.summary_writer.add_scalar('loss/{}'.format(mode), loss, global_item_idx)
                self.summary_writer.add_scalar('pitcher_loss/{}'.format(mode), pitcher_loss, global_item_idx)
                if self.use_batter:
                    self.summary_writer.add_scalar('batter_loss/{}'.format(mode), batter_loss, global_item_idx)
                    if batter_has_hit_loss is not None:
                        self.summary_writer.add_scalar('batter_has_hit_loss/{}'.format(mode), batter_has_hit_loss,
                                                       global_item_idx)

            if global_item_idx % self.args.grad_summary_every == 0 \
                    and self.summary_writer is not None and mode == 'train' \
                    and self.args.grad_summary and global_item_idx != 0:
                for name, p in self.model.named_parameters():
                    if p.grad is not None and p.grad.data is not None:
                        self.summary_writer.add_histogram('grad/{}'.format(name), p.grad.data,
                                                          (epoch * n_iters) + batch_idx)
                        self.summary_writer.add_histogram('weight/{}'.format(name), p.data,
                                                          (epoch * n_iters) + batch_idx)

            if self.scheduler is not None and mode == 'train':
                self.scheduler.step()

            if iter_since_grad_accum == self.args.n_grad_accum and mode == 'train':
                # print('OPTIMIZER STEP')
                if self.torch_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad()
                iter_since_grad_accum = 1
            else:
                iter_since_grad_accum += 1

            last_batch_end_time = time.time()

        if iter_since_grad_accum > 1 and mode == 'train':
            if self.torch_amp:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            self.optimizer.zero_grad()

        if self.rank == 0:
            print('Writing pitcher predictions...')
            pitcher_write_data = torch.cat(pitcher_write_data).numpy()
            pitcher_write_df = pd.DataFrame(pitcher_write_data, columns=self.pitcher_headers)
            pitcher_write_df.to_csv(self.pred_fp_tmplt.format(mode, 'pitcher', epoch), index=False)

            if len(batter_write_data) > 0:
                print('Writing batter predictions...')
                batter_write_data = torch.cat(batter_write_data).numpy()
                batter_write_df = pd.DataFrame(batter_write_data, columns=self.batter_headers)
                batter_write_df.to_csv(self.pred_fp_tmplt.format(mode, 'batter', epoch), index=False)

                if len(batter_has_hit_write_data) > 0:
                    print('Writing batter has hit predictions...')
                    batter_has_hit_write_data = torch.cat(batter_has_hit_write_data).numpy()
                    batter_has_hit_write_df = pd.DataFrame(batter_has_hit_write_data,
                                                           columns=self.batter_has_hit_headers)
                    batter_has_hit_write_df.to_csv(self.pred_fp_tmplt.format(mode, 'batter_has_hit', epoch), index=False)

            print('Done writing preds!')

    def read_model_args(self):
        if self.single_model:
            model_basedir = os.path.split(os.path.split(self.model_ckpt)[0])[0]
            model_args_fp = os.path.join(model_basedir, 'args.txt')
            model_args = read_model_args(model_args_fp)
            model_args.config_dir = '/home/czh/nvme1/SportsAnalytics/config'

            entity_type_d = {et: idx for idx, et in enumerate(model_args.type)}
            reverse_entity_type_d = {idx: et for idx, et in enumerate(model_args.type)}
            print('entity_type_d: {}'.format(entity_type_d))
            print('reverse_entity_type_d: {}'.format(reverse_entity_type_d))

            model_args.entity_type_d = entity_type_d
            model_args.reverse_entity_type_d = reverse_entity_type_d

            self.args.entity_type_d = entity_type_d
            self.args.reverse_entity_type_d = reverse_entity_type_d

            batter_args = model_args
            pitcher_args = model_args
        else:
            batter_basedir = os.path.split(os.path.split(self.batter_ckpt)[0])[0]
            batter_args_fp = os.path.join(batter_basedir, 'args.txt')
            batter_args = read_model_args(batter_args_fp)
            batter_args.config_dir = '/home/czh/nvme1/SportsAnalytics/config'

            pitcher_basedir = os.path.split(os.path.split(self.pitcher_ckpt)[0])[0]
            pitcher_args_fp = os.path.join(pitcher_basedir, 'args.txt')
            pitcher_args = read_model_args(pitcher_args_fp)
            pitcher_args.config_dir = '/home/czh/nvme1/SportsAnalytics/config'

            if self.entity_models:
                entity_type_d = {et: idx for idx, et in enumerate(batter_args.type)}
                reverse_entity_type_d = {idx: et for idx, et in enumerate(batter_args.type)}
                print('entity_type_d: {}'.format(entity_type_d))
                print('reverse_entity_type_d: {}'.format(reverse_entity_type_d))

                batter_args.entity_type_d = entity_type_d
                batter_args.reverse_entity_type_d = reverse_entity_type_d
                pitcher_args.entity_type_d = entity_type_d
                pitcher_args.reverse_entity_type_d = reverse_entity_type_d

                self.args.entity_type_d = entity_type_d
                self.args.reverse_entity_type_d = reverse_entity_type_d

        return batter_args, pitcher_args

    def make_dataset(self, mode):
        if self.single_model:
            dataset = FLDataset(self.batter_args, mode, gamestate_vocab=self.gamestate_vocab, finetune=True,
                                ab_data_cache=self.ab_data_cache)
            batter_dataset = dataset
            pitcher_dataset = dataset
        else:
            if self.use_batter:
                batter_dataset = FLDataset(self.batter_args, mode, gamestate_vocab=self.gamestate_vocab, finetune=True,
                                           ab_data_cache=self.ab_data_cache)
            else:
                batter_dataset = None
            if self.use_pitcher:
                pitcher_dataset = FLDataset(self.pitcher_args, mode, gamestate_vocab=self.gamestate_vocab, finetune=True,
                                            ab_data_cache=self.ab_data_cache)
            else:
                pitcher_dataset = None

        dataset = FLDatasetFinetune(self.args, mode, pitcher_dataset, batter_dataset,
                                    self.pitcher_args, self.batter_args,
                                    game_pk_to_date_d=self.game_pk_to_date_d,
                                    team_str_to_id_d=self.team_str_to_id_d)
        return dataset

    def make_model(self):
        map_location = {'cuda:0': 'cpu'}
        if self.single_model:
            pitcher_model = FLModel(self.pitcher_args, self.gamestate_vocab, make_pred_heads=False)
            if self.task_specific:
                print('{}\n{}\n{}'.format(
                    '*' * len('* Not loading model weights - training task specific model *'),
                    '* Not loading model weights - training task specific model *',
                    '*' * len('* Not loading model weights - training task specific model *'),
                ))
            else:
                print('\tLoading ckpt from {}...'.format(self.model_ckpt))
                raw_model_ckpt = torch.load(self.model_ckpt, map_location=map_location)
                model_ckpt = {
                    k: v for k, v in raw_model_ckpt.items()
                    if not k.endswith('clf_head.bias') and not k.endswith('clf_head.weight')
                       and not k.endswith('pred_head.bias') and not k.endswith('pred_head.weight')
                }
                pitcher_model.load_state_dict(model_ckpt)
            batter_model = None
        else:
            if self.use_batter:
                print('** Making Batter Model **')
                batter_model = FLModel(self.batter_args, self.gamestate_vocab, make_pred_heads=False)
                if self.task_specific:
                    print('{}\n{}\n{}'.format(
                        '*' * len('* Not loading batter weights - training task specific model *'),
                        '* Not loading batter weights - training task specific model *',
                        '*' * len('* Not loading batter weights - training task specific model *'),
                    ))
                else:
                    raw_batter_ckpt = torch.load(self.batter_ckpt, map_location=map_location)
                    # print('raw_batter_ckpt.keys(): {}'.format(raw_batter_ckpt.keys()))

                    if self.entity_models:
                        batter_ckpt = {
                            k: v for k, v in raw_batter_ckpt.items()
                            if not k.endswith('clf_head.bias') and not k.endswith('clf_head.weight')
                               and not k.endswith('pred_head.bias') and not k.endswith('pred_head.weight')
                        }
                    else:
                        # batter_ckpt = {k: v for k, v in raw_batter_ckpt.items() if 'clf_head' not in k and 'prefix_position_embd' not in k}
                        batter_ckpt = {k: v for k, v in raw_batter_ckpt.items() if 'pred_head' not in k and 'prefix_position_embd' not in k}
                    batter_model.load_state_dict(batter_ckpt)
            else:
                batter_model = None


            print('** Making FL Pitcher Model **')
            pitcher_model = FLModel(self.pitcher_args, self.gamestate_vocab, make_pred_heads=False)

            if self.task_specific:
                print('{}\n{}\n{}'.format(
                    '*' * len('* Not loading pitcher weights - training task specific model *'),
                    '* Not loading pitcher weights - training task specific model *',
                    '*' * len('* Not loading pitcher weights - training task specific model *'),
                ))
            else:
                raw_pitcher_ckpt = torch.load(self.pitcher_ckpt, map_location=map_location)

                if self.entity_models:
                    pitcher_ckpt = {
                        k: v for k, v in raw_pitcher_ckpt.items()
                        if not k.endswith('clf_head.bias') and not k.endswith('clf_head.weight')
                           and not k.endswith('pred_head.bias') and not k.endswith('pred_head.weight')
                    }
                else:
                    # pitcher_ckpt = {k: v for k, v in raw_pitcher_ckpt.items() if 'clf_head' not in k and 'prefix_position_embd' not in k}
                    pitcher_ckpt = {k: v for k, v in raw_pitcher_ckpt.items() if 'pred_head' not in k and 'prefix_position_embd' not in k}
                pitcher_model.load_state_dict(pitcher_ckpt)

        model = ModelFinetuner(self.args, pitcher_model, batter_model, self.pitcher_args, self.batter_args)
        return model

    def group_gather(self, xs):
        # print('=' * 30)
        # print('Raw xs:')
        # for x in xs:
        #     print('\t{}'.format(x.shape))
        to_unsqueeze = [False if len(x.shape) > 1 else True for x in xs]
        xs = [x.unsqueeze(-1) if tu else x for x, tu in zip(xs, to_unsqueeze)]
        # print('Group gather sizes:')
        # for x in xs:
        #     print('\t{}'.format(x.shape))
        x_sizes = [x.shape[-1] for x in xs]
        xs = torch.cat(xs, dim=-1)
        xs = self.gather(xs)
        # print('Gathered xs: {}'.format(xs.shape))
        xs = torch.split(xs, x_sizes, dim=-1)
        xs = [x.squeeze(-1) if tu else x for x, tu in zip(xs, to_unsqueeze)]
        # print('Split xs:')
        # for x in xs:
        #     print('\t{}'.format(x.shape))
        # print('=' * 30)
        return xs

    def gather(self, x):
        n_x = torch.tensor([x.shape[0]], device=x.device)
        n_x_list = [torch.zeros_like(n_x) for _ in range(self.world_size)]
        dist.all_gather(n_x_list, n_x)
        n_x = torch.cat(n_x_list, dim=0).contiguous()
        max_size = n_x.max() + 1

        indicator = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)

        if x.shape[0] != max_size:
            x_padding = torch.zeros(max_size - x.shape[0], *x.shape[1:], device=x.device, dtype=x.dtype)
            indicator_padding = torch.zeros(max_size - x.shape[0], device=x.device, dtype=torch.bool)

            x = torch.cat([x, x_padding], dim=0).contiguous()
            indicator = torch.cat([indicator, indicator_padding], dim=0).contiguous()

        x_list = [torch.zeros_like(x) for _ in range(self.world_size)]
        dist.all_gather(x_list, x)
        x = torch.cat(x_list, dim=0).contiguous()

        indicator_list = [torch.zeros_like(indicator) for _ in range(self.world_size)]
        dist.all_gather(indicator_list, indicator)
        indicator = torch.cat(indicator_list, dim=0).contiguous()

        x = x[indicator == 1]
        return x

