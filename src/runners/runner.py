__author__ = 'Connor Heaton'

import os
import time
import json
import math
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

from ..utils import jsonKeys2int
from ..vocabularies import GeneralGamestateDeltaVocabByBen
from ..datasets import FLDataset
from ..models import FLModel


class Runner(object):
    def __init__(self, gpu, mode, args):
        self.rank = gpu
        self.mode = mode
        self.args = args

        print('Initializing Runner on device {}...'.format(gpu))
        if self.args.on_cpu:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda:{}'.format(self.rank))
            torch.cuda.set_device(self.device)

        self.world_size = len(self.args.gpus)
        print('** Runner.world_size: {} **'.format(self.world_size))
        print('\ttorch.cuda.device_count(): {}'.format(torch.cuda.device_count()))
        # torch.manual_seed(self.args.seed)
        dist.init_process_group('nccl',
                                world_size=self.world_size,
                                rank=self.rank)

        self.out = args.out
        self.attn_mask_type = getattr(self.args, 'attn_mask_type', 'bidirectional')
        self.type = getattr(self.args, 'type', 'batter')
        # self.model_type = getattr(self.args, 'model_type', 'mt')
        self.lr = getattr(self.args, 'lr', 1e-5)
        self.l2 = getattr(self.args, 'l2', 0.0001)
        self.log_every = getattr(self.args, 'log_every', 100)
        self.save_model_every = getattr(self.args, 'save_model_every', -1)
        self.save_preds_every = getattr(self.args, 'save_preds_every', 5)
        self.pred_dir = os.path.join(self.args.out, 'preds')
        self.log_dir = os.path.join(self.args.out, 'logs')
        self.n_warmup_iters = getattr(self.args, 'n_warmup_iters', -1)
        self.torch_amp = getattr(self.args, 'torch_amp', True)
        self.use_intermediate_data = getattr(self.args, 'use_intermediate_data', False)
        self.ab_data_dir = getattr(self.args, 'ab_data', '/home/czh/sata1/SportsAnalytics/ab_seqs/ab_seqs_v10')
        self.intermediate_data_dir = os.path.join(self.ab_data_dir, 'intermediate_data')
        self.log_fp_tmplt = os.path.join(self.log_dir,
                                         '{}_{}_{}_preds_e{}.csv')  # train/dev, cntxt/cmpltn, gsd/event/ptype, e
        self.boab_can_be_masked = getattr(self.args, 'boab_can_be_masked', True)
        self.reduced_event_map = getattr(self.args, 'reduced_event_map', False)

        self.context_only = getattr(self.args, 'context_only', False)
        self.context_max_len = getattr(self.args, 'context_max_len', 378)
        self.completion_max_len = getattr(self.args, 'completion_max_len', 128)

        self.do_group_gathers = getattr(self.args, 'do_group_gathers', False)
        self.player_cls = getattr(self.args, 'player_cls', False)
        self.n_games_context = getattr(self.args, 'n_games_context', 15)
        self.n_games_completion = getattr(self.args, 'n_games_completion', 1)
        print('Runner player_cls: {}'.format(self.player_cls))
        print('n_games_context: {}'.format(self.n_games_context))
        print('n_games_completion: {}'.format(self.n_games_completion))
        self.v2_player_attn = getattr(self.args, 'v2_player_attn', False)
        self.v2_attn_max_n_batter = 9 * self.n_games_context
        self.v2_attn_max_n_pitcher = self.n_games_context
        self.v2_attn_offset = self.v2_attn_max_n_batter + self.v2_attn_max_n_pitcher

        self.use_ball_data = getattr(self.args, 'use_ball_data', True)
        self.gsd_weight = getattr(self.args, 'gsd_weight', 10)
        self.event_weight = getattr(self.args, 'event_weight', 1)
        self.ptype_weight = getattr(self.args, 'ptype_weight', 1)
        self.thrown_pitch_weight = getattr(self.args, 'thrown_pitch_weight', 1)
        self.batted_ball_weight = getattr(self.args, 'batted_ball_weight', 1)

        self.use_player_id = getattr(self.args, 'use_player_id', True)
        self.extended_test_logs = getattr(self.args, 'extended_test_logs', False)
        self.use_explicit_test_masked_indices = getattr(self.args, 'use_explicit_test_masked_indices', False)
        self.warm_start = getattr(self.args, 'warm_start', False)

        player_id_map_fp = getattr(self.args, 'player_id_map_fp',
                                   '/home/czh/nvme1/SportsAnalytics/config/all_player_id_mapping.json')
        self.player_id_map = json.load(open(player_id_map_fp), object_hook=jsonKeys2int)

        self.ptype_hot = self.use_ball_data and self.predict_ball_data
        if self.ptype_hot:
            total_weight = self.gsd_weight + self.event_weight + self.ptype_weight + \
                           self.thrown_pitch_weight + self.batted_ball_weight

            self.gsd_weight_pct = self.gsd_weight / total_weight
            self.event_weight_pct = self.event_weight / total_weight
            self.ptype_weight_pct = self.ptype_weight / total_weight
            self.thrown_pitch_weight_pct = self.thrown_pitch_weight / total_weight
            self.batted_ball_weight_pct = self.batted_ball_weight / total_weight
        else:
            total_weight = self.gsd_weight + self.event_weight
            self.gsd_weight_pct = self.gsd_weight / total_weight
            self.event_weight_pct = self.event_weight / total_weight
            self.ptype_weight_pct = 0.0
            self.thrown_pitch_weight_pct = 0.0
            self.batted_ball_weight_pct = 0.0

        if self.rank == 0:
            print('gsd_weight_pct: {0:.2f}'.format(self.gsd_weight_pct))
            print('event_weight_pct: {0:.2f}'.format(self.event_weight_pct))
            print('ptype_weight_pct: {0:.2f}'.format(self.ptype_weight_pct))
            print('thrown_pitch_weight_pct: {0:.2f}'.format(self.thrown_pitch_weight_pct))
            print('batted_ball_weight_pct: {0:.2f}'.format(self.batted_ball_weight_pct))

        if self.mode == 'test':
            print('MTRunner in test mode, setting log_every=1...')
            self.log_every = 1

        if self.torch_amp:
            print('** Using Torch AMP **')
            self.scaler = torch.cuda.amp.GradScaler()

        if not os.path.exists(self.pred_dir) and self.rank == 0 and not self.mode != 'inspect':
            os.makedirs(self.pred_dir)

        if self.use_intermediate_data and self.rank == 0 and not os.path.exists(
                self.intermediate_data_dir) and not self.mode != 'inspect':
            os.makedirs(self.intermediate_data_dir)

        self.pred_fp_tmplt = os.path.join(self.pred_dir, '{}_preds_epoch_{}.csv')

        print('MTRunner on device {} creating gamestate vocab...'.format(self.rank))
        gamestate_vocab_bos_inning_no = getattr(self.args, 'gamestate_vocab_bos_inning_no', False)
        gamestate_vocab_bos_score_diff = getattr(self.args, 'gamestate_vocab_bos_score_diff', False)
        gamestate_vocab_bos_base_occ = getattr(self.args, 'gamestate_vocab_bos_base_occ', True)
        gamestate_vocab_use_balls_strikes = getattr(self.args, 'gamestate_vocab_use_balls_strikes', True)
        gamestate_vocab_use_base_occupancy = getattr(self.args, 'gamestate_vocab_use_base_occupancy', True)
        gamestate_vocab_use_score_diff = getattr(self.args, 'gamestate_vocab_use_score_diff', True)
        gamestate_vocab_use_outs = getattr(self.args, 'gamestate_vocab_use_outs', True)
        gamestate_n_innings = getattr(self.args, 'gamestate_n_innings', 10)
        gamestate_max_score_diff = getattr(self.args, 'gamestate_max_score_diff', 6)
        # GeneralGamestateDeltaVocabByBen
        self.gamestate_vocab = GeneralGamestateDeltaVocabByBen(
            bos_inning_no=gamestate_vocab_bos_inning_no, max_inning_no=gamestate_n_innings,
            bos_score_diff=gamestate_vocab_bos_score_diff, bos_max_score_diff=gamestate_max_score_diff,
            bos_base_occ=gamestate_vocab_bos_base_occ, balls_delta=gamestate_vocab_use_balls_strikes,
            strikes_delta=gamestate_vocab_use_balls_strikes, outs_delta=gamestate_vocab_use_outs,
            score_delta=gamestate_vocab_use_score_diff, base_occ_delta=gamestate_vocab_use_base_occupancy,
        )

        self.args.n_gamestate_tokens = len(self.gamestate_vocab)
        self.args.n_gamestate_bos_tokens = len(self.gamestate_vocab.bos_vocab)
        self.args.gsd_mask_id = self.gamestate_vocab.mask_id
        if self.rank == 0:
            self.gamestate_vocab.save_vocab(self.out)

        self.args.n_gamestate_tokens = len(self.gamestate_vocab)
        self.args.n_gamestate_bos_tokens = len(self.gamestate_vocab.bos_vocab)
        self.args.gsd_mask_id = self.gamestate_vocab.mask_id
        if self.rank == 0:
            self.gamestate_vocab.save_vocab(self.out)

        print('Runner on device {} creating dataset...'.format(self.rank))
        self.dataset = FLDataset(self.args, self.mode, self.gamestate_vocab)
        print('\tlen(dataset): {}'.format(len(self.dataset)))
        if self.args.on_cpu:
            data_sampler = None
        else:
            data_sampler = torch.utils.data.distributed.DistributedSampler(self.dataset,
                                                                           num_replicas=args.world_size,
                                                                           rank=self.rank,
                                                                           shuffle=True if self.mode == 'train' else False, )
        self.data_loader = DataLoader(self.dataset, batch_size=self.args.batch_size, shuffle=False,
                                      num_workers=self.args.n_data_workers if self.mode == 'train' else self.args.n_data_workers_else,
                                      pin_memory=True, sampler=data_sampler,
                                      drop_last=False, persistent_workers=False)
        self.n_iters = int(math.ceil(len(self.dataset) / (self.args.batch_size * len(self.args.gpus))))
        self.aux_dataset = None
        self.aux_data_loader = None
        self.aux_n_iters = None
        if self.mode == 'train' and (self.args.dev or self.args.dev_every > 0):
            print('Runner on device {} creating auxiliary dataset...'.format(self.rank))
            self.aux_dataset = FLDataset(self.args, 'dev', self.gamestate_vocab)

            if self.args.on_cpu:
                aux_data_sampler = None
            else:
                aux_data_sampler = torch.utils.data.distributed.DistributedSampler(self.aux_dataset,
                                                                                   num_replicas=args.world_size,
                                                                                   rank=self.rank,
                                                                                   shuffle=False)

            self.aux_data_loader = DataLoader(self.aux_dataset, batch_size=self.args.batch_size, shuffle=False,
                                              num_workers=int(self.args.n_data_workers_else), pin_memory=True,
                                              sampler=aux_data_sampler, drop_last=False, persistent_workers=False)
            self.aux_n_iters = int(math.ceil(len(self.aux_dataset) / (self.args.batch_size * len(self.args.gpus))))

        print('FLRunner on device {} creating model...'.format(self.rank))
        self.model = FLModel(self.args, self.gamestate_vocab)

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
            if self.mode != 'train':
                self.start_epoch = model_epoch
                self.n_epochs = model_epoch + 1
            elif self.mode == 'train' and self.warm_start:
                self.start_epoch += 1

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
            # if self.rank == 0:
            #     for idx, (n, p) in enumerate(self.model.named_parameters()):
            #         print('{}: {}'.format(idx, n))

            for n, p in param_optimizer:
                if any(nd in n for nd in no_decay):
                    no_decay_parms.append(p)
                else:
                    reg_parms.append(p)

            optimizer_grouped_parameters = [
                {'params': reg_parms, 'weight_decay': self.l2},
                {'params': no_decay_parms, 'weight_decay': 0.0},
            ]
            if self.rank == 0:
                print('n parms: {}'.format(len(param_optimizer)))
                print('len(optimizer_grouped_parameters[0]): {}'.format(len(optimizer_grouped_parameters[0]['params'])))
                print('len(optimizer_grouped_parameters[1]): {}'.format(len(optimizer_grouped_parameters[1]['params'])))

            if self.rank == 0:
                print('Making Adam optimizer...')
            # self.optimizer = optim.Adam(optimizer_grouped_parameters, lr=self.lr,
            #                             betas=(0.9, 0.95))  # , betas=(0.9, 0.95)
            self.optimizer = optim.AdamW(optimizer_grouped_parameters, lr=self.lr,
                                         betas=(0.9, 0.95))

            if self.n_warmup_iters > 0:
                self.scheduler = get_constant_schedule_with_warmup(self.optimizer,
                                                                   num_warmup_steps=self.n_warmup_iters)

        if self.reduced_event_map:
            pitch_event_map_fp = getattr(self.args, 'reduced_pitch_event_map_fp',
                                         '/home/czh/nvme1/SportsAnalytics/config/reduced_pitch_event_id_mapping.json')
            event_count_fp = '/home/czh/nvme1/SportsAnalytics/config/reduced_event_counts.json'
        else:
            pitch_event_map_fp = getattr(self.args, 'pitch_event_map_fp',
                                         '/home/czh/nvme1/SportsAnalytics/config/pitch_event_id_mapping.json')
            event_count_fp = '/home/czh/nvme1/SportsAnalytics/config/event_counts.json'

        pitch_type_config_fp = getattr(self.args, 'pitch_type_config_fp',
                                       '/home/czh/nvme1/SportsAnalytics/config/pitch_type_id_mapping.json')

        self.pitch_event_map = json.load(open(pitch_event_map_fp))
        # event_counts_d = json.load(open(event_count_fp))
        # self.event_counts = [event_counts_d[k] for k in self.pitch_event_map.keys()]
        # event_count_threshold = int(max(self.event_counts) * self.max_weight_ratio)
        # self.event_counts = [max(ec, event_count_threshold) for ec in self.event_counts]
        # # self.event_counts = [v for v in event_counts_d.values()]
        # self.event_counts.insert(0, 1)
        # self.event_counts.insert(0, 1)
        self.pitch_type_map = json.load(open(pitch_type_config_fp))

        # gsd_counts_fp = '/home/czh/nvme1/SportsAnalytics/config/gsd_counts.json'
        # gsd_counts_d = json.load(open(gsd_counts_fp))
        # self.gsd_counts = [gsd_counts_d.get(str(v), 1) for v in range(len(self.gamestate_vocab))]
        # gsd_count_threshold = int(max(self.gsd_counts) * self.max_weight_ratio)
        # self.gsd_counts = [max(gc, gsd_count_threshold) for gc in self.gsd_counts]
        # self.gsd_counts.extend([1 for _ in self.gamestate_vocab.bos_vocab.keys()])
        # print('len(gsd_counts): {}'.format(len(self.gsd_counts)))

        all_columns = ['entity_id', 'pitcher_id', 'batter_id', 'label', 'label_p', 'pred', 'pred_p']
        self.pitch_event_columns = [c for c in all_columns]
        self.ptype_columns = [c for c in all_columns[1:]]
        self.gsd_columns = [c for c in all_columns]

        thrown_pitch_components = [
            'pitch_speed', 'release_x', 'release_y', 'release_z', 'spin_rate', 'extension',
            'vel_x', 'vel_y', 'vel_z', 'acc_x', 'acc_y', 'acc_z', 'plate_x', 'plate_z'
        ]
        self.thrown_pitch_columns = ['pitcher_id', 'batter_id']
        self.thrown_pitch_columns.extend(['{}_label'.format(tpc) for tpc in thrown_pitch_components])
        self.thrown_pitch_columns.extend(['{}_pred'.format(tpc) for tpc in thrown_pitch_components])

        batted_ball_components = [
            'hc_x', 'hc_y', 'hit_dist', 'launch_speed', 'launch_angle'
        ]
        self.batted_ball_columns = ['pitcher_id', 'batter_id']
        self.batted_ball_columns.extend(['{}_label'.format(bbc) for bbc in batted_ball_components])
        self.batted_ball_columns.extend(['{}_pred'.format(bbc) for bbc in batted_ball_components])

        if self.mode == 'test' and self.extended_test_logs:
            print('Adding extra log headers...')
            self.pitch_event_columns.extend(['pad', 'mask'])
            self.pitch_event_columns.extend(['{}_p'.format(class_name) for class_name in self.pitch_event_map.keys()])
            self.ptype_columns.extend(['pad', 'mask'])
            self.ptype_columns.extend(['{}_p'.format(class_name) for class_name in self.pitch_type_map.keys()])
            self.gsd_columns.extend(['{}_p'.format(class_name) for class_name in self.gamestate_vocab.vocab.keys()])
            # if self.boab_can_be_masked:
            self.gsd_columns.extend(['{}_p'.format(class_name) for class_name in self.gamestate_vocab.bos_vocab.keys()])

        if self.mode == 'identify':
            print('self.out: {}'.format(self.out))
            if not os.path.exists(self.out) and self.rank == 0:
                os.makedirs(self.out)
            self.out_season_dir_tmplt = os.path.join(args.out, '{}')
            self.out_filename_tmplt = '{}-{}-{}.npy'

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
                    dist.barrier()

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

        n_total_iters = (self.n_epochs + 1) * n_iters
        context_gsd_data = []
        context_event_data = []
        context_ptype_data = []
        completion_gsd_data = []
        completion_event_data = []
        completion_ptype_data = []

        context_thrown_pitch_data = []
        context_batted_ball_data = []
        completion_thrown_pitch_data = []
        completion_batted_ball_data = []

        event_criterion = None
        gsd_criterion = None
        # return_real_mask = True if mode == 'test' else False
        return_real_mask = False if mode == 'train' else True
        do_masking = False if self.mode == 'identify' else True
        # return_real_mask = False
        last_batch_end_time = None
        iter_since_grad_accum = 1
        for batch_idx, batch_data in enumerate(dataset):
            # if batch_idx > 5:
            #     print('breaking!')
            #     break

            global_item_idx = (epoch * n_iters) + batch_idx
            batch_start_time = time.time()
            # for k, v in batch_data.items():
            #     print('\tk: {} type(v): {}'.format(k, type(v)))
            context_inputs = {
                k[len('context_'):]: v.to(self.device, non_blocking=True) for k, v in batch_data.items()
                if k.startswith('context_') and not k == 'context_masked_indices'
            }
            # context_inputs['pitch_event_labels'] = context_inputs['pitch_event_ids']
            # context_inputs['pitch_type_labels'] = context_inputs['pitch_types']
            if not self.context_only:
                completion_inputs = {
                    k[len('completion_'):]: v.to(self.device, non_blocking=True) for k, v in batch_data.items()
                    if k.startswith('completion_') and not k == 'completion_masked_indices'
                }
            else:
                completion_inputs = {}
            # completion_inputs['pitch_event_labels'] = completion_inputs['pitch_event_ids']
            # completion_inputs['pitch_type_labels'] = completion_inputs['pitch_types']
            # print("context_inputs['pitch_event_labels']: {}".format(context_inputs['pitch_event_labels']))
            context_inputs['rv_thrown_pitch_data_labels'] = context_inputs['rv_thrown_pitch_data'].detach().clone()
            context_inputs['rv_batted_ball_data_labels'] = context_inputs['rv_batted_ball_data'].detach().clone()
            if not self.context_only:
                completion_inputs['rv_thrown_pitch_data_labels'] = completion_inputs['rv_thrown_pitch_data'].detach().clone()
                completion_inputs['rv_batted_ball_data_labels'] = completion_inputs['rv_batted_ball_data'].detach().clone()

            if self.use_player_id:
                custom_entity_id = batch_data['custom_entity_id'].to(self.device, non_blocking=True)
            else:
                custom_entity_id = None

            if mode == 'test' and self.use_explicit_test_masked_indices:
                context_masked_indices = batch_data['context_masked_indices'].to(self.device, non_blocking=True)
                completion_masked_indices = batch_data['completion_masked_indices'].to(self.device, non_blocking=True)
            else:
                context_masked_indices = None
                completion_masked_indices = None

            if self.player_cls:
                # custom_pitcher_id = batch_data['custom_pitcher_id'].to(self.device, non_blocking=True)
                # custom_batter_ids = batch_data['custom_batter_ids'].to(self.device, non_blocking=True)
                custom_player_ids = batch_data['custom_player_ids'].to(self.device, non_blocking=True)

            else:
                custom_pitcher_id = None
                custom_batter_ids = None
                custom_player_ids = None

            if self.rank == 0 and epoch == 0 and batch_idx == 0:  # and epoch == 0 and batch_idx == 0
                print('Context inputs:')
                for k, v in context_inputs.items():
                    print('\tk: {} v: {} min: {} max: {}'.format(k, v.shape, v.min(), v.max()))
                print('Completion inputs:')
                for k, v in completion_inputs.items():
                    print('\tk: {} v: {} min: {} max: {}'.format(k, v.shape, v.min(), v.max()))

                if context_masked_indices is not None:
                    print('context_masked_indices:\n{}'.format(context_masked_indices))
                    print('context pct masked: {}'.format(context_masked_indices.sum() /
                                                          (context_masked_indices.shape[0] *
                                                           context_masked_indices.shape[1])))
                    print('completion_masked_indices:\n{}'.format(completion_masked_indices))
                    print('completion pct masked: {}'.format(completion_masked_indices.sum() /
                                                             (completion_masked_indices.shape[0] *
                                                              completion_masked_indices.shape[1])))
                # if custom_pitcher_id is not None and custom_batter_ids is not None:
                #     print('\tcustom_pitcher_id: {}'.format(custom_pitcher_id.shape))
                #     print('\tcustom_batter_ids: {}'.format(custom_batter_ids.shape))
                if custom_player_ids is not None:
                    print('\tcustom_player_ids: {}'.format(custom_player_ids.shape))

            # input('okty')
            entity_type_ids = batch_data['entity_type_id'].to(self.device, non_blocking=True).view(-1)
            mask_seed = self.args.seed if mode == 'test' else None
            if self.torch_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(
                        context_inputs, completion_inputs,
                        return_real_mask=return_real_mask, mask_seed=mask_seed,
                        custom_player_ids=custom_player_ids,
                        entity_type_ids=entity_type_ids,
                        return_embds=self.mode == 'identify',
                        do_masking=do_masking
                    )
            else:
                with torch.autograd.detect_anomaly():
                    outputs = self.model(
                        context_inputs, completion_inputs,
                        return_real_mask=return_real_mask, mask_seed=mask_seed,
                        custom_player_ids=custom_player_ids,
                        entity_type_ids=entity_type_ids,
                        return_embds=self.mode == 'identify',
                        do_masking=do_masking
                    )

            context_gsd_xent_loss, context_gsd_preds = outputs[0], None if outputs[1] is None else outputs[1].detach()
            context_event_xent_loss, context_event_preds = outputs[2], None if outputs[3] is None else outputs[3].detach()
            context_ptype_xent_loss, context_ptype_preds = outputs[4], None if outputs[5] is None else outputs[5].detach()
            context_masked_idxs = outputs[6]

            completion_gsd_xent_loss, completion_gsd_preds = outputs[7], None if outputs[8] is None else outputs[8].detach()
            completion_event_xent_loss, completion_event_preds = outputs[9], None if outputs[10] is None else outputs[10].detach()
            completion_ptype_xent_loss, completion_ptype_preds = outputs[11], None if outputs[12] is None else outputs[12].detach()
            completion_masked_idxs = outputs[13]

            context_thrown_pitch_data_loss, context_thrown_pitch_data_preds = outputs[14], outputs[15]
            context_batted_ball_data_loss, context_batted_ball_data_preds = outputs[16], outputs[17]
            completion_thrown_pitch_data_loss, completion_thrown_pitch_data_preds = outputs[18], outputs[19]
            completion_batted_ball_data_loss, completion_batted_ball_data_preds = outputs[20], outputs[21]
            context_masked_pitcher_idxs, context_masked_batter_idxs = outputs[22], outputs[23]
            completion_masked_pitcher_idxs, completion_masked_batter_idxs = outputs[24], outputs[25]

            context_masked_entity_types = entity_type_ids.view(-1, 1).expand(-1, self.context_max_len)[context_masked_idxs]
            completion_masked_entity_types = entity_type_ids.view(-1, 1).expand(-1, self.completion_max_len)[completion_masked_idxs]

            if self.attn_mask_type == 'bidirectional':
                context_gsd_labels = context_inputs['state_delta_labels'].detach()[context_masked_idxs]
                context_event_labels = context_inputs['pitch_event_labels'].detach()[context_masked_idxs]
                context_ptype_labels = context_inputs['pitch_type_labels'].detach()[context_masked_idxs][context_masked_pitcher_idxs]
                # context_pad_mask = context_inputs['model_src_pad_mask'].detach()
                # print('context_event_labels: {}'.format(context_event_labels))

                context_pitcher_ids = context_inputs['pitcher_id'].detach()[context_masked_idxs]
                context_batter_ids = context_inputs['batter_id'].detach()[context_masked_idxs]

                context_thrown_pitch_labels = \
                context_inputs['rv_thrown_pitch_data_labels'].detach()[context_masked_idxs][context_masked_pitcher_idxs]
                context_batted_ball_labels = context_inputs['rv_batted_ball_data_labels'].detach()[context_masked_idxs][
                    context_masked_batter_idxs]

                if not self.context_only:
                    completion_gsd_labels = completion_inputs['state_delta_labels'].detach()[completion_masked_idxs]
                    completion_event_labels = completion_inputs['pitch_event_labels'].detach()[completion_masked_idxs]
                    completion_ptype_labels = completion_inputs['pitch_type_labels'].detach()[completion_masked_idxs][completion_masked_pitcher_idxs]
                    # completion_pad_mask = completion_inputs['model_src_pad_mask'].detach()

                    completion_pitcher_ids = completion_inputs['pitcher_id'].detach()[completion_masked_idxs]
                    completion_batter_ids = completion_inputs['batter_id'].detach()[completion_masked_idxs]

                    completion_thrown_pitch_labels = completion_inputs['rv_thrown_pitch_data_labels'].detach()[completion_masked_idxs][completion_masked_pitcher_idxs]
                    completion_batted_ball_labels = completion_inputs['rv_batted_ball_data_labels'].detach()[completion_masked_idxs][completion_masked_batter_idxs]

            else:
                indicator = context_inputs['model_src_pad_mask'] == 0
                indicator2 = context_inputs['state_delta_labels'].detach() != self.gamestate_vocab.boab_id
                indicator = indicator * indicator2
                context_gsd_labels = context_inputs['state_delta_labels'].detach()[indicator]
                context_event_labels = context_inputs['pitch_event_labels'].detach()[indicator]
                context_ptype_labels = context_inputs['pitch_type_labels'].detach()[indicator]

                context_pitcher_ids = context_inputs['pitcher_id'].detach()[indicator]
                context_batter_ids = context_inputs['batter_id'].detach()[indicator]

                indicator = completion_inputs['model_src_pad_mask'] == 0
                indicator2 = completion_inputs['state_delta_labels'].detach() != self.gamestate_vocab.boab_id
                indicator = indicator * indicator2
                completion_gsd_labels = completion_inputs['state_delta_labels'].detach()[indicator]
                completion_event_labels = completion_inputs['pitch_event_labels'].detach()[indicator]
                completion_ptype_labels = completion_inputs['pitch_type_labels'].detach()[indicator]

                completion_pitcher_ids = completion_inputs['pitcher_id'].detach()[indicator]
                completion_batter_ids = completion_inputs['batter_id'].detach()[indicator]

            context_avg_n_masked = None
            if context_masked_idxs is not None:
                context_item_n_masked = context_masked_idxs.sum()
                context_avg_n_masked = context_item_n_masked / context_masked_idxs.shape[0]

            completion_avg_n_masked = None
            if completion_masked_idxs is not None:
                completion_item_n_masked = completion_masked_idxs.sum()
                completion_avg_n_masked = completion_item_n_masked / completion_masked_idxs.shape[0]

            # print('context_ptype_labels min: {} max: {}'.format(context_ptype_labels.min(), context_ptype_labels.max()))
            # print('context_ptype_labels: {}'.format(context_ptype_labels))
            # print('context_ptype_labels[0]: {}'.format(context_ptype_labels[0]))

            # input('okty')

            ptype_hot = True if self.use_ball_data and (context_ptype_xent_loss is not None
                                                        or completion_ptype_xent_loss is not None
                                                        or context_thrown_pitch_data_loss is not None
                                                        or context_batted_ball_data_loss is not None) else False

            if mode == 'identify':
                loss = torch.tensor(-1.0)
            else:
                if self.context_only:
                    gsd_xent_loss = context_gsd_xent_loss
                    event_xent_loss = context_event_xent_loss

                    xent_loss = (gsd_xent_loss * self.gsd_weight_pct) + (event_xent_loss * self.event_weight_pct)

                    if ptype_hot:
                        if context_ptype_xent_loss is not None:
                            ptype_xent_loss = context_ptype_xent_loss
                            xent_loss += (ptype_xent_loss * self.ptype_weight_pct)

                        if context_thrown_pitch_data_loss is not None:
                            thrown_pitch_loss = context_thrown_pitch_data_loss
                            xent_loss += (thrown_pitch_loss * self.thrown_pitch_weight_pct)

                        if context_batted_ball_data_loss is not None:
                            batted_ball_loss = context_batted_ball_data_loss
                            xent_loss += (batted_ball_loss * self.batted_ball_weight_pct)
                else:
                    gsd_xent_loss = (context_gsd_xent_loss + completion_gsd_xent_loss) / 2
                    event_xent_loss = (context_event_xent_loss + completion_event_xent_loss) / 2

                    xent_loss = (gsd_xent_loss * self.gsd_weight_pct) + (event_xent_loss * self.event_weight_pct)

                    if ptype_hot:
                        if context_ptype_xent_loss is not None and completion_ptype_xent_loss is not None:
                            ptype_xent_loss = (context_ptype_xent_loss + completion_ptype_xent_loss) / 2
                            xent_loss += (ptype_xent_loss * self.ptype_weight_pct)

                        if context_thrown_pitch_data_loss is not None and completion_thrown_pitch_data_loss is not None:
                            thrown_pitch_loss = (context_thrown_pitch_data_loss + completion_thrown_pitch_data_loss) / 2
                            xent_loss += (thrown_pitch_loss * self.thrown_pitch_weight_pct)

                        if context_batted_ball_data_loss is not None and completion_batted_ball_data_loss is not None:
                            batted_ball_loss = (context_batted_ball_data_loss + completion_batted_ball_data_loss) / 2
                            xent_loss += (batted_ball_loss * self.batted_ball_weight_pct)

                loss = xent_loss

                if mode == 'train':
                    if self.torch_amp:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()

            if global_item_idx % self.log_every == 0 or (mode == 'train-dev'
                                                         and global_item_idx % int(self.log_every / 5) == 0) \
                    or torch.isnan(loss) or self.mode == 'identify':
                if mode != 'identify':
                    if self.do_group_gathers:
                        context_gsd_preds, context_gsd_labels, context_event_preds, context_event_labels = self.group_gather([context_gsd_preds, context_gsd_labels, context_event_preds, context_event_labels])
                    else:
                        context_gsd_preds = self.gather(context_gsd_preds)
                        context_gsd_labels = self.gather(context_gsd_labels)
                        context_event_preds = self.gather(context_event_preds)
                        context_event_labels = self.gather(context_event_labels)

                    if not self.context_only:
                        if self.do_group_gathers:
                            completion_gsd_preds, completion_gsd_labels, completion_event_preds, completion_event_labels = self.group_gather([completion_gsd_preds, completion_gsd_labels, completion_event_preds, completion_event_labels])
                        else:
                            completion_gsd_preds = self.gather(completion_gsd_preds)
                            completion_gsd_labels = self.gather(completion_gsd_labels)
                            completion_event_preds = self.gather(completion_event_preds)
                            completion_event_labels = self.gather(completion_event_labels)

                    if ptype_hot:
                        if context_ptype_preds is not None:
                            if self.do_group_gathers:
                                context_ptype_preds, context_ptype_labels = self.group_gather([context_ptype_preds, context_ptype_labels])
                            else:
                                context_ptype_preds = self.gather(context_ptype_preds)
                                context_ptype_labels = self.gather(context_ptype_labels)
                        if completion_ptype_preds is not None:
                            if self.do_group_gathers:
                                completion_ptype_preds, completion_ptype_labels = self.group_gather([completion_ptype_preds, completion_ptype_labels])
                            else:
                                completion_ptype_preds = self.gather(completion_ptype_preds)
                                completion_ptype_labels = self.gather(completion_ptype_labels)

                        if context_thrown_pitch_data_preds is not None:
                            if self.do_group_gathers:
                                context_thrown_pitch_data_preds, context_thrown_pitch_labels = self.group_gather([context_thrown_pitch_data_preds, context_thrown_pitch_labels])
                            else:
                                context_thrown_pitch_data_preds = self.gather(context_thrown_pitch_data_preds)
                                context_thrown_pitch_labels = self.gather(context_thrown_pitch_labels)

                        if completion_thrown_pitch_data_preds is not None:
                            if self.do_group_gathers:
                                completion_thrown_pitch_data_preds, completion_thrown_pitch_labels = self.group_gather([completion_thrown_pitch_data_preds, completion_thrown_pitch_labels])
                            else:
                                completion_thrown_pitch_data_preds = self.gather(completion_thrown_pitch_data_preds)
                                completion_thrown_pitch_labels = self.gather(completion_thrown_pitch_labels)

                        if context_batted_ball_data_preds is not None:
                            if self.do_group_gathers:
                                context_batted_ball_data_preds, context_batted_ball_labels = self.group_gather([context_batted_ball_data_preds, context_batted_ball_labels])
                            else:
                                context_batted_ball_data_preds = self.gather(context_batted_ball_data_preds)
                                context_batted_ball_labels = self.gather(context_batted_ball_labels)

                        if completion_batted_ball_data_preds is not None:
                            if self.do_group_gathers:
                                completion_batted_ball_data_preds, completion_batted_ball_labels = self.group_gather([completion_batted_ball_data_preds, completion_batted_ball_labels])
                            else:
                                completion_batted_ball_data_preds = self.gather(completion_batted_ball_data_preds)
                                completion_batted_ball_labels = self.gather(completion_batted_ball_labels)

                # context_masked_idxs = self.gather(context_masked_idxs)
                context_pitcher_ids = self.gather(context_pitcher_ids)
                context_batter_ids = self.gather(context_batter_ids)
                # context_pad_mask = self.gather(context_pad_mask)
                context_masked_entity_types = self.gather(context_masked_entity_types)
                completion_masked_entity_types = self.gather(completion_masked_entity_types)

                if not self.context_only:
                    # completion_masked_idxs = self.gather(completion_masked_idxs)
                    completion_pitcher_ids = self.gather(completion_pitcher_ids)
                    completion_batter_ids = self.gather(completion_batter_ids)
                    # completion_t = self.gather(completion_t)
                    # completion_pad_mask = self.gather(completion_pad_mask)

                if do_masking and self.use_ball_data and self.predict_ball_data:
                    context_masked_pitcher_idxs = self.gather(context_masked_pitcher_idxs)
                    context_masked_batter_idxs = self.gather(context_masked_batter_idxs)
                    if not self.context_only:
                        completion_masked_pitcher_idxs = self.gather(completion_masked_pitcher_idxs)
                        completion_masked_batter_idxs = self.gather(completion_masked_batter_idxs)

                if self.mode == 'identify':
                    game_pks = batch_data['completion_first_game_pk'].to(self.device, non_blocking=True)
                    game_years = batch_data['completion_first_game_year'].to(self.device, non_blocking=True)

                    agg_embds = outputs[-1]
                    # print('\traw agg_embds: {}'.format(agg_embds.shape))
                    # agg_embds = agg_embds.view(-1, agg_embds.shape[-1])
                    # print('\treshaped agg_embds: {}'.format(agg_embds.shape))
                    player_ids = batch_data['real_player_ids'].to(self.device, non_blocking=True)
                    # player_ids = player_ids.view(-1)
                    # game_pks = torch.cat(
                    #     [game_pks.unsqueeze(-1) for _ in range(10)], dim=-1
                    # ).view(-1)
                    # game_years = torch.cat(
                    #     [game_years.unsqueeze(-1) for _ in range(10)], dim=-1
                    # ).view(-1)

                    agg_embds = self.gather(agg_embds)
                    player_ids = self.gather(player_ids)
                    if batch_idx == 0 and self.rank == 0:
                        print('agg_embds: {}'.format(agg_embds.shape))

                    if self.do_group_gathers:
                        game_pks, game_years, entity_type_ids = self.group_gather([game_pks, game_years, entity_type_ids])
                    else:
                        game_pks = self.gather(game_pks)
                        game_years = self.gather(game_years)
                        entity_type_ids = self.gather(entity_type_ids)

                    # print('player_ids: {}'.format(player_ids.shape))
                    # print('game_pks: {}'.format(game_pks.shape))
                    # print('game_years: {}'.format(game_years.shape))
                    # print('agg_embds: {}'.format(agg_embds.shape))
                    # print('entity_type: {}'.format(entity_type.shape))
                    # input('okty')

                    if self.rank == 0:
                        for item_idx in range(agg_embds.shape[0]):
                            item_entity_type = entity_type_ids[item_idx].item()
                            item_entity_type_str = self.args.reverse_entity_type_d[item_entity_type]

                            if item_entity_type_str == 'pitcher':
                                start_idx = 0
                                end_idx = 1
                            else:
                                if self.v2_player_attn:
                                    start_idx = self.v2_attn_max_n_pitcher
                                    end_idx = start_idx + 10
                                else:
                                    start_idx = 1
                                    end_idx = 10

                            # print('item_entity_type: {}'.format(item_entity_type))
                            # print('start_idx: {} end_idx: {}'.format(start_idx, end_idx))

                            for player_idx in range(start_idx, end_idx):
                                item_embd = agg_embds[item_idx, player_idx].cpu().numpy()
                                item_player_id = player_ids[item_idx, player_idx].item()
                                # print('player_idx: {}'.format(player_idx))
                                # print('item_embd: {}'.format(item_embd.shape))
                                # print('item_player_id: {}'.format(item_player_id))
                                # input('okty')

                                if item_player_id > 0:
                                    season_dir = self.out_season_dir_tmplt.format(game_years[item_idx].item())
                                    if not os.path.exists(season_dir):
                                        os.makedirs(season_dir)

                                    item_fp = os.path.join(
                                        season_dir,
                                        self.out_filename_tmplt.format(item_entity_type_str, item_player_id,
                                                                       game_pks[item_idx].item())
                                    )
                                    np.save(item_fp, item_embd)

                else:
                    batch_context_gsd_write_data = self.make_pred_write_data(context_gsd_preds, context_gsd_labels,
                                                                             context_pitcher_ids, context_batter_ids,
                                                                             entity_type_ids=context_masked_entity_types)
                    batch_context_event_write_data = self.make_pred_write_data(context_event_preds,
                                                                               context_event_labels,
                                                                               context_pitcher_ids, context_batter_ids,
                                                                               entity_type_ids=context_masked_entity_types)

                    if not self.context_only:
                        batch_completion_gsd_write_data = self.make_pred_write_data(completion_gsd_preds,
                                                                                    completion_gsd_labels,
                                                                                    completion_pitcher_ids,
                                                                                    completion_batter_ids,
                                                                                    entity_type_ids=completion_masked_entity_types)
                        batch_completion_event_write_data = self.make_pred_write_data(completion_event_preds,
                                                                                      completion_event_labels,
                                                                                      completion_pitcher_ids,
                                                                                      completion_batter_ids,
                                                                                      entity_type_ids=completion_masked_entity_types)
                    # print('context_pitcher_ids:\n\t{}'.format(context_pitcher_ids))
                    # print('batch_context_gsd_write_data:\n\t{}'.format(batch_context_gsd_write_data))
                    # print('batch_context_gsd_write_data.dtype: {}'.format(batch_context_gsd_write_data.dtype))
                    if self.rank == 0:
                        context_gsd_data.append(batch_context_gsd_write_data.cpu())
                        context_event_data.append(batch_context_event_write_data.cpu())
                        if not self.context_only:
                            completion_gsd_data.append(batch_completion_gsd_write_data.cpu())
                            completion_event_data.append(batch_completion_event_write_data.cpu())

                    if ptype_hot:
                        if context_ptype_preds is not None:
                            batch_context_ptype_write_data = self.make_pred_write_data(context_ptype_preds,
                                                                                       context_ptype_labels,
                                                                                       context_pitcher_ids[context_masked_pitcher_idxs],
                                                                                       context_batter_ids[context_masked_pitcher_idxs])
                        else:
                            batch_context_ptype_write_data = None

                        if completion_ptype_preds is not None:
                            batch_completion_ptype_write_data = self.make_pred_write_data(completion_ptype_preds,
                                                                                          completion_ptype_labels,
                                                                                          completion_pitcher_ids[completion_masked_pitcher_idxs],
                                                                                          completion_batter_ids[completion_masked_pitcher_idxs])
                        else:
                            batch_completion_ptype_write_data = None

                        if context_thrown_pitch_data_preds is not None:
                            batch_context_thrown_pitch_write_data = self.make_pred_write_data(
                                context_thrown_pitch_data_preds, context_thrown_pitch_labels,
                                context_pitcher_ids[context_masked_pitcher_idxs],
                                context_batter_ids[context_masked_pitcher_idxs],
                                rv_preds=True
                            )
                        else:
                            batch_context_thrown_pitch_write_data = None

                        if context_batted_ball_data_preds is not None:
                            batch_context_batted_ball_write_data = self.make_pred_write_data(
                                context_batted_ball_data_preds, context_batted_ball_labels,
                                context_pitcher_ids[context_masked_batter_idxs],
                                context_batter_ids[context_masked_batter_idxs],
                                rv_preds=True
                            )
                        else:
                            batch_context_batted_ball_write_data = None

                        if completion_thrown_pitch_data_preds is not None:
                            batch_completion_thrown_pitch_write_data = self.make_pred_write_data(
                                completion_thrown_pitch_data_preds, completion_thrown_pitch_labels,
                                completion_pitcher_ids[completion_masked_pitcher_idxs],
                                completion_batter_ids[completion_masked_pitcher_idxs],
                                rv_preds=True
                            )
                        else:
                            batch_completion_thrown_pitch_write_data = None

                        if completion_batted_ball_data_preds is not None:
                            batch_completion_batted_ball_write_data = self.make_pred_write_data(
                                completion_batted_ball_data_preds, completion_batted_ball_labels,
                                completion_pitcher_ids[completion_masked_batter_idxs],
                                completion_batter_ids[completion_masked_batter_idxs],
                                rv_preds=True
                            )
                        else:
                            batch_completion_batted_ball_write_data = None

                        if self.rank == 0:
                            if batch_context_ptype_write_data is not None:
                                context_ptype_data.append(batch_context_ptype_write_data.cpu())

                            if batch_completion_ptype_write_data is not None:
                                completion_ptype_data.append(batch_completion_ptype_write_data.cpu())

                            # print('batch_context_thrown_pitch_write_data: {}'.format(batch_context_thrown_pitch_write_data.shape))
                            # print('batch_context_batted_ball_write_data: {}'.format(batch_context_batted_ball_write_data.shape))
                            # print('batch_completion_thrown_pitch_write_data: {}'.format(batch_completion_thrown_pitch_write_data.shape))
                            # print('batch_completion_batted_ball_write_data: {}'.format(batch_completion_batted_ball_write_data.shape))

                            if batch_context_thrown_pitch_write_data is not None:
                                context_thrown_pitch_data.append(batch_context_thrown_pitch_write_data.cpu())
                            if batch_context_batted_ball_write_data is not None:
                                context_batted_ball_data.append(batch_context_batted_ball_write_data.cpu())
                            if batch_completion_thrown_pitch_write_data is not None:
                                completion_thrown_pitch_data.append(batch_completion_thrown_pitch_write_data.cpu())
                            if batch_completion_batted_ball_write_data is not None:
                                completion_batted_ball_data.append(batch_completion_batted_ball_write_data.cpu())

                if torch.isnan(loss):
                    print('*** NAN LOSS ***')
                    print('custom_player_ids: {}'.format(custom_player_ids))

                    for i in range(completion_batter_ids.shape[0]):
                        completion_batter_ids[i] = self.player_id_map[int(completion_batter_ids[i].item())]
                        completion_pitcher_ids[i] = self.player_id_map[int(completion_pitcher_ids[i].item())]
                    # print('context_gsd_preds:\n{}'.format(context_gsd_preds))
                    # print('context_gsd_labels:\n{}'.format(context_gsd_labels))
                    #
                    # sm = nn.Softmax(dim=-1)
                    # context_gsd_probs = sm(context_gsd_preds)
                    # print('context_gsd_probs:\n{}'.format(context_gsd_probs))
                    #
                    # context_gsd_labels = context_gsd_labels.unsqueeze(-1)
                    #
                    # label_probs = torch.gather(context_gsd_probs, -1, context_gsd_labels)
                    # print('label_probs:\n{}\n\tmin: {} max: {}'.format(label_probs,
                    #                                                    label_probs.min(),
                    #                                                    label_probs.max()))
                    # labels_w_min_prob = context_gsd_labels[context_gsd_probs == label_probs.min()]
                    # labels_w_max_prob = context_gsd_labels[context_gsd_probs == label_probs.max()]
                    # print('labels_w_min_prob: {}'.format(labels_w_min_prob))
                    # print('labels_w_max_prob: {}'.format(labels_w_max_prob))
                    sm = nn.Softmax(dim=-1)
                    completion_gsd_probs = sm(completion_gsd_preds)
                    completion_gsd_labels = completion_gsd_labels.unsqueeze(-1)
                    print('completion_gsd_labels: {}'.format(completion_gsd_labels.shape))

                    gsd_prob_sum = completion_gsd_probs.sum(-1)
                    bad_pred_idxs = torch.isnan(gsd_prob_sum).nonzero()
                    print('bad_pred_idxs: {}'.format(bad_pred_idxs.shape))
                    print('labels w bad preds: {}'.format(completion_gsd_labels[bad_pred_idxs]))
                    print('batters w bad preds: {}'.format(completion_batter_ids.view(-1, 1)[bad_pred_idxs]))
                    print('pitchers w bad preds: {}'.format(completion_pitcher_ids.view(-1, 1)[bad_pred_idxs]))
                    # print('t w bad preds: {}'.format(completion_t.view(-1, 1)[bad_pred_idxs]))
                    print('context game_pk_t:\n{}'.format(batch_data['context_game_pk_t']))
                    print('completion game_pk_t:\n{}'.format(batch_data['completion_game_pk_t']))

            # input('okty')
            if global_item_idx % self.args.print_every == 0 and self.rank == 0:
                batch_elapsed_time = time.time() - batch_start_time
                if last_batch_end_time is not None:
                    time_btw_batches = batch_start_time - last_batch_end_time
                else:
                    time_btw_batches = 0.0

                print_str = '{0}- epoch:{1}/{2} iter:{3}/{4} loss:{5:2.2f}'
                print_str = print_str.format(mode, epoch, self.n_epochs, batch_idx, n_iters, loss)

                print_str = '{0} GSD xent: {1:.2f}/{2:.2f} Event xent: {3:.2f}/{4:.2f} Ptype xent: {5:.2f}/{6:.2f}'.format(
                    print_str,
                    -1.0 if context_gsd_xent_loss is None else context_gsd_xent_loss,
                    -1.0 if completion_gsd_xent_loss is None else completion_gsd_xent_loss,
                    -1.0 if context_event_xent_loss is None else context_event_xent_loss,
                    -1.0 if completion_event_xent_loss is None else completion_event_xent_loss,
                    -1.0 if context_ptype_xent_loss is None else context_ptype_xent_loss,
                    -1.0 if completion_ptype_xent_loss is None else completion_ptype_xent_loss,
                )

                print_str = '{0} Pitch RV: {1:.2f}/{2:.2f} Ball RV: {3:.2f}/{4:.2f}'.format(
                    print_str,
                    -1.0 if context_thrown_pitch_data_loss is None else context_thrown_pitch_data_loss,
                    -1.0 if completion_thrown_pitch_data_loss is None else completion_thrown_pitch_data_loss,
                    -1.0 if context_batted_ball_data_loss is None else context_batted_ball_data_loss,
                    -1.0 if completion_batted_ball_data_loss is None else completion_batted_ball_data_loss,
                )

                if context_avg_n_masked is not None and completion_avg_n_masked is not None:
                    print_str = '{0} N Mask: {1:.2f} / {2:.2f}'.format(print_str, context_avg_n_masked,
                                                                       completion_avg_n_masked)

                u_entity_types, entity_type_counts = np.unique(entity_type_ids.cpu().numpy(), return_counts=True)
                n_total_entities = sum(entity_type_counts)
                entity_type_counts = [etc / n_total_entities for etc in entity_type_counts]
                entity_count_d = dict(zip(u_entity_types, entity_type_counts))

                for entity_type in self.type:
                    entity_id = self.args.entity_type_d[entity_type]
                    entity_pct = entity_count_d.get(entity_id, 0.0)
                    print_str = '{0} {1}: {2:.2f}%'.format(print_str, entity_type, entity_pct * 100)

                print_str = '{0} Time: {1:.2f}s ({2:.2f}s)'.format(print_str, batch_elapsed_time, time_btw_batches)
                print(print_str)

            if (global_item_idx % self.args.summary_every == 0 and self.summary_writer is not None) or (
                    mode == 'train-dev' and self.summary_writer is not None and global_item_idx % int(
                self.args.summary_every / 3) == 0):

                self.summary_writer.add_scalar('loss/{}'.format(mode), loss, global_item_idx)
                self.summary_writer.add_scalar('context_gsd_xent_loss/{}'.format(mode), context_gsd_xent_loss,
                                               global_item_idx)
                self.summary_writer.add_scalar('context_event_xent_loss/{}'.format(mode), context_event_xent_loss,
                                               global_item_idx)

                if not self.context_only:
                    self.summary_writer.add_scalar('completion_gsd_xent_loss/{}'.format(mode), completion_gsd_xent_loss,
                                                   global_item_idx)
                    self.summary_writer.add_scalar('completion_event_xent_loss/{}'.format(mode), completion_event_xent_loss,
                                                   global_item_idx)

                if ptype_hot:
                    self.summary_writer.add_scalar('context_ptype_xent_loss/{}'.format(mode), context_ptype_xent_loss,
                                                   global_item_idx)
                    self.summary_writer.add_scalar('context_thrown_pitch_data_loss/{}'.format(mode),
                                                   context_thrown_pitch_data_loss,
                                                   global_item_idx)
                    self.summary_writer.add_scalar('context_batted_ball_data_loss/{}'.format(mode),
                                                   context_batted_ball_data_loss,
                                                   global_item_idx)

                    if not self.context_only:
                        self.summary_writer.add_scalar('completion_ptype_xent_loss/{}'.format(mode),
                                                       completion_ptype_xent_loss,
                                                       global_item_idx)

                        self.summary_writer.add_scalar('completion_thrown_pitch_data_loss/{}'.format(mode),
                                                       completion_thrown_pitch_data_loss,
                                                       global_item_idx)

                        self.summary_writer.add_scalar('completion_batted_ball_data_loss/{}'.format(mode),
                                                       completion_batted_ball_data_loss,
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

            if iter_since_grad_accum == self.args.n_grad_accum and mode == 'train':
                # print('OPTIMIZER STEP')
                if self.torch_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                if self.scheduler is not None:
                    self.scheduler.step()
                self.optimizer.zero_grad()
                iter_since_grad_accum = 1
            else:
                iter_since_grad_accum += 1

            last_batch_end_time = time.time()
            # input('okty')

        # self.log_fp_tmplt = os.path.join(self.log_dir,
        #              '{}_{}_{}_preds_e{}.csv')      # train/dev, cntxt/cmpltn, gsd/event/ptype, e
        if self.rank == 0:
            if len(context_gsd_data) > 0:
                print('Writing context gsd logs...')
                context_gsd_data = torch.cat(context_gsd_data).numpy()
                context_gsd_data_df = pd.DataFrame(context_gsd_data, columns=self.gsd_columns)
                context_gsd_data_df.to_csv(
                    self.log_fp_tmplt.format(mode, 'context', 'gsd', epoch), index=False
                )

            if len(context_event_data) > 0:
                print('Writing context event logs...')
                context_event_data = torch.cat(context_event_data).numpy()
                context_event_data_df = pd.DataFrame(context_event_data, columns=self.pitch_event_columns)
                context_event_data_df.to_csv(
                    self.log_fp_tmplt.format(mode, 'context', 'event', epoch), index=False
                )

            if len(context_ptype_data) > 0:
                print('Writing context ptype logs...')
                context_ptype_data = torch.cat(context_ptype_data).numpy()
                context_ptype_data_df = pd.DataFrame(context_ptype_data, columns=self.ptype_columns)
                context_ptype_data_df.to_csv(
                    self.log_fp_tmplt.format(mode, 'context', 'ptype', epoch), index=False
                )

            if len(context_thrown_pitch_data) > 0:
                print('Writing context thrown pitch data...')
                context_thrown_pitch_data = torch.cat(context_thrown_pitch_data).numpy()
                context_thrown_pitch_data_df = pd.DataFrame(context_thrown_pitch_data, columns=self.thrown_pitch_columns)
                context_thrown_pitch_data_df.to_csv(
                    self.log_fp_tmplt.format(mode, 'context', 'thrown_pitch', epoch), index=False
                )

            if len(context_batted_ball_data) > 0:
                print('Writing context batted ball data...')
                context_batted_ball_data = torch.cat(context_batted_ball_data).numpy()
                context_batted_ball_data_df = pd.DataFrame(context_batted_ball_data, columns=self.batted_ball_columns)
                context_batted_ball_data_df.to_csv(
                    self.log_fp_tmplt.format(mode, 'context', 'batted_ball', epoch), index=False
                )

            if len(completion_gsd_data) > 0:
                print('Writing completion gsd logs...')
                completion_gsd_data = torch.cat(completion_gsd_data).numpy()
                completion_gsd_data_df = pd.DataFrame(completion_gsd_data, columns=self.gsd_columns)
                completion_gsd_data_df.to_csv(
                    self.log_fp_tmplt.format(mode, 'completion', 'gsd', epoch), index=False
                )

            if len(completion_event_data) > 0:
                print('Writing completion event logs...')
                completion_event_data = torch.cat(completion_event_data).numpy()
                completion_event_data_df = pd.DataFrame(completion_event_data, columns=self.pitch_event_columns)
                completion_event_data_df.to_csv(
                    self.log_fp_tmplt.format(mode, 'completion', 'event', epoch), index=False
                )

            if len(completion_ptype_data) > 0:
                print('Writing completion ptype logs...')
                completion_ptype_data = torch.cat(completion_ptype_data).numpy()
                completion_ptype_data_df = pd.DataFrame(completion_ptype_data, columns=self.ptype_columns)
                completion_ptype_data_df.to_csv(
                    self.log_fp_tmplt.format(mode, 'completion', 'ptype', epoch), index=False
                )

            if len(completion_thrown_pitch_data) > 0:
                print('Writing completion thrown pitch data...')
                completion_thrown_pitch_data = torch.cat(completion_thrown_pitch_data).numpy()
                completion_thrown_pitch_data_df = pd.DataFrame(completion_thrown_pitch_data,
                                                               columns=self.thrown_pitch_columns)
                completion_thrown_pitch_data_df.to_csv(
                    self.log_fp_tmplt.format(mode, 'completion', 'thrown_pitch', epoch), index=False
                )

            if len(completion_batted_ball_data) > 0:
                print('Writing completion batted ball data...')
                completion_batted_ball_data = torch.cat(completion_batted_ball_data).numpy()
                completion_batted_ball_data_df = pd.DataFrame(completion_batted_ball_data,
                                                              columns=self.batted_ball_columns)
                completion_batted_ball_data_df.to_csv(
                    self.log_fp_tmplt.format(mode, 'completion', 'batted_ball', epoch), index=False
                )

            print('\tAll done writing logs!')

    def make_pred_write_data(self, pred_logits, labels, pitcher_ids, batter_ids, rv_preds=False, entity_type_ids=None):
        if self.attn_mask_type != 'bidirectional':
            pred_logits = pred_logits.view(-1, pred_logits.shape[-1])
            labels = labels.view(-1)
            pitcher_ids = pitcher_ids.view(-1)
            batter_ids = batter_ids.view(-1)

        if rv_preds:
            write_data_components = [
                pitcher_ids.unsqueeze(-1), batter_ids.unsqueeze(-1),
                labels, pred_logits,
            ]
        else:
            softmax = nn.Softmax(dim=-1)
            _, pred_ids = torch.topk(pred_logits, 1)
            # if self.ldam_loss:
            #     all_pred_probs = softmax(pred_logits * self.ldam_s)
            # else:
            all_pred_probs = softmax(pred_logits)
            label_probs = torch.gather(all_pred_probs, -1, labels.unsqueeze(-1).type(torch.int64))
            pred_probs = torch.gather(all_pred_probs, -1, pred_ids.type(torch.int64))
            if entity_type_ids is not None:
                write_data_components = [entity_type_ids.unsqueeze(-1), pitcher_ids.unsqueeze(-1), batter_ids.unsqueeze(-1),
                                         labels.unsqueeze(-1), label_probs, pred_ids, pred_probs.float()]
                # for wdc_idx, wdc in enumerate(write_data_components):
                #     print('\nwdc {}: {}'.format(wdc_idx, wdc.shape))
            else:
                write_data_components = [pitcher_ids.unsqueeze(-1), batter_ids.unsqueeze(-1),
                                         labels.unsqueeze(-1), label_probs, pred_ids, pred_probs.float()]
            if self.mode == 'test' and self.extended_test_logs:
                write_data_components.append(all_pred_probs.float())
        write_data = torch.cat(write_data_components, dim=-1)
        return write_data

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
        # print('\tx on device {}: {}'.format(self.rank, x.shape))
        n_x = torch.tensor([x.shape[0]], device=x.device)
        # print('\tn_x on device {}: {}'.format(self.rank, n_x))
        n_x_list = [torch.zeros_like(n_x) for _ in range(self.world_size)]
        dist.all_gather(n_x_list, n_x)
        n_x = torch.cat(n_x_list, dim=0).contiguous()
        # dist.barrier()
        max_size = n_x.max() + 1

        indicator = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
        # print('\tindicator on device {}: {}'.format(self.rank, indicator.shape))

        if x.shape[0] != max_size:
            # print('\tPadding x on device {} (raw shape: {})'.format(self.rank, x.shape))
            x_padding = torch.zeros(max_size - x.shape[0], *x.shape[1:], device=x.device, dtype=x.dtype)
            indicator_padding = torch.zeros(max_size - x.shape[0], device=x.device, dtype=torch.bool)

            x = torch.cat([x, x_padding], dim=0).contiguous()
            # print('\t\tNew shape of x: {}'.format(x.shape))
            indicator = torch.cat([indicator, indicator_padding], dim=0).contiguous()
            # print('\t\tNew shape of indicator: {}'.format(indicator.shape))

        # dist.barrier()
        x_list = [torch.zeros_like(x) for _ in range(self.world_size)]
        dist.all_gather(x_list, x)
        x = torch.cat(x_list, dim=0).contiguous()
        # if self.rank == 0:
        #     print('\tRaw aggregated x: {}'.format(x.shape))

        indicator_list = [torch.zeros_like(indicator) for _ in range(self.world_size)]
        dist.all_gather(indicator_list, indicator)
        indicator = torch.cat(indicator_list, dim=0).contiguous()
        # if self.rank == 0:
        #     print('\tAggregated indicator: {}'.format(indicator.shape))
        # dist.barrier()
        # print('\tmoving to cpu...')
        # x = x.cpu()
        # print('\tx on cpu...')
        # indicator = indicator.cpu()
        # print('\tindicator on cpu...')

        x = x[indicator == 1]
        # if self.rank == 0:
        #     print('\tAggregated x after considering indicator: {}'.format(x.shape))

        # dist.barrier()
        # print('\treturning')
        return x