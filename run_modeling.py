__author__ = 'Connor Heaton'

import argparse
import datetime
import torch
import os

import numpy as np
import torch.multiprocessing as mp

from src import Runner
from src.utils import str2bool


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ab_data', default='/home/czh/md0/ab_seqs/ab_seqs_v17',
                        help='Dir where data can be found')
    parser.add_argument('--career_data',
                        default='/home/czh/sata1/SportsAnalytics/player_career_data_08182022',
                        help='Where to find player career data')
    parser.add_argument('--model_type', default='mt')
    parser.add_argument('--type', default=['team_batting', 'pitcher'],
                        type=str, nargs='+',
                        help='Type of data to model. batter, pitcher, team_batting or team_fielding')

    parser.add_argument('--do_group_gathers', default=False, type=str2bool)
    parser.add_argument('--predict_pitches', default=False, type=str2bool)
    parser.add_argument('--predict_ball_data', default=False, type=str2bool)
    parser.add_argument('--player_cls', default=False, type=str2bool)
    parser.add_argument('--use_intermediate_data', default=False, type=str2bool)
    # parser.add_argument('--record_norm_values_fp',
    # default='/home/czh/nvme1/SportsAnalytics/config/max_vals_08162022.json')
    parser.add_argument('--pitch_event_map_fp',
                        default='../config/pitch_event_id_mapping.json')
    parser.add_argument('--reduced_pitch_event_map_fp',
                        default='../config/reduced_pitch_event_id_mapping.json')
    parser.add_argument('--intermediate_pitch_event_map_fp',
                        default='../config/pitch_event_intermediate_id_mapping.json')
    parser.add_argument('--explicit_test_set_dir',
                        default='/home/czh/sata1/SportsAnalytics/whole_game_records/fl_test_sets')

    parser.add_argument('--pretrained_embeddings', default=None)
    parser.add_argument('--starting_pitcher_only', default=False, type=str2bool)
    parser.add_argument('--pitcher_avg_inning_entry_fp',
                        default='../data/pitcher_avg_inning_entry.json')
    parser.add_argument('--prepend_entity_type_id', default=False, type=str2bool)
    parser.add_argument('--v2_player_attn', default=False, type=str2bool)
    parser.add_argument('--v2_attn_max_n_batter', default=24)
    parser.add_argument('--context_only', default=False, type=str2bool)

    parser.add_argument('--batter_ab_threshold_custom_id', default=40)
    parser.add_argument('--pitcher_ab_threshold_custom_id', default=200)
    parser.add_argument('--batter_pre2021_n_ab_d_fp', default='../data/batter_n_ab_pre_2021.json')
    parser.add_argument('--pitcher_pre2021_n_ab_d_fp', default='../data/pitcher_n_ab_pre_2021.json')

    parser.add_argument('--norm_first', default=False, type=str2bool)
    parser.add_argument('--ordinal_pos_embeddings', default=False, type=str2bool)
    parser.add_argument('--use_ball_data', default=False, type=str2bool)
    parser.add_argument('--mask_ball_data', default=False, type=str2bool)
    parser.add_argument('--drop_ball_data', default=False, type=str2bool)
    parser.add_argument('--pitcher_career_n_ball_data', default=47, type=int)
    parser.add_argument('--batter_career_n_ball_data', default=47, type=int)
    parser.add_argument('--else_n_ball_data', default=117, type=int)

    parser.add_argument('--gsd_weight', default=10, type=int)
    parser.add_argument('--event_weight', default=1, type=int)
    parser.add_argument('--ptype_weight', default=1, type=int)
    parser.add_argument('--thrown_pitch_weight', default=1, type=int)
    parser.add_argument('--batted_ball_weight', default=1, type=int)

    parser.add_argument('--train', default=True, type=str2bool)
    parser.add_argument('--dev', default=False, type=str2bool)
    parser.add_argument('--test', default=True, type=str2bool)
    parser.add_argument('--inspect', default=False, type=str2bool)
    parser.add_argument('--identify', default=False, type=str2bool)
    parser.add_argument('--ben_vocab', default=True, type=str2bool)
    parser.add_argument('--extended_test_logs', default=False, type=str2bool)
    parser.add_argument('--use_explicit_test_masked_indices', default=False, type=str2bool)
    parser.add_argument('--do_loss_weighting', default=False, type=str2bool)
    parser.add_argument('--causal_classifier', default=False, type=str2bool)
    parser.add_argument('--causal_classifier_alpha', default=3.0, type=float)
    parser.add_argument('--intermediate_cls_projection', default=True, type=str2bool)

    parser.add_argument('--xent_supcon_mixture', default=False, type=str2bool)
    parser.add_argument('--gsd_supcon_dim', default=64, type=int)
    parser.add_argument('--event_supcon_dim', default=56, type=int)
    parser.add_argument('--ptype_supcon_dim', default=48, type=int)

    parser.add_argument('--only_sos_statistics', default=False, type=str2bool)
    parser.add_argument('--reduced_event_map', default=False, type=str2bool)
    parser.add_argument('--ldam_loss', default=False, type=str2bool)
    parser.add_argument('--ldam_max_m', default=0.5, type=float)
    parser.add_argument('--ldam_s', default=10, type=int)
    parser.add_argument('--max_weight_ratio', default=0.001, type=float)

    parser.add_argument('--eqlv2_loss', default=False, type=str2bool)
    parser.add_argument('--eqlv2_gamma', default=12, type=int)
    parser.add_argument('--eqlv2_mu', default=0.8, type=float)
    parser.add_argument('--eqlv2_alpha', default=4.0, type=float)

    parser.add_argument('--v2_encoder', default=True, type=str2bool)
    parser.add_argument('--boab_can_be_masked', default=False, type=str2bool)
    parser.add_argument('--use_player_id', default=False, type=str2bool)
    parser.add_argument('--player_id_mask_pct', default=0.15, type=float)
    parser.add_argument('--player_id_map_fp', default='../config/all_player_id_mapping.json')
    parser.add_argument('--xent_label_smoothing', default=0.0, type=float)

    parser.add_argument('--n_games_context', default=15, type=int,
                        help='How many games to include in the context of each constructed record')
    parser.add_argument('--context_max_len', default=464, type=int)
    parser.add_argument('--chop_context_p', default=0.05, type=float)
    parser.add_argument('--n_ab_peak_ahead', default=0, type=int,
                        help='How many at-bats from the start of the initial completion to transfer to the context')
    parser.add_argument('--n_games_completion', default=1, type=int,
                        help='How many games to include in the completion of each constructed record')
    parser.add_argument('--completion_max_len', default=50, type=int)

    parser.add_argument('--raw_pitcher_data_dim', default=415, type=int)  # all scopes = 552, last_15|this_game = 274
    parser.add_argument('--raw_batter_data_dim', default=441, type=int)  # all scopes = 578, last_15|this_game = 274
    parser.add_argument('--raw_matchup_data_dim', default=411, type=int)  # all scopes = 411, this_game = 137, career|season=274
    parser.add_argument('--n_thrown_pitch_metrics', default=14, type=int)
    parser.add_argument('--n_batted_ball_metrics', default=5, type=int)
    parser.add_argument('--attn_mask_type', default='bidirectional')

    parser.add_argument('--torch_amp', default=True, type=str2bool)
    parser.add_argument('--tie_weights', default=True, type=str2bool)

    parser.add_argument('--out', default='..//out/fl_modeling', help='Directory to put output')
    parser.add_argument('--config_dir', default='../config', help='Directory to find config shit')

    parser.add_argument('--batter_data_scopes_to_use',
                        default=['career', 'season', 'last15'],     # , 'this_game'
                        type=str, nargs='+', )
    parser.add_argument('--pitcher_data_scopes_to_use',
                        default=['career', 'season', 'last15'],     # , 'this_game'
                        type=str, nargs='+', )
    parser.add_argument('--matchup_data_scopes_to_use',
                        default=['career', 'season', 'last15'],               # , 'this_game'
                        type=str, nargs='+', )

    parser.add_argument('--batter_data_scope_sizes',
                        default=[167, 137, 137],  # , 137
                        type=int, nargs='+', )
    parser.add_argument('--pitcher_data_scope_sizes',
                        default=[141, 137, 137],  # , 137
                        type=int, nargs='+', )
    parser.add_argument('--matchup_data_scope_sizes',
                        default=[137, 137, 137],  # , 137
                        type=int, nargs='+', )

    parser.add_argument('--dataset_size_train', default=50000, type=int)
    parser.add_argument('--dataset_size_else', default=5000, type=int)
    parser.add_argument('--epochs', default=135, type=int, help='# epochs to train for')
    parser.add_argument('--batch_size', default=16, type=int, help='Batch size to use')
    parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate')
    parser.add_argument('--l2', default=0.00001, type=float)
    parser.add_argument('--general_dropout_prob', default=0.10, type=float)
    parser.add_argument('--token_mask_pct', default=0.15, type=float)
    parser.add_argument('--mask_override_prob', default=0.15, type=float)
    parser.add_argument('--n_grad_accum', default=1, type=int)
    parser.add_argument('--n_warmup_iters', default=500, type=int)
    parser.add_argument('--distribution_based_player_sampling_prob', default=0.35, type=float)

    # ARCHITECTURE PARMS
    parser.add_argument('--n_layers', default=12, type=int)
    parser.add_argument('--gsd_embd_dim', default=384, type=int)
    parser.add_argument('--event_embd_dim', default=256, type=int)
    parser.add_argument('--pitch_type_embd_dim', default=72, type=int)
    parser.add_argument('--gsd_n_attn', default=12, type=int)
    parser.add_argument('--event_n_attn', default=8, type=int)
    parser.add_argument('--pitch_type_n_attn', default=6, type=int)

    parser.add_argument('--n_attn', default=12, type=int)
    parser.add_argument('--n_proj_layers', default=2, type=int)
    parser.add_argument('--n_raw_data_proj_layers', default=2, type=int)
    parser.add_argument('--complete_embd_dim', default=768, type=int)

    parser.add_argument('--n_ab_pitch_no_embds', default=25, type=int)
    parser.add_argument('--n_stadium_ids', default=35, type=int)
    parser.add_argument('--stadium_embd_dim', default=32, type=int)
    parser.add_argument('--handedness_embd_dim', default=5, type=int)

    parser.add_argument('--gamestate_vocab_bos_inning_no', default=False, type=str2bool)
    parser.add_argument('--gamestate_vocab_bos_score_diff', default=False, type=str2bool)
    parser.add_argument('--gamestate_vocab_bos_base_occ', default=True, type=str2bool)
    parser.add_argument('--gamestate_vocab_use_balls_strikes', default=True, type=str2bool)
    parser.add_argument('--gamestate_vocab_use_base_occupancy', default=True, type=str2bool)
    parser.add_argument('--gamestate_vocab_use_inning_no', default=False, type=str2bool)
    parser.add_argument('--gamestate_vocab_use_inning_topbot', default=False, type=str2bool)
    parser.add_argument('--gamestate_vocab_use_score_diff', default=True, type=str2bool)
    parser.add_argument('--gamestate_n_innings', default=10, type=int)
    parser.add_argument('--gamestate_max_score_diff', default=6, type=int)
    parser.add_argument('--gamestate_vocab_use_outs', default=True, type=str2bool)

    parser.add_argument('--batter_min_game_to_be_included_in_dataset', default=45, type=int)
    parser.add_argument('--pitcher_min_game_to_be_included_in_dataset', default=8, type=int)

    # logging parms
    parser.add_argument('--seed', default=16, type=int)
    parser.add_argument('--ckpt_file', default=None)
    parser.add_argument('--ckpt_file_tmplt', default='model_{}e.pt')
    parser.add_argument('--warm_start', default=False, type=str2bool)
    parser.add_argument('--print_every', default=1, type=int)
    parser.add_argument('--log_every', default=25, type=int)
    parser.add_argument('--save_model_every', default=5, type=int)
    parser.add_argument('--save_preds_every', default=1, type=int)
    parser.add_argument('--summary_every', default=100, type=int)
    parser.add_argument('--dev_every', default=1, type=int)
    parser.add_argument('--arg_out_file', default='args.txt', help='File to write cli args to')
    parser.add_argument('--verbosity', default=0, type=int)
    parser.add_argument('--grad_summary', default=True, type=str2bool)
    parser.add_argument('--grad_summary_every', default=250, type=int)
    parser.add_argument('--bad_data_fps', default=[
        '1975/169798-41.json', '1977/173943-53.json', '1978/174397-8.json', '1978/175813-10.json',
        '1980/179652-66.json', '1980/179658-48.json', '1983/185150-13.json', '1983/185150-16.json',
        '1984/186501-4.json', '1985/188314-53.json', '1986/190666-65.json', '1986/192066-44.json', '1986/192356-1.json',
        '1988/194576-8.json', '1988/195877-19.json', '1989/197115-54.json', '1989/197909-28.json',
        '1989/198065-44.json', '1990/199316-20.json', '1990/199952-37.json', '1990/199952-46.json',
        '1990/200431-33.json', '1991/201584-46.json', '1991/201584-48.json', '1991/201903-56.json',
        '1991/202405-10.json', '1991/202406-30.json', '1992/203018-73.json', '1992/203039-69.json',
        '1992/203093-69.json', '1992/203221-84.json', '1992/203390-76.json', '1992/203429-33.json',
        '1992/203448-13.json', '1992/203647-54.json', '1992/203690-4.json', '1992/203739-76.json',
        '1992/204367-40.json', '1992/204390-32.json', '1992/204410-6.json', '1992/204422-59.json',
        '1992/204433-47.json', '1992/204555-3.json', '1992/204835-71.json', '1992/204835-75.json',
        '1993/206841-62.json', '1993/207279-14.json', '1994/207423-84.json', '1994/207491-19.json',
        '1994/207885-45.json', '1994/207930-59.json', '1994/207945-68.json', '1994/208003-80.json',
        '1994/208226-53.json', '1994/208440-62.json', '1994/208450-58.json', '1994/208496-8.json',
        '1995/208922-54.json', '1995/209075-60.json', '1995/209334-5.json', '1995/209407-35.json',
        '1995/210131-21.json', '1995/210226-15.json', '1995/210831-46.json', '1996/211006-65.json',
        '1996/211078-37.json', '1996/211359-59.json', '1996/211583-43.json', '1996/211680-74.json',
        '1996/211876-63.json', '1996/212305-57.json', '1996/212501-57.json', '1996/212681-16.json',
        '1996/212836-49.json', '1997/213481-5.json', '1997/213499-52.json', '1997/213627-55.json',
        '1997/213639-63.json', '1997/213684-19.json', '1997/213751-46.json', '1997/213784-6.json',
        '1997/214446-82.json', '1997/214512-55.json', '1997/214621-5.json', '1997/214622-9.json', '2000/4300-38.json',
        '2003/14928-67.json', '2003/15139-63.json', '2003/15329-32.json', '2003/16017-52.json', '2003/16052-61.json',
        '2003/16074-89.json', '2003/16607-47.json', '2003/16748-30.json', '2004/20064-75.json', '2004/20112-66.json',
        '2004/20355-25.json', '2004/20669-38.json', '2004/20690-63.json', '2004/20751-2.json', '2004/20809-71.json',
        '2004/20931-25.json', '2004/20959-60.json', '2004/20964-23.json', '2004/20970-9.json', '2004/21499-82.json',
        '2004/21853-38.json', '2004/22366-44.json', '2005/22706-86.json', '2005/22963-54.json', '2005/23032-42.json',
        '2005/23192-35.json', '2005/23333-44.json', '2005/23402-64.json', '2005/23417-59.json', '2005/23489-87.json',
        '2005/23519-52.json', '2005/23786-25.json', '2005/23897-12.json', '2005/23966-50.json', '2005/24054-10.json',
        '2005/24243-78.json', '2005/24868-83.json', '2006/40075-29.json', '2006/40085-58.json', '2006/40208-60.json',
        '2006/40323-22.json', '2006/40657-5.json', '2006/41120-78.json', '2006/41676-40.json', '2006/41712-62.json',
        '2006/41786-87.json', '2006/41964-58.json', '2006/42070-27.json', '2007/68307-66.json', '2007/68364-64.json',
        '2007/68374-4.json', '2007/68413-23.json', '2007/68436-68.json', '2007/68511-50.json', '2007/68766-20.json',
        '2007/68781-23.json', '2007/68793-41.json', '2007/68847-41.json', '2007/68847-46.json', '2007/69093-3.json',
        '2007/69189-49.json', '2007/69373-77.json', '2007/69585-17.json', '2007/69620-40.json', '2007/69808-7.json',
        '2007/69983-36.json', '2007/70352-70.json', '2007/70388-61.json', '2007/70418-47.json', '2008/233930-1.json',
        '2008/233933-57.json', '2008/233945-62.json', '2008/234122-66.json', '2008/234234-64.json',
        '2008/234321-48.json', '2008/234326-22.json', '2008/234541-52.json', '2008/234569-42.json',
        '2008/234761-71.json', '2008/235138-66.json', '2008/235139-46.json', '2008/235236-53.json',
        '2008/236132-50.json', '2009/244205-26.json', '2009/244214-8.json', '2009/244284-60.json',
        '2009/244348-38.json', '2009/244457-66.json', '2009/244500-28.json', '2009/244785-5.json',
        '2009/244869-93.json', '2009/244873-9.json', '2009/245296-31.json', '2009/245404-21.json',
        '2009/245447-45.json', '2009/245553-58.json', '2009/246091-13.json', '2009/246573-18.json',
        '2010/263813-58.json', '2010/263834-10.json', '2010/263906-47.json', '2010/263908-31.json',
        '2010/263912-4.json', '2010/264121-62.json', '2010/264212-13.json', '2010/264278-11.json',
        '2010/264287-32.json', '2010/264292-64.json', '2010/264385-42.json', '2010/264586-35.json',
        '2010/264704-11.json', '2010/264957-62.json', '2010/265002-15.json', '2010/265741-73.json',
        '2011/286984-44.json', '2011/287006-7.json', '2011/287207-69.json', '2011/287275-34.json',
        '2011/287314-29.json', '2011/287359-64.json', '2011/287477-27.json', '2011/287495-47.json',
        '2011/287913-71.json', '2011/288280-11.json', '2011/288287-13.json', '2011/288355-52.json',
        '2011/288476-42.json', '2011/288886-55.json', '2012/317797-75.json', '2012/317947-48.json',
        '2012/318485-54.json', '2012/319562-64.json', '2012/319839-53.json', '2012/319985-52.json',
        '2012/320096-74.json', '2013/346792-15.json', '2013/346964-82.json', '2013/347543-40.json',
        '2013/347754-70.json', '2014/380894-26.json', '2015/414020-81.json', '2015/414264-19.json',
        '2015/415933-70.json', '2016/447752-28.json', '2016/448381-44.json', '2017/492054-24.json',
        '2017/492354-17.json', '2018/529755-56.json', '2018/529812-66.json', '2019/567172-14.json',
        '2021/632200-82.json'
    ], type=str, nargs='+',
                        help='FPs w/ corrupted data (likely b/c statcast)')

    # hardware parms
    parser.add_argument('--gpus', default=[0, 1], help='Which GPUs to use', type=int, nargs='+')
    parser.add_argument('--port', default='12345', help='Port to use for DDP')
    parser.add_argument('--on_cpu', default=False, type=str2bool)
    parser.add_argument('--n_data_workers', default=5, help='# threads used to fetch data', type=int)
    parser.add_argument('--n_data_workers_else', default=5, help='# threads used to fetch data', type=int)

    args = parser.parse_args()

    print('args:\n{}'.format(args))
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.random.manual_seed(args.seed)

    run_modes = []
    if args.train:
        run_modes.append('train')

        curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        print('*' * len('* Model time ID: {} *'.format(curr_time)))
        print('* Model time ID: {} *'.format(curr_time))
        print('*' * len('* Model time ID: {} *'.format(curr_time)))

        args.out = os.path.join(args.out, curr_time)
        os.makedirs(args.out)

        args.tb_dir = os.path.join(args.out, 'tb_dir')
        os.makedirs(args.tb_dir)

        args.model_save_dir = os.path.join(args.out, 'models')
        os.makedirs(args.model_save_dir)

        args.model_log_dir = os.path.join(args.out, 'logs')
        os.makedirs(args.model_log_dir)

        args.arg_out_file = os.path.join(args.out, args.arg_out_file)
        args_d = vars(args)
        with open(args.arg_out_file, 'w+') as f:
            for k, v in args_d.items():
                f.write('{} = {}\n'.format(k, v))
    if args.dev:
        run_modes.append('dev')
    if args.test:
        run_modes.append('test')
    if args.inspect:
        run_modes.append('inspect')
    if args.identify:
        run_modes.append('identify')

    if not args.inspect:
        if (args.dev or args.test) and not args.train:
            args.out = os.path.dirname(args.ckpt_file)
            if os.path.basename(args.out) == 'models':
                args.out = os.path.dirname(args.out)

    entity_type_d = {et: idx for idx, et in enumerate(args.type)}
    reverse_entity_type_d = {idx: et for idx, et in enumerate(args.type)}
    print('entity_type_d: {}'.format(entity_type_d))
    print('reverse_entity_type_d: {}'.format(reverse_entity_type_d))

    args.entity_type_d = entity_type_d
    args.reverse_entity_type_d = reverse_entity_type_d

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = args.port
    for mode in run_modes:
        print('Creating {} distributed models for {}...'.format(len(args.gpus), mode))
        mp.spawn(Runner, nprocs=len(args.gpus), args=(mode, args))

    print('All done :)')

