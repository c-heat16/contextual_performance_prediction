__author__ = 'Connor Heaton'
import os
import torch
import datetime
import argparse
import multiprocessing

import numpy as np
import torch.multiprocessing as mp

from src import FinetuneRunner
from src.utils import str2bool, get_game_pk_to_date_d, get_team_name_mapping


def make_ab_data_cache(data_dir):
    cache = {}
    for season in os.listdir(data_dir):
        print('Reading data for {}...'.format(season))
        for idx, int_filename in enumerate(os.listdir(os.path.join(data_dir, season))):
            data = torch.load(os.path.join(data_dir, season, int_filename))
            cache['{}/{}'.format(season, int_filename[:-3])] = data
            if idx % 20000 == 0:
                print('\tRead {} items...'.format(idx))

    return cache


def read_ab_data_cache(cache_dir, data_dir):
    cache_fp = os.path.join(cache_dir, 'cache.pt')
    if os.path.exists(cache_fp):
        print('Cache exists, loading from disk...')
        cache = torch.load(cache_fp)
    else:
        print('Creating cache...')
        cache = make_ab_data_cache(data_dir)
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        torch.save(cache, cache_fp)

    return cache


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_types', default='mt_data2vec')
    parser.add_argument('--single_model', default=False, type=str2bool)
    parser.add_argument('--entity_models', default=True, type=str2bool)
    parser.add_argument('--model_ckpt', default='../model_dir/models/model.pt')
    parser.add_argument('--batter_ckpt', default='../batter/models/model.pt')
    parser.add_argument('--pitcher_ckpt', default='../pitcher/models/model.pt')
    parser.add_argument('--out', default='/home/czh/nvme1/SportsAnalytics/out/fl_finetuning', help='Directory to put output')

    parser.add_argument('--do_ln', default=False, type=str2bool)
    parser.add_argument('--loss_type', default='mse')
    parser.add_argument('--do_group_gathers', default=False, type=str2bool)
    parser.add_argument('--task_specific', default=False, type=str2bool)

    parser.add_argument('--cyclic_lr', default=False, type=str2bool)
    parser.add_argument('--cosine_lr', default=False, type=str2bool)

    parser.add_argument('--pitcher_targets', default=['k', 'h'], type=str, nargs='*')       # , 'r', 'barrel', 'quab'
    parser.add_argument('--pitcher_scalars',
                        # default=[16, 16, 11],
                        default=[17, 17],       # , 12, 20, 17
                        type=float, nargs='*')
    parser.add_argument('--batter_targets',
                        default=['k', 'h'],     # , 'r', 'barrel', 'quab'
                        # default=[],
                        type=str, nargs='*')
    parser.add_argument('--batter_scalars',
                        # default=[5, 6, 8],
                        default=[6, 7],         # , 9, 7, 7
                        # default=[],
                        type=float, nargs='*')

    parser.add_argument('--batter_weight', default=1, type=float)
    parser.add_argument('--pitcher_weight', default=10, type=float)
    parser.add_argument('--batter_has_hit_weight', default=1, type=float)

    parser.add_argument('--pitcher_avg_entry_fp',
                        default='/home/czh/nvme1/SportsAnalytics/data/pitcher_avg_inning_entry.json')
    parser.add_argument('--only_starting_pitchers', default=False, type=str2bool)
    parser.add_argument('--binary_hit_preds', default=False, type=str2bool)

    parser.add_argument('--use_intermediate_whole_game_data', default=False, type=str2bool)
    parser.add_argument('--intermediate_whole_game_data_dir', default='/home/czh/md0/intermediate_whole_game_data/')
    parser.add_argument('--torch_amp', default=True, type=str2bool)
    parser.add_argument('--db_fp', default='/home/czh/nvme1/SportsAnalytics/database/mlb.db')

    parser.add_argument('--splits_dir',
                        default='/home/czh/md0/whole_game_records/ar_game_splits_8_26_2022'
                        )

    parser.add_argument('--whole_game_record_dir',
                        default='/home/czh/md0/whole_game_records',
                        help='Where to find player career data')
    parser.add_argument('--career_data',
                        # default='/home/czh/sata1/SportsAnalytics/player_career_data',
                        default='/home/czh/sata1/SportsAnalytics/player_career_data_08182022',
                        help='Where to find player career data')
    parser.add_argument('--single_pitcher_batter_completion', default=False, type=str2bool)

    parser.add_argument('--use_matchup_data', default=False, type=str2bool)
    parser.add_argument('--matchup_data_dim', default=274, type=int)

    # general modeling parms
    parser.add_argument('--train', default=True, type=str2bool)
    parser.add_argument('--dev', default=False, type=str2bool)
    parser.add_argument('--test', default=False, type=str2bool)
    parser.add_argument('--epochs', default=6, type=int, help='# epochs to train for')
    parser.add_argument('--batch_size', default=13, type=int,
                        help='Batch size to use')  # batters ~15 pitchers ~112 both ~13
    parser.add_argument('--lr', default=0.00005, type=float, help='Learning rate')
    parser.add_argument('--l2', default=0.0, type=float)
    parser.add_argument('--general_dropout_prob', default=0.25, type=float)
    parser.add_argument('--n_grad_accum', default=1, type=int)
    parser.add_argument('--n_warmup_iters', default=500, type=int)

    # logging parms
    parser.add_argument('--seed', default=16, type=int)
    parser.add_argument('--ckpt_file',
                        default=None,
                        # default='/home/czh/nvme1/SportsAnalytics/out/fl_finetuning/20220326-091717/models/model_19e.pt',
                        )
    parser.add_argument('--ckpt_file_tmplt', default='model_{}e.pt')
    parser.add_argument('--warm_start', default=False, type=str2bool)
    parser.add_argument('--print_every', default=1, type=int)
    parser.add_argument('--log_every', default=25, type=int)
    parser.add_argument('--save_model_every', default=1, type=int)
    parser.add_argument('--save_preds_every', default=10, type=int)
    parser.add_argument('--summary_every', default=45, type=int)
    parser.add_argument('--dev_every', default=1, type=int)
    parser.add_argument('--arg_out_file', default='args.txt', help='File to write cli args to')
    parser.add_argument('--verbosity', default=0, type=int)
    parser.add_argument('--grad_summary', default=True, type=str2bool)
    parser.add_argument('--grad_summary_every', default=200, type=int)

    # hardware parms
    parser.add_argument('--gpus', default=[0, 1], help='Which GPUs to use', type=int, nargs='+')
    parser.add_argument('--port', default='12345', help='Port to use for DDP')
    parser.add_argument('--on_cpu', default=False, type=str2bool)
    parser.add_argument('--n_data_workers', default=5, help='# threads used to fetch data', type=int)
    parser.add_argument('--n_data_workers_else', default=6, help='# threads used to fetch data', type=int)
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

    parser.add_argument('--intermediate_ab_data_dir',
                        default='/home/czh/md0/ab_seqs/ab_seqs_v16/intermediate_data')
    parser.add_argument('--ab_cache_dir',
                        default='/home/czh/md0/ab_cache')

    parser.add_argument('--game_tensor_dir',
                        default='/home/czh/md0/game_tensors')

    parser.add_argument('--game_pk_to_date_fp',
                        default='/home/czh/nvme1/SportsAnalytics/data/game_pk_to_date.json')
    parser.add_argument('--team_str_to_id_fp',
                        default='/home/czh/nvme1/SportsAnalytics/data/team_str_to_id.json')

    args = parser.parse_args()
    args.world_size = len(args.gpus)

    args.use_pitcher = True if len(args.pitcher_targets) > 0 else False
    args.use_batter = True if len(args.batter_targets) > 0 else False

    assert len(args.pitcher_targets) == len(args.pitcher_scalars)
    assert len(args.batter_targets) == len(args.batter_scalars)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.random.manual_seed(args.seed)
    run_modes = []
    if args.train:
        run_modes.append('train')
        if args.ckpt_file is not None:
            curr_time = os.path.split(os.path.split(os.path.split(args.ckpt_file)[0])[0])[1]
        else:
            curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        # directories should only be made if training new model
        print('*' * len('* Model time ID: {} *'.format(curr_time)))
        print('* Model time ID: {} *'.format(curr_time))
        print('*' * len('* Model time ID: {} *'.format(curr_time)))

        args.out = os.path.join(args.out, curr_time)
        if not os.path.exists(args.out):
            os.makedirs(args.out)

        args.tb_dir = os.path.join(args.out, 'tb_dir')
        if not os.path.exists(args.tb_dir):
            os.makedirs(args.tb_dir)

        args.model_save_dir = os.path.join(args.out, 'models')
        if not os.path.exists(args.model_save_dir):
            os.makedirs(args.model_save_dir)

        args.pred_dir = os.path.join(args.out, 'preds')
        if not os.path.exists(args.pred_dir):
            os.makedirs(args.pred_dir)

        args.model_log_dir = os.path.join(args.out, 'logs')
        if not os.path.exists(args.model_log_dir):
            os.makedirs(args.model_log_dir)

        args.arg_out_file = os.path.join(args.out, args.arg_out_file)
        if not os.path.exists(args.arg_out_file):
            args_d = vars(args)
            with open(args.arg_out_file, 'w+') as f:
                for k, v in args_d.items():
                    f.write('{} = {}\n'.format(k, v))
    if args.dev:
        run_modes.append('dev')

        if args.ckpt_file is not None:
            curr_time = os.path.split(os.path.split(os.path.split(args.ckpt_file)[0])[0])[1]

            args.out = os.path.join(args.out, curr_time)
            args.tb_dir = os.path.join(args.out, 'tb_dir')
            args.model_save_dir = os.path.join(args.out, 'models')
            args.model_log_dir = os.path.join(args.out, 'logs')
            args.pred_dir = os.path.join(args.out, 'preds')

    if args.test:
        run_modes.append('test')

        if args.ckpt_file is not None:
            curr_time = os.path.split(os.path.split(os.path.split(args.ckpt_file)[0])[0])[1]

            args.out = os.path.join(args.out, curr_time)
            args.tb_dir = os.path.join(args.out, 'tb_dir')
            args.model_save_dir = os.path.join(args.out, 'models')
            args.model_log_dir = os.path.join(args.out, 'logs')
            args.pred_dir = os.path.join(args.out, 'preds')

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = args.port
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"

    print('Creating game_pk_to_date_d...')
    game_pk_to_date_d = get_game_pk_to_date_d(args.db_fp, args.game_pk_to_date_fp)
    print('Creating team_str_to_id_d...')
    team_str_to_id_d = get_team_name_mapping(args.db_fp, args.team_str_to_id_fp)
    # input('team_str_to_id_d: {}'.format(team_str_to_id_d))

    # intermediate_ab_data = make_ab_data_cache(args.intermediate_ab_data_dir)
    # intermediate_ab_data = read_ab_data_cache(args.ab_cache_dir, args.intermediate_ab_data_dir)
    intermediate_ab_data = None
    print(args)

    # multiprocessing.set_start_method('spawn')
    mp.set_start_method('spawn')
    # mp.set_sharing_strategy('file_system')
    for mode in run_modes:
        print('Creating {} distributed models for {}...'.format(len(args.gpus), mode))

        mp.spawn(FinetuneRunner, nprocs=len(args.gpus), args=(mode, args, game_pk_to_date_d,
                                                                team_str_to_id_d, intermediate_ab_data))

    print('Finished!')
