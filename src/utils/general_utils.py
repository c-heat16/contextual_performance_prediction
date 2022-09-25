__author__ = 'Connor Heaton'

import os
import os.path
import json
import sqlite3
import argparse
from argparse import Namespace


def try_cast_int(x):
    try:
        x = int(x)
    except:
        pass

    return x


def jsonKeys2int(x):
    if isinstance(x, dict):
        return {try_cast_int(k): v for k, v in x.items()}
    return x


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() == 'none':
        return None
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def query_db(db_fp, query, args=None):
    conn = sqlite3.connect(db_fp, check_same_thread=False)
    c = conn.cursor()
    if args is None:
        c.execute(query)
    else:
        c.execute(query, args)
    rows = c.fetchall()

    return rows


def get_game_pk_to_date_d(db_fp, cache_fp=None):
    if cache_fp is not None and os.path.exists(cache_fp):
        d = json.load(open(cache_fp), object_hook=jsonKeys2int)
    else:
        query_str = 'select distinct game_pk, game_date from statcast where game_year >= 2000'
        res = query_db(db_fp, query_str)

        d = {int(r[0]): r[1] for r in res}
        with open(cache_fp, 'w+') as f:
            f.write(json.dumps(d, indent=2))

    return d


def get_team_name_mapping(db_fp, cache_fp=None):
    if cache_fp is not None and os.path.exists(cache_fp):
        d = json.load(open(cache_fp), object_hook=jsonKeys2int)
    else:
        query_str = 'select distinct home_team from statcast where game_year >= 2000'
        res = query_db(db_fp, query_str)

        d = {r[0]: idx for idx, r in enumerate(res)}
        with open(cache_fp, 'w+') as f:
            f.write(json.dumps(d, indent=2))
    return d


def boolify(s):
    if s == 'True':
        return True
    if s == 'False':
        return False
    raise ValueError("huh?")


def autoconvert(s):
    if s in ['[BOS]', '[EOS]']:
        return s
    for fn in (boolify, int, float):
        try:
            return fn(s)
        except ValueError:
            pass

    if s[0] == '[' and s[-1] == ']':
        s = s[1:-1]
        s = [ss.strip().strip('\'') for ss in s.split(',')]

    return s


def read_model_args(fp):
    m_args = {}

    with open(fp, 'r') as f:
        for line in f:
            line = line.strip()
            if not line == '':
                arg, val = line.split('=')
                arg = arg.strip()
                val = val.strip()

                val = autoconvert(val)
                m_args[arg] = val

    m_args = Namespace(**m_args)

    return m_args


def calc_ptb(ptb_d):
    h = ptb_d.get('h', 0)
    hr = ptb_d.get('hr', 0)
    bb = ptb_d.get('bb', 0)
    hbp = ptb_d.get('hbp', 0)
    ibb = ptb_d.get('ibb', 0)

    ptb = 0.89 * (1.255 * (h - hr) + 4 * hr) + 0.56 * (bb + hbp - ibb)
    return ptb


def is_barrel(launch_speed, launch_angle):
    barrel = False
    if launch_angle <= 50 and launch_speed >= 98 and launch_speed * 1.5 - launch_angle >= 117 and \
            launch_speed + launch_angle >= 124:
        barrel = True

    return barrel


def is_quab(pitches):
    """
    https://gamechanger.zendesk.com/hc/en-us/articles/214493703-Quality-At-Bats-
    """
    # AB over 6 pitches
    if len(pitches) >= 6:
        return True

    # hard hit ball
    if pitches[-1]['launch_speed'] is not None and pitches[-1]['launch_speed'] >= 95:
        return True

    # extra base hit
    if pitches[-1]['events'] in ['double', 'home_run', 'triple']:
        return True

    # walk
    if pitches[-1]['events'] in ['walk', 'hit_by_pitch', 'intent_walk']:
        return True

    # sac bunt or sac fly
    if pitches[-1]['events'] in ['sac_fly', 'sac_bunt']:
        return True

    pitches_after_two_strikes = 0
    start_counting = False
    for pitch in pitches:
        if pitch['strikes'] == 2:
            start_counting = True

        if start_counting:
            pitches_after_two_strikes += 1

    if pitches_after_two_strikes >= 3:
        return True

    return False


