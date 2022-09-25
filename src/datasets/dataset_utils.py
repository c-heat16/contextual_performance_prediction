__author__ = 'Connor Heaton'

import math


def parse_json(j, j_type, max_value_data=None):
    if j_type == 'pitcher':
        avoid_keys = ['__id__', 'throws', 'first_name', 'last_name']
        handedness = j['throws']
    elif j_type == 'batter':
        avoid_keys = ['__id__', 'stand', 'first_name', 'last_name']
        handedness = j['stand']
    else:
        avoid_keys = []
        handedness = None

    ball_keys = [
        # pitcher career
        'career_fb_pct', 'career_sl_pct', 'career_ct_pct', 'career_cb_pct', 'career_ch_pct', 'career_sf_pct',
        'career_kn_pct', 'career_xx_pct', 'career_fbv', 'career_slv', 'career_ctv', 'career_cbv', 'career_chv',
        'career_sfv', 'career_knv', 'career_zone_pct', 'career_swstr_pct', 'career_pull_pct', 'career_cent_pct',
        'career_oppo_pct', 'career_soft_pct', 'career_med_pct', 'career_hard_pct', 'career_wfb_avg', 'career_wfb_sum',
        'career_wsl_avg', 'career_wsl_sum', 'career_wct_avg', 'career_wct_sum', 'career_wcb_avg', 'career_wcb_sum',
        'career_wch_avg', 'career_wch_sum', 'career_wsf_avg', 'career_wsf_sum', 'career_wkn_avg', 'career_wkn_sum',
        'career_gb_sum', 'career_gb_avg', 'career_fb_sum', 'career_fb_avg', 'career_ld_sum', 'career_ld_avg',
        # batter career
        'career_fb_pct_pitch', 'career_sl_pct', 'career_ct_pct', 'career_cb_pct', 'career_ch_pct',
        'career_sf_pct_pitch', 'career_kn_pct', 'career_xx_pct', 'career_fbv', 'career_slv', 'career_ctv', 'career_cbv',
        'career_chv', 'career_sfv', 'career_knv', 'career_zone_pct', 'career_swstr_pct', 'career_pull_pct',
        'career_cent_pct', 'career_oppo_pct', 'career_soft_pct', 'career_med_pct', 'career_hard_pct',
        'career_o_swing_pct', 'career_z_swing_pct', 'career_o_contact_pct', 'career_z_contact_pct', 'career_wfb_avg',
        'career_wfb_sum', 'career_wsl_avg', 'career_wsl_sum', 'career_wct_avg', 'career_wct_sum', 'career_wcb_avg',
        'career_wcb_sum', 'career_wch_avg', 'career_wch_sum', 'career_wsf_avg', 'career_wsf_sum', 'career_wkn_avg',
        'career_wkn_sum', 'career_gb_sum', 'career_gb_avg', 'career_fb_sum', 'career_fb_avg', 'career_ld_sum',
        'career_ld_avg',
        # season/last15/this_game
        'fc_pct', 'ff_pct', 'sl_pct', 'ch_pct', 'cu_pct', 'si_pct', 'fs_pct', 'ft_pct', 'kc_pct', 'po_pct',
        'in_pct', 'sc_pct', 'fa_pct', 'ep_pct', 'kn_pct', 'fo_pct', 'un_pct', 'fc_cnt', 'ff_cnt', 'sl_cnt',
        'ch_cnt', 'cu_cnt', 'si_cnt', 'fs_cnt', 'ft_cnt', 'kc_cnt', 'po_cnt', 'in_cnt', 'sc_cnt', 'fa_cnt',
        'ep_cnt', 'kn_cnt', 'fo_cnt', 'un_cnt', 'avg_sz_bot', 'avg_sz_top', 'x_pos_swing_pcts',
        'z_pos_swing_pcts', 'x_pos_hit_pcts', 'z_pos_hit_pcts', 'x_pos_pitch_pcts', 'z_pos_pitch_pcts',
        'avg_vx0', 'avg_vy0', 'avg_vz0', 'avg_ax', 'avg_ay', 'avg_az', 'avg_hist_distance', 'avg_launch_speed',
        'avg_launch_angle', 'avg_spin_rate', 'hit_xpos_res', 'hit_ypos_res'
    ]

    # print('j_type: {} data_scopes_to_use: {}'.format(j_type, data_scopes_to_use))

    if max_value_data is not None:
        # print('MAX VALUE DATA BEING USED')
        type_max_value_data = max_value_data[j_type]
    else:
        type_max_value_data = {}

    # data = []
    data = {'handedness': handedness}
    for k1, v1 in j.items():
        k1_norm_vals = type_max_value_data.get(k1, {})
        if k1 not in avoid_keys:
            # print('key {} is being parsed...'.format(k1))
            k1_data = []
            ball_data = []
            for k2, v2 in v1.items():
                if type(v2) == list:
                    k2_norm_val = k1_norm_vals.get(k2, [1.0])
                    k2_norm_val = [nv if nv != 0.0 else 1.0 for nv in k2_norm_val]
                    processed_list = [float(item_val) / norm_val if not math.isnan(item_val) and not item_val == math.inf else 1.0
                         for item_val, norm_val in zip(v2, k2_norm_val)]
                    if any(k2.endswith(bk) for bk in ball_keys):
                        ball_data.extend(processed_list)
                    else:
                        k1_data.extend(processed_list)
                else:
                    if v2 == math.inf:
                        new_v2 = 1.0
                    elif math.isnan(v2):
                        new_v2 = 0.0
                    else:
                        k2_norm_val = k1_norm_vals.get(k2, 1.0)
                        new_v2 = v2 / k2_norm_val if k2_norm_val != 0 else v2
                    if any(k2.endswith(bk) for bk in ball_keys):
                        ball_data.append(float(new_v2))
                    else:
                        k1_data.append(float(new_v2))
            # print('len(k1_data): {}'.format(len(k1_data)))
            # print('len(ball_data): {}'.format(len(ball_data)))
            k1_data.extend(ball_data)
            # print('len(k1_data) after combining: {}'.format(len(k1_data)))
            data[k1] = k1_data
        # elif k1 in avoid_keys:
        #     print('key {} in keys to avoid...'.format(k1))
        # elif k1 not in data_scopes_to_use:
        #     print('key {} is in data scopes not to use...'.format(k1))

    if math.nan in data:
        input('j: {}'.format(j))

    return data


def parse_w_sign(s):
    if type(s) == int:
        pass
    elif s in [None]:  # , 'N/A'
        s = 0
    elif s[0] in ['+', '-']:
        s = int(s[1:])
    else:
        s = int(s)

    return s


def compare_states(curr_state, last_state, fill_value):
    attrs_to_compare = ['balls', 'strikes', 'outs', 'on_1b', 'on_2b', 'on_3b', 'score']
    delta = {}
    if last_state is None:
        for attr_name in attrs_to_compare:
            # attr_val = curr_state.get(attr_name, 0)
            # if attr_name != 'score':
            #     delta[attr_name] = str(attr_val) if attr_val == 0 else '+{}'.format(attr_val)
            # else:
            #     delta[attr_name] = '0'
            delta[attr_name] = fill_value
    else:
        for attr_name in attrs_to_compare:
            curr_val = parse_w_sign(curr_state.get(attr_name, 0))
            last_val = parse_w_sign(last_state.get(attr_name, 0))
            change_val = curr_val - last_val
            if change_val > 0:
                change_str = '+{}'.format(change_val)
            else:
                change_str = str(change_val)
            delta[attr_name] = change_str

    if delta['balls'][0] == '-':
        delta['strikes'] = '-1'
    if delta['strikes'][0] == '-':
        delta['balls'] = '-1'

    return delta, curr_state


def find_state_deltas_v2(n_balls, n_strikes, outs_when_up, inning_no, inning_topbot, next_state, batter_score,
                         on_1b, on_2b, on_3b, last_state=None, pitch_description=None, use_swing_status=False):
    # Check for "'its just a hunk of metal' commissioner base runner"
    if next_state['inning'] == 'N/A':
        next_state['inning'] = inning_no + 1
    if (inning_no < int(next_state['inning'])) or (inning_no and inning_topbot != next_state['inning_topbot']):
        next_state['on_2b'] = 0
        # print('Setting on_2b in next_state to 0...')

    deltas = []
    p_state_dict = {}
    this_last_state = last_state
    for p_idx, (balls, strikes) in enumerate(zip(n_balls, n_strikes)):
        # print('*** Evaluating pitch {} ***'.format(p_idx))
        p_state_dict = {'balls': balls,
                        'strikes': strikes,
                        'outs': outs_when_up,
                        'on_1b': on_1b,
                        'on_2b': on_2b,
                        'on_3b': on_3b,
                        'score': batter_score}
        # print('p_state_dict:\n{}'.format(json.dumps(p_state_dict, indent=2)))
        # print('this_last_state:\n{}'.format(json.dumps(this_last_state, indent=2)))
        p_delta, this_last_state = compare_states(p_state_dict, this_last_state, '[BOI]')
        # print('p_delta:\n{}'.format(json.dumps(p_delta, indent=2)))
        deltas.append(p_delta)

    p_state_dict = {}
    if next_state['home_score'] == 'N/A':
        p_state_dict['balls'] = 0
        p_state_dict['strikes'] = 0
        p_state_dict['on_1b'] = on_1b
        p_state_dict['on_2b'] = on_2b
        p_state_dict['on_3b'] = on_3b
        p_state_dict['outs'] = outs_when_up
        p_state_dict['score'] = batter_score + 1
    else:
        p_state_dict['balls'] = 0
        p_state_dict['strikes'] = 0
        p_state_dict['on_1b'] = 0 if next_state['on_1b'] in [None, 0] else 1
        p_state_dict['on_2b'] = 0 if next_state['on_2b'] in [None, 0] else 1
        p_state_dict['on_3b'] = 0 if next_state['on_3b'] in [None, 0] else 1
        p_state_dict['outs'] = next_state['outs_when_up'] if next_state[
                                                                 'inning_topbot'].lower() == inning_topbot.lower() else 3
        p_state_dict['score'] = next_state['bat_score'] if next_state[
                                                               'inning_topbot'].lower() == inning_topbot.lower() else \
            next_state['fld_score']

    # print('*** COMPARING FINAL STATE ***')
    # print('p_state_dict:\n{}'.format(json.dumps(p_state_dict, indent=2)))
    # print('this_last_state:\n{}'.format(json.dumps(this_last_state, indent=2)))
    # print('deltas: {}'.format(deltas))
    p_delta, this_last_state = compare_states(p_state_dict, this_last_state, '[BOI]')
    deltas.append(p_delta)
    # print('p_delta:\n{}'.format(json.dumps(p_delta, indent=2)))
    # input('*' * 25)

    return deltas, this_last_state


def fill_null(data, fill_val, norm_val=1.0):
    data = [x if x is not None else fill_val for x in data]
    data = [x if x <= norm_val else norm_val for x in data]
    # data = [x if x is not None and x <= norm_val else norm_val for x in data]
    data = [x / norm_val if x is not None else fill_val for x in data]
    return data


def parse_pitches_updated(pitch_j, norm_vals=None):
    if norm_vals is None:
        norm_vals = {}
    pitch_info = [[p['balls'], p['strikes'], p['pitch_type'], p['release_speed'],
                   p['plate_x'], p['plate_z'], p['release_pos_x'], p['release_pos_y'],
                   p['release_pos_z'], p['release_spin_rate'], p['release_extension'],
                   p['hc_x'], p['hc_y'], p['vx0'], p['vy0'], p['vz0'], p['ax'], p['ay'], p['az'],
                   p['hit_distance_sc'], p['launch_speed'], p['launch_angle'], p['events'],
                   p['description']] for p in pitch_j]
    balls, strikes, pitch_type, pitch_speed, plate_x, plate_z, release_x, release_y, release_z, spin_rate, extension, \
        hc_x, hc_y, vx0, vy0, vz0, ax, ay, az, hit_dist, launch_speed, launch_angle, events, description = map(
            list, zip(*pitch_info))
    # print('len(pitch_speed): {}'.format(len(pitch_speed)))
    # print('len(release_x): {}'.format(len(release_x)))
    # print('len(hc_x): {}'.format(len(hc_x)))

    pitch_speed = fill_null(pitch_speed, 0, norm_val=norm_vals.get('release_speed', 1.0))
    release_x = fill_null(release_x, 0, norm_val=norm_vals.get('release_pos_x', 1.0))
    release_y = fill_null(release_y, 0, norm_val=norm_vals.get('release_pos_y', 1.0))
    release_z = fill_null(release_z, 0, norm_val=norm_vals.get('release_pos_z', 1.0))
    spin_rate = fill_null(spin_rate, 0, norm_val=norm_vals.get('release_spin_rate', 1.0))
    extension = fill_null(extension, 0, norm_val=norm_vals.get('release_extension', 1.0))
    hc_x = fill_null(hc_x, 0, norm_val=norm_vals.get('hc_x', 1.0))
    hc_y = fill_null(hc_y, 0, norm_val=norm_vals.get('hc_y', 1.0))
    vx0 = fill_null(vx0, 0, norm_val=norm_vals.get('vx0', 1.0))
    vy0 = fill_null(vy0, 0, norm_val=norm_vals.get('vy0', 1.0))
    vz0 = fill_null(vz0, 0, norm_val=norm_vals.get('vz0', 1.0))
    ax = fill_null(ax, 0, norm_val=norm_vals.get('ax', 1.0))
    ay = fill_null(ay, 0, norm_val=norm_vals.get('ay', 1.0))
    az = fill_null(az, 0, norm_val=norm_vals.get('az', 1.0))
    hit_dist = fill_null(hit_dist, 0, norm_val=norm_vals.get('hit_distance_sc', 1.0))
    launch_speed = fill_null(launch_speed, 0, norm_val=norm_vals.get('launch_speed', 1.0))
    launch_angle = fill_null(launch_angle, 0, norm_val=norm_vals.get('launch_angle', 1.0))
    plate_x = fill_null(plate_x, 0, norm_val=norm_vals.get('plate_x', 1.0))
    plate_z = fill_null(plate_z, 0, norm_val=norm_vals.get('plate_z', 1.0))

    events = [ev if ev is not None else 'none' for ev in events]

    rv_inputs = [pitch_speed, release_x, release_y, release_z, spin_rate, extension, hc_x, hc_y, vx0, vy0, vz0,
                 ax, ay, az, hit_dist, launch_speed, launch_angle, plate_x, plate_z]
    rv_inputs = list(zip(*rv_inputs))
    # print('len(rv_inputs): {}'.format(len(rv_inputs)))

    return balls, strikes, pitch_type, rv_inputs, events, description


def read_file_lines(fp, bad_data_fps=None):
    if bad_data_fps is None:
        bad_data_fps = []
    lines = []
    with open(fp, 'r') as f:
        for line in f:
            line = line.strip()
            if not line == '' and line not in bad_data_fps:
                lines.append(line)

    return lines


def group_ab_files_by_game(ab_files):
    files_by_game = []
    last_game_pk = None

    for ab_fp in ab_files:
        ab_game_pk = int(ab_fp[5:].split('-')[0])

        if ab_game_pk != last_game_pk:
            files_by_game.append([])

        files_by_game[-1].append(ab_fp)
        last_game_pk = ab_game_pk

    return files_by_game




