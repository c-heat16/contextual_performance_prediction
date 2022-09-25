__author__ = 'Connor Heaton'

import math


def ab_ending_token(gamestate):
    non_bs_values = [v for k, v in gamestate.items() if k not in ['balls', 'strikes']]
    n_non_null = [0 if v in ['0', '[BOI]', '[EOAB]', '[BOAB]'] else 1 for v in non_bs_values]

    ab_ender = False if sum(n_non_null) == 0 else True
    return ab_ender


def calc_entropy(count_list):
    n_total = sum(count_list)
    pct_list = [cnt/n_total for cnt in count_list]
    entropy_components = [-1 * p * math.log(p, 2) for p in pct_list]
    entropy = sum(entropy_components)

    return entropy
