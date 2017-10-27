from collections import OrderedDict
from copy import deepcopy
import itertools
import pandas as pd

from new_tuner import *
from profiler import *
from utils import *

import dmagellan
from dmagellan.utils.py_utils.utils import build_inv_index, tokenize_strings
from dmagellan.tokenizer.whitespacetokenizer import WhiteSpaceTokenizer
from dmagellan.utils.cy_utils.stringcontainer import StringContainer
from dmagellan.blocker.overlap.overlapblocker import OverlapBlocker


def grid_search(command, command_params, metadata,
                do_cartesian=False, sample_size=0.1, repeat=1):


    args = process_args(command, command_params)
    # input_tables, input_keys, search_params
    input_tables = metadata['input_tables']
    input_keys = metadata['input_keys']
    search_params = metadata['search_params']

    # preprocess the search values to get the configuration settings
    if not check_search_vals(args, search_params, do_cartesian):
        raise ValueError('Check the search values')

    if len(input_tables) == 2:
        ltable, rtable = args[input_tables[0]], args[input_tables[1]]
        ltable[input_keys[0]] = list(range(len(ltable)))
        rtable[input_keys[1]] = list(range(len(rtable)))
        lstopwords = get_stopwords(ltable, input_keys[0])
        rstopwords = get_stopwords(rtable, input_keys[1])
        s_ltable, s_rtable = sample_tables(ltable, rtable, sample_size,
                                           lstopwords=lstopwords, rstopwords=rstopwords)
        args[input_keys[0]] = s_ltable
        args[input_keys[1]] = s_rtable
    else:
        candset = args[input_tables[0]]
        s_candset = sample_table(candset, sample_size)
        args[input_tables[0]] = s_candset

    if not do_cartesian or len(input_tables) == 1:
        config_setting = get_config_setting(args, search_params, do_cartesian)
        best_config, result = do_grid_search(command, args, search_params,
                                             config_setting, repeat=repeat)
        print('Returning best config as : ' + str(best_config))
    else:
        # do staged tuning
        # ltable, rtable = args[input_tables[0]], args[input_tables[1]]
        search_values = get_search_values(args, search_params)
        copy_search_values = deepcopy(search_values)
        copy_search_values[0] = [1]
        args = set_search_values(args, search_params, copy_search_values)

        config_setting = get_config_setting(args, search_params, do_cartesian)
        best_config, result = do_grid_search(command, args, search_params,
                                             config_setting, repeat=repeat)
        print('Best config after stage 1: ' + str(best_config))

        b = best_config[1]
        copy_search_values = deepcopy(search_values)
        copy_search_values[1] = [b]
        args = set_search_values(args, search_params, copy_search_values)

        config_setting = get_config_setting(args, search_params, do_cartesian)
        best_config, result = do_grid_search(command, args, search_params,
                                             config_setting, repeat=repeat)
        print('Best config after stage 2: ' + str(best_config))
    return best_config, result


def get_search_values(args, search_params):
    values = []
    for param in search_params:
        values.append(args[param])
    return values
def set_search_values(args, search_params, values):
    i = 0
    for param in search_params:
        args[param] = values[i]
        i += 1
    return args
def process_args(command, command_params):
    # get the command arguments, by analysing command's prototype
    fun_args = get_func_args(command)

    # get default values of command arguments, by analysing command's prototype
    def_args = get_default_args(command)

    # remove 'self'
    if 'self' in fun_args:
        fun_args.remove('self')

    # required args
    req_args = set(fun_args).difference(def_args.keys())

    # get the missing args
    mis_args = set(req_args).difference(command_params)

    # missing args must be empty
    if mis_args:
        raise ValueError('The following args. are missing: {0}'.str(mis_args))

    out_args = def_args
    for key, val in command_params.iteritems():
        out_args[key] = val

    return out_args



def check_search_vals(args, search_vals, do_cartesian):
    if len(search_vals) > 1 and not do_cartesian:
        l = args[search_vals[0]]
        for i in range(1, len(search_vals)):
            if len(args[search_vals[i]]) != l:
                return False
        return True
    else:
        return True

def get_config_setting(args, search_values, do_cartesian):
    values = []
    for v in search_values:
        values.append(args[v])
    if do_cartesian:
        config_setting = list(itertools.product(*values))
    else:
        config_setting = zip(*values)
    return config_setting


def do_grid_search(command, args, search_params, config_setting, result=OrderedDict(),
                   repeat=1):
    best_runtime = float('inf')
    best_config = None
    for config in config_setting:
        print('Trying config: ' + str(config))
        if not instance(config, tuple):
            config = tuple(config)
        if config not in result:
            runtime = 0
            for i in range(len(search_params)):
                args[search_params[i]] = config[i]
            for i in range(repeat):
                tmp = time_command(command, args)
                runtime += tmp
            runtime = float(runtime)/repeat
            config_dict = OrderedDict()
            for i in range(len(search_params)):
                config_dict[search_params[i]] = config[i]
            config_dict['runtime'] = runtime
            result[config] = config_dict
        else:
            runtime = result[config].get('runtime')
        print('Runtime: ' + str(runtime))
        if runtime < best_runtime:
            best_runtime = runtime
            best_config = config
    return best_config, result

    pass