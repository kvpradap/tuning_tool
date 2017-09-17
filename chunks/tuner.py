from profiler import *
from utils import *
from copy import deepcopy


from collections import OrderedDict
import itertools

import pandas as pd

import logging
logger = logging.getLogger(__name__)
from new_tuner import  *

import dmagellan
from dmagellan.utils.py_utils.utils import build_inv_index, tokenize_strings
from dmagellan.tokenizer.whitespacetokenizer import WhiteSpaceTokenizer
from dmagellan.utils.cy_utils.stringcontainer import StringContainer
from dmagellan.blocker.overlap.overlapblocker import OverlapBlocker

def grid_search_overlap(input_tables, params_command, params_grid,
                        nbins=10, do_cartesian=False,
                        repeat=1):
    ob = OverlapBlocker()
    command = ob.block_tables
    args = process_args(command, input_tables, params_command, params_grid)
    if not check_param_grid(params_grid, do_cartesian):
        raise ValueError('Check the parameter grid')
    sample_size = 0.1
    ltable, rtable = input_tables['ltable'], input_tables['rtable']
    ltable['id'] = list(range(len(ltable)))
    rtable['id'] = list(range(len(rtable)))
    if args['word_level'] == True:
        tokenizer = WhiteSpaceTokenizer()
    else:
        tokenizer = QgramTokenizer(qval=args['q_val'])
    lid = args['l_key']
    l_block_attr = args['l_block_attr']
    s_ltable = sample_ltable(ltable, lid, l_block_attr, nbins,
                             sample_size)
    ob = OverlapBlocker()
    p = ob.process_and_tokenize_ltable(ltable, lid, l_block_attr,
                                       tokenizer, [])
    inv_index = build_inv_index([p])
    if args['word_level'] == True:
        tokenizer = WhiteSpaceTokenizer()
    else:
        tokenizer = QgramTokenizer(qval=args['q_val'])

    rid = args['r_key']
    r_block_attr = args['r_block_attr']

    s_rtable = sample_rtable(rtable, rid, r_block_attr, tokenizer, nbins,
                             sample_size, inv_index)
    args['ltable'] = s_ltable
    args['rtable'] = s_rtable


    # do staged tuning
    ltable, rtable = input_tables['ltable'], input_tables['rtable']
    copy_params_grid = deepcopy(params_grid)
    keys = params_grid.keys()
    copy_params_grid[keys[0]] = [1]

    config_setting = get_config_setting(copy_params_grid, do_cartesian)
    best_config, result = do_grid_search(command, args,
                                         params_grid.keys(),
                                         config_setting,
                                         repeat=repeat)
    print('best config after first stage: ' + str(best_config))
    print(result[best_config])
    b = best_config[1]
    copy_params_grid = deepcopy(params_grid)
    copy_params_grid[keys[1]] = [b]
    config_setting = get_config_setting(copy_params_grid, do_cartesian)
    best_config, result = do_grid_search(command, args,
                                         params_grid.keys(),
                                         config_setting,
                                         repeat=repeat)
    print('best_config: ' + str(best_config))
    return best_config, result


def grid_search(command, input_tables, params_command, params_grid,
                do_cartesian=False,
                repeat=1):

    # preprocess the parameters of the command
    # command, input_tables, params_command, params_grid):
    args = process_args(command, input_tables, params_command, params_grid)

    # preprocess the param grid to get the configuration settings
    # param grid
    if not check_param_grid(params_grid, do_cartesian):
        raise ValueError('Check the parameter grid')
    sample_size = 0.1
    if len(input_tables) == 2:

        ltable, rtable = input_tables['ltable'], input_tables['rtable']
        ltable['id'] = list(range(len(ltable)))
        rtable['id'] = list(range(len(rtable)))
        lstopwords = get_stopwords(ltable, 'id')
        rstopwords = get_stopwords(rtable, 'id')
        start = time.time()
        s_ltable, s_rtable = sample_tables(ltable, rtable, sample_size,
                                           lstopwords=lstopwords, rstopwords=rstopwords)
        n1 = len(pd.unique(s_ltable.id))
        n2 = len(pd.unique(s_rtable.id))
        print('Sampling took: ' + str(time.time()-start))
        args['ltable'] = s_ltable
        args['rtable'] = s_rtable
    else:
        candset = input_tables['candset']
        s_candset = sample_table(candset, sample_size)
        args['candset'] = s_candset

    if not do_cartesian or len(input_tables) == 1:
        config_setting = get_config_setting(params_grid, do_cartesian)
        # grid search for each config, repeat settings
        best_config, result = do_grid_search(command, args,
                                             params_grid.keys(),
                                             config_setting,
                                             repeat=repeat)
        print('Returning best config as: ' + str(best_config))
    else:
        # do staged tuning
        ltable, rtable = input_tables['ltable'], input_tables['rtable']
        copy_params_grid = deepcopy(params_grid)
        keys = params_grid.keys()
        copy_params_grid[keys[0]] = [1]

        config_setting = get_config_setting(copy_params_grid, do_cartesian)
        best_config, result = do_grid_search(command, args,
                                         params_grid.keys(),
                                         config_setting,
                                             repeat=repeat)
        print('best config after first stage: ' + str(best_config))
        print(result[best_config])
        b = best_config[1]
        copy_params_grid = deepcopy(params_grid)
        copy_params_grid[keys[1]] = [b]
        config_setting = get_config_setting(copy_params_grid, do_cartesian)
        best_config, result = do_grid_search(command, args,
                                         params_grid.keys(),
                                         config_setting,
                                             repeat=repeat)
        print('best_config: ' + str(best_config))
        return best_config, result
    # finally, return the result.
    print('Returning best config as: ' + str(best_config))
    return best_config, result

def process_args(command, input_tables, params_command, params_grid):
    # get the command arguments, by analysing command's prototype
    fun_args = get_func_args(command)

    # get default values of command arguments, by analysing command's prototype
    def_args = get_default_args(command)

    # remove 'self'
    if 'self' in fun_args:
        fun_args.remove('self')

    # required args
    req_args = set(fun_args).difference(def_args.keys())

    # get the missing params
    mis_args = set(req_args).difference(params_command)
    mis_args = set(mis_args).difference(input_tables.keys())
    mis_args = set(mis_args).difference(params_grid)

    # missing args must be empty
    if mis_args:
        raise ValueError('The following args. are missing: {0}'.str(mis_args))

    out_args = def_args
    for key, val in params_command.iteritems():
        out_args[key] = val

    return out_args


def check_param_grid(param_grid, do_cartesian):
    if len(param_grid) > 1 and not do_cartesian:
        keys = param_grid.keys()
        l = len(param_grid[keys[0]])
        for i in range(1, len(keys)):
            if len(param_grid[keys[i]]) != l:
                return False
        return True
    else:
        return True

def get_config_setting(param_grid, do_cartesian):
    for key, val in param_grid.iteritems():
        if not isinstance(val, list):
            param_grid[key] = [val]
    values = param_grid.values()
    if do_cartesian:
        config_setting = list(itertools.product(*values))
    else:
        config_setting = zip(*values)

    return config_setting


def do_grid_search(command, args, grid_param_keys, config_setting,
                    result=OrderedDict(), repeat=1):
    best_runtime = float('inf')
    best_config = None
    # result = OrderedDict()
    for config in config_setting:
        print('Trying config:' + str(config))
        # if not isinstance(config, list):
        #     config = [config]
        if not isinstance(config, tuple):
            config = tuple(config)

        if config not in result:
            runtime = 0
            for i in range(len(grid_param_keys)):
                args[grid_param_keys[i]] = config[i]
            for i in range(repeat):
                tmp = time_command(command, args)
                runtime += tmp
            runtime = float(runtime)/repeat
            config_dict = OrderedDict()
            for i in range(len(grid_param_keys)):
                config_dict[grid_param_keys[i]] = config[i]
            config_dict['runtime'] = runtime
            result[config] = config_dict
        else:
            runtime = result[config].get('runtime')
        print('Runtime: ' + str(runtime))
        if runtime < best_runtime:
            best_runtime = runtime
            best_config = config
    return best_config, result


# def do_grid_search(command, input_tables, args, grid_param_keys, config_setting, repeat):
#     fun_name, cls_name = get_func_class_name(command)
#     # if cls_name != None:
#     #     sample_sizes = sample_size_setting[fun_name][cls_name]
#     # else:
#     #     sample_sizes = sample_size_setting[fun_name]
#     sample_sizes = get_sample_size_setting(len(input_tables))
#
#     prev_best_config, prev_best_runtime = -1, -1
#     result = OrderedDict()
#     for size in sample_sizes:
#         i = 0
#         curr_best_config, curr_best_runtime = -1, -1
#         if not isinstance(size, (list, tuple)):
#             size = [size]
#         if len(input_tables) == 2:
#             keys = input_tables.keys()
#             ltable, rtable = input_tables[keys[0]], input_tables[keys[1]]
#             a, b = sample_tables(ltable, rtable, size)
#             input_tables[keys[0]] = a
#             input_tables[keys[1]] = b
#
#             if fun_name == 'downsample' and args['size'] > len(
#                     sampled_table_b):
#                 args['size'] = len(sampled_table_b)
#
#
#         else:
#             for key in input_tables.keys():
#                 args[key] = sample_table(input_tables[key], size[i])
#         result[tuple(size)] = OrderedDict()
#         for config in config_setting:
#             runtime = 0
#             for i in range(len(grid_param_keys)):
#                 args[grid_param_keys[i]] = config[i]
#             for i in range(repeat):
#                 # tmp = time_command(command, args)
#                 tmp = i
#                 runtime += tmp
#             runtime = float(runtime)/repeat
#             config_dict = OrderedDict()
#             for i in range(len(grid_param_keys)):
#                 config_dict[grid_param_keys[i]] = config[i]
#             config_dict['runtime'] = runtime
#             result[tuple(size)] = config_dict
#
#             if curr_best_runtime == -1 or runtime < curr_best_runtime:
#                 curr_best_runtime = runtime
#                 curr_best_config = config
#
#         if prev_best_runtime == -1 or (curr_best_config != prev_best_config):
#             prev_best_runtime = curr_best_runtime
#             prev_best_config = curr_best_config
#         else:
#             return prev_best_config
#     return prev_best_config, result
