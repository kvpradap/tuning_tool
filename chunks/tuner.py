from profiler import *
from utils import *

from collections import OrderedDict

import logging
logger = logging.getLogger(__name__)


def grid_search(command, input_tables, params_command, params_grid, do_cartesian=False,
                repeat=1):

    # preprocess the parameters of the command
    # command, input_tables, params_command, params_grid):
    args = process_args(command, input_tables, params_command, params_grid)

    # preprocess the param grid to get the configuration settings
    # param grid
    if not check_param_grid(params_grid, do_cartesian):
        raise ValueError('Check the parameter grid')
    config_setting = get_config_setting(params_grid, do_cartesian)

    # grid search for each config, repeat settings
    best_config, result = do_grid_search(command, input_tables, args, params_grid.keys(),
                             config_setting, repeat)


    # finally, return the result.
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
    print(mis_args)

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




def do_grid_search(command, input_tables, args, grid_param_keys, config_setting, repeat):
    fun_name, cls_name = get_func_class_name(command)
    # if cls_name != None:
    #     sample_sizes = sample_size_setting[fun_name][cls_name]
    # else:
    #     sample_sizes = sample_size_setting[fun_name]
    sample_sizes = get_sample_size_setting(len(input_tables))

    prev_best_config, prev_best_runtime = -1, -1
    result = OrderedDict()
    for size in sample_sizes:
        i = 0
        curr_best_config, curr_best_runtime = -1, -1
        if not isinstance(size, (list, tuple)):
            size = [size]
        for key in input_tables.keys():
            args[key] = sample_table(input_tables[key], size[i])
            i += 1
            if i == 2 and fun_name == 'downsample' and args['size'] > len(sampled_table_b):
                args['size'] = len(sampled_table_b)
        result[tuple(size)] = OrderedDict()
        for config in config_setting:
            runtime = 0
            for i in range(len(grid_param_keys)):
                args[grid_param_keys[i]] = config[i]
            for i in range(repeat):
                # tmp = time_command(command, args)
                tmp = i
                runtime += tmp
            runtime = float(runtime)/repeat
            config_dict = OrderedDict()
            for i in range(len(grid_param_keys)):
                config_dict[grid_param_keys[i]] = config[i]
            config_dict['runtime'] = runtime
            result[tuple(size)] = config_dict

            if curr_best_runtime == -1 or runtime < curr_best_runtime:
                curr_best_runtime = runtime
                curr_best_config = config

        if prev_best_runtime == -1 or (curr_best_config != prev_best_config) or \
                curr_best_runtime < 30:
            prev_best_runtime = curr_best_runtime
            prev_best_config = curr_best_config
        else:
            return prev_best_config
    return prev_best_config, result





















    pass