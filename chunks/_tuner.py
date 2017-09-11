from profiler import *
from utils import *
import logging
logger = logging.getLogger(__name__)
def grid_search(command, input_args, param_grid, do_cartesian=False, repeat=3):
    args = preprocess_args(command, input_args, param_grid)
    if 'num_chunks' in args:
        logger.info('Handle single param')
        result = handle_single_param(command, args, repeat)
    elif 'nltable_chunks' in args and 'nrtable_chunks' in args:
        logger.info('Handle two param')
        result = handle_two_param(command, args, repeat, do_cartesian)
    else:
        raise ValueError('Currently only searching diff. number of chunks is supported')

    return result




def handle_single_param(command, args, repeat):
    config_setting = args['num_chunks']
    function_name, class_name = get_class_func_name(command)
    if class_name is not None:
        sample_sizes = sample_size_setting[function_name][class_name]
    else:
        sample_sizes = sample_size_setting[function_name]
    prev_best_config, prev_best_runtime = -1, -1
    size = sample_sizes[-1]
    for size in sample_sizes:
        curr_best_config, curr_best_runtime = -1, -1
        for config in config_setting:
            runtime = 0
            for i in range(repeat):
                sampled_table = sample_table(candset, config)
                args['candset'] = sampled_table
                runtime += time_command(command, args)
            runtime = float(runtime)/repeat
            logger.info('size:{0}, config:{1}, avg.runtime:{2}'.format(size, config,
                                                                       runtime))
            if curr_best_runtime == -1 or runtime < curr_best_runtime:
                curr_best_runtime = runtime
                curr_best_config = config
        if prev_best_runtime == -1 or (curr_best_config != prev_best_config and
                                           runtime_norm(curr_best_runtime,
                                                        prev_best_runtime) > 0.05):
            prev_best_runtime = curr_best_runtime
            prev_best_config = curr_best_config
        else:
            return size
    return size

def handle_two_param(command, args, repeat, do_cartesian):
    ltable_setting = args['nltable_chunks']
    if not isinstance(ltable_setting, list):
        ltable_setting = [ltable_setting]
    rtable_setting = args['nrtable_chunks']
    if not isinstance(rtable_setting, list):
        rtable_setting = [rtable_setting]
    if do_cartesian:
        config_setting = list(itertools.product(ltable_setting, rtable_setting))
    else:
        assert (len(ltable_setting) == len(rtable_setting))
        config_setting = zip(ltable_setting, rtable_setting)
    function_name, class_name = get_class_func_name(command)
    if class_name is not None:
        sample_sizes = sample_size_setting[function_name][class_name]
    else:
        sample_sizes = sample_size_setting[function_name]
    prev_best_config, prev_best_runtime = -1, -1
    # size = sample_sizes[-1]
    ltable = args['ltable']
    rtable = args['rtable']

    for size in sample_sizes:
        curr_best_config, curr_best_runtime = -1, -1
        for config in config_setting:
            runtime = 0
            args['nltable_chunks'] = config[0]
            args['nrtable_chunks'] = config[1]
            for i in range(repeat):
                sampled_table_a, sampled_table_b = sample_tables(ltable,
                                                                 rtable,
                                                                 size)
                logger.info('s_a:{0}, s_b:{1}'.format(len(sampled_table_a), len(sampled_table_b)))
                if function_name == 'downsample':
                    if args['size'] > len(sampled_table_b):
                        args['size'] = len(sampled_table_b)
                args['ltable'] = sampled_table_a
                args['rtable'] = sampled_table_b
                runtime = 0
                # runtime += time_command(command, args)
            runtime = float(runtime) / repeat
            logger.info('size:{0}, config:{1}, avg.runtime:{2}'.format(size, config,
                                                                       runtime))
            if curr_best_runtime == -1 or runtime < curr_best_runtime:
                curr_best_runtime = runtime
                curr_best_config = config
        if prev_best_runtime == -1 or (curr_best_config != prev_best_config and
                                               runtime_norm(curr_best_runtime,
                                                            prev_best_runtime) > 0.05) \
                or curr_best_runtime < 30:
            prev_best_runtime = curr_best_runtime
            prev_best_config = curr_best_config
        else:
            return prev_best_config
    return prev_best_config


def runtime_norm(r1, r2):
    return float(abs(r1-r2))/min(r1, r2)

def preprocess_args(command, input_args, param_grid):

    default_args = get_default_args(command)
    function_args = get_func_args(command)

    if 'self' in function_args:
        function_args.remove('self')
    required_args = set(function_args).difference(default_args.keys())
    missing_args = set(required_args).difference(input_args.keys())
    missing_args = set(missing_args).difference(param_grid.keys())

    if len(missing_args):
        raise ValueError('The following args are required: ' + str(missing_args))
    args = default_args
    for key, value in input_args.iteritems():
        args[key] = value
    for key, value in param_grid.iteritems():
        args[key] = value

    return args