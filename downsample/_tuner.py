from profiler import *

from utils import *

logger = logging.getLogger(__name__)

from sampler import *
from dmagellan.sampler.downsample.downsample import downsample_dk


def tune_for_ds(input_tables, params_command, params_grid,
                sample_proportion_a=0.1,
                sample_proportion_b=0.01,
                n_bins=50,
                do_cartesian=False, repeat=1, seed=0):
    command = downsample_dk
    args = process_args(command, input_tables, params_command, params_grid)
    if not check_param_grid(params_grid, do_cartesian):
        raise ValueError('Check the parameter grid')

    # -------- #
    # 1. check if the tables need to be swapped or not.
    print('Checking whether the tables need to be swapped....')
    should_swap, s_ltable, s_rtable = do_table_swap(command, input_tables, args,
                                                    sample_proportion_a,
                                                    sample_proportion_b,
                                                    n_bins, repeat, seed)
    print('should_swap: {0}'.format(should_swap))
    args['swap'] = should_swap
    args['ltable'] = s_ltable
    args['rtable'] = s_rtable
    #-------#
    # 2. tune rtable partitions
    copy_params_grid = deepcopy(params_grid)
    copy_params_grid["nltable_chunks"] = [4]
    n_r = 4
    max_n = 100
    args['nlcunks'] = 4
    min_runtime = float('inf')
    nr_config = 4
    epsilon = 1
    count = 1
    while n_r <= max_n:
        args['nrcunks'] = n_r
        runtime = 0
        for i in range(repeat):
            tmp = time_command(command, args)
            runtime += tmp
        runtime = runtime/repeat
        if runtime + epsilon < min_runtime:
            min_runtime = runtime
            nr_config = nr
        else:
            break
        count += 1
        n_r = 4*count
    if nr_config == 4:
        print('Got nrtable partitions as {0}'.format(n_r))
    else:
        pass
    #     print('Doing binary search')
    #     # do a binary search
    #     l = nr_config/4.0
    #     r = nr_config
    #     while l < r:
    #         m = l + (r-l)//2
    #         for i in range(repeat):
    #             tmp = time_command(command, args)
    #             runtime += tmp
    #         runtime = runtime/repeat
    #         if runtime + epsilon < min_runtime:
    #             min_runtime = runtime
    #             nr_config = nr
    #             l = m + 1
    #         else:
    #             r = m - 1
    #     print('Got nr config as '.format((nr_config)))
    # # copy_params_grid["nrtable_chunks"] = [4]



    # config_setting = get_config_setting(copy_params_grid, do_cartesian)
    # best_config, result = do_grid_search(command, args, copy_params_grid.keys(),
    #                                      config_setting, repeat=repeat)
    # print('Best config after the first stage: {0}'.format(best_config))
    # print(result[best_config])
    #------#
    # 3. tune ltable partitions
    # copy_params_grid = deepcopy(params_grid)
    # b_rtable_partitions = best_config[1]
    # copy_params_grid['nrtable_chunks'] = [b_rtable_partitions]
    # config_setting = get_config_setting(copy_params_grid, do_cartesian)
    # best_config, result = do_grid_search(command, args, copy_params_grid.keys(),
    #                                      config_setting, repeat=repeat)

    copy_params_grid = deepcopy(params_grid)
    n_l = 4
    max_n = 100
    args['nrcunks'] = nr_config
    min_runtime = float('inf')
    nl_config = 4
    epsilon = 1
    count = 1
    while n_l <= max_n:
        args['nlcunks'] = n_l
        runtime = 0
        for i in range(repeat):
            tmp = time_command(command, args)
            runtime += tmp
        runtime = runtime/repeat
        if runtime + epsilon < min_runtime:
            min_runtime = runtime
            nl_config = n_l
        else:
            break
        count += 1
        n_l = 4*count
    # if nl_config == 4:
    #     print('Got nrtable partitions as {0}'.format(n_l))
    # else:
    #     print('Doing binary search')
    #     # do a binary search
    #     l = nl_config/4.0
    #     r = nl_config
    #
    #     while l < r:
    #         m = l + (r-l)//2
    #         runtime = 0
    #         for i in range(repeat):
    #             tmp = time_command(command, args)
    #             runtime += tmp
    #         runtime = runtime/repeat
    #         if runtime + epsilon < min_runtime:
    #             min_runtime = runtime
    #             nr_config = n_l
    #             l = m + 1
    #         else:
    #             r = m - 1
    #     print('Got nr config as '.format((nr_config)))



    print('Best config after the second stage: {0}'.format(best_config))
    print(result[best_config])
    return best_config, result


def do_table_swap(command, input_tables, args, sample_proportion_a, sample_proportion_b,
                  n_bins, repeat, seed):
    ltable, rtable = input_tables['ltable'], input_tables['rtable']
    lid = args['l_key']
    rid = args['r_key']
    lstopwords = args['lstopwords']
    rstopwords = args['rstopwords']
    s_ltable_before = sample_stratified_length(ltable, lid, nbins=n_bins, seed=seed,
                                               sample_proportion=sample_proportion_a)
    s_rtable_before = sample_stratified_probelen(rtable, s_ltable_before, rid, lid,
                                                 nbins=n_bins,
                                                 seed=seed,
                                                 sample_proportion=sample_proportion_b,
                                                 lstopwords=lstopwords,
                                                 rstopwords=rstopwords)
    args['ltable'] = s_ltable_before
    args['rtable'] = s_rtable_before
    # copy_params_grid = deepcopy(params_grid)
    # copy_params_grid["nltable_chunks"] = [4]
    # copy_params_grid["nrtable_chunks"] = [4]
    args['nltable_chunks'] = 4
    args['nrtable_chunks'] = 4
    args['swap'] = True
    runtime = 0
    for i in range(repeat):
        tmp = time_command(command, args)
        runtime += tmp
    before_swap = runtime / float(repeat)

    s_rtable_after = sample_stratified_length(rtable, rid, nbins=n_bins, seed=seed,
                                              sample_proportion=sample_proportion_a)

    s_ltable_after = sample_stratified_probelen(ltable, s_rtable_after, lid, rid,
                                                nbins=n_bins,
                                                seed=seed,
                                                sample_proportion=sample_proportion_b,
                                                lstopwords=lstopwords,
                                                rstopwords=rstopwords)
    args['ltable'] = s_ltable_after
    args['rtable'] = s_rtable_after
    args['swap'] = True
    # copy_params_grid = deepcopy(params_grid)
    # copy_params_grid["nltable_chunks"] = [4]
    # copy_params_grid["nrtable_chunks"] = [4]
    args['nltable_chunks'] = 4
    args['nrtable_chunks'] = 4
    runtime = 0
    for i in range(repeat):
        tmp = time_command(command, args)
        runtime += tmp
    after_swap = runtime / float(repeat)
    if before_swap < after_swap:
        return (False, s_ltable_before, s_rtable_before)
    else:
        return (True, s_ltable_after, s_rtable_after)


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