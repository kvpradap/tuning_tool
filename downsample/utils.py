import inspect
import math
import logging
import dmagellan
import copy
from dmagellan.sampler.downsample.downsample import downsample_dk
from dmagellan.utils.py_utils.utils import get_stopwords_for_downsample
logger = logging.getLogger(__name__)
def get_default_args(func):
    """
    returns a dictionary of arg_name:default_values for the input function
    """
    args, varargs, keywords, defaults = inspect.getargspec(func)
    return dict(zip(args[-len(defaults):], defaults))

def get_func_args(func):
    return inspect.getargspec(func)[0]


def get_func_class_name(command):
    name = command.__name__
    cls_name = None
    if name == 'block_tables':
        cls_name = command.im_class.__name__
    return name, cls_name

def sample_table(table, proportion):
    num_tuples = int(math.floor(len(table)*proportion))
    if num_tuples > len(table):
        num_tuples = len(table)
    sampled_table = table.sample(num_tuples)
    sampled_table.sort_index(inplace=True)
    return sampled_table

def sample_tables(A, B, proportion, lid='id', rid='id', lstopwords=[], rstopwords=[]):

    num_tuples_b = int(math.floor(len(B)*proportion))
    print(len(A), len(B), len(A), num_tuples_b)
    if num_tuples_b > len(B):
        num_tuples_b = len(B)

    A1 = copy.deepcopy(A)
    B1 = copy.deepcopy(B)
    A1.reset_index(inplace=True, drop=True)
    B1.reset_index(inplace=True, drop=True)
    A1['_pos'] = list(range(len(A1)))
    B1['_pos'] = list(range(len(A1)))
    sampled_table_a, sampled_table_b = downsample_dk(A1, B1, lid, rid, size=num_tuples_b,
                                                     y=1,
                                                     lstopwords=lstopwords,
                                                     rstopwords=rstopwords, compute=True)
    #sampled_table_a.sort_index(inplace=True)
    #sampled_table_b.sort_index(inplace=True)
    sampled_table_a = sampled_table_a.sort_values('_pos')
    sampled_table_b = sampled_table_b.sort_values('_pos')
    print(len(sampled_table_a), len(sampled_table_b))
    sampled_table_a.drop('_pos', axis=1, inplace=True)
    sampled_table_b.drop('_pos', axis=1, inplace=True)
    return sampled_table_a, sampled_table_b

def get_stopwords(table, id):
    return get_stopwords_for_downsample(table, id)




sample_size_setting = {
    'block_tables':{
        'AttrEquivalenceBlocker':[(0.1, 0.1), (0.3, 0.3), (0.5, 0.5), (0.7, 0.7), (0.9,
                                                                                   0.9)],
        # 'RuleBasedBlocker':[(1, 0.1), (1, 0.3), (1, 0.5), (1, 0.7), (1, 0.9)],
        #'RuleBasedBlocker':[(1, 0.1), (1, 0.3), (1, 0.5), (1, 0.7), (1, 0.9)],
        'RuleBasedBlocker':[(1, 0.1)],
        'BlacBoxBlocker':[(0.01, 0.01), (0.05, 0.05), (0.1, 0.1), (0.15, 0.15), (0.2,
                                                                                 0.2)],
        'OverlapBlocker':[(1, 0.01), (1, 0.05), (1, 0.1), (1, 0.3), (1, 0.5), (1, 0.7),
                          (1, 0.9)],
    },
    'block_candset': [0.01, 0.05, 0.1, 0.15, 0.2],
    'downsample':[(1, 0.2), (1, 0.3), (1, 0.5), (1, 0.7), (1, 0.9)],
    'extract_feature_vecs': [0.01, 0.05, 0.1, 0.15, 0.2],
    'predict':[0.01, 0.05, 0.1, 0.15, 0.2]
}

def get_sample_size_setting(len_input_tables):
    s = [0.1]
    if len_input_tables > 1:
        return zip(s, s)
    return s

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
