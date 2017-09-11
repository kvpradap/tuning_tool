import inspect
import math
import logging
logger = logging.getLogger(__name__)
def get_default_args(func):
    """
    returns a dictionary of arg_name:default_values for the input function
    """
    args, varargs, keywords, defaults = inspect.getargspec(func)
    return dict(zip(args[-len(defaults):], defaults))

def get_func_args(func):
    return inspect.getargspec(func)[0]


def get_class_func_name(command):
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

def sample_tables(A, B, proportions):
    prop_a, prop_b = proportions[0], proportions[1]
    num_tuples_a = int(math.floor(len(A)*prop_a))
    num_tuples_b = int(math.floor(len(B)*prop_b))
    print(len(A), len(B), num_tuples_a, num_tuples_b)
    if num_tuples_a > len(A):
        num_tuples_a = len(A)
    if num_tuples_b > len(B):
        num_tuples_b = len(B)
    sampled_table_a = A.sample(num_tuples_a)
    sampled_table_b = B.sample(num_tuples_b)
    sampled_table_a.sort_index(inplace=True)
    sampled_table_b.sort_index(inplace=True)
    print(len(sampled_table_a), len(sampled_table_b))
    return sampled_table_a, sampled_table_b





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
