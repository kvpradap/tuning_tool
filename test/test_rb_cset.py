import pandas as pd
import time
import sys
import psutil
import os
import sys

import dmagellan
from dask import threaded, multiprocessing
from dask.diagnostics import *

from chunks.tuner import *
# from dmagellan.blocker.rulebased.rule_based_blocker import RuleBasedBlocker
from dmagellan.blocker.rulebased.rule_based_blocker import RuleBasedBlocker
from dmagellan.feature.autofeaturegen import get_features_for_blocking
from dmagellan.feature.simfunctions import get_sim_funs_for_blocking
from dmagellan.feature.tokenizers import get_tokenizers_for_blocking

import logging
logging.basicConfig(level=logging.INFO)
A = pd.read_csv('../datasets/sample_citeseer_100k.csv')
B = pd.read_csv('../datasets/sample_dblp_100k.csv')
C = pd.read_csv('../datasets/candset_100k_overlap_blktbls_citeseer_th_2.csv')

A.reset_index(inplace=True, drop=True)
B.reset_index(inplace=True, drop=True)


s = A.title.str.len().sort_values().index
A1 = A.reindex(s)
A1 = A1.reset_index(drop=True)

s = B.title.str.len().sort_values().index
B1 = B.reindex(s)
B1 = B1.reset_index(drop=True)

rb = RuleBasedBlocker()
feature_table = get_features_for_blocking(A, B)
sim = get_sim_funs_for_blocking()
tok = get_tokenizers_for_blocking()

block_f = get_features_for_blocking(A1, B1)
_ = rb.add_rule(['title_title_lev_dist(ltuple, rtuple) > 6'], block_f)

rb.set_table_attrs(['title'], ['title'])

input_tables = OrderedDict()
input_tables['candset'] = C

input_args = OrderedDict()
input_args['ltable'] = A
input_args['rtable'] = B
input_args['key'] = '_id'
input_args['l_key'] = 'id'
input_args['r_key'] = 'id'
input_args['fk_ltable'] = 'l_id'
input_args['fk_rtable'] = 'r_id'
input_args['compute'] = True
input_args['show_progress'] = False
input_args['scheduler'] = multiprocessing.get

param_grid = OrderedDict()
param_grid['nchunks'] = [2, 4]
# param_grid['nrtable_chunks'] = [2, 1]

# (command, input_tables, params_command, params_grid, do_cartesian=False,
#                 repeat=1)
bestconfig, result = grid_search(rb.block_candset, input_tables, input_args, param_grid,
                          repeat=1)
print(bestconfig)
print(result.keys())
for k, v in result.iteritems():
    print(v['runtime'])
# print(result)