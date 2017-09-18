import pandas as pd
import time
import sys
import psutil
import os
import sys

sys.path.append('/scratch/pradap/python-work/tuning_tool')
import dmagellan
from dask import threaded, multiprocessing
from dask.diagnostics import *

from chunks.tuner import *
from chunks.new_tuner import *
# from dmagellan.blocker.rulebased.rule_based_blocker import RuleBasedBlocker
from dmagellan.blocker.overlap.overlapblocker import OverlapBlocker
import logging


A = pd.read_csv('../datasets/sample_citeseer_100k.csv')
B = pd.read_csv('../datasets/sample_dblp_100k.csv')

A.reset_index(inplace=True, drop=True)
B.reset_index(inplace=True, drop=True)

input_tables = OrderedDict()
input_tables['ltable'] = A
input_tables['rtable'] = B

input_args = OrderedDict()
input_args['l_key'] = 'id'
input_args['r_key'] = 'id'
input_args['compute'] = True
input_args['show_progress'] = False
input_args['scheduler'] = threaded.get
input_args['l_block_attr'] = 'title'
input_args['r_block_attr'] = 'title'
input_args['overlap_size'] = 2
input_args['rem_stop_words'] = False

param_grid = OrderedDict()
param_grid['nltable_chunks'] = [1, 2, 4, 8, 16, 32]
param_grid['nrtable_chunks'] = [1, 2, 4, 8, 16, 32]

# def grid_search_overlap(input_tables, params_command, params_grid,
#                         nbins=10, do_cartesian=False,
#                         repeat=1):
start = time.time()
result = grid_search_overlap(input_tables, input_args, param_grid, do_cartesian=True)
end = time.time()
