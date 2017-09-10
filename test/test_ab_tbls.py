import dmagellan
import pandas as pd
from dmagellan.blocker.attrequivalence.attr_equiv_blocker import AttrEquivalenceBlocker
from chunks.tuner import *
import logging
logging.basicConfig(level=logging.INFO)

A = pd.read_csv('../datasets/citeseer.csv')
B = pd.read_csv('../datasets/dblp.csv')
ab = AttrEquivalenceBlocker()

input_args = {'ltable':A, 'rtable':B,
        'l_block_attr':'title', 'r_block_attr':'title',
        'l_key':'id', 'r_key':'id',
        'compute':True
     }
param_grid = {'nltable_chunks': [1], 'nrtable_chunks': [4]}
result = grid_search(ab.block_tables, input_args, param_grid, repeat=1)
print(result)