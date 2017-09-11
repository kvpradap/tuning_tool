#!/scratch/pradap/local/share/anaconda2/bin/python
import pandas as pd
import time
import sys
import psutil
import os
import sys

from dask import threaded, multiprocessing
from dask.diagnostics import *

from dmagellan.blocker.rulebased.rule_based_blocker import RuleBasedBlocker
from dmagellan.feature.autofeaturegen import get_features_for_blocking
from dmagellan.feature.simfunctions import get_sim_funs_for_blocking
from dmagellan.feature.tokenizers import get_tokenizers_for_blocking

import py_entitymatching as em


# sys.path.append('/scratch/pradap/python-work/mur_expts/scaling-expts')
#ProgressBar().register()
if not len(sys.argv) == 3:
    print('Syntax: ....py num_lchunks num_rchunks')
    print(sys.argv)
    sys.exit(-1)


num_lchunks = int(sys.argv[1])
num_rchunks = int(sys.argv[2])
print('n_lchunks: {0}, n_rchunks: {1}, n_workers: 4'.format(num_lchunks, num_rchunks))

path = '../datasets/'

print('--------------------------------------------------------')
print('DASK; CITESEER; RULE-BASED BLOCKER; 100K')
print("Mem. usage before reading:{0} (GB)".format( psutil.virtual_memory().used/1e9))
A = pd.read_csv(path + 'sample_citeseer_100k.csv')
B = pd.read_csv(path + 'sample_dblp_100k.csv')
B = B.sample(10000)
A.reset_index(inplace=True, drop=True)
B.reset_index(inplace=True, drop=True)
s = A.title.str.len().sort_values().index
A1 = A.reindex(s)
A1 = A1.reset_index(drop=True)

s = B.title.str.len().sort_values().index
B1 = B.reindex(s)
B1 = B1.reset_index(drop=True)

print("Mem. usage after reading:{0} (GB)".format(psutil.virtual_memory().used/1e9))

rb = RuleBasedBlocker()
feature_table = get_features_for_blocking(A, B)
sim = get_sim_funs_for_blocking()
tok = get_tokenizers_for_blocking()


#feature_string = """jaccard(wspace(ltuple['title'].lower()), wspace(rtuple[
#'title'].lower()))"""
 
#feature = em.get_feature_fn(feature_string, tok, sim)
#em.add_feature(feature_table, 'jac_ws_title_title', feature)
#rb.add_rule(['jac_ws_title_title(ltuple, rtuple) < 0.8'], feature_table)
block_f = get_features_for_blocking(A1, B1)
_ = rb.add_rule(['title_title_lev_dist(ltuple, rtuple) > 6'], block_f)

rb.set_table_attrs(['title'], ['title'])

memUsageBefore = psutil.virtual_memory().used/1e9
timeBefore = time.time()


C = rb.block_tables(A1, B1, 'id', 'id', nltable_chunks=num_lchunks, nrtable_chunks=num_rchunks, compute=False,
                    show_progress=False)
with Profiler() as prof, CacheProfiler() as cprof, ResourceProfiler() as rprof:
    D = C.compute(get=multiprocessing.get)
# f = 'skew_'+str(num_lchunks)+'_'+str(num_rchunks)+'.html'

visualize([prof, cprof, rprof], save=True, file_path=f, show=False)

timeAfter = time.time()
memUsageAfter = psutil.virtual_memory().used/1e9

print('Mem.usage (after reading): {0} (GB), Mem.usage (after rule-based blocking: {1} (GB), diff: {2} (GB)'.format(memUsageBefore, memUsageAfter, memUsageAfter-memUsageBefore))
print('Time. diff: {0} (secs)'.format(timeAfter-timeBefore))
print(len(D))

# writing the dataset.
#print('Writing the dataset to disk.')
#filename = 'candset_100k_overlap_blktbls_citeseer_th_{0}.csv'.format(overlap_threshold)
#D.to_csv(path+filename, index=False)
print('--------------------------------------------------------')


