import time
import math
from dask.diagnostics import *


class Timer(object):
    def __enter__(self):
        self.start = time.time()
        return self
    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start


def execute(p):
    args = ()
    f, kwargs = None, None
    t = None
    if isinstance(p, (list, tuple)):
        f, kwargs = (p[0], p[1])
    # f(*args, **kwargs)
    try:
        with Timer() as t, Profiler() as prof, CacheProfiler() as cprof, ResourceProfiler() as rprof:
            _ = f(*args, **kwargs)
        f = 'rb_'+str(kwargs['nltable_chunks']) +'_'+str(kwargs['nrtable_chunks'])+'.html'
        visualize([prof, cprof, rprof], save=True, file_path=f, show=False)

    finally:
        return t.interval


def time_command(command, kwargs):
    p = (command, kwargs)
    return execute(p)


