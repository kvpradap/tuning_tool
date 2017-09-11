import time
import math
from dask.diagnostics import *


class Timer(object):
    def __enter__(self):
        self.start = time.clock()
        return self
    def __exit__(self, *args):
        self.end = time.clock()
        self.interval = self.end - self.start


def execute(p):
    args = ()
    f, kwargs = None, None
    t = None
    if isinstance(p, (list, tuple)):
        f, kwargs = (p[0], p[1])
        kwargs['compute'] = False
    try:
        # with Timer() as t:
        with Profiler() as prof:
            C = f(*args, **kwargs)
            _ = C.get(kwargs['scheduler'])

    finally:
        return 5.0


def time_command(command, kwargs):
    p = (command, kwargs)
    return execute(p)


