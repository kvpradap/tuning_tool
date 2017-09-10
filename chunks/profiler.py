import time
import math


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
    try:
        with Timer() as t:
            _ = f(*args, **kwargs)
    finally:
        return t.interval


def time_command(command, kwargs):
    p = (command, kwargs)
    return execute(p)


