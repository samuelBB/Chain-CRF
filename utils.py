import os
import time
import errno
from os.path import isdir
from contextlib import contextmanager

import numpy as np


@contextmanager
def timed(msg, obj=None):
    start = time.time()
    yield
    total = time.time() - start
    print '\n[DONE] {} - {:.4f} sec'.format(msg, total)
    if obj:
        obj.train_time = total



def memoize(f):
    """
    memoization decorator (https://wiki.python.org/moin/PythonDecoratorLibrary)
    """
    cache = {}
    def _memoize(*args, **kwargs):
        key = str(args) + str(kwargs)
        if key not in cache:
            cache[key] = f(*args, **kwargs)
        return cache[key]
    return _memoize


def mkdir_p(path):
    """
    bash `mkdir -p`
    """
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and isdir(path):
            pass
        else:
            raise


def softmax(x):
    """
    softmax for 1D np-arrays
    """
    e = np.exp(x - np.max(x))
    return e / e.sum()