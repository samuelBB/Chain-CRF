import os
import time
import errno
from os.path import isdir
import multiprocessing as mp
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


def worker(f, Q, results, lock, setup):
    setup()
    while True:
        try:
            x = Q.get()
            Q.task_done()
            if not x:
                return
            y = f(x)
            with lock:
                results.append(y)
        except Exception as e:
            print(e)


def pmap(f, tasks, n_jobs=mp.cpu_count(), setup=lambda: None):
    Q       = mp.JoinableQueue()
    results = mp.Manager().list()
    lock    = mp.Lock()
    procs   = [mp.Process(target=worker, args=(f, Q, results, lock, setup))
               for _ in range(min(n_jobs, len(tasks)))]
    map(mp.Process.start, procs)
    map(Q.put, tasks + [None] * len(procs))
    Q.join()
    return results._getvalue()


def softmax(x):
    """
    softmax for 1D np-arrays
    """
    e = np.exp(x - np.max(x))
    return e / e.sum()