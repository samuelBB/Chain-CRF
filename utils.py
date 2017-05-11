import time
from contextlib import contextmanager


@contextmanager
def timed(msg):
    start = time.time()
    yield
    print '\n{} - {:.4f} secs\n'.format(msg, time.time() - start)


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