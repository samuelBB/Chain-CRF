from itertools import izip

import numpy as np


class Evaluator:
    def __init__(self, hamming=True, zero1=True):
        self.metrics = []
        if hamming:
            self.metrics.append(self.hamming)
        if zero1:
            self.metrics.append(self.zero1)
        self.metric_names = [m.__name__ for m in self.metrics]

    def hamming(self, Ys):
        return np.mean([np.mean([y_t != y_p for y_t, y_p in izip(*ys)])
            for ys in izip(*Ys)])

    def zero1(self, Ys):
        return np.mean([not np.array_equal(*ys) for ys in izip(*Ys)])

    def __call__(self, *ys):
        return [m(ys) for m in self.metrics]

    def get_names(self, vs):
        return zip(self.metric_names, vs)


if __name__ == '__main__':
    # import random
    # y_true = [np.random.randint(0, 27, random.randint(3, 14)) for _ in range(1000)]
    # y_pred = [y for y in y_true]

    y_true = [[1, 2, 3, 4], [5, 4], [7, 8, 9]]
    y_pred = [[1, 2, 3, 5], [4, 5], [7, 8, 9]]

    ev = Evaluator()
    losses = ev(y_true, y_pred)
    print losses
    print ev.get_names(losses)