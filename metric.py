import random
from functools import lru_cache

from de import DE


def tnr(crit, score, label):
    b = 1 - (score < crit)
    tn_fp = (label == 0).sum()
    tn = (((1 - b) * (1 - label)) == 1).sum()

    if tn_fp > 0:
        return (tn / tn_fp) * 100
    else:
        return 0


def tpr(crit, score, label):
    b = 1 - (score < crit)
    tp_fn = (label == 1).sum()
    tp = ((b * label) == 1).sum()

    if tp_fn > 0:
        return (tp / tp_fn) * 100
    else:
        return 0


class Fitness:
    def __init__(self, score, label):
        self.score = score
        self.label = label

    @lru_cache(maxsize=None)
    def __call__(self, crit):
        if tpr(crit, self.score, self.label) >= 95:
            return tnr(crit, self.score, self.label)
        else:
            return 0


class Sampler:
    def __init__(self):
        pass

    def __call__(self):
        return 2 * (random.random() - 0.5)


def tnr_at_tpr95(score, label):
    de = DE(Fitness(score, label), Sampler(), np=8)
    crit, max_fitness = de.fit()
    print(f"crit:{crit}, max_fitness:{max_fitness}")
    return max_fitness
