import random
from functools import lru_cache

from de import DE


def tnr(crit, score, label):
    b = 1 - (score < crit)
    tn_fp = (label == 0).sum()
    tn = ((b * label) == 0).sum()
    return (tn / tn_fp) * 100


def tpr(crit, score, label):
    b = 1 - (score < crit)
    tp_fn = (label == 1).sum()
    tp = ((b * label) == 1).sum()
    return (tp / tp_fn) * 100


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
    _, max_fitness = de.fit()
    return max_fitness
