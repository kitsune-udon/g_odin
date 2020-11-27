import random
from functools import lru_cache

from sklearn.metrics import roc_auc_score

from de import DE


def tnr(crit, prediction, label):
    tn_fp = (label == 0).sum()
    tn = (((1 - prediction) * (1 - label)) == 1).sum()

    if tn_fp > 0:
        return (tn / tn_fp) * 100
    else:
        return 100


def tpr(crit, prediction, label):
    tp_fn = (label == 1).sum()
    tp = ((prediction * label) == 1).sum()

    if tp_fn > 0:
        return (tp / tp_fn) * 100
    else:
        return 100


class Fitness:
    def __init__(self, score, label):
        self.score = score
        self.label = label

    @lru_cache(maxsize=None)
    def __call__(self, crit):
        prediction = self.score > crit
        if tpr(crit, prediction, self.label) >= 95:
            return tnr(crit, prediction, self.label)
        else:
            return 0


class Sampler:
    def __init__(self):
        pass

    def __call__(self):
        return 2 * (random.random() - 0.5)


def tnr_at_tpr95(score, label):
    de = DE(Fitness(score, label), Sampler(), np=16, n_iter=30)
    crit, max_fitness = de.fit()

    return crit, max_fitness


def auroc(score, label):
    if (label == 0).all() or (label == 1).all():
        return 0
    return roc_auc_score(label, -score)
