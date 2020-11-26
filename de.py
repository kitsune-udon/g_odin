import random


def pick(xs, i):
    def op(x, y, z):
        if x + y >= z:
            return x + y - z
        else:
            return x + y

    def iter(depth, remain):
        if depth == 0:
            r = [random.randint(0, remain-1)]
            return r
        else:
            j = random.randint(0, remain-1)
            r = [op(x+1, j, remain) for x in iter(depth-1, remain-1)]
            r.append(j)
            return r

    n = len(xs)
    assert n >= 4
    assert 0 <= i and i < n
    return [xs[op(x+1, i, n)] for x in iter(2, n-1)]


class DE:
    def __init__(self, target, sampler, f=0.8, cr=0.9, np=4, n_iter=10):
        self.target = target
        self.sampler = sampler
        self.f = f
        self.cr = cr
        self.np = np
        self.n_iter = n_iter

    def fit(self):
        pop = []
        for _ in range(self.np):
            pop.append(self.sampler())

        for _ in range(self.n_iter):
            pop_succ = []
            for i, x in enumerate(pop):
                a, b, c = pick(pop, i)

                if random.random() < self.cr:
                    y = a + self.f * (b - c)
                else:
                    y = x

                if self.target(y) > self.target(x):
                    z = y
                else:
                    z = x
                pop_succ.append(z)

            pop = pop_succ

        max_fitness, max_i = -float('inf'), -1
        for i in range(len(pop)):
            fitness = self.target(pop[i])
            if fitness > max_fitness:
                max_fitness = fitness
                max_i = i

        return pop[max_i], max_fitness
