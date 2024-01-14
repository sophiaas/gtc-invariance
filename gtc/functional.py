import torch
import numpy as np


def triple_correlation(x):
    d = x.shape[0]
    TC = torch.zeros((d, d))
    for i in range(d):
        for j in range(d):
            for k in range(d):
                TC[j, k] += x[i] * x[(i - j) % d] * x[(i - k) % d]
    return TC


def autocorrelation(x):
    d = x.shape[0]
    AC = torch.zeros((d))
    for i in range(d):
        for j in range(d):
                AC[j] += x[i] * x[(i - j) % d]
    return AC


def triple_correlation_batch(X):
    all_TCs = []
    for x in X:
        all_TCs.append(triple_correlation(x))
    return torch.stack(all_TCs)


def triple_correlation_group(x, cayley_table):
    N = cayley_table.shape[0]
    elements = np.arange(N)
    TC = torch.zeros((N, N))
    for i in range(N):
        for j in range(N):
            for k in range(N):
                g = elements[i]
                g1 = elements[j]
                g2 = elements[k]
                gg1 = cayley_table[g, g1]
                gg2 = cayley_table[g, g2]
                TC[j, k] += x[g] * x[gg1] * x[gg2]
    return TC


def get_cayley_table(group):
    elements = [x._element for x in group.elements]
    n = len(elements)
    cayley_table = np.zeros((n, n), dtype=int)
    for i, e1 in enumerate(elements):
        for j, e2 in enumerate(elements):
            combined = group._combine(e1, e2)
            for k, check in enumerate(elements):
                if group._equal(combined, check):
                    break
            cayley_table[i, j] = k
    return cayley_table


def triple_correlation_group_vectorized_batch(x, cayley_table):
    b, k, d = x.shape
    x = x.reshape((b * k, d))
    nexts = x[:, cayley_table]
    mult = x.unsqueeze(1) * x[:, cayley_table.swapaxes(0, 1)]
    TC = torch.bmm(mult, nexts)
    TC = TC.reshape((b, k, d, d))
    return TC