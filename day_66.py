"""
This problem was asked by Square.

Assume you have access to a function toss_biased() which returns 0 or 1 with a probability 
that's not 50-50 (but also not 0-100 or 100-0). You do not know the bias of the coin.

Write a function to simulate an unbiased coin toss.
"""
import random

def toss_biased():
    bias = 3
    if random.randint(0, 9) < bias:
        return False
    return True

def toss_unbiased():
    """
    We want a situation where prob of series of events is same
    using the biased_func. If p is prob of biased_func,
    (p)(1-p) = (1-p)(p)  --> 
    So we can base our outputs using this information only.
    """
    toss1 = toss_biased()
    toss2 = toss_biased()

    if toss1 and not toss2:
        return True
    if not toss1 and toss2:
        return False

    return toss_unbiased()

if __name__ == '__main__':
    n_iter = 10000

    counts = {'0':0, '1':0}
    for i in range(n_iter):
        if toss_unbiased():
            counts['1'] += 1
        else:
            counts['0'] += 1

    print(counts)

