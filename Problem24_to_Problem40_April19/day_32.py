"""
This problem was asked by Jane Street.

Suppose you are given a table of currency exchange rates, represented as a 2D array. Determine whether there is
a possible arbitrage: that is, whether there is some sequence of trades you can make, starting with some amount A of any
currency, so that you can end up with some amount greater than A of that currency.

There are no transaction costs and you can trade fractional quantities.
"""

# SIDE NOTE : This is basically Bellman-Ford, we are trying to see if there are any negative cycles.
# https://www.dailycodingproblem.com/blog/how-to-find-arbitrage-opportunities-in-python/

from math import log

def arbitrage(n_cur, exchange):
    src = 0
    dist = [float('inf')] * n_cur

    n_edg = len(exchange)

    dist[src] = 0
    log_exchange = [(u, v, log(wt)) for u,v,wt in exchange]

    for i in range(n_cur - 1):
        for u, v, wt in log_exchange:
            if dist[v] > dist[u] + wt:
                dist[v] = dist[u] + wt

    for u, v, wt in log_exchange:
        if dist[v] > dist[u] + wt:
            print(u, v, wt, dist[u], dist[v])
            return True

    return False

if __name__ == '__main__':
    n_cur = 4
    exchange = [(0, 1, 70.0), (1, 0, 1.0/70),
                (1, 2, 10.0), (2, 1, 1.0/10),
                (2, 3, 40.0), (3, 2, 1.0/40),
                (0, 3, 100.0), (3, 0, 1.0/100)]
    print(arbitrage(n_cur, exchange))
    
    exchange = [(0, 1, 70.0), (1, 0, 1.0/70),
                (1, 2, 10.0), (2, 1, 1.0/10),
                (2, 3, 40.0), (3, 2, 1.0/40),
                (0, 3, 28000.0), (3, 0, 1.0/28000)]
    print(arbitrage(n_cur, exchange))
