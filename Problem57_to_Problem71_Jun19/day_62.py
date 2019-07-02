"""
This problem was asked by Facebook.

There is an N by M matrix of zeroes. Given N and M, write a function to count the number of ways of starting at the top-left corner and getting to the bottom-right corner. You can only move right or down.

For example, given a 2 by 2 matrix, you should return 2, since there are two ways to get to the bottom-right:

Right, then down
Down, then right
Given a 5 by 5 matrix, there are 70 ways to get to the bottom-right.
"""

def n_ways(n, m):
    if n == 1 or m == 1:
        return 1

    return n_ways(n-1, m) + n_ways(n, m-1)

if __name__ == '__main__':
    ns = [1, 2, 5]
    ms = [2, 3, 5]

    for n in ns:
        for m in ms:
            print(n, m, n_ways(n, m))
