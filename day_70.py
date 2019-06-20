"""
This problem was asked by Microsoft.

A number is considered perfect if its digits sum up to exactly 10.

Given a positive integer n, return the n-th perfect number.

For example, given 1, you should return 19. Given 2, you should return 28.
"""

def n_perfect(n):
    n_mod = sum([int(i) for i in str(n)]) % 10

    return int(str(n)+str(10-n_mod))

if __name__ == '__main__':
    n = [1, 2, 9, 15]
    for i in n:
        print(n, n_perfect(i))
