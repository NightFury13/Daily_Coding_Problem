"""
This problem was asked by Amazon.

There exists a staircase with N steps, and you can climb up either 1 or 2 steps at a time. Given N, write a function
that returns the number of unique ways you can climb the staircase. The order of the steps matters.

For example, if N is 4, then there are 5 unique ways:

    1, 1, 1, 1
    2, 1, 1
    1, 2, 1
    1, 1, 2
    2, 2
    What if, instead of being able to climb 1 or 2 steps at a time, you could climb any number from a set of positive
    integers X? For example, if X = {1, 3, 5}, you could climb 1, 3, or 5 steps at a time.
"""

def climb_ways(n):
    if n == 1:
        return [[1]]
    if n == 2:
        return [[1,1], [2]]

    return [[1]+way for way in climb_ways(n-1)] + [[2]+way for way in climb_ways(n-2)]

if __name__ == '__main__':
    n = 4
    print(n)
    ways = climb_ways(n)
    for way in ways:
        print way

    n = 7
    print(n)
    ways = climb_ways(n)
    for way in ways:
        print way
