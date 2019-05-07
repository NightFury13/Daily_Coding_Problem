"""
This problem was asked by Google.

Given a list of integers S and a target number k, write a function that returns a subset of S that adds up to k. If such
a subset cannot be made, then return null.

Integers can appear more than once in the list. You may assume all numbers in the list are positive.

For example, given S = [12, 1, 61, 5, 9, 2] and k = 24, return [12, 9, 2, 1] since it sums up to 24.
"""

def sub_sum_k(arr, k):
    if len(arr) == 0:
        if k == 0:
            return True
        else:
            return False

    return sub_sum_k(arr[:-1], k-arr[-1]) or sub_sum_k(arr[:-1], k)

if __name__ == '__main__':
    arr = [12, 1, 61, 5, 9, 2]
    print(arr)
    
    k = 24
    print(k, sub_sum_k(arr, k))
    k = 4
    print(k, sub_sum_k(arr, k))
