"""
This problem was asked by Facebook.

Given a multiset of integers, return whether it can be partitioned into two subsets whose sums are the same.

For example, given the multiset {15, 5, 20, 10, 35, 15, 10}, it would return true, since we can split it up into {15, 5, 10, 15, 10} and {20, 35}, which both add up to 55.

Given the multiset {15, 5, 20, 10, 35}, it would return false, since we can't split it up into two subsets that add up to the same sum.
"""

def sub_sum(arr, k):
    if k == 0:
        return True
    if len(arr) == 0:
        return False

    return sub_sum(arr[1:], k-arr[0]) or sub_sum(arr[1:], k)

def set_split(arr):
    sum_arr = sum(arr)

    if sum_arr % 2 != 0:
        return False

    # Now find subset of elements whose sum is sum_arr/2
    return sub_sum(arr, sum_arr/2)

if __name__ == '__main__':
    arr = [15, 5, 20, 10, 35, 15, 10]
    print(arr, set_split(arr))
    arr = [15, 5, 20, 10, 35]
    print(arr, set_split(arr))

