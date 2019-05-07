"""
This problem was asked by Google.

Given an array of integers where every integer occurs three times except for one integer, which only occurs once, find
and return the non-duplicated integer.

For example, given [6, 1, 3, 3, 3, 6, 6], return 1. Given [13, 19, 13, 13], return 19.

Do this in O(N) time and O(1) space.
"""
import numpy as np

# O(N) time and O(N) space
def non_triplet(arr):
    set_sum = np.sum(list(set(arr)))
    return (3*set_sum - np.sum(arr))/2


if __name__ == '__main__':
    inp_arr = [6, 1, 3, 3, 3, 6, 6]
    print(inp_arr, non_triplet(inp_arr))

    inp_arr = [13, 19, 13, 13]
    print(inp_arr, non_triplet(inp_arr))

