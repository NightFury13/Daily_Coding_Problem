"""
Given an array of integers, return a new array such that each element at index i of the new array is the product of all
the numbers in the original array except the one at i.

For example, if our input was [1, 2, 3, 4, 5], the expected output would be [120, 60, 40, 30, 24]. If our input was [3,
2, 1], the expected output would be [2, 3, 6]
"""
import numpy as np

def not_prod(arr):
    tot_prod = np.prod(arr)
    prod_arr = [tot_prod/num for num in arr]

    return prod_arr

def not_prod_nodiv(arr):
    l_arr = len(arr)
    left, right = np.ones(l_arr), np.ones(l_arr)
    for i in range(1, l_arr):
        left[i] = left[i-1]*arr[i-1]
        right[l_arr-1-i] = right[l_arr-i]*arr[l_arr-i]

    prod_arr = [left[i]*right[i] for i in range(l_arr)]
    return prod_arr
