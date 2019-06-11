"""
This problem was asked by Amazon.

Given an array of numbers, find the maximum sum of any contiguous subarray of the array.

For example, given the array [34, -50, 42, 14, -5, 86], the maximum sum would be 137, since we would take elements 42,
14, -5, and 86.

Given the array [-5, -1, -8, -9], the maximum sum would be 0, since we would not take any elements.

Do this in O(N) time.
"""

def max_subarr_sum(arr):
    max_sum = 0

    cur_sum = 0
    for ele in arr:
        if cur_sum + ele > 0:
            cur_sum += ele
            if cur_sum > max_sum:
                max_sum = cur_sum
        else:
            cur_sum = 0

    return max_sum

if __name__ == '__main__':
    in_arrs = [[34, -50, 42, 14, -5, 86], [-5, -1, -8, -9]]

    for in_arr in in_arrs:
        print(in_arr, max_subarr_sum(in_arr))
