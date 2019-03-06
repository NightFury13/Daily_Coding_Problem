"""
Given an array of integers, find the first missing positive integer in linear time and constant space. In other words,
find the lowest positive integer that does not exist in the array. The array can contain duplicates and negative numbers
as well.

For example, the input [3, 4, -1, 1] should give 2. The input [1, 2, 0] should give 3.

You can modify the input array in-place.
"""

"""
SOLUTIONS :
    Brute force - search for all numbers n+1 if array of size n [ O(n^2)]
    Sort - then simple linear search [ O(logn + n) ]
    Hash - keep a dict of all +ve int, then do another linear iteration of N+1 elements in hash [ O(n) + O(n) space ]

    Best -
    1) Segregate positive numbers from others i.e., move all non-positive numbers to left side. In the following code,
       segregate() function does this part.
    2) Now we can ignore non-positive elements and consider only the part of array which contains all positive
       elements. We traverse the array containing all positive numbers and to mark presence of an element x, we
       change the sign of value at index x to negative. We traverse the array again and print the first index which
       has positive value.
"""

def segregate(arr):
    pos_idx = 0
    for i in range(len(arr)):
        if arr[i] < 1:
            arr[i], arr[pos_idx] = arr[pos_idx], arr[i]
            pos_idx = pos_idx+1

    return pos_idx

# Making general solution for finding Kth missing int
def missing_int(arr, k):
    pos_idx = segregate(arr)
    arr = arr[pos_idx:]
    l_arr = len(arr)
    for i in range(l_arr):
        if arr[i] - 1 < l_arr:
            arr[arr[i]-1] = -1 * abs(arr[arr[i]-1])

    miss_ctr = 0
    for i in range(l_arr):
        if arr[i] > 0:
            miss_ctr += 1
            if miss_ctr is k:
                return i+1

if __name__ == '__main__':
    for k, arr in ((1, [3,4,-1,1]), (2, [3,4,-1,1]), (3, [1,2,0]), (1, [3,5,-1,5,0,2,7,1])):
        print(arr, k, missing_int(arr, k))
