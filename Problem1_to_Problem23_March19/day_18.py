"""
This problem was asked by Google.

Given an array of integers and a number k, where 1 <= k <= length of the array, compute the maximum values of each
subarray of length k.

For example, given array = [10, 5, 2, 7, 8, 7] and k = 3, we should get: [10, 7, 8, 8], since:

    10 = max(10, 5, 2)
    7 = max(5, 2, 7)
    8 = max(2, 7, 8)
    8 = max(7, 8, 7)

Do this in O(n) time and O(k) space. You can modify the input array in-place and you do not need to store the
results. You can simply print them out as you compute them.
"""

# Not O(n)
def max_val(arr, k):
    max_val = max(arr[:k])
    print(max_val)

    for i in range(len(arr)-k):
        if arr[i+k] > max_val:
            max_val = arr[i+k]
            print(max_val)
        elif arr[i] != max_val:
            print(max_val)
        else:
            max_val = max(arr[i+1:i+1+k])
            print(max_val)

if __name__ == '__main__':
    arr = [10, 5, 2, 7, 8, 7]
    k = 3

    max_val(arr, k)
