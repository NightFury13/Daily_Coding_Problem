"""
This problem was asked by Airbnb.

Given a list of integers, write a function that returns the largest sum of non-adjacent numbers. Numbers can be 0 or
negative.

For example, [2, 4, 6, 2, 5] should return 13, since we pick 2, 6, and 5. [5, 1, 1, 5] should return 10, since we pick
5 and 5.

Follow-up: Can you do this in O(N) time and constant space?
"""

def maxsum(arr):
    if not arr:
        return 0
    elif len(arr) == 1:
        return arr[0]

    sum = [0] * (len(arr)+1)
    sum[0] = 0
    sum[1] = arr[0]
    sum[2] = max(arr[0], arr[1])

    for i in range(3, len(arr)+1):
        sum[i] = max(sum[i-2] + arr[i-1], sum[i-1])

    return sum[-1]

if __name__ == '__main__':
    arr = [2,4,6,2,5]
    print(arr, maxsum(arr))
    arr = [5,1,1,5]
    print(arr, maxsum(arr))

