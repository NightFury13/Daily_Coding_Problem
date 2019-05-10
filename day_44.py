"""
This problem was asked by Google.

We can determine how "out of order" an array A is by counting the number of inversions it has. Two elements A[i] and
A[j] form an inversion if A[i] > A[j] but i < j. That is, a smaller element appears after a larger element.

Given an array, count the number of inversions it has. Do this faster than O(N^2) time.

You may assume each element in the array is distinct.

For example, a sorted list has zero inversions. The array [2, 4, 1, 3, 5] has three inversions: (2, 1), (4, 1), and (4,
3). The array [5, 4, 3, 2, 1] has ten inversions: every distinct pair forms an inversion.
"""

def get_min_idx(arr):
    min_idx = 0
    for i in range(1, len(arr)):
        if arr[i] < arr[min_idx]:
            min_idx = i

    return min_idx

def get_inversions(arr):
    n_inv = 0
    l_arr = len(arr)
    for i in range(l_arr):
        # Note this is min_idx of arr[i:] not arr[:]
        min_idx = get_min_idx(arr[i:])
        n_inv += min_idx
        val = arr.pop(i+min_idx)
        arr = [val]+arr

    return n_inv

if __name__ == '__main__':
    arrs = [[2, 4, 1, 3, 5], [5, 4, 3, 2, 1], [1, 2, 3, 4, 5]]

    for arr in arrs:
        print(arr)
        print('Inversions : '+str(get_inversions(arr)))
