"""
This problem was asked by Amazon.

An sorted array of integers was rotated an unknown number of times.

Given such an array, find the index of the element in the array in faster than linear time. If the element doesn't exist in the array, return null.

For example, given the array [13, 18, 25, 2, 8, 10] and the element 8, return 4 (the index of 8 in the array).

You can assume all the integers in the array are unique.
"""

def log_search(arr, ele, st=0, end=None):
    if not end:
        end = len(arr)-1
    if st > end:
        return 'null'

    mid = (st+end)/2

    if ele == arr[mid]:
        return mid
    if ele > arr[mid]:
        return log_search(arr, ele, mid+1, end)
    return log_search(arr[:mid], ele, st, mid-1)

def get_index(arr, ele):
    pivot = len(arr)/2

    cliff = 0
    if pivot != 0:
        while arr[pivot] > arr[pivot-1] and arr[pivot] < arr[pivot+1]:
            iterations += 1
            if arr[pivot] > arr[0]:
                pivot = (len(arr)+pivot)/2
                if pivot == len(arr)-1:
                    pivot = 0
                    break
            else:
                pivot = pivot/2
        cliff = pivot

    if ele == arr[cliff]:
        return cliff
    elif ele > arr[0]:
        return log_search(arr[0:cliff], ele)
    else:
        idx = log_search(arr[cliff:], ele)
        if idx == 'null':
            return idx
        return cliff+idx
    

if __name__ == '__main__':
    arr = [13, 18, 25, 2, 8, 10]

    ele = 8
    print(str(ele)+' occurs at index '+str(get_index(arr, ele))+' in '+str(arr))
    ele = 18
    print(str(ele)+' occurs at index '+str(get_index(arr, ele))+' in '+str(arr))
    ele = 10
    print(str(ele)+' occurs at index '+str(get_index(arr, ele))+' in '+str(arr))
    ele = 2
    print(str(ele)+' occurs at index '+str(get_index(arr, ele))+' in '+str(arr))
