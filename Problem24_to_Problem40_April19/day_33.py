"""
This problem was asked by Microsoft.

Compute the running median of a sequence of numbers. That is, given a stream of numbers, print out the median of the
list so far on each new element.

Recall that the median of an even-numbered list is the average of the two middle numbers.

    For example, given the sequence [2, 1, 5, 7, 2, 0, 5], your algorithm should print out:

    2
    1.5
    2
    3.5
    2
    2
    2
"""

def insert_into(ele, sorted_arr):
    lo = 0
    hi = len(sorted_arr)

    while lo < hi:
        mid = (hi+lo)/2
        if ele < sorted_arr[mid]:
            hi = mid
        else:
            lo = mid+1

    return sorted_arr[:lo]+[ele]+sorted_arr[lo:]

def run_median(arr):
    sorted_arr = []

    for i in range(len(arr)):
        sorted_arr = insert_into(arr[i], sorted_arr)

        # Odd sized list
        if i%2==0:
            print(float(sorted_arr[i/2]))
        else:
            print(float(sorted_arr[i/2]+sorted_arr[(i/2)+1])/2)
    return

if __name__ == '__main__':
    arr = [2,1,5,7,2,0,5]

    print(arr)
    run_median(arr)
