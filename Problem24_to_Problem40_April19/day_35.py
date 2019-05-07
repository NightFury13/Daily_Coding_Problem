"""
This problem was asked by Google.

Given an array of strictly the characters 'R', 'G', and 'B', segregate the values of the array so that all the Rs come
first, the Gs come second, and the Bs come last. You can only swap elements of the array.

Do this in linear time and in-place.

For example, given the array ['G', 'B', 'R', 'R', 'B', 'R', 'G'], it should become ['R', 'R', 'R', 'G', 'G', 'B', 'B'].
"""

def swap_sort(arr):
    st_ptr = 0
    end_ptr = len(arr)-1

    while True:
        while arr[st_ptr] is 'R' and st_ptr < end_ptr:
            st_ptr += 1
        while arr[end_ptr] is not 'R' and st_ptr <= end_ptr:
            end_ptr -= 1

        if st_ptr > end_ptr:
            break

        arr[st_ptr], arr[end_ptr] = arr[end_ptr], arr[st_ptr]

    end_ptr = len(arr)-1

    while True:
        while arr[st_ptr] is not 'B' and st_ptr < end_ptr:
            st_ptr += 1
        while arr[end_ptr] is 'B' and st_ptr <= end_ptr:
            end_ptr -= 1

        if st_ptr > end_ptr:
            break

        arr[st_ptr], arr[end_ptr] = arr[end_ptr], arr[st_ptr]

    return arr


if __name__ == '__main__':
    in_arr = ['G', 'B', 'R', 'R', 'B', 'R', 'G']

    print(in_arr)
    print(swap_sort(in_arr))
