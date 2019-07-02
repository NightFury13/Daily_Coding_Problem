"""
This problem was asked by Facebook.

Given a list of integers, return the largest product that can be made by multiplying any three integers.

For example, if the list is [-10, -10, 5, 2], we should return 500, since that's -10 * -10 * 5.

You can assume the list has at least three integers.
"""

def max_product(arr):
    n_max1, n_max2, n_max3 = 0, 0, 0
    p_max1, p_max2 = 0, 0

    for val in arr:
        if val > 0:
            if val > p_max1:
                p_max3 = p_max2
                p_max2 = p_max1
                p_max1 = val
            elif val > p_max2:
                p_max3 = p_max2
                p_max2 = val
            elif val > p_max3:
                p_max3 = val
        else:
            if val < n_max1:
                n_max2 = n_max1
                n_max1 = val
            elif val < n_max2:
                n_max2 = val

    return max(p_max1*p_max2*p_max3, n_max1*n_max2*p_max1)

if __name__ == '__main__':
    arr = [-10, -10, 5, 2]
    print(arr, max_product(arr))

    arr = [-10, 10, 5, 2, -20, 20]
    print(arr, max_product(arr))
