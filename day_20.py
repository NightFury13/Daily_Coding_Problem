"""
This problem was asked by Google.

Given two singly linked lists that intersect at some point, find the intersecting node. The lists are non-cyclical.

For example, given A = 3 -> 7 -> 8 -> 10 and B = 99 -> 1 -> 8 -> 10, return the node with value 8.

In this example, assume nodes with the same value are the exact same node objects.

Do this in O(M + N) time (where M and N are the lengths of the lists) and constant space.
"""

def intersection(A, B):
    if len(A) > len(B):
        diff = len(A) - len(B)
        longer_list = A
        shorter_list = B
    else:
        diff = len(B) - len(A)
        longer_list = B
        shorter_list = A

    pt_1, pt_2 = longer_list[diff], shorter_list[0]
    for i in range(1, len(shorter_list)):
        if pt_1 == pt_2:
            return pt_1
        pt_1 = longer_list[diff+i]
        pt_2 = shorter_list[i]

    return None

if __name__ == '__main__':
    A = [3, 7, 8, 10]
    B = [99, 1, 8, 10]

    print(intersection(A,B))
