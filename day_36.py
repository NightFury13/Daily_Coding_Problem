"""
This problem was asked by Dropbox.

Given the root to a binary search tree, find the second largest node in the tree.
"""

# Naive Solution (MaxHeap)

# >heapq._heapify_max(bst)
# >n=2
# >for i in range(n):
# >  ans = heapq.heappop(bst)
# > return ans


class Node:
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def anti_in_order(root, order_arr, k):
    if root:
        anti_in_order(root.right, order_arr, k)
        if len(order_arr) < k:
            order_arr.append(root.val)
        else:
            return
        anti_in_order(root.left, order_arr, k)

def k_max(bst, k):
    max_arr = []
    anti_in_order(bst, max_arr, k)
    return max_arr[-1]


if __name__ == '__main__':
    bst = Node(10, Node(5, Node(3), Node(6)), Node(12, None, Node(14, Node(13))))

    print(k_max(bst, k=2))

