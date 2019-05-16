"""
This problem was asked by Google.

A unival tree (which stands for "universal value") is a tree where all nodes under it have the same value.

Given the root to a binary tree, count the number of unival subtrees.

For example, the following tree has 5 unival subtrees:

       0
      / \
     1   0
        / \
       1   0
      / \
     1   1
"""

class Node:
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
        self.is_unival = False

def unival_ct(node):
    node_ct = 0

    # Leaf nodes are unival
    if not node.left or not node.right:
        node.is_unival = True
        node_ct = 1
        return node_ct

    # Recurse
    left_ct = unival_ct(node.left)
    right_ct = unival_ct(node.right)
    
    if node.left.is_unival and node.right.is_unival and \
            node.left.val == node.right.val:
        node.is_unival = True
        node_ct = 1
    
    return node_ct + left_ct + right_ct

if __name__ == '__main__':
    head = Node(0)
    head.left = Node(1)
    head.left.left = Node(1)
    head.left.right = Node(1)
    head.right = Node(0)
    head.right.left = Node(1)
    head.right.left.left = Node(1)
    head.right.left.right = Node(1)
    head.right.right = Node(0)

    print(unival_ct(head))

