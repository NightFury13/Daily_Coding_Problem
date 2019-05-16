"""
This problem was asked by Google.

Given pre-order and in-order traversals of a binary tree, write a function to reconstruct the tree.

For example, given the following preorder traversal:

    [a, b, d, e, c, f, g]

And the following inorder traversal:

    [d, b, e, a, f, c, g]

You should return the following tree:

        a
       / \
      b   c
     / \ / \
    d  e f  g
"""


class Node:
    def __init__(self, val, left=None, right=None):
        self.val = val 
        self.left = left
        self.right = right

def create_tree(preorder, inorder):
    if not preorder:
        return None

    root_val = preorder[0]
    
    root_inorder_idx = inorder.index(root_val)
    left_inorder = inorder[:root_inorder_idx]
    right_inorder = inorder[root_inorder_idx+1:]

    left_preorder = [val for val in left_inorder]
    left_preorder.sort(key=lambda x: preorder.index(x))
    right_preorder = [val for val in right_inorder]
    right_preorder.sort(key=lambda x: preorder.index(x))

    tree = Node(root_val, \
            create_tree(left_preorder, left_inorder), \
            create_tree(right_preorder, right_inorder))

    return tree


def print_postorder(tree):
    if tree:
        print_postorder(tree.left)
        print_postorder(tree.right)
        print(tree.val)

if __name__ == '__main__':
    preorder = ['a', 'b', 'd', 'e', 'c', 'f', 'g']
    inorder = ['d', 'b', 'e', 'a', 'f', 'c', 'g']

    tree = create_tree(preorder, inorder)

    print(preorder, inorder)
    print('postorder :')
    print_postorder(tree)
