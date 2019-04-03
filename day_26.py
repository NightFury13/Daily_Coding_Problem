"""
This problem was asked by Google.

Given a singly linked list and an integer k, remove the kth last element from the list. k is guaranteed to be smaller
than the length of the list.

The list is very long, so making more than one pass is prohibitively expensive.

Do this in constant space and in one pass.
"""

class Node:
    def __init__(self, val, next_node=None):
        self.val = val
        self.next = next_node

def remove_k_last(ll_head, k):
    front_ptr = ll_head

    for i in range(k):
        front_ptr = front_ptr.next

    if not front_ptr:
        ll_head = ll_head.next
    else:
        shadow_ptr = ll_head
        while front_ptr.next:
            front_ptr = front_ptr.next
            shadow_ptr = shadow_ptr.next

        shadow_ptr.next = shadow_ptr.next.next

    return ll_head

def print_list(ll_head):
    out_str = [str(ll_head.val)]

    while ll_head.next:
        ll_head = ll_head.next
        out_str.append(str(ll_head.val))

    print('('+') -> ('.join(out_str)+')')

if __name__ == '__main__':
    ll_head = Node(1, Node(2, Node(3, Node(4, Node(5)))))
    print_list(ll_head)

    k = 2
    print('k = '+str(k))
    ll_head = remove_k_last(ll_head, k)
    print_list(ll_head)
    k = 3
    print('k = '+str(k))
    ll_head = remove_k_last(ll_head, k)
    print_list(ll_head)
    k = 3
    print('k = '+str(k))
    ll_head = remove_k_last(ll_head, k)
    print_list(ll_head)
