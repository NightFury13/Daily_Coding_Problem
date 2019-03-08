"""
This problem was asked by Twitter.

Implement an autocomplete system. That is, given a query string s and a set of all possible query strings, return all
strings in the set that have s as a prefix.

For example, given the query string de and the set of strings [dog, deer, deal], return [deer, deal].

Hint: Try preprocessing the dictionary into a more efficient data structure to speed up queries.
"""


class Node:
    def __init__(self, val=None, children=[], c_vals=[]):
        self.val = val
        self.child = children
        self.c_vals = c_vals

def create_tree(head, all_s):
    for word in all_s:
        ptr = Node(word[0])
        head.child.append(ptr)
        head.c_vals.append(word[0])
        for char in word[1:]:
            if char not in ptr.c_vals:
                ptr.c_vals.append(char)
                c_node = Node(char)
                ptr.child.append(c_node)

            ptr = [c_node for c_node in ptr.child if c_node.val is char][0]
        # Add EOW char ' '
        ptr.c_vals.append(' ')
        ptr.child.append(Node(' '))

    return head

def subtree_walk(ptr, prefix):
    suggestions = []

    node_stack = ptr.child
    word_stack = [prefix+node.val for node in node_stack]
    
    while node_stack:
        pop_node = node_stack.pop(0)
        pop_word = word_stack.pop(0)

        if pop_node.val == ' ':
            suggestions.append(pop_word.strip())
            print(suggestions)
        else:
            child_nodes = pop_node.child
            add_word_stack = [pop_word+node.val for node in child_nodes]

            node_stack += child_nodes
            word_stack += add_word_stack

    return suggestions

def autocomplete(s, s_tree_head):
    prefix = ''
    ptr = s_tree_head
    for char in s:
        if char in ptr.c_vals:
            prefix += char
            ptr = [node for node in ptr.child if node.val is char][0]
        else:
            return []

    return subtree_walk(ptr, prefix)


if __name__ == '__main__':
    s = 'de'
    all_s = ['dog', 'deer', 'deal']
    print('All Strings : '+ str(all_s))
    print('Query : '+ s)

    head = Node()
    s_tree_head = create_tree(head, all_s)
    print(s_tree_head.child)

    vals = autocomplete(s, s_tree_head)
    print('AutoComplete : '+ str(vals))
