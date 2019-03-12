# Problem #16
This problem was asked by Twitter.

You run an e-commerce website and want to record the last N order ids in a log. Implement a data structure to accomplish
this, with the following API:

>record(order_id): adds the order_id to the log
>get_last(i): gets the ith last element from the log. i is guaranteed to be smaller than or equal to N.
>You should be as efficient with time and space as possible.

```python
class API:
    def __init__(self, N):
        self.log = [None] * N

    def record(self, order_id):
        self.log = [order_id]+self.log[:-1]

    def get_last(self, i):
        return self.log[-i]

if __name__ == '__main__':
    api = API(4)

    for i in range(10):
        api.record(i)
        if i % 3:
            log = []
            for j in range(1,5):
                log.append(api.get_last(j))
            print(log)
```
----------------------------------------------------------------
# Problem #15
This problem was asked by Facebook.

Given a stream of elements too large to store in memory, pick a random element from the stream with uniform probability.


>SOLUTION : This genre of problems is known as Reservoir Sampling (randomly select k elements from a 'large' set of n elements)
>readup : https://www.geeksforgeeks.org/select-a-random-number-from-stream-with-o1-space/
>readup : https://www.geeksforgeeks.org/reservoir-sampling/
----------------------------------------------------------------
# Problem #14
This problem was asked by Google.

The area of a circle is defined as (pi)r^2. Estimate (pi) to 3 decimal places using a Monte Carlo method.

Hint: The basic equation of a circle is x2 + y2 = r2.

```python
# READUP : https://www.geeksforgeeks.org/estimating-value-pi-using-monte-carlo/

import random

def pi_estimate(n_iter):
    circle_pts = 0
    square_pts = 0
    iter_size = 1000000

    for epoch in range(n_iter):
        for sample in range(iter_size):
            rand_x = random.random()
            rand_y = random.random()

            if (rand_x*rand_x)+(rand_y*rand_y) <= 1:
                circle_pts += 1
            square_pts += 1

    pi = 4*(float(circle_pts)/square_pts)

    return pi

if __name__ == '__main__':
    for i in range(1, 10):
        print('Iter '+str(i), pi_estimate(i))
```
----------------------------------------------------------------
# Problem #13
This problem was asked by Amazon.

Given an integer k and a string s, find the length of the longest substring that contains at most k distinct characters.

For example, given s = "abcba" and k = 2, the longest substring with k distinct characters is "bcb".

```python
def longest_subs(s, k):
    chars_map = {}

    cur_st = 0
    longest_st = 0
    longest_len = 0

    for i, char in enumerate(s):
        if char in chars_map:
            chars_map[char] += 1
        else:
            chars_map[char] = 1

        if len(chars_map) <= k:
            if i-cur_st+1 > longest_len:
                longest_len = i-cur_st+1
                longest_st = cur_st
        else:
            if chars_map[s[cur_st]] == 1:
                chars_map.pop(s[cur_st])
            else:
                chars_map[s[cur_st]] -= 1
            cur_st += 1

    return (s[longest_st:longest_st+longest_len])

if __name__ == '__main__':
    s = 'abcba'
    k = 2
    print(s, k, '->', longest_subs(s, k))
    s = 'abcadcacacaca'
    k = 3
    print(s, k, '->', longest_subs(s, k))
```
----------------------------------------------------------------

# Problem #12
This problem was asked by Amazon.

There exists a staircase with N steps, and you can climb up either 1 or 2 steps at a time. Given N, write a function
that returns the number of unique ways you can climb the staircase. The order of the steps matters.

For example, if N is 4, then there are 5 unique ways:

    1, 1, 1, 1
    2, 1, 1
    1, 2, 1
    1, 1, 2
    2, 2

EXTRA : What if, instead of being able to climb 1 or 2 steps at a time, you could climb any number from a set of positive integers X? For example, if X = {1, 3, 5}, you could climb 1, 3, or 5 steps at a time.

```python
def climb_ways(n):
    if n == 1:
        return [[1]]
    if n == 2:
        return [[1,1], [2]]

    return [[1]+way for way in climb_ways(n-1)] + [[2]+way for way in climb_ways(n-2)]

if __name__ == '__main__':
    n = 4
    print(n)
    ways = climb_ways(n)
    for way in ways:
        print way

    n = 7
    print(n)
    ways = climb_ways(n)
    for way in ways:
        print way
```
----------------------------------------------------------------
# Problem #11
This problem was asked by Twitter.

Implement an autocomplete system. That is, given a query string s and a set of all possible query strings, return all
strings in the set that have s as a prefix.

For example, given the query string de and the set of strings [dog, deer, deal], return [deer, deal].

Hint: Try preprocessing the dictionary into a more efficient data structure to speed up queries.

```python
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
```
----------------------------------------------------------------
# Problem #10

This problem was asked by Apple.

Implement a job scheduler which takes in a function f and an integer n, and calls f after n milliseconds.

```python
import sched
import time


def printer():
    print("Executed at : "+ str(time.time()))

def job_sched(scheduler, f, n):
    scheduler.enter(float(n)/100, 1, f, ())
    scheduler.run()


if __name__ == '__main__':
    scheduler = sched.scheduler(time.time, time.sleep)
    print("Start at : "+ str(time.time()))
    job_sched(scheduler, printer, 10)
```
----------------------------------------------------------------

# Problem #9

This problem was asked by Airbnb.

Given a list of integers, write a function that returns the largest sum of non-adjacent numbers. Numbers can be 0 or
negative.

For example, [2, 4, 6, 2, 5] should return 13, since we pick 2, 6, and 5. [5, 1, 1, 5] should return 10, since we pick
5 and 5.

Follow-up: Can you do this in O(N) time and constant space?

```python
def maxsum(arr):
    if not arr:
        return 0
    elif len(arr) == 1:
        return arr[0]

    sum = [0] * (len(arr)+1)
    sum[0] = 0
    sum[1] = arr[0]
    sum[2] = max(arr[0], arr[1])

    for i in range(3, len(arr)+1):
        sum[i] = max(sum[i-2] + arr[i-1], sum[i-1])

    return sum[-1]

if __name__ == '__main__':
    arr = [2,4,6,2,5]
    print(arr, maxsum(arr))
    arr = [5,1,1,5]
    print(arr, maxsum(arr))
```
----------------------------------------------------------------

# Problem #8

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

```python
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
    #head.left.left = Node(1)
    #head.left.right = Node(1)
    head.right = Node(0)
    head.right.left = Node(1)
    head.right.left.left = Node(1)
    head.right.left.right = Node(1)
    head.right.right = Node(0)

    print(unival_ct(head))
```
----------------------------------------------------------------

# Problem #7

This problem was asked by Facebook.

Given the mapping a = 1, b = 2, ... z = 26, and an encoded message, count the number of ways it can be decoded.

For example, the message '111' would give 3, since it could be decoded as 'aaa', 'ka', and 'ak'.

You can assume that the messages are decodable. For example, '001' is not allowed.

```python
def count_decode(message):
    arr = [0] * (len(message)+1)
    
    arr[0] = 1
    arr[1] = 1

    for i in range(2, len(message)+1):
        arr[i] = 0

        if message[i-1] > '0':
            arr[i] = arr[i-1]
        if message[i-2] == '1' or (message[i-2] == '2' and message[i-1] < '7'):
            arr[i] += arr[i-2]

    return arr[-1]

if __name__ == '__main__':
    mes = '111'
    print(mes, count_decode(mes))
    mes = '131'
    print(mes, count_decode(mes))
    mes = '1312'
    print(mes, count_decode(mes))
    mes = '1013'
    print(mes, count_decode(mes))
```
----------------------------------------------------------------

# Problem #6

This problem was asked by Google.

An XOR linked list is a more memory efficient doubly linked list. Instead of each node holding next and prev fields, it
holds a field named both, which is an XOR of the next node and the previous node. Implement an XOR linked list; it has
an add(element) which adds the element to the end, and a get(index) which returns the node at index.

If using a language that has no pointers (such as Python), you can assume you have access to get_pointer and
dereference_pointer functions that converts between nodes and memory addresses.


> https://www.geeksforgeeks.org/xor-linked-list-a-memory-efficient-doubly-linked-list-set-1/
----------------------------------------------------------------

# Problem #5

cons(a, b) constructs a pair, and car(pair) and cdr(pair) returns the first and last element of that pair. For example,
car(cons(3, 4)) returns 3, and cdr(cons(3, 4)) returns 4.

Given this implementation of cons:
```python
    def cons(a, b):
        def pair(f):
           return f(a, b)
        return pair
```
Implement car and cdr.

```python
def cons(a, b):
    def pair(f):
        return f(a, b)
    return pair

def car(f):
    def left(a, b):
        return a
    return f(left)

def cdr(f):
    def right(a, b):
        return b
    return f(right)


if __name__ == '__main__':
    print(car(cons(3, 4)))
    print(cdr(cons(3, 4)))
```
----------------------------------------------------------------

# Problem #4

Given an array of integers, find the first missing positive integer in linear time and constant space. In other words,
find the lowest positive integer that does not exist in the array. The array can contain duplicates and negative numbers
as well.

For example, the input [3, 4, -1, 1] should give 2. The input [1, 2, 0] should give 3.

You can modify the input array in-place.



>SOLUTIONS :
>    Brute force - search for all numbers n+1 if array of size n [ O(n^2)]
>    Sort - then simple linear search [ O(logn + n) ]
>    Hash - keep a dict of all +ve int, then do another linear iteration of N+1 elements in hash [ O(n) + O(n) space ]

>    Best -
>    1) Segregate positive numbers from others i.e., move all non-positive numbers to left side. In the following code,
>       segregate() function does this part.
>    2) Now we can ignore non-positive elements and consider only the part of array which contains all positive
>       elements. We traverse the array containing all positive numbers and to mark presence of an element x, we
>       change the sign of value at index x to negative. We traverse the array again and print the first index which
>       has positive value.

```python
def segregate(arr):
    pos_idx = 0
    for i in range(len(arr)):
        if arr[i] < 1:
            arr[i], arr[pos_idx] = arr[pos_idx], arr[i]
            pos_idx = pos_idx+1

    return pos_idx

# Making general solution for finding Kth missing int
def missing_int(arr, k):
    pos_idx = segregate(arr)
    arr = arr[pos_idx:]
    l_arr = len(arr)
    for i in range(l_arr):
        if arr[i] - 1 < l_arr:
            arr[arr[i]-1] = -1 * abs(arr[arr[i]-1])

    miss_ctr = 0
    for i in range(l_arr):
        if arr[i] > 0:
            miss_ctr += 1
            if miss_ctr is k:
                return i+1

if __name__ == '__main__':
    for k, arr in ((1, [3,4,-1,1]), (2, [3,4,-1,1]), (3, [1,2,0]), (1, [3,5,-1,5,0,2,7,1])):
        print(arr, k, missing_int(arr, k))
```
----------------------------------------------------------------

# Problem #3

Given the root to a binary tree, implement serialize(root), which serializes the tree into a string, and deserialize(s),
which deserializes the string back into the tree.

For example, given the following Node class
```python
class Node:
        def __init__(self, val, left=None, right=None):
            self.val = val
            self.left = left
            self.right = right
```
The following test should pass:
    `node = Node('root', Node('left', Node('left.left')), Node('right'))`
    `assert deserialize(serialize(node)).left.left.val == 'left.left'`


```python
class Node:
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def serialize(node):
    if not node:
        return '()'
    if not node.left and not node.right:
        return '("'+node.val+'")'
    return '("'+node.val+'", '+serialize(node.left)+', '+serialize(node.right)+')'

def deserialize(tree_str):
    code_string = 'tree = '+ tree_str.replace(', ()','').replace('(','Node(')
    print("Deserialized : "+code_string)
    exec code_string
    return tree

if __name__ == '__main__':
    node = Node('root', Node('left', Node('left.left')), Node('right'))
    print("Serialized : "+serialize(node))
    assert deserialize(serialize(node)).left.left.val == 'left.left'
```
----------------------------------------------------------------
# Problem #2

Given an array of integers, return a new array such that each element at index i of the new array is the product of all
the numbers in the original array except the one at i.

For example, if our input was [1, 2, 3, 4, 5], the expected output would be [120, 60, 40, 30, 24]. If our input was [3,
2, 1], the expected output would be [2, 3, 6]

```python
import numpy as np

def not_prod(arr):
    tot_prod = np.prod(arr)
    prod_arr = [tot_prod/num for num in arr]

    return prod_arr

def not_prod_nodiv(arr):
    l_arr = len(arr)
    left, right = np.ones(l_arr), np.ones(l_arr)
    for i in range(1, l_arr):
        left[i] = left[i-1]*arr[i-1]
        right[l_arr-1-i] = right[l_arr-i]*arr[l_arr-i]

    prod_arr = [left[i]*right[i] for i in range(l_arr)]
    return prod_arr
```
----------------------------------------------------------------
# Problem #1

Given a list of numbers and a number k, return whether any two numbers from the list add up to k.

For example, given [10, 15, 3, 7] and k of 17, return true since 10 + 7 is 17.

>NOTE : 
>If negative numbers are also to be handled, just add the smallest 
>number of the set to all elements and carry on with below algos

```python
# Brute Force
def sum_true_bf(ele_list, k):
    for i in range(len(ele_list)):
        for j in range(i+1, len(ele_list)):
            if ele_list[i]+ele_list[j] == k:
                return True
    return False

# Single Pass
def sum_true_f(ele_list, k):
    diff_list = []
    for i in range(len(ele_list)):
        diff = k - ele_list[i]
        if diff in diff_list:
            return True
        diff_list.append(ele_list[i])
    return False
```
----------------------------------------------------------------









