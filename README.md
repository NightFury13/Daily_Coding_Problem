----------------------------------------------------------------
# Problem #42
This problem was asked by Google.

Given a list of integers S and a target number k, write a function that returns a subset of S that adds up to k. If such
a subset cannot be made, then return null.

Integers can appear more than once in the list. You may assume all numbers in the list are positive.

For example, given S = [12, 1, 61, 5, 9, 2] and k = 24, return [12, 9, 2, 1] since it sums up to 24.

```python
def sub_sum_k(arr, k):
    if len(arr) == 0:
        if k == 0:
            return True
        else:
            return False

    return sub_sum_k(arr[:-1], k-arr[-1]) or sub_sum_k(arr[:-1], k)

if __name__ == '__main__':
    arr = [12, 1, 61, 5, 9, 2]
    print(arr)
    
    k = 24
    print(k, sub_sum_k(arr, k))
    k = 4
    print(k, sub_sum_k(arr, k))
```
----------------------------------------------------------------
# Problem #41
This problem was asked by Facebook.

Given an unordered list of flights taken by someone, each represented as (origin, destination) pairs, and a starting
airport, compute the person's itinerary. If no such itinerary exists, return null. If there are multiple possible
itineraries, return the lexicographically smallest one. All flights must be used in the itinerary.

For example, given the list of flights [('SFO', 'HKO'), ('YYZ', 'SFO'), ('YUL', 'YYZ'), ('HKO', 'ORD')] and starting
airport 'YUL', you should return the list ['YUL', 'YYZ', 'SFO', 'HKO', 'ORD'].

Given the list of flights [('SFO', 'COM'), ('COM', 'YYZ')] and starting airport 'COM', you should return null.

Given the list of flights [('A', 'B'), ('A', 'C'), ('B', 'C'), ('C', 'A')] and starting airport 'A', you should return
the list ['A', 'B', 'C', 'A', 'C'] even though ['A', 'C', 'A', 'B', 'C'] is also a valid itinerary. However, the first
one is lexicographically smaller.

```python
def valid_iten(flights, iten):
    if not flights:
        return iten

    cur_airport = iten[-1]
    outbound_flights = [(idx, flt) for idx, flt in enumerate(flights) if flt[0] is cur_airport]

    for idx, flt in sorted(outbound_flights, key=lambda x:x[1]):
        iten.append(flt[1])
        remain_flights = flights[:idx]+flights[idx+1:]

        if valid_iten(remain_flights, iten) != 'null':
            return iten

        iten.pop()

    return 'null'

if __name__ == '__main__':
    flights = [[('SFO', 'HKO'), ('YYZ', 'SFO'), ('YUL', 'YYZ'), ('HKO', 'ORD')],
            [('SFO', 'COM'), ('COM', 'YYZ')],
            [('A', 'B'), ('A', 'C'), ('B', 'C'), ('C', 'A')]
            ]

    origins = ['YUL', 'COM', 'A']

    for flts, orig in zip(flights, origins):
        print(flts)
        print('Origin : '+orig)
        iten = valid_iten(flts, [orig])
        if iten != 'null':
            print('Itinerary : '+'-->'.join(iten))
        else:
            print('Itinerary : null')
```
----------------------------------------------------------------
# Problem #40
This problem was asked by Google.

Given an array of integers where every integer occurs three times except for one integer, which only occurs once, find
and return the non-duplicated integer.

For example, given [6, 1, 3, 3, 3, 6, 6], return 1. Given [13, 19, 13, 13], return 19.

Do this in O(N) time and O(1) space.

```python
import numpy as np

# O(N) time and O(N) space
def non_triplet(arr):
    set_sum = np.sum(list(set(arr)))
    return (3*set_sum - np.sum(arr))/2


if __name__ == '__main__':
    inp_arr = [6, 1, 3, 3, 3, 6, 6]
    print(inp_arr, non_triplet(inp_arr))

    inp_arr = [13, 19, 13, 13]
    print(inp_arr, non_triplet(inp_arr))
```
----------------------------------------------------------------
# Problem #37
This problem was asked by Google.

The power set of a set is the set of all its subsets. Write a function that, given a set, generates its power set.

For example, given the set {1, 2, 3}, it should return {{}, {1}, {2}, {3}, {1, 2}, {1, 3}, {2, 3}, {1, 2, 3}}.

You may also use a list or array to represent a set.

```python
def make_powerset(in_set):
    powset_len = pow(2, len(in_set))

    powerset = []
    for i in range(powset_len):
        subset = []
        
        for ch_id, char in enumerate(format(i, 'b').zfill(3)):
            if char is '1':
                subset.append(in_set[ch_id])
        powerset.append(subset)

    return powerset

if __name__ == '__main__':
    in_set = [1, 2, 3]

    print('Input : '+str(in_set))
    print('Powerset : '+str(make_powerset(in_set)))
```
----------------------------------------------------------------
# Problem #36
This problem was asked by Dropbox.

Given the root to a binary search tree, find the second largest node in the tree.

```python
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
```
----------------------------------------------------------------
# Problem #35
This problem was asked by Google.

Given an array of strictly the characters 'R', 'G', and 'B', segregate the values of the array so that all the Rs come
first, the Gs come second, and the Bs come last. You can only swap elements of the array.

Do this in linear time and in-place.

For example, given the array ['G', 'B', 'R', 'R', 'B', 'R', 'G'], it should become ['R', 'R', 'R', 'G', 'G', 'B', 'B'].

```python
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
```
----------------------------------------------------------------
# Problem #33
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

```python
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
```
----------------------------------------------------------------
# Problem #32
This problem was asked by Jane Street.

Suppose you are given a table of currency exchange rates, represented as a 2D array. Determine whether there is
a possible arbitrage: that is, whether there is some sequence of trades you can make, starting with some amount A of any
currency, so that you can end up with some amount greater than A of that currency.

There are no transaction costs and you can trade fractional quantities.

```python
# SIDE NOTE : This is basically Bellman-Ford, we are trying to see if there are any negative cycles.
# https://www.dailycodingproblem.com/blog/how-to-find-arbitrage-opportunities-in-python/

from math import log

def arbitrage(n_cur, exchange):
    src = 0
    dist = [float('inf')] * n_cur

    n_edg = len(exchange)

    dist[src] = 0
    log_exchange = [(u, v, log(wt)) for u,v,wt in exchange]

    for i in range(n_cur - 1):
        for u, v, wt in log_exchange:
            if dist[v] > dist[u] + wt:
                dist[v] = dist[u] + wt

    for u, v, wt in log_exchange:
        if dist[v] > dist[u] + wt:
            print(u, v, wt, dist[u], dist[v])
            return True

    return False

if __name__ == '__main__':
    n_cur = 4
    # The inputs should be create a cycle to test this. Ideally, they should be complete.
    exchange = [(0, 1, 70.0), (1, 0, 1.0/70),
                (1, 2, 10.0), (2, 1, 1.0/10),
                (2, 3, 40.0), (3, 2, 1.0/40),
                (0, 3, 100.0), (3, 0, 1.0/100)]
    print(arbitrage(n_cur, exchange))
    
    exchange = [(0, 1, 70.0), (1, 0, 1.0/70),
                (1, 2, 10.0), (2, 1, 1.0/10),
                (2, 3, 40.0), (3, 2, 1.0/40),
                (0, 3, 28000.0), (3, 0, 1.0/28000)]
    print(arbitrage(n_cur, exchange))
```
----------------------------------------------------------------
# Problem #31
This problem was asked by Google.

The edit distance between two strings refers to the minimum number of character insertions, deletions, and substitutions
required to change one string to the other. For example, the edit distance between "kitten" and "sitting" is three:
substitute the "k" for "s", substitute the "e" for "i", and append a "g".

Given two strings, compute the edit distance between them.

```python
def editdistance(str1, str2, l1, l2, DP):
    for i in range(l1+1):
        for j in range(l2+1):
            if i == 0:
                DP[i][j] = j
            elif j == 0:
                DP[i][j] = i
            elif str1[i-1] == str2[j-1]:
                DP[i][j] = DP[i-1][j-1]
            else:
                DP[i][j] = 1 + min(DP[i-1][j-1], DP[i][j-1], DP[i-1][j])

    return DP[l1][l2]


if __name__ == '__main__':
    for str1, str2 in (('kitten', 'sitting'), ('mohit', 'rohit'), ('teapot', 'eats')):
        l1 = len(str1)
        l2 = len(str2)

        DP = [[0 for j in range(l2+1)] for i in range(l1+1)]

        print(str1, str2, editdistance(str1, str2, l1, l2, DP))
```
----------------------------------------------------------------
# Problem #30
This problem was asked by Facebook.

You are given an array of non-negative integers that represents a two-dimensional elevation map where each element is
unit-width wall and the integer is the height. Suppose it will rain and all spots between two walls get filled up.

Compute how many units of water remain trapped on the map in O(N) time and O(1) space.

For example, given the input [2, 1, 2], we can hold 1 unit of water in the middle.

Given the input [3, 0, 1, 3, 0, 5], we can hold 3 units in the first index, 2 in the second, and 3 in the fourth index
(we cannot hold 5 since it would run off to the left), so we can trap 8 units of water.

```python
def max_fill(wall):
    water_fill = 0
    max_left, max_right = 0, 0

    left_pt, right_pt = 0, len(wall)-1

    while left_pt < right_pt:
        if wall[left_pt] < wall[right_pt]:
            if wall[left_pt] > max_left:
                max_left = wall[left_pt]
            water_fill += max_left - wall[left_pt]
            left_pt += 1
        else:
            if wall[right_pt] > max_right:
                max_right = wall[right_pt]
            water_fill += max_right - wall[right_pt]
            right_pt -= 1

    return water_fill

if __name__ == '__main__':
    walls = [[2, 1, 2], [3, 0, 1, 3, 0, 5], [6, 3, 0, 1, 3, 0, 5]]

    for wall in walls:
        print(wall, 'Max Fill : ', max_fill(wall))
```

----------------------------------------------------------------
# Problem #29
This problem was asked by Amazon.

Run-length encoding is a fast and simple method of encoding strings. The basic idea is to represent repeated successive
characters as a single count and character. For example, the string "AAAABBBCCDAA" would be encoded as "4A3B2C1D2A".

Implement run-length encoding and decoding. You can assume the string to be encoded have no digits and consists solely
of alphabetic characters. You can assume the string to be decoded is valid.

```python
DIGITS = [str(i) for i in range(10)]

def encode(in_str):
    enc = ''

    ctr = 1
    for i in range(1, len(in_str)):
        if in_str[i] == in_str[i-1]:
            ctr+=1
        else:
            enc += str(ctr)+in_str[i-1]
            ctr = 1
    enc+= str(ctr)+in_str[-1]

    return enc

def decode(in_str):
    dec = ''

    idx = 0
    repeat = ''
    while idx < len(in_str):
        if in_str[idx] in DIGITS:
            repeat += in_str[idx]
        else:
            dec+=(in_str[idx]*int(repeat))
            repeat = ''
        idx+=1

    return dec

if __name__ == '__main__':
    in_str = 'AAAABBBCCDAA'

    print('In : '+in_str+' | Encoded : '+encode(in_str)+' | Decoded : '+decode(encode(in_str))+' | Correct? : '+str(in_str==decode(encode(in_str))))
```
----------------------------------------------------------------
# Problem #28
This problem was asked by Palantir.

Write an algorithm to justify text. Given a sequence of words and an integer line length k, return a list of strings
which represents each line, fully justified.

More specifically, you should have as many words as possible in each line. There should be at least one space between
each word. Pad extra spaces when necessary so that each line has exactly length k. Spaces should be distributed as
equally as possible, with the extra spaces, if any, distributed starting from the left.

If you can only fit one word on a line, then you should pad the right-hand side with spaces.

Each word is guaranteed not to be longer than k.

For example, given the list of words ["the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"] and k = 16,
you should return the following:

    ["the  quick brown", # 1 extra space on the left
    "fox  jumps  over", # 2 extra spaces distributed evenly
    "the   lazy   dog"] # 4 extra spaces distributed evenly

```python
def fix_line(word_list, k):
    n_words = len(word_list)
    if n_words < 2:
        return word_list

    cur_len = len(''.join(word_list))

    repeat = (k-cur_len)/(n_words-1)
    word_list = [word_list[0]] + [(' '*repeat)+sub_word for sub_word in word_list[1:]]

    extra = k - len(''.join(word_list))
    for i in range(extra):
        word_list[i] = word_list[i]+' '

    return word_list

def justify(words, k):
    text = []

    line_len = len(words[0])
    text.append([words[0]])
    for idx, word in enumerate(words[1:]):
        if line_len+len(' '+word) <= k:
            text[-1].append(' '+word)
            line_len += len(' '+word)

            # Last word reached
            if idx == len(words)-2:
                text[-1] = fix_line(text[-1], k)
        else:
            # Fix last complete line
            text[-1] = fix_line(text[-1], k)

            # Place word in new line
            text.append([word])
            line_len = len(word)
    
    text = [''.join(line) for line in text]
    return text

if __name__ == '__main__':
    all_words = (["the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"],
                ["my", "name", "is", "mohit", "jain.", "What", "is", "your", "name?"])
    all_k = (16, 12)

    for words, k in zip(all_words, all_k):
        print(words, k)
        text = justify(words, k)
        for line in text:
            print(line)
```
----------------------------------------------------------------
# Problem #27
This problem was asked by Facebook.

Given a string of round, curly, and square open and closing brackets, return whether the brackets are balanced
(well-formed).

For example, given the string "([])[]({})", you should return true.

Given the string "([)]" or "((()", you should return false.

```python
def well_formed(in_str):
    stack = []
    b_map = {')':'(', '}':'{', ']':'['}

    for bracket in in_str:
        if bracket not in b_map:
            stack.append(bracket)
        elif stack[-1] == b_map[bracket]:
            stack.pop(-1)
        else:
            return False

    if stack:
        return False
    return True

if __name__ == '__main__':
    in_strs = ['([])[]({})', '([)]', '((()']

    for in_str in in_strs:
        print(in_str, well_formed(in_str))
```
----------------------------------------------------------------
# Problem #26
This problem was asked by Google.

Given a singly linked list and an integer k, remove the kth last element from the list. k is guaranteed to be smaller
than the length of the list.

The list is very long, so making more than one pass is prohibitively expensive.

Do this in constant space and in one pass.

```python
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
```
----------------------------------------------------------------
# Problem #25
This problem was asked by Facebook.

Implement regular expression matching with the following special characters:

    . (period) which matches any single character
    * (asterisk) which matches zero or more of the preceding element

That is, implement a function that takes in a string and a valid regular expression and returns whether or not the
string matches the regular expression.

For example, given the regular expression "ra." and the string "ray", your function should return true. The same
regular expression on the string "raymond" should return false.

Given the regular expression ".*at" and the string "chat", your function should return true. The same regular
expression on the string "chats" should return false.

```python
def re(in_str, re_exp):
    if not re_exp:
        return not in_str

    prefix_match = bool(in_str) and re_exp[0] in (in_str[0], '.')

    if len(re_exp) > 1 and re_exp[1] == '*':
        return prefix_match and (re(in_str[1:], re_exp[2:]) or re(in_str[1:], re_exp))
    else:
        return prefix_match and re(in_str[1:], re_exp[1:])

if __name__ == '__main__':
    in_strs = ['ray', 'raymond', 'chat', 'chats']
    re_exps = ['ra.', 'ra.', '.*at', '.*at']

    for in_str, re_exp in zip(in_strs, re_exps):
        print(re_exp, in_str, re(in_str, re_exp))
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

```
----------------------------------------------------------------
# Problem #24
This problem was asked by Google.

Implement locking in a binary tree. A binary tree node can be locked or unlocked only if all of its descendants or
ancestors are not locked.

Design a binary tree node class with the following methods:

    is_locked, which returns whether the node is locked
    lock, which attempts to lock the node. If it cannot be locked, then it should return false. Otherwise, it should
    lock it and return true.
    unlock, which unlocks the node. If it cannot be unlocked, then it should return false. Otherwise, it should unlock
    it and return true.

You may augment the node to add parent pointers or any other property you would like. You may assume the class is
used in a single-threaded program, so there is no need for actual locks or mutexes. Each method should run in O(h),
where h is the height of the tree.

```python
class LockNode:
    def __init__(self, val, lock_state=False, left=None, right=None, parent=None):
        self.val = val
        self.lock = lock_state
        self.left = left
        self.right = right
        self.parent = parent
        self.ct_locked_child = 0

    def is_locked(self):
        return self.lock

    def _lockable_or_unlockable(self):
        if self.ct_locked_child > 0:
            return False

        cur_node = self.parent
        while cur_node:
            if cur_node.lock:
                return False
            cur_node = cur_node.parent

        return True

    def lock(self):
        if self._lockable_or_unlockable():
            self.lock = True

            cur_node = self.parent
            while cur_node:
                cur_node.ct_locked_child += 1
                cur_node = cur_node.parent
            
            return True
        else:
            return False

    def unlock(self):
        if self._lockable_or_unlockable():
            self.lock = False

            cur_node = self.parent
            while cur_node:
                cur_node.ct_locked_child -= 1
                cur_node = cur_node.parent

            return True
        else:
            return False
```
----------------------------------------------------------------
# Problem #23
This problem was asked by Google.

You are given an M by N matrix consisting of booleans that represents a board. Each True boolean represents a wall. Each
False boolean represents a tile you can walk on.

Given this matrix, a start coordinate, and an end coordinate, return the minimum number of steps required to reach the
end coordinate from the start. If there is no possible path, then return null. You can move up, left, down, and right.
You cannot move through walls. You cannot wrap around the edges of the board.

For example, given the following board:

    [[f, f, f, f],
    [t, t, f, t],
    [f, f, f, f],
    [f, f, f, f]]

and start = (3, 0) (bottom left) and end = (0, 0) (top left), the minimum number of steps required to reach the end
is 7, since we would need to go through (1, 2) because there is a wall everywhere else on the second row

```python
import numpy as np

def is_valid(x, y, board, visited):
    b_xmax, b_ymax = len(board), len(board[0])

    if x < b_xmax and \
            y < b_ymax and \
            x >= 0 and \
            y >= 0 and \
            visited[x][y] < 0 and \
            not board[x][y]:
                return True
    return False

def min_steps(board, start, end):
    # Start or End itself is a wall
    if board[start[0]][start[1]] or board[end[0]][end[1]]:
        return -1

    visited = [ [-1 for i in range(len(board[0]))] for j in range(len(board)) ]
    queue = [start]

    visited[start[0]][start[1]] = 0

    while queue:
        node_x, node_y = queue.pop(0)
        cur_depth = visited[node_x][node_y]

        if (node_x, node_y) == end:
            print(np.array(visited))
            return cur_depth

        if is_valid(node_x+1, node_y, board, visited):
            visited[node_x+1][node_y] = cur_depth+1
            queue.append((node_x+1, node_y))
        if is_valid(node_x, node_y+1, board, visited):
            visited[node_x][node_y+1] = cur_depth+1
            queue.append((node_x, node_y+1))
        if is_valid(node_x-1, node_y, board, visited):
            visited[node_x-1][node_y] = cur_depth+1
            queue.append((node_x-1, node_y))
        if is_valid(node_x, node_y-1, board, visited):
            visited[node_x][node_y-1] = cur_depth+1
            queue.append((node_x, node_y-1))
            
    return -1


if __name__ == '__main__':
    board = [[False, False, False, False], 
            [True, True, False, True],
            [False, False, False, False],
            [False, False, False, False]]

    start = (3,0)
    end = (0,0)

    print(min_steps(board, start, end))
```
----------------------------------------------------------------
# Problem #22
This problem was asked by Microsoft.

Given a dictionary of words and a string made up of those words (no spaces), return the original sentence in a list. If
there is more than one possible reconstruction, return any of them. If there is no possible reconstruction, then return
null.

>For example, given the set of words 'quick', 'brown', 'the', 'fox', and the string "thequickbrownfox", you should return
>['the', 'quick', 'brown', 'fox'].

>Given the set of words 'bed', 'bath', 'bedbath', 'and', 'beyond', and the string "bedbathandbeyond", return either
>['bed', 'bath', 'and', 'beyond] or ['bedbath', 'and', 'beyond'].

```python
def breakdown(st, vocab, broken=[[]]):
    for i in range(len(st)):
        if st[:i+1] in vocab:
            broken[-1].append(st[:i+1])
            if i+1 == len(st):
                broken.append([])
                return broken
            broken = breakdown(st[i+1:], vocab, broken)

    return broken


if __name__ == '__main__':
    vocab = ['bed', 'bath', 'bedbath', 'and', 'beyond']
    st = 'bedbathandbeyond'
    print(st, vocab)
    broken = breakdown(st, vocab, [[]])
    print(broken)
    
    vocab = ['quick', 'brown', 'the', 'fox']
    st = 'thequickbrownfoxx'
    print(st, vocab)
    broken = breakdown(st, vocab, [[]])
    print(broken)
```

----------------------------------------------------------------
# Problem #21
This problem was asked by Snapchat.

Given an array of time intervals (start, end) for classroom lectures (possibly overlapping), find the minimum number of
rooms required.

For example, given [(30, 75), (0, 50), (60, 150)], you should return 2.

```python
def schedule(intervals):
    sorted_intv = sorted(intervals, key=lambda x: x[1])

    rooms = [[sorted_intv[0]]]

    for interval in sorted_intv[1:]:
        placed = False
        for room in rooms:
            if room[-1][1] < interval[0]:
                room.append(interval)
                placed = True
                break
        if not placed:
            rooms.append([interval])
    
    print(rooms)
    return len(rooms)

if __name__ == '__main__':
    intervals = [(30, 75), (0, 50), (60, 150)]
    print(intervals)

    print("Rooms : " +str(schedule(intervals)))
```
----------------------------------------------------------------
# Problem #20
This problem was asked by Google.

Given two singly linked lists that intersect at some point, find the intersecting node. The lists are non-cyclical.

For example, given A = 3 -> 7 -> 8 -> 10 and B = 99 -> 1 -> 8 -> 10, return the node with value 8.

In this example, assume nodes with the same value are the exact same node objects.

Do this in O(M + N) time (where M and N are the lengths of the lists) and constant space.

```python
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
```
----------------------------------------------------------------
# Problem #19
This problem was asked by Facebook.

A builder is looking to build a row of N houses that can be of K different colors. He has a goal of minimizing cost
while ensuring that no two neighboring houses are of the same color.

Given an N by K matrix where the nth row and kth column represents the cost to build the nth house with kth color,
return the minimum cost which achieves this goal.

```python
import numpy as np

def min_cost(nk_mat):
    n_row, n_col = nk_mat.shape
    costs = np.zeros((n_row, n_col))

    costs[0] = nk_mat[0]
    for row in range(1, n_row):
        for col in range(n_col):
            costs[row][col] = nk_mat[row][col] + np.min(np.concatenate([costs[row-1][:col], costs[row-1][col+1:]]))

    return np.min(costs[-1])
            

if __name__ == '__main__':
    arr = [[1,2,3], [4,5,6], [7,8,9]]
    print(np.array(arr))
    print(min_cost(np.array(arr)))
    
    arr = [[1,4,7], [2,5,8], [3,6,9]]
    print(np.array(arr))
    print(min_cost(np.array(arr)))
```
----------------------------------------------------------------
# Problem #18
This problem was asked by Google.

Given an array of integers and a number k, where 1 <= k <= length of the array, compute the maximum values of each
subarray of length k.

For example, given array = [10, 5, 2, 7, 8, 7] and k = 3, we should get: [10, 7, 8, 8], since:

    10 = max(10, 5, 2)
    7 = max(5, 2, 7)
    8 = max(2, 7, 8)
    8 = max(7, 8, 7)

Do this in O(n) time and O(k) space. You can modify the input array in-place and you do not need to store the
results. You can simply print them out as you compute them.

```python
# Not O(n)
def max_val(arr, k):
    max_val = max(arr[:k])
    print(max_val)

    for i in range(len(arr)-k):
        if arr[i+k] > max_val:
            max_val = arr[i+k]
            print(max_val)
        elif arr[i] != max_val:
            print(max_val)
        else:
            max_val = max(arr[i+1:i+1+k])
            print(max_val)

if __name__ == '__main__':
    arr = [10, 5, 2, 7, 8, 7]
    k = 3

    max_val(arr, k)
```
----------------------------------------------------------------
# Problem #17
This problem was asked by Google.

Suppose we represent our file system by a string in the following manner:

The string "dir\n\tsubdir1\n\tsubdir2\n\t\tfile.ext" represents:

>dir
>    subdir1
>    subdir2
>        file.ext
The directory dir contains an empty sub-directory subdir1 and a sub-directory subdir2 containing a file file.ext.

The string "dir\n\tsubdir1\n\t\tfile1.ext\n\t\tsubsubdir1\n\tsubdir2\n\t\tsubsubdir2\n\t\t\tfile2.ext" represents:

>dir
>    subdir1
>        file1.ext
>        subsubdir1
>    subdir2
>        subsubdir2
>            file2.ext
The directory dir contains two sub-directories subdir1 and subdir2. subdir1 contains a file file1.ext and an empty second-level sub-directory subsubdir1. subdir2 contains a second-level sub-directory subsubdir2 containing a file file2.ext.

We are interested in finding the longest (number of characters) absolute path to a file within our file system. For example, in the second example above, the longest absolute path is "dir/subdir2/subsubdir2/file2.ext", and its length is 32 (not including the double quotes).

Given a string representing the file system in the above format, return the length of the longest absolute path to a file in the abstracted file system. If there is no file in the system, return 0.

Note:

The name of a file contains at least a period and an extension.

The name of a directory or sub-directory will not contain a period.

```python
def longest_abs_path(in_path):
    chunks = in_path.split('\n')

    cur_path_prefix_lens = []
    cur_path_prefix = []
    longest_abs_path = []
    longest_abs_path_len = -1

    for f_name in chunks:
        depth = 0
        while f_name[depth] == '\t':
            depth += 1
        if depth < len(cur_path_prefix):
            cur_path_prefix = cur_path_prefix[:depth]
            cur_path_prefix_lens = cur_path_prefix_lens[:depth]

        cur_path_prefix.append(f_name.strip())
        cur_path_prefix_lens.append(len(f_name.strip()) + 1)

        if '.' in f_name:
            if sum(cur_path_prefix_lens) > longest_abs_path_len:
                longest_abs_path = '/'.join(cur_path_prefix)
                longest_abs_path_len = sum(cur_path_prefix_lens)

    return longest_abs_path


if __name__ == '__main__':
    in_str = 'dir\n\tsubdir1\n\t\tfile1.ext\n\t\tsubsubdir1\n\tsubdir2\n\t\tsubsubdir2\n\t\t\tfile2.ext'

    print(longest_abs_path(in_str))
```
----------------------------------------------------------------
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









