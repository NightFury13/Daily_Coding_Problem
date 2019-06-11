"""
This problem was asked by Google.

Implement an LRU (Least Recently Used) cache. It should be able to be initialized with a cache size n, and contain the following methods:

set(key, value): sets key to value. If there are already n items in the cache and we are adding a new item, then it should also remove the least recently used item.
get(key): gets the value at key. If no such key exists, return null.
Each operation should run in O(1) time.
"""

class DLL:
    def __init__(self, root):
        self.root = root
        self.tail = root
        self.size = 0 # Since the first node is a Null-Node

    def push(self, node):
        self.tail.child = node
        node.parent = self.tail
        self.tail = node

        self.size += 1
        return

    def pop(self, node):
        self.size -= 1

        if self.root.child == node:
            self.root.child = node.child
            self.root.parent = None
            return
        if self.tail == node:
            self.tail = node.parent
            self.tail.child = None
            return

        node.parent.child = node.child
        node.child.parent = node.parent
        return

class LLNode:
    def __init__(self, val, parent=None, child=None):
        self.val = val
        self.parent = parent
        self.child = child

class LRU:
    def __init__(self, n):
        self.cache = {}
        self.queue = DLL(LLNode('Null'))
        self.allowed_max_size = n

    def set(self, key, val):
        if key in self.cache:
            self.queue.pop(self.cache[key]['node'])

        key_node = LLNode(key)
        self.queue.push(key_node)

        if self.queue.size > self.allowed_max_size: 
            self.cache.pop(self.queue.root.child.val) # Root is Null-Node
            self.queue.pop(self.queue.root.child)

        self.cache[key] = {'val':val, 'node':key_node}
        return

    def get(self, key):
        if key not in self.cache:
            return 'null'

        self.queue.pop(self.cache[key]['node'])
        key_node = LLNode(key)
        self.queue.push(key_node)
        self.cache[key]['node'] = key_node
        return self.cache[key]['val']

def queue_to_list(queue):
    out_arr = []
    node = queue.root
    while node.child:
        out_arr.append(node.val)
        node = node.child
    out_arr.append(node.val)

    return out_arr

if __name__ == '__main__':
    lru_cache = LRU(3)

    print(5, lru_cache.get(5))
    lru_cache.set(1, 'A')
    lru_cache.set(2, 'B')
    lru_cache.set(3, 'C')
    print(2, lru_cache.get(2))
    lru_cache.set(1, 'A-new')
    lru_cache.set(4, 'D')
    print(3, lru_cache.get(3))
    print(1, lru_cache.get(1))
    print(2, lru_cache.get(2))
    print(4, lru_cache.get(4))
