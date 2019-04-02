"""
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
"""

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
