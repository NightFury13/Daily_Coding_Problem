"""
This problem was asked by Apple.

Implement a queue using two stacks. Recall that a queue is a FIFO (first-in, first-out) 
data structure with the following methods: enqueue, which inserts an element into the queue, 
and dequeue, which removes it.
"""

class Stack:
    def __init__(self, stack_list=None):
        # IMPORTANT! Default instantiation for mutables workaround below
        self.stack = stack_list if stack_list is not None else []

    def push(self, ele):
        self.stack.append(ele)
        return

    def pop(self):
        if self.stack:
            return self.stack.pop(-1)
        return None

class StackQueue:
    def __init__(self, stackA=Stack(), stackB=Stack()):
        self.stackA = stackA
        self.stackB = stackB

    def enqueue(self, ele):
        val = self.stackA.pop()
        while val is not None:
            self.stackB.push(val)
            val = self.stackA.pop()
        
        self.stackA.push(ele)
        
        val = self.stackB.pop()
        while val is not None:
            self.stackA.push(val)
            val = self.stackB.pop()

        return
    
    def dequeue(self):
        return self.stackA.pop()


if __name__ == '__main__':
    st_queue = StackQueue()
    vals = [1,5,2,7,3,9,4]
    for i, val in enumerate(vals):
        st_queue.enqueue(val)
        print('Enqueue', val, st_queue.stackA.stack)
        if i%3==0:
            pop_val = st_queue.dequeue()
            print('Dequeue', pop_val, st_queue.stackA.stack)
