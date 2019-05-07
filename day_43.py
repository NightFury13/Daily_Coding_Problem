"""
This problem was asked by Amazon.

Implement a stack that has the following methods:

    push(val), which pushes an element onto the stack
    pop(), which pops off and returns the topmost element of the stack. If there are no elements in the stack, then it
    should throw an error or return null.
    max(), which returns the maximum value in the stack currently. If there are no elements in the stack, then it should
    throw an error or return null.

Each method should run in constant time.
"""


class MaxStack:
    def __init__(self):
        self.stack = []
        self.max = None

    def push(self, val):
        if not self.max:
            self.stack.append(val)
            self.max = val
        elif val < self.max:
            self.stack.append(val)
        else:
            self.stack.append(val + self.max)
            self.max = val

    def pop(self):
        if not self.stack:
            return 'null'

        top_val = self.stack.pop()
        if top_val < self.max:
            return top_val
        else:
            self.max = top_val - self.max
            return top_val - self.max

    def getmax(self):
        return self.max


if __name__ == '__main__':
    arr = [1, 2, 5, 3, 3, 4]

    m_stack = MaxStack()

    for ele in arr:
        m_stack.push(ele)
        print(ele, m_stack.getmax(), m_stack.stack)

    arr.reverse()

    for ele in arr:
        print('Pop : '+str(m_stack.pop()))
        print(ele, m_stack.getmax(), m_stack.stack)

