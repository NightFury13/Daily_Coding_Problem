"""
This problem was asked by Twitter.

You run an e-commerce website and want to record the last N order ids in a log. Implement a data structure to accomplish
this, with the following API:

    record(order_id): adds the order_id to the log
    get_last(i): gets the ith last element from the log. i is guaranteed to be smaller than or equal to N.
    You should be as efficient with time and space as possible.
"""

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
