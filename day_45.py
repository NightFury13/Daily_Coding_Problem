"""
This problem was asked by Two Sigma.

Using a function rand5() that returns an integer from 1 to 5 (inclusive) with uniform probability, implement a function
rand7() that returns an integer from 1 to 7 (inclusive).

Upgrade to premium and get in-depth solutions to every problem.
"""

# Source : https://www.geeksforgeeks.org/generate-integer-from-1-to-7-with-equal-probability/

import random

def rand5():
    return random.randint(1, 5)

def rand7():
    val = 5*rand5() + rand5() - 5
    if val < 22:
        return val%7 + 1
    return rand7()
    

if __name__ == '__main__':
    ctr = {}
    for i in range(10000):
        val = rand7()
        if val not in ctr:
            ctr[val] = 1
        else:
            ctr[val] += 1

    print("After 10k iterations")
    print(ctr)
