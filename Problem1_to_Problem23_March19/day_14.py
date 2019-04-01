"""
This problem was asked by Google.

The area of a circle is defined as (pi)r^2. Estimate (pi) to 3 decimal places using a Monte Carlo method.

Hint: The basic equation of a circle is x2 + y2 = r2.
"""

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
