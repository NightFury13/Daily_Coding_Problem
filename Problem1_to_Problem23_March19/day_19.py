"""
This problem was asked by Facebook.

A builder is looking to build a row of N houses that can be of K different colors. He has a goal of minimizing cost
while ensuring that no two neighboring houses are of the same color.

Given an N by K matrix where the nth row and kth column represents the cost to build the nth house with kth color,
return the minimum cost which achieves this goal.
"""
# Imports
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
