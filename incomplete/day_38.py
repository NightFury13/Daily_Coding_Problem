"""
This problem was asked by Microsoft.

You have an N by N board. Write a function that, given N, returns the number of possible arrangements of the board where
N queens can be placed on the board without threatening each other, i.e. no two queens share the same row, column, or
diagonal.
"""

def n_queens(N):
    board = [[0 for i in range(N)] for j in range(N)]

    

if __name__ == '__main__':
    N = 4

    print(N, n_queens(N))
