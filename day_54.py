"""
This problem was asked by Dropbox.

Sudoku is a puzzle where you're given a partially-filled 9 by 9 grid with digits. The objective is 
to fill the grid with the constraint that every row, column, and box (3 by 3 subgrid) must contain 
all of the digits from 1 to 9.

Implement an efficient sudoku solver
"""
# Imports
import numpy as np

def check(game, col, row, val):
    # Check row-col
    for i in range(9):
        if game[i][row] == val:
            return False
        if game[col][i] == val:
            return False
    
    # Check 3x3 block
    for i in range(3):
        for j in range(3):
            if game[3*(col/3)+i][3*(row/3)+j] == val:
                return False

    return True

def get_empty_cell(game):
    for i in range(9):
        for j in range(9):
            if game[i][j] == 0:
                return (i,j)
    return False

def sudoku_solver(game):
    empty_cell = get_empty_cell(game)
    # Game solved!
    if not empty_cell:
        return True
    col, row = empty_cell

    for val in range(1, 10):
        if check(game, col, row, val):
            game[col][row] = val
            
            if sudoku_solver(game):
                return True

            game[col][row] = 0

    return False

if __name__ == '__main__':
    game = np.zeros((9,9))
    game[0][0] = 9
    game[3][2] = 8
    game[8][1] = 7
    game[1][3] = 6
    game[4][4] = 5
    game[8][5] = 4
    game[0][8] = 3
    game[4][7] = 2
    game[8][6] = 1

    print(game)
    print(sudoku_solver(game))
    print(game)
