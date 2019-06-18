"""
This problem was asked by Google.

A knight's tour is a sequence of moves by a knight on a chessboard such that all squares are visited once.

Given N, write a function to return the number of knight's tours on an N by N chessboard.
"""

def is_valid(board, move, n):
    if 0 <= move[0] < n and 0 <= move[1] < n and \
        board[move[0]][move[1]] is None:
            return True
    return False

def valid_moves(board, cur_r, cur_c, n):
    moves = [(-2, -1), (-2, 1), (-1, 2), (1, 2),
             (2, -1), (2, 1), (-1, -2), (1, -2)]

    val_moves = [(cur_r+del_r, cur_c+del_c) for del_r, del_c in moves]
    return [move for move in val_moves if is_valid(board, move, n)]

def n_tour_helper(board, cur_tour, n):
    if len(cur_tour) == n*n:
        return 1
    
    count = 0
    cur_r, cur_c = cur_tour[-1]
    for r,c in valid_moves(board, cur_r, cur_c, n):
        cur_tour.append((r,c))
        board[r][c] = len(cur_tour)
        count += n_tour_helper(board, cur_tour, n)
        board[r][c] = None
        cur_tour.pop()
    return count

def get_n_tours(n):
    count = 0

    for i in range(n):
        for j in range(n):
            board = [[None for i in range(n)] for j in range(n)]
            board[i][j] = 0
            count += n_tour_helper(board, [(i, j)], n)
    return count


if __name__ == '__main__':
    ns = [1, 4]

    for n in ns:
        print(n, get_n_tours(n))
