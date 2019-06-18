"""
This problem was asked by Google.

On our special chessboard, two bishops attack each other if they share the same diagonal. This includes bishops that have another bishop located between them, i.e. bishops can attack through pieces.

You are given N bishops, represented as (row, column) tuples on a M by M chessboard. Write a function to count the number of pairs of bishops that attack each other. The ordering of the pair doesn't matter: (1, 2) is considered the same as (2, 1).

For example, given M = 5 and the list of bishops:

(0, 0)
(1, 2)
(2, 2)
(4, 0)
The board would look like this:

[b 0 0 0 0]
[0 0 b 0 0]
[0 0 b 0 0]
[0 0 0 0 0]
[b 0 0 0 0]
You should return 2, since bishops 1 and 3 attack each other, as well as bishops 3 and 4.
"""

import numpy as np

def n_attacks_helper(board, cood, m):
    r_skew, c_skew = cood
    deltas = ((-1, -1), (-1, 1), (1, -1), (1, 1))

    att_ct = 0
    for i in range(1, m):
        for c_r, c_c in deltas:
            r_cood, c_cood = r_skew+(c_r*i), c_skew+(c_c*i) 
            if (0 <= r_cood < m) and (0 <= c_cood < m):
                att_ct += board[r_cood][c_cood]
    return att_ct

def n_attacks(m, bishops):
    board = np.zeros((m,m))
    for cood in bishops:
        board[cood[0]][cood[1]] = 1

    att_ct = 0
    for cood in bishops:
        att_ct += n_attacks_helper(board, cood, m)

    print(board)
    return att_ct/2

if __name__ == '__main__':
    m = 5
    bishops = [(0,0), (1,2), (2,2), (4,0)]
    print(n_attacks(m, bishops))

    m = 4
    bishops = [(0,0), (2,0), (1,1), (3,3)]
    print(n_attacks(m, bishops))
