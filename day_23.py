"""
This problem was asked by Google.

You are given an M by N matrix consisting of booleans that represents a board. Each True boolean represents a wall. Each
False boolean represents a tile you can walk on.

Given this matrix, a start coordinate, and an end coordinate, return the minimum number of steps required to reach the
end coordinate from the start. If there is no possible path, then return null. You can move up, left, down, and right.
You cannot move through walls. You cannot wrap around the edges of the board.

For example, given the following board:

    [[f, f, f, f],
    [t, t, f, t],
    [f, f, f, f],
    [f, f, f, f]]

and start = (3, 0) (bottom left) and end = (0, 0) (top left), the minimum number of steps required to reach the end
is 7, since we would need to go through (1, 2) because there is a wall everywhere else on the second row
"""
import numpy as np

def is_valid(x, y, board, visited):
    b_xmax, b_ymax = len(board), len(board[0])

    if x < b_xmax and \
            y < b_ymax and \
            x >= 0 and \
            y >= 0 and \
            visited[x][y] < 0 and \
            not board[x][y]:
                return True
    return False

def min_steps(board, start, end):
    # Start or End itself is a wall
    if board[start[0]][start[1]] or board[end[0]][end[1]]:
        return -1

    visited = [ [-1 for i in range(len(board[0]))] for j in range(len(board)) ]
    queue = [start]

    visited[start[0]][start[1]] = 0

    while queue:
        node_x, node_y = queue.pop(0)
        cur_depth = visited[node_x][node_y]

        if (node_x, node_y) == end:
            print(np.array(visited))
            return cur_depth

        if is_valid(node_x+1, node_y, board, visited):
            visited[node_x+1][node_y] = cur_depth+1
            queue.append((node_x+1, node_y))
        if is_valid(node_x, node_y+1, board, visited):
            visited[node_x][node_y+1] = cur_depth+1
            queue.append((node_x, node_y+1))
        if is_valid(node_x-1, node_y, board, visited):
            visited[node_x-1][node_y] = cur_depth+1
            queue.append((node_x-1, node_y))
        if is_valid(node_x, node_y-1, board, visited):
            visited[node_x][node_y-1] = cur_depth+1
            queue.append((node_x, node_y-1))
            
    return -1


if __name__ == '__main__':
    board = [[False, False, False, False], 
            [True, True, False, True],
            [False, False, False, False],
            [False, False, False, False]]

    start = (3,0)
    end = (0,0)

    print(min_steps(board, start, end))
