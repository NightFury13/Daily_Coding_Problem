"""
This problem was asked by Facebook.

You are given an array of non-negative integers that represents a two-dimensional elevation map where each element is
unit-width wall and the integer is the height. Suppose it will rain and all spots between two walls get filled up.

Compute how many units of water remain trapped on the map in O(N) time and O(1) space.

For example, given the input [2, 1, 2], we can hold 1 unit of water in the middle.

Given the input [3, 0, 1, 3, 0, 5], we can hold 3 units in the first index, 2 in the second, and 3 in the fourth index
(we cannot hold 5 since it would run off to the left), so we can trap 8 units of water.
"""

def max_fill(wall):
    water_fill = 0
    max_left, max_right = 0, 0

    left_pt, right_pt = 0, len(wall)-1

    while left_pt < right_pt:
        if wall[left_pt] < wall[right_pt]:
            if wall[left_pt] > max_left:
                max_left = wall[left_pt]
            water_fill += max_left - wall[left_pt]
            left_pt += 1
        else:
            if wall[right_pt] > max_right:
                max_right = wall[right_pt]
            water_fill += max_right - wall[right_pt]
            right_pt -= 1

    return water_fill

if __name__ == '__main__':
    walls = [[2, 1, 2], [3, 0, 1, 3, 0, 5], [6, 3, 0, 1, 3, 0, 5]]

    for wall in walls:
        print(wall, 'Max Fill : ', max_fill(wall))
