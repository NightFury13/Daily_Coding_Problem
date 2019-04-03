"""
This problem was asked by Facebook.

Given a string of round, curly, and square open and closing brackets, return whether the brackets are balanced
(well-formed).

For example, given the string "([])[]({})", you should return true.

Given the string "([)]" or "((()", you should return false.
"""

def well_formed(in_str):
    stack = []
    b_map = {')':'(', '}':'{', ']':'['}

    for bracket in in_str:
        if bracket not in b_map:
            stack.append(bracket)
        elif stack[-1] == b_map[bracket]:
            stack.pop(-1)
        else:
            return False

    if stack:
        return False
    return True

if __name__ == '__main__':
    in_strs = ['([])[]({})', '([)]', '((()']

    for in_str in in_strs:
        print(in_str, well_formed(in_str))
