"""
This problem was asked by Facebook.

Implement regular expression matching with the following special characters:

    . (period) which matches any single character
    * (asterisk) which matches zero or more of the preceding element

That is, implement a function that takes in a string and a valid regular expression and returns whether or not the
string matches the regular expression.

For example, given the regular expression "ra." and the string "ray", your function should return true. The same
regular expression on the string "raymond" should return false.

Given the regular expression ".*at" and the string "chat", your function should return true. The same regular
expression on the string "chats" should return false.
"""

def re(in_str, re_exp):
    if not re_exp:
        return not in_str

    prefix_match = bool(in_str) and re_exp[0] in (in_str[0], '.')

    if len(re_exp) > 1 and re_exp[1] == '*':
        return prefix_match and (re(in_str[1:], re_exp[2:]) or re(in_str[1:], re_exp))
    else:
        return prefix_match and re(in_str[1:], re_exp[1:])

if __name__ == '__main__':
    in_strs = ['ray', 'raymond', 'chat', 'chats']
    re_exps = ['ra.', 'ra.', '.*at', '.*at']

    for in_str, re_exp in zip(in_strs, re_exps):
        print(re_exp, in_str, re(in_str, re_exp))
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

