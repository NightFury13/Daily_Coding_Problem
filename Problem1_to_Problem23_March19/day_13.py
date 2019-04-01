"""
This problem was asked by Amazon.

Given an integer k and a string s, find the length of the longest substring that contains at most k distinct characters.

For example, given s = "abcba" and k = 2, the longest substring with k distinct characters is "bcb".
"""

def longest_subs(s, k):
    chars_map = {}

    cur_st = 0
    longest_st = 0
    longest_len = 0

    for i, char in enumerate(s):
        if char in chars_map:
            chars_map[char] += 1
        else:
            chars_map[char] = 1

        if len(chars_map) <= k:
            if i-cur_st+1 > longest_len:
                longest_len = i-cur_st+1
                longest_st = cur_st
        else:
            if chars_map[s[cur_st]] == 1:
                chars_map.pop(s[cur_st])
            else:
                chars_map[s[cur_st]] -= 1
            cur_st += 1

    return (s[longest_st:longest_st+longest_len])

if __name__ == '__main__':
    s = 'abcba'
    k = 2
    print(s, k, '->', longest_subs(s, k))
    s = 'abcadcacacaca'
    k = 3
    print(s, k, '->', longest_subs(s, k))
