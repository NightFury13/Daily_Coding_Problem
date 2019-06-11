"""
This problem was asked by Amazon.

Given a string, find the longest palindromic contiguous substring. If there are more than one with the maximum length,
return any one.

For example, the longest palindromic substring of "aabcdcb" is "bcdcb". The longest palindromic substring of "bananas"
is "anana".
"""

def longest_palin(in_str):
    str_len = len(in_str)
    palin_dp = [[False for i in range(str_len)] for j in range(str_len)]

    longest = ''

    # DP init
    for i in range(str_len):
        palin_dp[i][i] = True
        
        if i < str_len-1 and in_str[i] == in_str[i+1]:
            palin_dp[i][i+1] = True

    # DP expand
    for palin_len in range(3, str_len+1):
        for i in range(str_len-palin_len+1):
            j = i+palin_len-1

            if palin_dp[i+1][j-1] and in_str[i] == in_str[j]:
                palin_dp[i][j] = True
                
                if len(in_str[i:j+1]) > len(longest):
                    longest = in_str[i:j+1]

    return longest

if __name__ == '__main__':
    in_strs = ['aabcdcb', 'bananas', 'palinilap']

    for in_str in in_strs:
        print(in_str, longest_palin(in_str))
