"""
This problem was asked by Google.

The edit distance between two strings refers to the minimum number of character insertions, deletions, and substitutions
required to change one string to the other. For example, the edit distance between "kitten" and "sitting" is three:
substitute the "k" for "s", substitute the "e" for "i", and append a "g".

Given two strings, compute the edit distance between them.
"""

def editdistance(str1, str2, l1, l2, DP):
    for i in range(l1+1):
        for j in range(l2+1):
            if i == 0:
                DP[i][j] = j
            elif j == 0:
                DP[i][j] = i
            elif str1[i-1] == str2[j-1]:
                DP[i][j] = DP[i-1][j-1]
            else:
                DP[i][j] = 1 + min(DP[i-1][j-1], DP[i][j-1], DP[i-1][j])

    return DP[l1][l2]


if __name__ == '__main__':
    for str1, str2 in (('kitten', 'sitting'), ('mohit', 'rohit'), ('teapot', 'eats')):
        l1 = len(str1)
        l2 = len(str2)

        DP = [[0 for j in range(l2+1)] for i in range(l1+1)]

        print(str1, str2, editdistance(str1, str2, l1, l2, DP))
