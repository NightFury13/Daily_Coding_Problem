"""
Given a 2D matrix of characters and a target word, write a function that returns whether the word can be found in the matrix by going left-to-right, or up-to-down.

For example, given the following matrix:

[['F', 'A', 'C', 'I'],
 ['O', 'B', 'Q', 'P'],
 ['A', 'N', 'O', 'B'],
 ['M', 'A', 'S', 'S']]
and the target word 'FOAM', you should return true, since it's the leftmost column. Similarly, given the target word 'MASS', you should return true, since it's the last row.
"""

def has_word_util(matrix, i, j, word):
    max_i = len(matrix)-1
    max_j = len(matrix[0])-1

    if not word:
        return True
    
    if  (j<max_j and matrix[i][j+1] != word[0]) and (i<max_i and matrix[i+1][j] != word[0]):
        return False
    elif (j<max_j and matrix[i][j+1] == word[0]) and (i<max_i and matrix[i+1][j] == word[0]):
        return has_word_util(matrix, i, j+1, word[1:]) or has_word_util(matrix, i+1, j, word[1:])
    elif (j<max_j and matrix[i][j+1] == word[0]):
        return has_word_util(matrix, i, j+1, word[1:])
    elif (i<max_i and matrix[i+1][j] == word[0]):
        return has_word_util(matrix, i+1, j, word[1:])
    else:
        return False

def has_word(matrix, word):
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            if matrix[i][j] == word[0]:
                if has_word_util(matrix, i, j, word[1:]):
                    return True
    return False

if __name__ == '__main__':
    matrix = [['F', 'A', 'C', 'I'], ['O', 'B', 'Q', 'P'], ['A', 'N', 'O', 'B'], ['M', 'A', 'S', 'S']]
    words = ['FOAM', 'MASS', 'QOBS', 'FOAB']

    mat_str = """[['F', 'A', 'C', 'I'],
 ['O', 'B', 'Q', 'P'],
 ['A', 'N', 'O', 'B'],
 ['M', 'A', 'S', 'S']]"""
    print(mat_str)
    for word in words:
        print(word, has_word(matrix, word))

