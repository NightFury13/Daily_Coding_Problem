"""
This problem was asked by Amazon.

Given a string s and an integer k, break up the string into multiple lines such that each line has a length of k or less. 
You must break it up so that words don't break across lines. Each line has to have the maximum possible amount of words. 
If there's no way to break the text up, then return null.

You can assume that there are no spaces at the ends of the string and that there is exactly one space between each word.

For example, given the string "the quick brown fox jumps over the lazy dog" and k = 10, you should return: 
    ["the quick", "brown fox", "jumps over", "the lazy", "dog"]. No string in the list has a length of more than 10.
"""

def break_text(line, k):
    words = line.split()

    split_lines = []
    cur_line = ''
    for word in words:
        word_len = len(word)
        if word_len > k:
            return 'null'
        elif len(cur_line)+word_len+1 > k:
            split_lines.append(cur_line)
            cur_line = word
        else:
            cur_line += ' '+word
            cur_line = cur_line.strip()
    split_lines.append(cur_line)

    return split_lines

if __name__ == '__main__':
    in_str = 'the quick brown fox jumps over the lazy dog'
    k = 10

    print(in_str, break_text(in_str, k))
