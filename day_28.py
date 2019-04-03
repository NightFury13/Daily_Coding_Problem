"""
This problem was asked by Palantir.

Write an algorithm to justify text. Given a sequence of words and an integer line length k, return a list of strings
which represents each line, fully justified.

More specifically, you should have as many words as possible in each line. There should be at least one space between
each word. Pad extra spaces when necessary so that each line has exactly length k. Spaces should be distributed as
equally as possible, with the extra spaces, if any, distributed starting from the left.

If you can only fit one word on a line, then you should pad the right-hand side with spaces.

Each word is guaranteed not to be longer than k.

For example, given the list of words ["the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"] and k = 16,
you should return the following:

    ["the  quick brown", # 1 extra space on the left
    "fox  jumps  over", # 2 extra spaces distributed evenly
    "the   lazy   dog"] # 4 extra spaces distributed evenly
"""

def fix_line(word_list, k):
    n_words = len(word_list)
    if n_words < 2:
        return word_list

    cur_len = len(''.join(word_list))

    repeat = (k-cur_len)/(n_words-1)
    word_list = [word_list[0]] + [(' '*repeat)+sub_word for sub_word in word_list[1:]]

    extra = k - len(''.join(word_list))
    for i in range(extra):
        word_list[i] = word_list[i]+' '

    return word_list

def justify(words, k):
    text = []

    line_len = len(words[0])
    text.append([words[0]])
    for idx, word in enumerate(words[1:]):
        if line_len+len(' '+word) <= k:
            text[-1].append(' '+word)
            line_len += len(' '+word)

            # Last word reached
            if idx == len(words)-2:
                text[-1] = fix_line(text[-1], k)
        else:
            # Fix last complete line
            text[-1] = fix_line(text[-1], k)

            # Place word in new line
            text.append([word])
            line_len = len(word)
    
    text = [''.join(line) for line in text]
    return text

if __name__ == '__main__':
    all_words = (["the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"],
                ["my", "name", "is", "mohit", "jain.", "What", "is", "your", "name?"])
    all_k = (16, 12)

    for words, k in zip(all_words, all_k):
        print(words, k)
        text = justify(words, k)
        for line in text:
            print(line)
