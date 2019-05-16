"""
This problem was asked by Microsoft.

Given a dictionary of words and a string made up of those words (no spaces), return the original sentence in a list. If
there is more than one possible reconstruction, return any of them. If there is no possible reconstruction, then return
null.

For example, given the set of words 'quick', 'brown', 'the', 'fox', and the string "thequickbrownfox", you should return
['the', 'quick', 'brown', 'fox'].

Given the set of words 'bed', 'bath', 'bedbath', 'and', 'beyond', and the string "bedbathandbeyond", return either
['bed', 'bath', 'and', 'beyond] or ['bedbath', 'and', 'beyond'].
"""

def breakdown(st, vocab, broken=[[]]):
    for i in range(len(st)):
        if st[:i+1] in vocab:
            broken[-1].append(st[:i+1])
            if i+1 == len(st):
                broken.append([])
                return broken
            broken = breakdown(st[i+1:], vocab, broken)

    return broken


if __name__ == '__main__':
    vocab = ['bed', 'bath', 'bedbath', 'and', 'beyond']
    st = 'bedbathandbeyond'
    print(st, vocab)
    broken = breakdown(st, vocab, [[]])
    print(broken)
    
    vocab = ['quick', 'brown', 'the', 'fox']
    st = 'thequickbrownfoxx'
    print(st, vocab)
    broken = breakdown(st, vocab, [[]])
    print(broken)
