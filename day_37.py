"""
This problem was asked by Google.

The power set of a set is the set of all its subsets. Write a function that, given a set, generates its power set.

For example, given the set {1, 2, 3}, it should return {{}, {1}, {2}, {3}, {1, 2}, {1, 3}, {2, 3}, {1, 2, 3}}.

You may also use a list or array to represent a set.
"""

def make_powerset(in_set):
    powset_len = pow(2, len(in_set))

    powerset = []
    for i in range(powset_len):
        subset = []
        
        for ch_id, char in enumerate(format(i, 'b').zfill(3)):
            if char is '1':
                subset.append(in_set[ch_id])
        powerset.append(subset)

    return powerset

if __name__ == '__main__':
    in_set = [1, 2, 3]

    print('Input : '+str(in_set))
    print('Powerset : '+str(make_powerset(in_set)))
