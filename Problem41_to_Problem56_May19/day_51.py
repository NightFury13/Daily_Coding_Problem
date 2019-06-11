"""
This problem was asked by Facebook.

Given a function that generates perfectly random numbers between 1 and k (inclusive), where k is an input, write a function that shuffles a deck of cards represented as an array using only swaps.

It should run in O(N) time.

Hint: Make sure each one of the 52! permutations of the deck is equally likely.
"""

# Fisher-Yates Shuffle Algo : https://www.geeksforgeeks.org/shuffle-a-given-array-using-fisher-yates-shuffle-algorithm/

import random

def randk(k):
    return random.randint(1, k)

def shuffle(cards):
    for i in range(52):
        # Get index between [i, 51] (inclusive)
        swap_with = i+randk(52-i)-1
        cards[i], cards[swap_with] = cards[swap_with], cards[i]

    return cards

if __name__ == '__main__':
    cards = range(52)
    print(cards)
    print(shuffle(cards))
    print(shuffle(cards))

