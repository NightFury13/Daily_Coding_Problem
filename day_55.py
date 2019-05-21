"""
This problem was asked by Microsoft.

Implement a URL shortener with the following methods:

    shorten(url), which shortens the url into a six-character alphanumeric string, such as zLg6wl.
    restore(short), which expands the shortened string into the original url. If no such shortened string exists, return null.

Hint: What if we enter the same URL twice?
"""
# Imports
import random

class Bitly:
    def __init__(self):
        self.map = {}
        self.rev_map = {}
        self.lexicon = [str(num) for num in range(10)] + \
                [chr(i) for i in range(65,91)] + \
                [chr(i) for i in range(97,123)]
        self.short_len = 6

    def shorten(self, url):
        if url not in self.map:
            # TODO : This would cause lots of collisions after a while! And how to handle a distributed system?
            short = ''.join(random.sample(self.lexicon, self.short_len))
            while short in self.rev_map:
                short = ''.join(random.sample(self.lexicon, self.short_len))

            self.map[url] = short
            self.rev_map[short] = url

        return self.map[url]
        
    def restore(self, short):
        if short in self.rev_map:
            return self.rev_map[short]
        else:
            return 'null'

if __name__ == '__main__':
    shortner = Bitly()

    urls = ['https://mail.google.com/mail/u/1/#inbox',
            'https://mail.google.com/mail/u/0/#label/Daily+Coding+Problem/FMfcgxwCgLnwDmvSLmHJklKFKLssFHxC',
            'https://music.youtube.com/watch?v=HWpZ_rOe_f0&list=RDMMKf781IbMQXk',
            'https://github.com/NightFury13/Daily_Coding_Problem',
            'https://github.com/NightFury13/Daily_Coding_Problem']

    for url in urls:
        shortner.shorten(url)

    print(shortner.map)
    print(shortner.rev_map)
