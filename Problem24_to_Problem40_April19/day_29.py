"""
This problem was asked by Amazon.

Run-length encoding is a fast and simple method of encoding strings. The basic idea is to represent repeated successive
characters as a single count and character. For example, the string "AAAABBBCCDAA" would be encoded as "4A3B2C1D2A".

Implement run-length encoding and decoding. You can assume the string to be encoded have no digits and consists solely
of alphabetic characters. You can assume the string to be decoded is valid.
"""

DIGITS = [str(i) for i in range(10)]

def encode(in_str):
    enc = ''

    ctr = 1
    for i in range(1, len(in_str)):
        if in_str[i] == in_str[i-1]:
            ctr+=1
        else:
            enc += str(ctr)+in_str[i-1]
            ctr = 1
    enc+= str(ctr)+in_str[-1]

    return enc

def decode(in_str):
    dec = ''

    idx = 0
    repeat = ''
    while idx < len(in_str):
        if in_str[idx] in DIGITS:
            repeat += in_str[idx]
        else:
            dec+=(in_str[idx]*int(repeat))
            repeat = ''
        idx+=1

    return dec

if __name__ == '__main__':
    in_str = 'AAAABBBCCDAA'

    print('In : '+in_str+' | Encoded : '+encode(in_str)+' | Decoded : '+decode(encode(in_str))+' | Correct? : '+str(in_str==decode(encode(in_str))))
