"""
This problem was asked by Facebook.

Given the mapping a = 1, b = 2, ... z = 26, and an encoded message, count the number of ways it can be decoded.

For example, the message '111' would give 3, since it could be decoded as 'aaa', 'ka', and 'ak'.

You can assume that the messages are decodable. For example, '001' is not allowed.
"""

def count_decode(message):
    arr = [0] * (len(message)+1)
    
    arr[0] = 1
    arr[1] = 1

    for i in range(2, len(message)+1):
        arr[i] = 0

        if message[i-1] > '0':
            arr[i] = arr[i-1]
        if message[i-2] == '1' or (message[i-2] == '2' and message[i-1] < '7'):
            arr[i] += arr[i-2]

    return arr[-1]

if __name__ == '__main__':
    mes = '111'
    print(mes, count_decode(mes))
    mes = '131'
    print(mes, count_decode(mes))
    mes = '1312'
    print(mes, count_decode(mes))
    mes = '1013'
    print(mes, count_decode(mes))
