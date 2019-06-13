"""
This problem was asked by Google.

Implement integer exponentiation. That is, implement the pow(x, y) function, where x and y are integers and returns x^y.

Do this faster than the naive method of repeated multiplication.

For example, pow(2, 10) should return 1024.
"""

def pow(x, y):
    if y == 0:
        return 1

    if y%2 == 0:
        return pow(x, y/2)*pow(x, y/2)
    else:
        return x*pow(x, y-1)

if __name__ == '__main__':
    xs = [2, 3, 4, 5]
    ys = [2, 3, 4, 5]

    for x in xs:
        for y in ys:
            print(str(x)+'^'+str(y)+' = '+str(pow(x, y)))
