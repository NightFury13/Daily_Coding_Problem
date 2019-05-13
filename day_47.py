"""
This problem was asked by Facebook.

Given a array of numbers representing the stock prices of a company in chronological order, write a function that
calculates the maximum profit you could have made from buying and selling that stock once. You must buy before you can
sell it.

For example, given [9, 11, 8, 5, 7, 10], you should return 5, since you could buy the stock at 5 dollars and sell it at
10 dollars.
"""

def max_profit(stocks):
    min_price = stocks[0]
    max_profit = 0

    for price in stocks:
        if price < min_price:
            min_price = price

        profit = price-min_price
        if profit > max_profit:
            max_profit = profit

    return max_profit

if __name__ == '__main__':
    stocks = [9, 11, 8, 5, 7, 10]
    print(stocks, max_profit(stocks))
