"""
This problem was asked by Facebook.

Given an unordered list of flights taken by someone, each represented as (origin, destination) pairs, and a starting
airport, compute the person's itinerary. If no such itinerary exists, return null. If there are multiple possible
itineraries, return the lexicographically smallest one. All flights must be used in the itinerary.

For example, given the list of flights [('SFO', 'HKO'), ('YYZ', 'SFO'), ('YUL', 'YYZ'), ('HKO', 'ORD')] and starting
airport 'YUL', you should return the list ['YUL', 'YYZ', 'SFO', 'HKO', 'ORD'].

Given the list of flights [('SFO', 'COM'), ('COM', 'YYZ')] and starting airport 'COM', you should return null.

Given the list of flights [('A', 'B'), ('A', 'C'), ('B', 'C'), ('C', 'A')] and starting airport 'A', you should return
the list ['A', 'B', 'C', 'A', 'C'] even though ['A', 'C', 'A', 'B', 'C'] is also a valid itinerary. However, the first
one is lexicographically smaller.
"""

def valid_iten(flights, iten):
    if not flights:
        return iten

    cur_airport = iten[-1]
    outbound_flights = [(idx, flt) for idx, flt in enumerate(flights) if flt[0] is cur_airport]

    for idx, flt in sorted(outbound_flights, key=lambda x:x[1]):
        iten.append(flt[1])
        remain_flights = flights[:idx]+flights[idx+1:]

        if valid_iten(remain_flights, iten) != 'null':
            return iten

        iten.pop()

    return 'null'

if __name__ == '__main__':
    flights = [[('SFO', 'HKO'), ('YYZ', 'SFO'), ('YUL', 'YYZ'), ('HKO', 'ORD')],
            [('SFO', 'COM'), ('COM', 'YYZ')],
            [('A', 'B'), ('A', 'C'), ('B', 'C'), ('C', 'A')]
            ]

    origins = ['YUL', 'COM', 'A']

    for flts, orig in zip(flights, origins):
        print(flts)
        print('Origin : '+orig)
        iten = valid_iten(flts, [orig])
        if iten != 'null':
            print('Itinerary : '+'-->'.join(iten))
        else:
            print('Itinerary : null')
