"""
This problem was asked by Snapchat.

Given an array of time intervals (start, end) for classroom lectures (possibly overlapping), find the minimum number of
rooms required.

For example, given [(30, 75), (0, 50), (60, 150)], you should return 2.
"""

def schedule(intervals):
    sorted_intv = sorted(intervals, key=lambda x: x[1])

    rooms = [[sorted_intv[0]]]

    for interval in sorted_intv[1:]:
        placed = False
        for room in rooms:
            if room[-1][1] < interval[0]:
                room.append(interval)
                placed = True
                break
        if not placed:
            rooms.append([interval])
    
    print(rooms)
    return len(rooms)

if __name__ == '__main__':
    intervals = [(30, 75), (0, 50), (60, 150)]
    print(intervals)

    print("Rooms : " +str(schedule(intervals)))
