"""
Given a list of numbers and a number k, return whether any two numbers from the list add up to k.

For example, given [10, 15, 3, 7] and k of 17, return true since 10 + 7 is 17.
"""

# If negative numbers are also to be handled, just add the smallest 
# number of the set to all elements and carry on with below algos

# Brute Force
def sum_true_bf(ele_list, k):
    for i in range(len(ele_list)):
        for j in range(i+1, len(ele_list)):
            if ele_list[i]+ele_list[j] == k:
                return True
    return False

# Single Pass
def sum_true_f(ele_list, k):
    diff_list = []
    for i in range(len(ele_list)):
        diff = k - ele_list[i]
        if diff in diff_list:
            return True
        diff_list.append(ele_list[i])
    return False
