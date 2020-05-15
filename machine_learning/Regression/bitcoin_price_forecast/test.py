tests = ['alan', '(mac()()ias)', 'se (la) ((come))', 'al((an)) (saicam) se ((la)) (c)(o)(m)(e)']


def invert_strings(string):
    stack = list()
    index = -1
    res = ''
    for char in string:
        if char == '(':
            stack.append('')
            index += 1
        elif char == ')':
            inv = stack[index][::-1]
            if index > 0:
                stack[index-1] += inv
            else:
                res += inv
            stack.pop()
            index -= 1
        elif index == -1:
            res += char
        else:
            stack[index] += char
    return res


for test in tests:
    print(invert_strings(test))
input()
import pandas as pd
from pandas import DataFrame

values = ['0.39 14.63', '0.26 20.12', '0.20 0.4', '.2622 .30']

df = DataFrame(values, columns=['%B'])

nums1, nums2 = list(), list()
for vals in df['%B'].values:
    nums = [float(i) for i in vals.split()]
    nums1.append(nums[0])
    nums2.append(nums[1])

df['%B'] = nums1
df.insert(list(df.columns).index('%B')+1, '%B2', nums2)

print(df)

tmp = df.iloc[-1]
df = df.shift(1)
df.iloc[0] = tmp

print(df)
