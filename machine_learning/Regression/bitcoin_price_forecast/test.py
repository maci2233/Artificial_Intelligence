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
