import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train = pd.read_csv('train.csv')

print(train.shape)

neg_list = ['GarageArea', 'SalePrice']
for item in neg_list:
    neg_item = train[item] <= 0
    print('for ' + item, 'there are ' + str(neg_item.sum()) + '<=0 outliers')


df = pd.read_csv('train.csv')
GA = df['GarageArea']
SP = df['SalePrice']
plt.scatter(GA, SP)

plt.axis()
# 设置title和x，y轴的label
plt.title("GarageArea And SalePrice")
plt.xlabel("GarageArea")
plt.ylabel("SalePrice")
p1=plt.scatter(GA,SP,marker='x',color='g',label='1',s=10)


plt.show()

df1 = df.copy()

df1 = df1[df1['GarageArea'] != 0]
print(df1)
GA = df1['GarageArea']
SP = df1['SalePrice']
plt.scatter(GA, SP)
plt.xlim(0, 1000)
plt.ylim(0, 600000)
plt.axis()
# 设置title和x，y轴的label
plt.title("GarageArea And SalePrice")
plt.xlabel("GarageArea")
plt.ylabel("SalePrice")
p2=plt.scatter(GA,SP,marker='x',color='g',label='1',s=10)


plt.show()

