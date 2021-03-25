'''
数据预处理
'''
import numpy as np
import pandas as pd

from pandas import read_csv
from pandas import DataFrame
from datetime import datetime
from matplotlib import pyplot
from pylab import mpl

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from pandas import concat
from scipy import interpolate, math
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

plt.rcParams['figure.figsize'] = (10.0, 5.0)  # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

dataset = pd.read_csv('../csv/Water Quality Record.csv', header=0, index_col=0, parse_dates=True)

values = dataset.values
groups = [0, 1, 2, 3]
# fig, axs = plt.subplots(1)

df = pd.DataFrame(dataset)  # 整体数据的全部字典类型
do = df['Dissolved Oxygen']  # 返回溶解氧那一列，用字典的方式

DO = []
for i in range(0, len(do)):
    DO.append([do[i]])
# scaler_DO = MinMaxScaler(feature_range=(0, 1))
# DO = scaler_DO.fit_transform(DO)
DO = np.array(DO).reshape(-1,1)
print(DO.shape)


model=IsolationForest(3000,0.25)

model.fit(DO)

a=model.predict(DO)

# a = np.array(a).reshape(-1,1)
#
# print(DO)
# print(a)
#
# fig, ax = plt.subplots(figsize=(10, 6))
# ax.plot(DO, color='blue', label='Normal')
# ax.plot(a,color='red', label='Anomaly')
# plt.show()
#


