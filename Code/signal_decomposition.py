import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler

from pandas import read_csv
from pandas import DataFrame
from datetime import datetime
from matplotlib import pyplot
from pylab import mpl

from pandas import concat
from PyEMD import EEMD, EMD, CEEMDAN

import matplotlib.pyplot as plt

def data_split(data, train_len, lookback_window):
    train = data[:train_len]  # 标志训练集
    test = data[train_len:]  # 标志测试集

    # X1[]代表移动窗口中的10个数
    # Y1[]代表相应的移动窗口需要预测的数
    # X2, Y2 同理

    X1, Y1 = [], []
    for i in range(lookback_window, len(train)):
        X1.append(train[i - lookback_window:i])
        Y1.append(train[i])
        Y_train = np.array(Y1)
        X_train = np.array(X1)

    X2, Y2 = [], []
    for i in range(lookback_window, len(test)):
        X2.append(test[i - lookback_window:i])
        Y2.append(test[i])
        y_test = np.array(Y2)
        X_test = np.array(X2)

    return (X_train, Y_train, X_test, y_test)


def data_split_LSTM(X_train, Y_train, X_test, y_test):  # data split f
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    Y_train = Y_train.reshape(Y_train.shape[0], 1)
    y_test = y_test.reshape(y_test.shape[0], 1)
    return (X_train, Y_train, X_test, y_test)



if __name__ == '__main__':
    dataset = pd.read_csv('C:/Users/Cy/Documents/Vscode/EEMD-LSTM/csv/NB/data-563-1.csv', header=0, index_col=0, parse_dates=True)
    data = dataset.values.reshape(-1)

    values = dataset.values
    groups = [0, 1, 2, 3]
    # fig, axs = plt.subplots(1)

    df = pd.DataFrame(dataset)  # 整体数据的全部字典类型
    do = df['all_time_change']  # 返回all_time_change那一列，用字典的方式
    print(do)
    DO = []
    for i in range(0, len(do)):
        DO.append([do[i]])

    scaler_DO = MinMaxScaler(feature_range=(0, 1))
    DO = scaler_DO.fit_transform(DO)   #归一化
    print("DO",DO.shape)

    c = int(len(DO) * .8)
    lookback_window = 2

    # #数组划分为不同的数据集
    l_X1_train, l_Y1_train, l_X1_test, l_Y1_test =data_split(DO, c, lookback_window) 
    l_X2_train, l_Y2_train, l_X2_test, l_Y2_test = data_split_LSTM(l_X1_train, l_Y1_train, l_X1_test, l_Y1_test)
    

    l_X2_train_svr = l_X2_train.reshape(l_X2_train.shape[0], -1)
    l_Y2_train_svr = l_Y2_train.reshape(-1,)
    l_X2_test_svr = l_X2_test.reshape(l_X2_test.shape[0], -1)


########################################################EMD
    emd = EMD()
    emd_imfs = emd.emd(DO.reshape(-1),None,8)
    print("emd_imfs",emd_imfs)

    i = 1
    plt.rc('font', family='Times New Roman')
    plt.subplot(len(emd_imfs)+1,1,i)
    plt.plot(DO)
    plt.ylabel("Signal")
    for emd_imf in emd_imfs:
        plt.subplot(len(emd_imfs)+1, 1, i+1)
        plt.plot(emd_imf,color = 'black')
        plt.ylabel("IMF "+str(i))
        i += 1
    plt.show()

########################################################EEMD
#     eemd = EEMD()
#     eemd.noise_seed(12345)
#     eemd_imfs = eemd.eemd(DO.reshape(-1),None,8)

#     i = 1
#     plt.subplot(len(eemd_imfs)+1,1,i)
#     plt.plot(DO)
#     for emd_imf in eemd_imfs:
#         plt.subplot(len(eemd_imfs)+1, 1, i+1)
#         plt.plot(emd_imf,color = 'black')
#         i += 1
#     # plt.plot(DO, "black")
#     plt.show()

# ########################################################CEEMDAN
    ceemdan = CEEMDAN()
    # ceemdan.noise_seed(12345)
    ceemdan_imfs = ceemdan.ceemdan(DO.reshape(-1),None,8)
    print("ceemdan_imfs",ceemdan_imfs)

    i = 1
    plt.rc('font', family='Times New Roman')
    plt.subplot(len(ceemdan_imfs)+1,1,i)
    plt.plot(DO)
    plt.ylabel("Signal")
    for emd_imf in ceemdan_imfs:
        plt.subplot(len(ceemdan_imfs)+1, 1, i+1)
        plt.plot(emd_imf,color = 'black')
        plt.ylabel("IMF "+str(i))
        i += 1
    # plt.plot(DO, "black")
    plt.show()