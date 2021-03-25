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
from PyEMD import EEMD

from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
from keras.layers import Activation

from scipy import interpolate, math
import matplotlib.pyplot as plt

from keras import Input, Model
from keras.layers import Dense
from keras.models import load_model

from evaluate_data import *


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


def imf_data(data, lookback_window):
    X1 = []
    for i in range(lookback_window, len(data)):
        X1.append(data[i - lookback_window:i])
    X1.append(data[len(data) - 1:len(data)])
    X_train = np.array(X1)
    return X_train


def visualize(history):
    plt.rcParams['figure.figsize'] = (10.0, 6.0)
    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()


def BP_Model(X_train, Y_train,i):
    filepath = 'res/' + str(i) + '-{epoch:02d}-{val_acc:.2f}.h5'
    checkpoint = ModelCheckpoint(filepath,
                                 monitor='loss',
                                 verbose=1,
                                 save_best_only=False,
                                 mode='auto',
                                 period=10)
    callbacks_list = [checkpoint]
    model = Sequential()
    model.add(32, input_shape=(X_train.shape[1],X_train.shape[2]))  # 已经确定10步长
    model.add(Dense(32))
    model.add(Activation('tanh'))
    model.compile(loss='mse', optimizer='adam')
    model.fit(X_train, Y_train, epochs=20, batch_size=32, validation_split=0.1, verbose=1, shuffle=True)
    return (model)


def plot_curve(true_data, predicted):
    rmse=format(RMSE(test,prediction),'.4f')
    mape=format(MAPE(test,prediction),'.4f')
    plt.plot(true_data, label='True data')
    plt.plot(predicted, label='Predicted data')
    plt.legend()
    plt.text(1, 1, 'RMSE:' + str(rmse)+' \n '+'MAPE:'+str(mape), color = "r",style='italic', wrap=True)
    # plt.text(2, 2, "RMSE:" + str(format(RMSE(true_data,predicted),'.4f'))+" \n "+"MAPE:"+str(format(MAPE(true_data,predicted),'.4f')), style='italic', ha='center', wrap=True)
    # plt.savefig('result_EEMD_LSTM_100.png')
    # plt.show()



if __name__ == '__main__':



    dataset = pd.read_csv('C:/Users/Cy/Documents/Vscode/EEMD-LSTM/csv/data-569-1.csv', header=0, index_col=0, parse_dates=True)
    # dataset = pd.read_csv('Water Quality Record.csv', header=0, index_col=0, parse_dates=True)
    data = dataset.values.reshape(-1)

    values = dataset.values
    groups = [0, 1, 2, 3]
    # fig, axs = plt.subplots(1)

    df = pd.DataFrame(dataset)  # 整体数据的全部字典类型
    do = df['all_time_change']  # 返回溶解氧那一列，用字典的方式
    # do = df['Dissolved Oxygen']

    DO = []
    for i in range(0, len(do)):
        DO.append([do[i]])
    scaler_DO = MinMaxScaler(feature_range=(0, 1))
    DO = scaler_DO.fit_transform(DO)

    eemd = EEMD()
    imfs = eemd.eemd(DO.reshape(-1),None,8)
    c = int(len(do) * .7)
    lookback_window = 2
    imfs_prediction = []

    # i = 1
    # for imf in imfs:
    #    plt.subplot(len(imfs), 1, i)
    #    plt.plot(imf)
    #    i += 1
    #
    # plt.savefig('res/result_imf.png')
    # plt.show()

    test = np.zeros([len(do) - c - lookback_window, 1])

    # i = 1
    # for imf in imfs:
    #     print('-' * 45)
    #     print('This is  ' + str(i) + '  time(s)')
    #     print('*' * 45)
    #     X1_train, Y1_train, X1_test, Y1_test = data_split(imf_data(imf, 1), c, lookback_window)
    #     X2_train, Y2_train, X2_test, Y2_test = data_split_LSTM(X1_train, Y1_train, X1_test, Y1_test)
    #     test += Y2_test
    #     model = load_model('EEMD-LSTM-B1-E100/EEMD-LSTM-imf' + str(i) + '-100.h5')
    #     prediction_Y = model.predict(X2_test)
    #     imfs_prediction.append(prediction_Y)
    #     i += 1;

    i = 1
    for imf in imfs:
       print('-'*45)
       print('This is  ' + str(i)  + '  time(s)')
       print('*'*45)
       X1_train, Y1_train, X1_test, Y1_test = data_split(imf_data(imf,1), c, lookback_window)
       X2_train, Y2_train, X2_test, Y2_test = data_split_LSTM(X1_train, Y1_train, X1_test, Y1_test)
       test += Y2_test
    
       model = BP_Model(X2_train,Y2_train,i)
       #model.save('EEMD-LSTM-imf' + str(i) + '.h5')
       prediction_Y = model.predict(X2_test)
       imfs_prediction.append(prediction_Y)
       i+=1;


    imfs_prediction = np.array(imfs_prediction)
    prediction = [0.0 for i in range(len(test))]
    prediction = np.array(prediction)
    for i in range(len(test)):
        t = 0.0
        for imf_prediction in imfs_prediction:
            t += imf_prediction[i][0]
        prediction[i] = t

    prediction = prediction.reshape(prediction.shape[0], 1)

    test = scaler_DO.inverse_transform(test)
    prediction = scaler_DO.inverse_transform(prediction)

    #####=====================EEMD-LSTM
    rmse_emd_lstm = RMSE1(test, prediction)
    mape_emd_lstm = MAPE1(test, prediction)
    mae_emd_lstm = MAE1(test, prediction)
    print("mae_eemd_bp:", mae_emd_lstm)
    print("rmse_eemd_bp:", rmse_emd_lstm)
    print("mape_eemd_bp:", mape_emd_lstm)

    
    #########################

    plt.figure(1,figsize=(15,5))
    plt.plot(test, "k", label="true", linewidth=1)
    # plt.plot(Y2_test_hat, "m", label="lstm", linewidth=1)
    plt.plot(prediction, "b", label="EEMD-BP", linewidth=1)
    # plt.plot(predict_lgbm, "y", label="LGBM", linewidth=1)
    # plt.plot(predict_xgboost, "r", label="xgboost", linewidth=1)
    # plt.plot(np.arange(len(y_test)), predict_mlp, "g", label="mlp")
    # plt.plot(predict_tcn, "r", label="NLSTM", linewidth=1)
    plt.xlabel("time(days)")
    plt.ylabel("tunnel settlement(mm)")
    plt.title("571")
    plt.legend(loc='best')

    plt.show()

