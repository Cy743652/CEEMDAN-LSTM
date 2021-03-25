import numpy as np
import pandas as pd

from pandas import read_csv
from pandas import DataFrame
from datetime import datetime
from matplotlib import pyplot
from pylab import mpl

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.ensemble import IsolationForest

from pandas import concat
from PyEMD import EEMD, EMD,CEEMDAN

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.layers import Dropout
from keras.layers import Activation

from scipy import interpolate, math
import matplotlib.pyplot as plt

from keras import Input, Model
from keras.layers import Dense
from keras.models import load_model

from evaluate_data import *
from sklearn import  svm      #### SVM回归####
from elm import *     ####ELM回归
from sklearn.neural_network import MLPRegressor   ###BP回归
from nested_lstm import NestedLSTM #####nlstm回归
from keras.layers import Dense, Activation, Conv1D, LSTM, Dropout, Reshape, Bidirectional

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


# def visualize(history):
#     plt.rcParams['figure.figsize'] = (10.0, 6.0)
#     # Plot training & validation loss values
#     plt.plot(history.history['loss'])
#     plt.plot(history.history['val_loss'])
#     plt.title('Model loss')
#     plt.ylabel('Loss')
#     plt.xlabel('Epoch')
#     plt.legend(['Train', 'Test'], loc='upper left')
#     plt.show()


# def NLSTM_Model(timestep):
#    batch_size, timesteps, input_dim = None, timestep, 1
#    i = Input(batch_shape=(batch_size, timesteps, input_dim))
# #    con1 = Conv1D(filters=32, kernel_size=5, padding='same', activation='relu',input_shape=(None, 1))(i)
#    # model.add(MaxPooling1D(pool_size=2,strides=2))
# #    con2 = Conv1D(filters=64, kernel_size=3, padding='valid', activation='relu')(con1)
#    x = NestedLSTM(32, depth=2, dropout=0, recurrent_dropout=0.0,)(i)

#    # x = LSTM(units=100, return_sequences=False)(con1)
#    # ############################################################################i
#    # a_left = Permute((2, 1), name='permute0_left')(x)
#    # a_left = Dense(timesteps, activation='linear', name='dense_left')(a_left)  # softmax
#    # a_left_probs = Permute((2, 1), name='permute1_left')(a_left)
#    # print("a_left_probs的大小：===============================", a_left_probs.shape)
#    # output_attention_mul_left = merge.concatenate([x, a_left_probs], name='merge')
#    # attention_mul_left = Flatten(name='flatten_left')(output_attention_mul_left)
#    # ######################################################################################

#    o = Dense(1)(x)
#    o = Activation('linear')(o)
#    # output_layer = x
#    # model = Sequential()
#    model = Model(inputs=[i], outputs=[o])
#    model.compile(optimizer='rmsprop', loss='mse', )
#    model.summary()
#    return model

def LSTM_Model(X_train, Y_train):
    model = Sequential()
    model.add(LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2])))   #已经确定10步长
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    model.fit(X_train, Y_train, epochs=20, batch_size=1, validation_split=0.1, verbose=1, shuffle=True)
    return model
# batch_size=1 有最优的结果, 但是需要更长的时间


def EMD_LSTM_Model(X_train, Y_train,i):
    filepath = 'res/' + str(i) + '-{epoch:02d}-{val_acc:.2f}.h5'
    checkpoint = ModelCheckpoint(filepath, monitor='loss',verbose=1,save_best_only=False,mode='auto',period=10)
    callbacks_list = [checkpoint]
    model = Sequential()
    model.add(Bidirectional(LSTM(32, input_shape=(X_train.shape[1], X_train.shape[2]),return_sequences = True)))# 已经确定10步长
   
    '''
    如果设置return_sequences = True，该LSTM层会返回每一个time step的h，
    那么该层返回的就是1个由多个h组成的2维数组了，如果下一层不是可以接收2维数组
    的层，就会报错。所以一般LSTM层后面接LSTM层的话，设置return_sequences = True，
    如果接全连接层的话，设置return_sequences = False。
    '''
    model.add(Bidirectional(LSTM(units=32)))
    model.add(Dense(1))
    model.add(Activation('tanh'))
    model.compile(loss='mse', optimizer='adam')
    model.fit(X_train, Y_train, epochs=20, batch_size=1, validation_split=0.1, verbose=2, shuffle=True)
    return (model)

def EEMD_LSTM_Model(X_train, Y_train,i):
    filepath = 'res/' + str(i) + '-{epoch:02d}-{val_acc:.2f}.h5'
    checkpoint = ModelCheckpoint(filepath, monitor='loss',verbose=1,save_best_only=False,mode='auto',period=10)
    callbacks_list = [checkpoint]
    model = Sequential()
    model.add(Bidirectional(LSTM(50,activation='tanh', input_shape=(X_train.shape[1], X_train.shape[2]))))  # 已经确定10步长
    # '''
    # ,return_sequences = True
    # 如果设置return_sequences = True，该LSTM层会返回每一个time step的h，
    # 那么该层返回的就是1个由多个h组成的2维数组了，如果下一层不是可以接收2维数组
    # 的层，就会报错。所以一般LSTM层后面接LSTM层的话，设置return_sequences = True，
    # 如果接全连接层的话，设置return_sequences = False。
    # '''
    model.add(Dense(50, activation='tanh'))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    model.fit(X_train, Y_train, epochs=20, batch_size=1, validation_split=0.1, verbose=2, shuffle=True)
    return model

def isolutionforest(DO):
    rng = np.random.RandomState(42)
    clf = IsolationForest(random_state=rng, contamination=0.025)  # contamination为异常样本比例
    clf.fit(DO)

    DO_copy = DO
    m = 0

    pre = clf.predict(DO)
    for i in range(len(pre)):
        if pre[i] == -1:
            DO_copy = np.delete(DO_copy, i - m, 0)
            print(i)
            m += 1
    return DO_copy



if __name__ == '__main__':

    dataset = pd.read_csv('C:/Users/Cy/Documents/Vscode/EEMD-LSTM/csv/ZH/data-184-1.csv', header=0, index_col=0, parse_dates=True)
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

    # plt.rc('font', family='Times New Roman')
    # plt.figure(1,figsize=(15,5))
    # plt.plot(DO, "black", label="true", linewidth=2.5,linestyle='--', marker='.')
    # plt.show()

    scaler_DO = MinMaxScaler(feature_range=(0, 1))
    DO = scaler_DO.fit_transform(DO)   #归一化
    print("DO",DO.shape)
    

    # DO = isolutionforest(DO)

  
    c = int(len(DO) * .7)
    lookback_window = 2

    # #数组划分为不同的数据集
    l_X1_train, l_Y1_train, l_X1_test, l_Y1_test =data_split(DO, c, lookback_window) 
    l_X2_train, l_Y2_train, l_X2_test, l_Y2_test = data_split_LSTM(l_X1_train, l_Y1_train, l_X1_test, l_Y1_test)
    

    l_X2_train_svr = l_X2_train.reshape(l_X2_train.shape[0], -1)
    l_Y2_train_svr = l_Y2_train.reshape(-1,)
    l_X2_test_svr = l_X2_test.reshape(l_X2_test.shape[0], -1)


    

    "svm回归"

    model_SVR = svm.SVR()  # svm回归
    model_SVR.fit(l_X2_train_svr, l_Y2_train_svr)

    predict_svr = model_SVR.predict(l_X2_test_svr) # svm训练预测



    # "ELM回归"
    # model_ELM = ELMRegressor()
    # model_ELM.fit(l_X2_train_svr, l_Y2_train_svr)

    # predict_elm = model_ELM.predict(l_X2_test_svr)


    "BPNN回归"
    model_BP = MLPRegressor(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(10,), random_state=1)
    model_BP.fit(l_X2_train_svr, l_Y2_train_svr)

    predict_bp = model_BP.predict(l_X2_test_svr)

    "LSTM回归"
    # model_DO_LSTM=LSTM_Model(l_X2_train, l_Y2_train)
    model_lstm = LSTM_Model(l_X2_train, l_Y2_train)
    predict_lstm=model_lstm.predict(l_X2_test)

    # "NLSTM回归"
    # model_nlstm = NLSTM_Model(lookback_window)
    # model_nlstm.fit(l_X2_train, l_Y2_train)
    # predict_nlstm = model_nlstm.predict(l_X2_test)


    #变回原来的值,inverse_transform
    # l_Y2_train=scaler_DO.inverse_transform(l_Y2_train)

    l_Y2_test = scaler_DO.inverse_transform(l_Y2_test)  # Y1_test 反归一化

    predict_svr = np.array(predict_svr).reshape(1,-1)
    predict_svr = scaler_DO.inverse_transform(predict_svr)  # SVM反归一化
    predict_svr = np.array(predict_svr).reshape(-1,1)

    # predict_elm = np.array(predict_elm).reshape(1,-1)
    # predict_elm = scaler_DO.inverse_transform(predict_elm) #elm反归一化
    # predict_elm = np.array(predict_elm).reshape(-1,1)

    predict_bp = np.array(predict_bp).reshape(1,-1)
    predict_bp = scaler_DO.inverse_transform(predict_bp)###BP反归一化
    predict_bp = np.array(predict_bp).reshape(-1,1)

    predict_lstm = scaler_DO.inverse_transform(predict_lstm)#####lstm反归一化

    # predict_nlstm = scaler_DO.inverse_transform(predict_nlstm)#########nlstm反归一化


 

    # Y2_test_hat=model_DO_LSTM.predict(l_X2_test)
    # Y2_test_hat=scaler_DO.inverse_transform(Y2_test_hat)
    # l_Y2_test=scaler_DO.inverse_transform(l_Y2_test)



    #################################################################EMD_LSTM
    emd = EMD()
    emd_imfs = emd.emd(DO.reshape(-1),None,8)
    emd_imfs_prediction = []

    i = 1
    plt.rc('font', family='Times New Roman')
    plt.subplot(len(emd_imfs)+1,1,i)
    plt.plot(DO,color = 'black')
    plt.ylabel("Signal")
    plt.title("EMD")
    for emd_imf in emd_imfs:
       plt.subplot(len(emd_imfs)+1, 1, i+1)
       plt.plot(emd_imf,color = 'black')
       plt.ylabel("IMF "+str(i))
       i += 1 
    plt.show()

    emd_test = np.zeros([len(do) - c - lookback_window, 1])

    i = 1
    for emd_imf in emd_imfs:
       print('-'*45)
       print('This is  ' + str(i)  + '  time(s)')
       print('*'*45)
       emd_X1_train, emd_Y1_train, emd_X1_test, emd_Y1_test = data_split(imf_data(emd_imf,1), c, lookback_window)
       emd_X2_train, emd_Y2_train, emd_X2_test, emd_Y2_test = data_split_LSTM(emd_X1_train, emd_Y1_train, emd_X1_test, emd_Y1_test)
       emd_test += emd_Y2_test
       model = EEMD_LSTM_Model(emd_X2_train,emd_Y2_train,i)
    #    model.save('EEMD-LSTM-imf' + str(i) + '.h5')
       emd_prediction_Y = model.predict(emd_X2_test)
       emd_imfs_prediction.append(emd_prediction_Y)
       i+=1

    emd_imfs_prediction = np.array(emd_imfs_prediction)
    emd_prediction = [0.0 for i in range(len(emd_test))]
    emd_prediction = np.array(emd_prediction)
    for i in range(len(emd_test)):
        emd_t = 0.0
        for emd_imf_prediction in emd_imfs_prediction:
            emd_t += emd_imf_prediction[i][0]
        emd_prediction[i] = emd_t

    emd_prediction = emd_prediction.reshape(emd_prediction.shape[0], 1)

    emd_prediction = scaler_DO.inverse_transform(emd_prediction)##########反归一化

    ################################################################EEMD_LSTM

    eemd = EEMD()
    eemd.noise_seed(12345)
    eemd_imfs = eemd.eemd(DO.reshape(-1),None,8)

    eemd_imfs_prediction = []


    i = 1
    plt.rc('font', family='Times New Roman')
    plt.subplot(len(eemd_imfs)+1,1,i)
    plt.plot(DO,color = 'black')
    plt.ylabel("Signal")
    plt.title("EEMD")
    for imf in eemd_imfs:
       plt.subplot(len(eemd_imfs)+1, 1, i+1)
       plt.plot(imf,color = 'black')
       plt.ylabel("IMF "+str(i))
       i += 1

    # # plt.savefig('result_imf.png')
    plt.show()

    test = np.zeros([len(DO) - c - lookback_window, 1])

    # i = 1
    # for imf in imfs:
    #     print('-' * 45)
    #     print('This is  ' + str(i) + '  time(s)')
    #     print('*' * 45)
    #     X1_train, Y1_train, X1_test, Y1_test = data_split(imf_data(imf, 1), c, lookback_window)
    #     X2_train, Y2_train, X2_test, Y2_test = data_split_LSTM(X1_train, Y1_train, X1_test, Y1_test)
    #     test += Y2_test
    #     model = load_model('../B16LSTM50tanh/EEMD-LSTM-imf' + str(i) + '-90.h5')
    #     prediction_Y = model.predict(X2_test)

    #     imfs_prediction.append(prediction_Y)
    #     i += 1;

    i = 1
    for imf in eemd_imfs:
       print('-'*45)
       print('This is  ' + str(i)  + '  time(s)')
       print('*'*45)
       X1_train, Y1_train, X1_test, Y1_test = data_split(imf_data(imf,1), c, lookback_window)
       X2_train, Y2_train, X2_test, Y2_test = data_split_LSTM(X1_train, Y1_train, X1_test, Y1_test)
       test += Y2_test
       model = EEMD_LSTM_Model(X2_train,Y2_train,i)
       #model.save('C:/Users/Cy/Documents/Vscode/EEMD-LSTM/Code/EEMD-LSTM-imf' + str(i) + '-50.h5')
       prediction_Y = model.predict(X2_test)
       eemd_imfs_prediction.append(prediction_Y)
       i+=1;


    eemd_imfs_prediction = np.array(eemd_imfs_prediction)
    
    eemd_prediction = [0.0 for i in range(len(test))]
    eemd_prediction = np.array(eemd_prediction)
    for i in range(len(test)):
        t = 0.0
        for imf_prediction in eemd_imfs_prediction:
            t += imf_prediction[i][0]
        eemd_prediction[i] = t

    eemd_prediction = eemd_prediction.reshape(eemd_prediction.shape[0], 1)

    #################################反归一化
    test = scaler_DO.inverse_transform(test)
    eemd_prediction = scaler_DO.inverse_transform(eemd_prediction)

    ################################################################CEEMDAN_LSTM

    ceemdan = CEEMDAN()
    ceemdan_imfs = ceemdan.ceemdan(DO.reshape(-1),None,8)

    ceemdan_imfs_prediction = []


    i = 1
    plt.rc('font', family='Times New Roman')
    plt.subplot(len(ceemdan_imfs)+1,1,i)
    plt.plot(DO,color = 'black')
    plt.ylabel("Signal")
    plt.title("CEEMDAN")
    for imf in ceemdan_imfs:
       plt.subplot(len(ceemdan_imfs)+1, 1, i+1)
       plt.plot(imf,color = 'black')
       plt.ylabel("IMF "+str(i))
       i += 1
    
    # # plt.savefig('result_imf.png')
    plt.show()

    test = np.zeros([len(DO) - c - lookback_window, 1])

    i = 1
    for imf in ceemdan_imfs:
       print('-'*45)
       print('This is  ' + str(i)  + '  time(s)')
       print('*'*45)
       X1_train, Y1_train, X1_test, Y1_test = data_split(imf_data(imf,1), c, lookback_window)
       X2_train, Y2_train, X2_test, Y2_test = data_split_LSTM(X1_train, Y1_train, X1_test, Y1_test)
       test += Y2_test
       model = EEMD_LSTM_Model(X2_train,Y2_train,i)
       #model.save('C:/Users/Cy/Documents/Vscode/EEMD-LSTM/Code/EEMD-LSTM-imf' + str(i) + '-50.h5')
       #model = Bidirectional(LSTM(64, return_sequences=True))(i)
       prediction_Y = model.predict(X2_test)
       ceemdan_imfs_prediction.append(prediction_Y)
       i+=1;


    ceemdan_imfs_prediction = np.array(ceemdan_imfs_prediction)
    
    ceemdan_prediction = [0.0 for i in range(len(test))]
    ceemdan_prediction = np.array(ceemdan_prediction)
    for i in range(len(test)):
        t = 0.0
        for imf_prediction in ceemdan_imfs_prediction:
            t += imf_prediction[i][0]
        ceemdan_prediction[i] = t

    ceemdan_prediction = ceemdan_prediction.reshape(ceemdan_prediction.shape[0], 1)

    #################################反归一化
    test = scaler_DO.inverse_transform(test)
    ceemdan_prediction = scaler_DO.inverse_transform(ceemdan_prediction)



##################################################evaluation
    #####=====================lstm
    rmse_lstm = RMSE1(l_Y2_test, predict_lstm)
    mape_lstm = MAPE1(l_Y2_test, predict_lstm)
    mae_lstm = MAE1(l_Y2_test, predict_lstm)
    print("mae_lstm:", mae_lstm)
    print("rmse_lstm:", rmse_lstm)
    print("mape_lstm:", mape_lstm)

    ##########################svr
    rmse_svr = RMSE1(l_Y2_test, predict_svr)
    mape_svr = MAPE1(l_Y2_test, predict_svr)
    mae_svr = MAE1(l_Y2_test, predict_svr)
    print("mae_svr:", mae_svr)
    print("rmse_svr:", rmse_svr)
    print("mape_svr:", mape_svr)

    #####=====================EEMD-LSTM
    rmse_eemd_lstm = RMSE1(l_Y2_test, eemd_prediction)
    mape_eemd_lstm = MAPE1(l_Y2_test, eemd_prediction)
    mae_eemd_lstm = MAE1(l_Y2_test, eemd_prediction)
    print("mae_eemd_lstm:", mae_eemd_lstm)
    print("rmse_eemd_lstm:", rmse_eemd_lstm)
    print("mape_eemd_lstm:", mape_eemd_lstm)

    #####=====================EMD-LSTM
    rmse_emd_lstm = RMSE1(l_Y2_test, emd_prediction)
    mape_emd_lstm = MAPE1(l_Y2_test, emd_prediction)
    mae_emd_lstm = MAE1(l_Y2_test, emd_prediction)
    print("mae_emd_lstm:", mae_emd_lstm)
    print("rmse_emd_lstm:", rmse_emd_lstm)
    print("mape_emd_lstm:", mape_emd_lstm)

    #####=====================CEEMDAN-LSTM
    rmse_ceemdan_lstm = RMSE1(l_Y2_test, ceemdan_prediction)
    mape_ceemdan_lstm = MAPE1(l_Y2_test, ceemdan_prediction)
    mae_ceemdan_lstm = MAE1(l_Y2_test, ceemdan_prediction)
    print("mae_ceemdan_lstm:", mae_ceemdan_lstm)
    print("rmse_ceemdan_lstm:", rmse_ceemdan_lstm)
    print("mape_ceemdan_lstm:", mape_ceemdan_lstm)

    ###################################BPNN
    rmse_bp = RMSE1(l_Y2_test, predict_bp)
    mape_bp = MAPE1(l_Y2_test, predict_bp)
    mae_bp = MAE1(l_Y2_test, predict_bp)
    print("mae_bp:", mae_bp)
    print("rmse_bp:", rmse_bp)
    print("mape_bp:", mape_bp)



    ###===============画图===========================
    plt.rc('font', family='Times New Roman')
    plt.figure(1,figsize=(15,5))
    plt.plot(predict_bp, "indianred", label="bp",linewidth=1)
    plt.plot(predict_svr, "tan", label="SVR", linewidth=1)
    plt.plot(predict_lstm, "lightsteelblue", label="lstm", linewidth=1)
    plt.plot(emd_prediction, "seagreen", label="EMD-LSTM", linewidth=1)
    # plt.plot(predict_elm, "r", label="elm", linewidth=1)
    plt.plot(eemd_prediction, "r", label="EEMD-LSTM", linewidth=2.5,linestyle='--',marker='^',markersize=6)
    plt.plot(l_Y2_test, "black", label="true", linewidth=2.5,linestyle='--', marker='.')
    plt.plot(ceemdan_prediction, "darkred", label="CEEMDAN-LSTM", linewidth=2.5,linestyle='--',marker='^',markersize=6)


    # plt.rc('font', family='Times New Roman')
    # plt.figure(1,figsize=(15,5))
    # plt.plot(predict_bp, "black", label="bp",linewidth=1)
    # plt.plot(predict_svr, "black", label="SVR", linewidth=1)
    # plt.plot(predict_lstm, "black", label="lstm", linewidth=1)
    # plt.plot(emd_prediction, "black", label="EMD-BiLSTM", linewidth=1)
    # # plt.plot(predict_elm, "r", label="elm", linewidth=1)
    # # plt.plot(eemd_prediction, "black", label="EEMD-LSTM", linewidth=1)
    # plt.plot(l_Y2_test, "black", label="true", linewidth=2.5,linestyle='--', marker='.')
    # # plt.plot(eemd_prediction, "black", label="EEMD-LSTM", linewidth=2.5,linestyle='--',marker='^',markersize=6)
    # plt.plot(ceemdan_prediction, "black", label="CEEMDAN-BiLSTM", linewidth=2.5,linestyle='--',marker='^',markersize=6)

    plt.grid(True, linestyle=":", color="gray", linewidth="0.5", axis='both')
    plt.xlabel("time(days)",fontsize=18)
    plt.ylabel("height(mm)",fontsize=18)
    plt.title("569")
    plt.legend(loc='best')

    plt.show()