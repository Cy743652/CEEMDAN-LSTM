 # -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 17:52:31 2018
@author: bigV
"""
import math
from sklearn.metrics import mean_squared_error #均方误差       MSE
from sklearn.metrics import mean_absolute_error #平方绝对误差  MAE
from sklearn.metrics import r2_score#R square #调用
# mean_squared_error(y_test,y_predict)
# mean_absolute_error(y_test,y_predict)
# r2_score(y_test,y_predict)

## array 版
# mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
def MAPE(true,predict):
    L1=int(true.shape[0])
    L2=int(predict.shape[0])
    #print(L1,L2)
    if L1==L2:
        #SUM1=sum(abs(true-predict)/abs(true))
        SUM=0.0
        for i in range(L1-1):
            SUM=(abs(true[i,0]-predict[i,0])/true[i,0])+SUM
        per_SUM=SUM*100.0
        mape=per_SUM/L1
        return mape
    else:
        print("error")

#list 版
def MAPE1(true,predict):
#    L1=int(true.shape[0])
#    L2=int(predict.shape[0])
    L1 = int(len(true))
    L2 = int(len(predict))
    #print(L1,L2)
    if L1==L2:
        #SUM1=sum(abs(true-predict)/abs(true))
        SUM=0.0
        for i in range(L1):
            #SUM=(abs(true[i,0]-predict[i,0])/true[i,0])+SUM
            SUM = abs((true[i]-predict[i])/true[i])+SUM
        per_SUM=SUM*100.0
        mape=per_SUM/L1
        return mape
    else:
        print("error")

def MSE(true_data, predict_data):
    testY = true_data
    testPredict = predict_data
    #mse = np.sum((predict_data-true_data)**2)/len(true_data) #跟数学公式一样的
    mse = mean_squared_error(testY[:,0], testPredict[:, 0])
    return mse

def MSE1(true_data, predict_data):
    testY = true_data
    testPredict = predict_data
    mse = mean_squared_error(testY[:], testPredict[:])
    return mse


def RMSE(true_data, predict_data):
    testY = true_data
    testPredict = predict_data
    # rmse = mse ** 0.5
    rmse = math.sqrt( mean_squared_error(testY[:,0], testPredict[:, 0]))
    return rmse

def RMSE1(true_data, predict_data):
    testY = true_data
    testPredict = predict_data
    rmse = math.sqrt( mean_squared_error(testY[:], testPredict[:]))
    return rmse


def MAE(true_data, predict_data):
    testY = true_data
    testPredict = predict_data
    #mae = np.sum(np.absolute(predict_data - true_data))/len(true_data)
    mae=mean_absolute_error(testY[:,0], testPredict[:, 0])
    return mae

def MAE1(true_data, predict_data):
    testY = true_data
    testPredict = predict_data
    mae=mean_absolute_error(testY[:], testPredict[:])
    return mae
def R2(true_data, predict_data):
    testY = true_data
    testPredict = predict_data
    r2=r2_score(testY[:], testPredict[:])
    return r2
def main():
    a = [1,2,3]
    b = [2,3,4]
    print(MAPE1(b,a))
    print(MAE1(b,a))
    print(RMSE1(b,a))



if __name__=="__main__":
    main()









    