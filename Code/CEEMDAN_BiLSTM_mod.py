import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest

from PyEMD import EEMD, EMD,CEEMDAN

from sklearn import svm      #### SVM回归####
from elm import *     ####ELM回归
from sklearn.neural_network import MLPRegressor   ###BP回归
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Activation, Conv1D, LSTM, Dropout, Reshape, Bidirectional

from evaluate_data import *


def data_split(data, train_len, lookback_window):

    X_all = []
    Y_all = []
    data = data.reshape(-1, )
    for i in range(lookback_window, len(data)):
        X_all.append(data[i - lookback_window:i])
        Y_all.append(data[i])

    X_train = X_all[:train_len]
    X_test = X_all[train_len:]

    Y_train = Y_all[:train_len]
    Y_test = Y_all[train_len:]

    return [np.array(X_train), np.array(Y_train), np.array(X_test), np.array(Y_test)]


def data_split_LSTM(data_regular):  # data split f
	X_train = data_regular[0].reshape(data_regular[0].shape[0], data_regular[0].shape[1], 1)
	Y_train = data_regular[1].reshape(data_regular[1].shape[0], 1)
	X_test = data_regular[2].reshape(data_regular[2].shape[0], data_regular[2].shape[1], 1)
	y_test = data_regular[3].reshape(data_regular[3].shape[0], 1)
	return [X_train, Y_train, X_test, y_test]


def load_data(file):
	dataset = pd.read_csv(file, header=0, index_col=0, parse_dates=True)

	df = pd.DataFrame(dataset)  # 整体数据的全部字典类型
	do = df['all_time_change']  # 返回all_time_change那一列，用字典的方式
	print(do)
	full_data = []
	for i in range(0, len(do)):
		full_data.append([do[i]])

	scaler_data = MinMaxScaler(feature_range=(0, 1))
	full_data = scaler_data.fit_transform(full_data)   #归一化
	print('Size of the Dataset: ', full_data.shape)

	return full_data, scaler_data


def Training_Prediction_ML(model, y_real, scaler, data, name):

	print(str(name) + ' Start.')

	model.fit(data[0], data[1])
	predict = model.predict(data[2])
	predict = scaler.inverse_transform(predict.reshape(-1, 1)).reshape(-1, )

	global result
	result += '\n\nMAE_' + name + ': {}'.format(MAE1(y_real, predict))
	result += '\nRMSE_' + name + ': {}'.format(RMSE1(y_real, predict))
	result += '\nMAPE_' + name + ': {}'.format(MAPE1(y_real, predict))
	result += '\nR2_' + name + ':{}'.format(R2(y_real, predict))
	print(str(name) + ' Complete.')

	return predict


def Training_Prediction_DL(model, y_real, scaler, data, name):

	print(str(name) + ' Start.')

	model.fit(data[0], data[1], epochs=20, batch_size=1, validation_split=0.1, verbose=1, shuffle=True)
	predict = model.predict(data[2])
	predict = scaler.inverse_transform(predict).reshape(-1, )

	global result
	result += '\n\nMAE_' + name + ': {}'.format(MAE1(y_real, predict))
	result += '\nRMSE_' + name + ': {}'.format(RMSE1(y_real, predict))
	result += '\nMAPE_' + name + ': {}'.format(MAPE1(y_real, predict))
	result += '\nR2_' + name + ':{}'.format(R2(y_real, predict))
	print(str(name) + ' Complete.')

	return predict


def model_SVR():
	return svm.SVR()


def model_ELM():
	return ELMRegressor()


def model_BPNN():
	return MLPRegressor(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(10,), random_state=1)


def model_LSTM(step_num):
	model = Sequential()
	model.add(LSTM(50, input_shape=(step_num, 1)))   #已经确定10步长
	model.add(Dense(1))
	model.compile(loss='mse', optimizer='adam')
	return model


def imf_data(data, lookback_window):
	X1 = []
	for i in range(lookback_window, len(data)):
		X1.append(data[i - lookback_window:i])
	X1.append(data[len(data) - 1:len(data)])
	X_train = np.array(X1)
	return X_train


def LSTM_Model(X_train, Y_train):
	model = Sequential()
	model.add(LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2])))   #已经确定10步长
	model.add(Dense(1))
	model.compile(loss='mse', optimizer='adam')
	model.fit(X_train, Y_train, epochs=20, batch_size=1, validation_split=0.1, verbose=1, shuffle=True)
	return model


def EMD_LSTM_Model(X_train, Y_train,i):
	filepath = 'res/' + str(i) + '-{epoch:02d}-{val_acc:.2f}.h5'
	checkpoint = ModelCheckpoint(filepath, monitor='loss',verbose=1,save_best_only=False,mode='auto',period=10)
	callbacks_list = [checkpoint]
	model = Sequential()
	# model.add(Bidirectional(LSTM(32, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences = True)))
	# model.add(Bidirectional(LSTM(units=32)))
	model.add(LSTM(32, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
	model.add(LSTM(units=32))
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
	# model.add(Bidirectional(LSTM(50,activation='tanh', input_shape=(X_train.shape[1], X_train.shape[2]))))
	model.add(LSTM(50, activation='tanh', input_shape=(X_train.shape[1], X_train.shape[2])))
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


def main():

	full_data, scaler = load_data('data-569-1.csv')

	training_set_split = int(len(full_data) * 0.8)
	lookback_window = 2

	global result
	result = '\nEvaluation.'

	# #数组划分为不同的数据集
	data_regular = data_split(full_data, training_set_split, lookback_window)
	data_regular_DL = data_split_LSTM(data_regular)
	y_real = scaler.inverse_transform(data_regular[3].reshape(-1, 1)).reshape(-1, )

	predict_svr = Training_Prediction_ML(model_SVR(), y_real, scaler, data_regular, 'SVR')
	predict_elm = Training_Prediction_ML(model_ELM(), y_real, scaler, data_regular, 'ELM')
	predict_bp = Training_Prediction_ML(model_BPNN(), y_real, scaler, data_regular, 'BPNN')

	predict_LSTM = Training_Prediction_DL(model_LSTM(lookback_window), y_real, scaler, data_regular_DL, 'LSTM')

	#################################################################EMD_LSTM

	emd = EMD()
	emd_imfs = emd.emd(full_data.reshape(-1), None, 8)
	emd_imfs_prediction = []

	# i = 1
	# plt.rc('font', family='Times New Roman')
	# plt.subplot(len(emd_imfs) + 1, 1, i)
	# plt.plot(full_data, color='black')
	# plt.ylabel('Signal')
	# plt.title('EMD')
	# for emd_imf in emd_imfs:
	# 	plt.subplot(len(emd_imfs) + 1, 1, i + 1)
	# 	plt.plot(emd_imf, color='black')
	# 	plt.ylabel('IMF ' + str(i))
	# 	i += 1
	# plt.show()

	test = np.zeros([len(full_data) - training_set_split - lookback_window, 1])

	i = 1
	for emd_imf in emd_imfs:
		print('-' * 45)
		print('This is  ' + str(i) + '  time(s)')
		print('*' * 45)

		data_imf = data_split_LSTM(data_split(imf_data(emd_imf, 1), training_set_split, lookback_window))

		test += data_imf[3]

		model = EEMD_LSTM_Model(data_imf[0], data_imf[1], i)
		# model.save('EEMD-LSTM-imf' + str(i) + '.h5')
		emd_prediction_Y = model.predict(data_imf[2])
		emd_imfs_prediction.append(emd_prediction_Y)
		i += 1

	emd_imfs_prediction = np.array(emd_imfs_prediction)
	emd_prediction = [0.0 for i in range(len(test))]
	emd_prediction = np.array(emd_prediction)
	for i in range(len(test)):
		emd_t = 0.0
		for emd_imf_prediction in emd_imfs_prediction:
			emd_t += emd_imf_prediction[i][0]
		emd_prediction[i] = emd_t

	emd_prediction = scaler.inverse_transform(emd_prediction.reshape(-1, 1)).reshape(-1, )

	result += '\n\nMAE_emd_lstm: {}'.format(MAE1(y_real, emd_prediction))
	result += '\nRMSE_emd_lstm: {}'.format(RMSE1(y_real, emd_prediction))
	result += '\nMAPE_emd_lstm: {}'.format(MAPE1(y_real, emd_prediction))
	result += '\nR2_emd_lstm: {}'.format(R2(y_real, emd_prediction))
	################################################################EEMD_LSTM

	# eemd = EEMD()
	# # eemd.noise_seed(12345)
	# eemd_imfs = eemd.eemd(full_data.reshape(-1), None, 8)
	# eemd_imfs_prediction = []
	#
	# # i = 1
	# # plt.rc('font', family='Times New Roman')
	# # plt.subplot(len(eemd_imfs) + 1, 1, i)
	# # plt.plot(full_data, color='black')
	# # plt.ylabel('Signal')
	# # plt.title('EEMD')
	# # for imf in eemd_imfs:
	# # 	plt.subplot(len(eemd_imfs) + 1, 1, i + 1)
	# # 	plt.plot(imf, color='black')
	# # 	plt.ylabel('IMF ' + str(i))
	# # 	i += 1
	# #
	# # # # plt.savefig('result_imf.png')
	# # plt.show()
	#
	# test = np.zeros([len(full_data) - training_set_split - lookback_window, 1])
	#
	# i = 1
	# for imf in eemd_imfs:
	# 	print('-' * 45)
	# 	print('This is  ' + str(i) + '  time(s)')
	# 	print('*' * 45)
	#
	# 	data_imf = data_split_LSTM(data_split(imf_data(imf, 1), training_set_split, lookback_window))
	#
	# 	test += data_imf[3]
	#
	# 	model = EEMD_LSTM_Model(data_imf[0], data_imf[1], i)
	#
	# 	prediction_Y = model.predict(data_imf[2])
	# 	eemd_imfs_prediction.append(prediction_Y)
	# 	i += 1
	#
	# eemd_imfs_prediction = np.array(eemd_imfs_prediction)
	#
	# eemd_prediction = [0.0 for i in range(len(test))]
	# eemd_prediction = np.array(eemd_prediction)
	# for i in range(len(test)):
	# 	t = 0.0
	# 	for imf_prediction in eemd_imfs_prediction:
	# 		t += imf_prediction[i][0]
	# 	eemd_prediction[i] = t
	#
	# eemd_prediction = scaler.inverse_transform(eemd_prediction.reshape(-1, 1)).reshape(-1, )
	#
	# result += '\n\nMAE_eemd_lstm: {}'.format(MAE1(y_real, eemd_prediction))
	# result += '\nRMSE_eemd_lstm: {}'.format(RMSE1(y_real, eemd_prediction))
	# result += '\nMAPE_eemd_lstm: {}'.format(MAPE1(y_real, eemd_prediction))
	# result += '\nR2_eemd_lstm: {}'.format(R2(y_real, eemd_prediction))
	################################################################CEEMDAN_LSTM

	ceemdan = CEEMDAN()
	ceemdan_imfs = ceemdan.ceemdan(full_data.reshape(-1), None, 8)
	ceemdan_imfs_prediction = []

	# i = 1
	# plt.rc('font', family='Times New Roman')
	# plt.subplot(len(ceemdan_imfs) + 1, 1, i)
	# plt.plot(full_data, color='black')
	# plt.ylabel('Signal')
	# plt.title('CEEMDAN')
	# for imf in ceemdan_imfs:
	# 	plt.subplot(len(ceemdan_imfs) + 1, 1, i + 1)
	# 	plt.plot(imf, color='black')
	# 	plt.ylabel('IMF ' + str(i))
	# 	i += 1
	#
	# # # plt.savefig('result_imf.png')
	# plt.show()

	test = np.zeros([len(full_data) - training_set_split - lookback_window, 1])

	i = 1
	for imf in ceemdan_imfs:
		print('-' * 45)
		print('This is  ' + str(i) + '  time(s)')
		print('*' * 45)

		data_imf = data_split_LSTM(data_split(imf_data(imf, 1), training_set_split, lookback_window))
		test += data_imf[3]
		model = EEMD_LSTM_Model(data_imf[0], data_imf[1], i)  # [X_train, Y_train, X_test, y_test]
		prediction_Y = model.predict(data_imf[2])
		ceemdan_imfs_prediction.append(prediction_Y)
		i += 1



	ceemdan_imfs_prediction = np.array(ceemdan_imfs_prediction)

	ceemdan_prediction = [0.0 for i in range(len(test))]
	ceemdan_prediction = np.array(ceemdan_prediction)
	for i in range(len(test)):
		t = 0.0
		for imf_prediction in ceemdan_imfs_prediction:
			t += imf_prediction[i][0]
		ceemdan_prediction[i] = t

	ceemdan_prediction = scaler.inverse_transform(ceemdan_prediction.reshape(-1, 1)).reshape(-1, )

	result += '\n\nMAE_ceemdan_lstm: {}'.format(MAE1(y_real, ceemdan_prediction))
	result += '\nRMSE_ceemdan_lstm: {}'.format(RMSE1(y_real, ceemdan_prediction))
	result += '\nMAPE_ceemdan_lstm: {}'.format(MAPE1(y_real, ceemdan_prediction))
	result += '\nR2_ceemdan_lstm: {}'.format(R2(y_real, ceemdan_prediction))
	##################################################evaluation

	print(result)

	###===============画图===========================
	# plt.rc('font', family='Times New Roman')
	# plt.figure(1, figsize=(15, 5))
	# plt.plot(y_real , 'black', label='true', linewidth=2.5, linestyle='--', marker='.')
	# plt.plot(predict_svr, 'tan', label='SVR', linewidth=1)
	# plt.plot(predict_bp, 'indianred', label='bp', linewidth=1)
	# plt.plot(predict_elm, 'khaki', label='elm', linewidth=1)
	# plt.plot(predict_LSTM, 'lightsteelblue', label='lstm', linewidth=1)
	# plt.plot(emd_prediction, 'seagreen', label='EMD-LSTM', linewidth=1)
	# # plt.plot(eemd_prediction, 'r', label='EEMD-LSTM', linewidth=2.5, linestyle='--', marker='^', markersize=6)
	# plt.plot(ceemdan_prediction, 'darkred', label='CEEMDAN-LSTM', linewidth=2.5, linestyle='--', marker='^',markersize=6)
	# plt.grid(True, linestyle=':', color='gray', linewidth='0.5', axis='both')
	# plt.xlabel('time(days)', fontsize=18)
	# plt.ylabel('height(mm)', fontsize=18)
	# plt.title('563')
	# plt.legend(loc='best')
	#
	# plt.show()

	plt.rc('font', family='Times New Roman')
	plt.figure(1, figsize=(15, 5))
	plt.plot(y_real , 'black', linewidth=2.5, linestyle='--', marker='.')
	plt.plot(predict_svr, 'tan', linewidth=1)
	plt.plot(predict_bp, 'indianred',linewidth=1)
	plt.plot(predict_elm, 'khaki', linewidth=1)
	plt.plot(predict_LSTM, 'lightsteelblue',linewidth=1)
	plt.plot(emd_prediction, 'seagreen', linewidth=1)
	# plt.plot(eemd_prediction, 'r', linewidth=2.5, linestyle='--', marker='^', markersize=2)
	plt.plot(ceemdan_prediction, 'red', linewidth=2.5, linestyle='--', marker='^',markersize=6)
	plt.legend(loc='best')

	plt.show()


if __name__ == '__main__':

	main()
