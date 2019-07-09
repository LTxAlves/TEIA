import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from math import sqrt
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from datetime import datetime
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

# load data
def LeCSV(arq):
    dataset = pd.read_csv(arq,  parse_dates = ['mdct'], index_col=0, usecols=['mdct', 'temp', 'hmdy', 'wdsp'])
    return dataset

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = pd.DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = pd.concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

csv = LeCSV("data/SP.csv")
print(csv.shape)
values = csv.values

# specify columns to plot
groups = [0, 1, 2]
i = 1
# plot each column
plt.figure()
for group in groups:
	plt.subplot(len(groups), 1, i)
	plt.plot(values[:, group])
	plt.title(csv.columns[group], y=0.5, loc='right')
	i += 1
plt.show()

valores = csv.to_numpy()

encoder = preprocessing.LabelEncoder()
valores[:,1] = encoder.fit_transform(valores[:,1])

# ensure all data is float
valores = valores.astype('float32')
# normalize features
scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(valores)
# frame as supervised learning
n_hours = 1
n_features = 3
reframed = series_to_supervised(scaled, n_hours, 1)
#Line for removing one of the last columns.
#We shall have only one of the 3 last columns, so we predict only one of the attributes
#reframed.drop(reframed.columns[[3,4]], axis=1, inplace=True)
print(reframed.shape)
print(reframed.head(5))

# split into train and test sets
values = reframed.to_numpy()
n_train_hours = 365 * 24 * 2
train = values[:n_train_hours, :]
test = values[n_train_hours:, :]

#split into input and outputs
n_obs = n_hours * n_features
train_x, train_y = values[:, :n_obs], values[:, -n_features]
val_x, val_y = values[:, :n_obs], values[:, -n_features]
print(train_x.shape, len(train_x), train_y.shape)

train_x = train_x.reshape((train_x.shape[0], n_hours, n_features))
val_x = val_x.reshape((val_x.shape[0], n_hours, n_features))
print(train_x.shape, train_y.shape, val_x.shape, val_y.shape)

# design network
model = Sequential()
model.add(LSTM(10, input_shape=(train_x.shape[1], train_x.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam', metrics=['acc'])

model.summary()
# fit network
history = model.fit(train_x, train_y, epochs=10, batch_size=588, validation_data=(val_x, val_y), verbose=2, shuffle=False)
# plot history
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.title('Erro Aritmético Médio')
plt.legend()
plt.show()

# make a prediction
yhat = model.predict(val_x)
print(yhat.shape)
print(val_y.shape)
val_x = val_x.reshape((val_x.shape[0], n_hours*n_features))
# invert scaling for forecast
inv_yhat = np.concatenate((yhat, val_x[:, (1-n_features):]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]
print(inv_yhat.shape)
# invert scaling for actual
val_y = val_y.reshape((len(val_y), 1))
inv_y = np.concatenate((val_y, val_x[:, (1-n_features):]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]
print(inv_y.shape)
# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)

plt.plot(inv_yhat[-100:], label='Previsão')
plt.plot(inv_y[-100:], label='Atual')
plt.title('Previsão')
plt.legend()
plt.show()