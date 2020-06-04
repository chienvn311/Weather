from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Flatten
import numpy as np
import pandas as pds
import matplotlib.pyplot as plt


weather_data = pds.read_csv("/Users/chienvn/PycharmProjects/Weather/Files/LSTM2.csv", header=0)
dataset = weather_data.values

encoder = LabelEncoder()
# Need encode wind direction
dataset[:, 4] = encoder.fit_transform(dataset[:, 4])
dataset[:, 10] = encoder.fit_transform(dataset[:, 10])
dataset[:, 16] = encoder.fit_transform(dataset[:, 16])
dataset = dataset.astype('float32')
scale = MinMaxScaler(feature_range=(0, 1))
scaled_dataset = scale.fit_transform(dataset)

n_hours = 2
n_features = 20
n_vars = 1 if type(scaled_dataset) is list else scaled_dataset.shape[1]
df = pds.DataFrame(scaled_dataset)
cols, names = list(), list()
for i in range(n_hours, 0, -1):
    cols.append(df.shift(i))
    names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
for i in range(0, 1):
    cols.append(df.shift(-i))
    if i == 0:
        names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
    else:
        names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
reframed_dataset = pds.concat(cols, axis=1)
reframed_dataset.columns = names
reframed_dataset.dropna(inplace=True)
values = reframed_dataset.values
# def to_superviser(data, n_in=1, n_out=1, dropnan=True):
#     n_vars = 1 if type(data) is list else data.shape[1]
#     df = pds.DataFrame(data)
#     cols, names = list(), list()
#     # input sequence (t-n, ... t-1)
#     for i in range(n_in, 0, -1):
#         cols.append(df.shift(i))
#         names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
#     # forecast sequence (t, t+1, ... t+n)
#     for i in range(0, n_out):
#         cols.append(df.shift(-i))
#         if i == 0:
#             names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
#         else:
#             names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
#     agg = pds.concat(cols, axis=1)
#     agg.columns = names
#     # drop rows with NaN values
#     if dropnan:
#         agg.dropna(inplace=True)
#     return agg
#
# values = to_superviser(scaled_dataset, n_hours, 1)

number_data = 350
train = values[:number_data, :]
test = values[number_data:, :]
n_obs = n_hours * n_features
train_X, train_y = train[:, :n_obs], train[:, -n_features]
test_X, test_y = test[:, :n_obs], test[:, -n_features]
train_X = train_X.reshape((train_X.shape[0], n_hours, n_features))
test_X = test_X.reshape((test_X.shape[0], n_hours, n_features))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
model = Sequential()
model.add(LSTM(200, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
history = model.fit(train_X, train_y, epochs=200, batch_size=72, validation_data=(test_X, test_y), verbose=2, shuffle=False)

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()

yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], n_hours*n_features))
inv_yhat = np.concatenate((yhat, test_X[:, -19:]), axis=1)
inv_yhat = scale.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:, 0]
test_y = test_y.reshape((len(test_y), 1))
inv_y = np.concatenate((test_y, test_X[:, -19:]), axis=1)
inv_y = scale.inverse_transform(inv_y)
inv_y = inv_y[:, 0]
rmse = np.sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)


