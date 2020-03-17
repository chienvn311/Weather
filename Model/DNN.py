from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import numpy as np
import getData


weather_data = getData.get_data()


dependence = np.array(weather_data['Rainfall'])
weatherData = weather_data.drop('Rainfall', axis=1)
independence = np.array(weatherData)
train, test, train_lb, test_lb = train_test_split(independence, dependence, test_size=0.25, random_state=42)

# print(train.shape)
# print(type(train))

NN_model = Sequential()
NN_model.add(Dense(128, kernel_initializer='normal', input_dim=train.shape[1], activation='relu'))


NN_model.add(Dense(256, kernel_initializer='normal', activation='relu'))
NN_model.add(Dense(256, kernel_initializer='normal', activation='relu'))
NN_model.add(Dense(256, kernel_initializer='normal', activation='relu'))


NN_model.add(Dense(1, kernel_initializer='normal', activation='linear'))
NN_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
NN_model.summary()


cp_name = 'Weights-{epoch:03d}--{val_loss:.5f}.hdf5'
check_point = ModelCheckpoint(cp_name, monitor='var_loss', verbose=1, save_best_only=True, mode='auto')
callback = [check_point]


NN_model.fit(train, train_lb, epochs=500, batch_size=32, validation_split=0.2, callbacks=callback)
NN_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])


prediction = NN_model.predict(test)
print(prediction)
print(test_lb)
mase = mean_absolute_error(test_lb, prediction)
print(mase)