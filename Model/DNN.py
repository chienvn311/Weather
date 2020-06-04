from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
import pandas as pds
import numpy as np
from matplotlib import pyplot


weather_data = pds.read_csv("/Users/chienvn/PycharmProjects/Weather/Files/weather_exclude.csv", header=0)
weather_data = pds.get_dummies(weather_data.drop('Date', axis=1))
weather_data = weather_data.drop('Rain_today_No', axis=1)
weather_data = weather_data.drop('Rain_Tomorrow_No', axis=1)
weather_data = weather_data.drop('Rainfall_Tomorrow', axis=1)

dependence = np.array(weather_data['Rain_Tomorrow_Yes'])
weather_data = weather_data.drop('Rain_Tomorrow_Yes', axis=1)
independence = np.array(weather_data)
train, test, train_lb, test_lb = train_test_split(independence, dependence, test_size=0.25, random_state=42)


model = Sequential()
model.add(Dense(128, activation='relu', kernel_initializer='glorot_uniform'))
model.add(Dense(50, activation='relu', kernel_initializer='glorot_uniform'))
model.add(Dense(1, activation='sigmoid'))
opt = SGD(lr=0.01, momentum=0.9)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
history = model.fit(train, train_lb, epochs=200, validation_data=(test, test_lb), verbose=0)
# history = model.fit(train, train_lb, epochs=200, verbose=0)

_, train_acc = model.evaluate(train, train_lb, verbose=0)
_, test_acc = model.evaluate(test, test_lb, verbose=0)
print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
# plot loss during training
pyplot.subplot(211)
pyplot.title('Loss')
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
# plot accuracy during training
pyplot.subplot(212)
pyplot.title('Accuracy')
pyplot.plot(history.history['accuracy'], label='train')
pyplot.plot(history.history['val_accuracy'], label='test')
pyplot.legend()
pyplot.show()


prediction = model.predict(test)
