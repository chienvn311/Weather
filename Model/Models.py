from sklearn import preprocessing
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.callbacks import ModelCheckpoint

class Learning:
    def __init__(self, train, test, train_lb):
        self.train = train
        self.test = test
        self.train_lb = train_lb

    def train_model(self):
        # linear_model = self.linear()
        logistic_model = self.logistic()
        knn_model = self.knn()
        svm_model = self.svm()
        rf_model = self.random_forest(estimators=1000, state=42)
        # linear_prediction = linear_model.predict(self.test)
        logistic_prediction = logistic_model.predict(self.test)
        knn_prediction = knn_model.predict(self.test)
        svm_prediction = svm_model.predict(self.test)
        rf_prediction = rf_model.predict(self.test)
        return logistic_prediction, knn_prediction, svm_prediction, rf_prediction

    def train_nn(self):
        return self.deep_neural_network().predict(self.test)

    def linear(self):
        model = LinearRegression()
        model.fit(self.train, self.train_lb)
        return model

    def logistic(self):
        model = LogisticRegression(solver='liblinear', random_state=0)
        model.fit(self.train, self.train_lb)
        return model

    def knn(self):
        model = KNeighborsRegressor(5)
        model.fit(self.train, self.train_lb)
        return model

    def svm(self):
        model = SVR(kernel='rbf')
        model.fit(self.train, self.train_lb)
        return model

    def random_forest(self, estimators, state):
        model = RandomForestRegressor(n_estimators=estimators, random_state=state)
        model.fit(self.train, self.train_lb)
        return model

    def deep_neural_network(self):
        model = Sequential()
        model.add(Dense(128, kernel_initializer='normal', input_dim=self.train.shape[1], activation='relu'))
        model.add(Dense(256, kernel_initializer='normal', activation='relu'))
        model.add(Dense(256, kernel_initializer='normal', activation='relu'))
        model.add(Dense(256, kernel_initializer='normal', activation='relu'))
        model.add(Dense(1, kernel_initializer='normal', activation='linear'))
        model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
        cp_name = 'Weights-{epoch:03d}--{val_loss:.5f}.hdf5'
        check_point = ModelCheckpoint(cp_name, monitor='var_loss', verbose=1, save_best_only=True, mode='auto')
        callback = [check_point]
        model.fit(self.train, self.train_lb, epochs=500, batch_size=32, validation_split=0.2, callbacks=callback)
        model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
        return model

