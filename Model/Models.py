from sklearn import preprocessing
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Flatten
from keras.callbacks import ModelCheckpoint
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Learning:
    def __init__(self, train, test, train_lb):
        self.train = train
        self.test = test
        self.train_lb = train_lb

    def train_model(self):
        # linear_model = self.linear()
        # logistic_model = self.linear()
        logistic_model = self.logistic()
        knn_model = self.knn()
        svm_model = self.svm()
        rf_model = self.random_forest(estimators=1000, state=42)
        # linear_prediction = linear_model.predict(self.test)
        logistic_prediction = logistic_model.predict(self.test)
        knn_prediction = knn_model.predict(self.test)
        svm_prediction = svm_model.predict(self.test)
        rf_prediction = rf_model.predict(self.test)

        # new = pd.DataFrame(self.test)
        # new_test = new[[5, 7, 10, 12, 15, 16]].to_numpy()
        # rf_prediction = rf_model.predict()
        return logistic_prediction, knn_prediction, svm_prediction, rf_prediction

    def train_single(self):
        return self.extreme_boosting().predict(self.test)

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
        model = SVR(kernel='linear')
        model.fit(self.train, self.train_lb)
        return model

    def random_forest(self, estimators, state):
        model = RandomForestRegressor(n_estimators=estimators, random_state=state)
        model.fit(self.train, self.train_lb)
        feat_importances = pd.Series(model.feature_importances_)
        feat_importances.nlargest(4).plot(kind='barh')
        return model

    def extreme_boosting(self):
        xgb.DMatrix(data=self.train, label=self.train_lb)
        gbm_param_grid = {
            'colsample_bytree': np.linspace(0.5, 0.9, 5),
            'n_estimators': [100, 200],
            'max_depth': [10, 15, 20, 25]
        }
        gbm = xgb.XGBRegressor()
        grid_mse = GridSearchCV(estimator=gbm, param_grid=gbm_param_grid, scoring='neg_mean_squared_error', cv=5,
                                verbose=1)
        grid_mse.fit(self.train, self.train_lb)
        return grid_mse

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
        model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
        history = model.fit(self.train, self.train_lb, epochs=50, batch_size=32, validation_split=0.2, callbacks=callback)
        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='test')
        plt.legend()
        plt.show()
        return model

