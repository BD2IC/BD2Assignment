import sys

import pandas as pd
import numpy as np
import sklearn as skl
import tensorflow as tf
import scikeras as sck
from tqdm import tqdm

from sklearn.model_selection import PredefinedSplit
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.regularizers import l1, l2, l1_l2
from scikeras.wrappers import KerasRegressor
from tensorflow.keras.callbacks import EarlyStopping
from collections import defaultdict

#Import the data
return_data = pd.read_pickle('E:\秘密基地\看一听\硕士研究生相关\Electives\Big Data in Finance II\BDF2\Data/returns_chars_panel.pkl')
return_data['date'] = pd.to_datetime(return_data['date'])
print(return_data)

macro_data = pd.read_pickle('E:\秘密基地\看一听\硕士研究生相关\Electives\Big Data in Finance II\BDF2\Data\macro_timeseries.pkl')
macro_data['date'] = pd.to_datetime(macro_data['date'])
print(macro_data)

original_data = pd.merge(return_data, macro_data, how='inner', on='date')
print(original_data)

#Split a small portion of data for experiments
exp_size = 0.00001
n_exp = int(original_data.shape[0] * exp_size)
exp_data = original_data.iloc[:n_exp, ]
X_exp = exp_data.drop(['date', 'permno', 'excess_ret'], axis=1)
y_exp = exp_data['excess_ret']
print(X_exp)
print(y_exp)
X = X_exp.values
y = y_exp.values

#Split the data manually (keep the data sequence in time-series)
train_size = 0.8
test_size = 1 - train_size

n_observations = X.shape[0]
n_train = int(n_observations * train_size)
X_train, X_test = X[:n_train, ], X[n_train:, ]
y_train, y_test = y[:n_train, ], y[n_train:, ]

#Standardise the data
standard_scaler = skl.preprocessing.StandardScaler()
standard_scaler.fit(X_train)
X_train = standard_scaler.transform(X_train)
X_test = standard_scaler.transform(X_test)

'''Define a hyperparameter space for neural networks with 3 hidden layers'''
'''complex_param_grid_nn_3 = {
                 'model__optimizer': ['adam', 'sgd', 'rmsprop', 'adagrad'],
                 'model__learning_rate': [0.1, 0.01, 0.001],
                 'model__activation_func': ['relu', 'tanh', 'sigmoid', 'elu'],
                 #'batch_size': [32, 64, 128, 256], #Don't know how to grid search this
                 #'epochs': [10, 20, 50], #Don't know how to grid search this
                 'model__neurons_n': [[64, 32, 16], [128, 64, 32], [256, 128, 64]],
                 'model__dropout_rate': [0.0, 0.2, 0.5],
                 'model__regularize_terms': [None,
                                    l1(0.1), l1(0.01), l1(0.001),
                                    l2(0.1), l2(0.01), l2(0.001),
                                    l1_l2(l1=0.1, l2=0.1), l1_l2(l1=0.01, l2=0.01),
                                    l1_l2(l1=0.001, l2=0.001)]
                 #'loss_func': ['mse', 'mae', 'mape', 'msle', tf.keras.losses.Huber()], #Should be the same as the 'scoring' input of GridSearchCV
                 #'metrics_func': [['mae'], ['mse'], ['mape'], ['msle'], [rmse_metric]] #Meaningless hyperparameter: Do not influence model performance
                 #'callbacks': [[early_stopping]] #Dangerous
                 }'''

simple_param_grid_nn_3 = {
                 'model__optimizer': ['adam', 'sgd'],
                 'model__learning_rate': [0.01, 0.001],
                 'model__activation_func': ['relu', 'sigmoid'],
                 'model__neurons_n': [[64, 32, 16], [128, 64, 32]],
                 'model__dropout_rate': [0.0, 0.5],
                 'model__regularize_terms': [None, l1(0.01), l2(0.01)]
                 }

#Neural Network 3 Function
def neural_net_3(input_shape, optimizer, learning_rate, activation_func, neurons_n, dropout_rate, regularize_terms, loss_func='mse', metrics_func=['mae']):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Dense(neurons_n[0], activation=activation_func, kernel_regularizer=regularize_terms))
    if dropout_rate > 0:
        model.add(Dropout(dropout_rate))
    model.add(Dense(neurons_n[1], activation=activation_func, kernel_regularizer=regularize_terms))
    if dropout_rate > 0:
        model.add(Dropout(dropout_rate))
    model.add(Dense(neurons_n[2], activation=activation_func, kernel_regularizer=regularize_terms))
    if dropout_rate > 0:
        model.add(Dropout(dropout_rate))
    model.add(Dense(1))
    optimizer_instance = tf.keras.optimizers.get(optimizer)
    optimizer_instance.learning_rate = learning_rate
    model.compile(optimizer=optimizer_instance, loss=loss_func, metrics=metrics_func)

    return model


#Index-generating function for rolling-window time-series cross validation
each_window_size = int(0.2 * X_train.shape[0])
n_train_window = int(0.75 * each_window_size)
n_test_window = each_window_size - n_train_window

def rolling_window_index_generator(X_train, train_window_size=n_train_window, test_window_size=n_test_window, step_size=1):
    n_samples = X_train.shape[0]
    indices = np.arange(n_samples)
    for start in range(0, n_samples - train_window_size - test_window_size + 1, step_size):
        train_end = start + train_window_size
        test_end = train_end + test_window_size

        if test_end <= n_samples:
            train_indices = indices[start:train_end]
            test_indices = indices[train_end:test_end]
            yield train_indices, test_indices


#Implement rolling-window time-series cross validation for hyperparameter grid search
val_scores_dict = defaultdict(list)

#Get a function to transfer the list types in params.items() to tuple types
#for further use when we use these tuples as the keys of val_scores_dict
def make_hashable(params):
    hashable_params = {}
    for key, value in params.items():
        if isinstance(value, list):
            hashable_params[key] = tuple(value)
        else:
            hashable_params[key] = value
    return hashable_params

for train_indices, test_indices in tqdm(rolling_window_index_generator(X_train=X_train, step_size=n_train_window), file=sys.stdout):
    X_train_cv, y_train_cv = X_train[train_indices], y_train[train_indices]
    X_val_cv, y_val_cv = X_train[test_indices], y_train[test_indices]

    current_fold = np.zeros(X_train_cv.shape[0] + X_val_cv.shape[0])
    current_fold[:X_train_cv.shape[0]] = -1 #-1 indicates training set
    current_fold[X_train_cv.shape[0]:] = 0 #0 indicates validation set
    ps = skl.model_selection.PredefinedSplit(current_fold)

    X_combined = np.vstack((X_train_cv, X_val_cv))
    y_combined = np.concatenate((y_train_cv, y_val_cv))

    NN_3 = KerasRegressor(model=neural_net_3, input_shape=(X_combined.shape[1],), verbose=2)
    #print("Available parameters for NN_3: ", NN_3.get_params().keys())
    grid = skl.model_selection.GridSearchCV(estimator=NN_3, param_grid=simple_param_grid_nn_3, scoring='neg_mean_squared_error', cv=ps, error_score='raise')
    grid.fit(X_combined, y_combined)

    #Record the best hyperparameters on average by validation scores across all validation observations (e.g., 1 validation set) within 1 rolling window
    for params, mean_score in zip(grid.cv_results_['params'], grid.cv_results_['mean_test_score']):
        hashable_params = make_hashable(params)
        val_scores_dict[tuple(hashable_params.items())].append(mean_score)

#Get the best hyperparamters on average by validation scores across all rolling windows (e.g., all validation sets)
mean_val_scores = {params: np.mean(scores) for params,scores in val_scores_dict.items()}
best_params = min(mean_val_scores, key=mean_val_scores.get)
optimised_params = dict(best_params)
print(f'Optimised hyperparameters: {optimised_params} & Average validation score: {mean_val_scores[best_params]}')


#Train models with optimised hyperparameters
NN_3 = neural_net_3(input_shape=(X_train.shape[1],),
                    optimizer=optimised_params['model__optimizer'],
                    learning_rate=optimised_params['model__learning_rate'],
                    activation_func=optimised_params['model__activation_func'],
                    dropout_rate=optimised_params['model__dropout_rate'],
                    neurons_n=optimised_params['model__neurons_n'],
                    regularize_terms=optimised_params['model__regularize_terms'])

NN3_history = NN_3.fit(X_train, y_train, epochs=10, batch_size=64, validation_split=0.2, shuffle=False, verbose=2)

y_pred_nn3 = NN_3.predict(X_test)
test_results_nn3 = NN_3.evaluate(X_test, y_test, verbose=0)
print(f'MSE on test set: {test_results_nn3[0]}')
print(f'MAE on test set: {test_results_nn3[1]}')

