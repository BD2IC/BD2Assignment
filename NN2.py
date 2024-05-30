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

#################################################### Data Preprocessing ####################################################
# Import the data
return_data = pd.read_pickle('returns_chars_panel.pkl')
return_data['date'] = pd.to_datetime(return_data['date'])
print(return_data)

macro_data = pd.read_pickle('macro_timeseries.pkl')
macro_data['date'] = pd.to_datetime(macro_data['date'])
print(macro_data)

original_data = pd.merge(return_data, macro_data, how='inner', on='date')
print(original_data)

# Preparation for Grid Search
# Split a small portion of data for experiments
exp_size = 0.00001
n_exp = int(original_data.shape[0] * exp_size)
exp_data = original_data.iloc[:n_exp, ]
X_exp = exp_data.drop(['ret','excess_ret','rfree','permno','date'], axis=1)
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

#################################################### Neural Network model with 4 hidden layers ####################################################
# Assume that model with 2 hidden layers performs the best OOS
simple_param_grid_nn_2 = {
    'model__optimizer': ['sgd', 'adam'],
    'model__learning_rate': [ 0.01, 0.001],
    'model__activation_func': ['relu', 'sigmoid'],
    'model__neurons_n': [[64, 32], [128, 64], [256, 128]],
    'model__dropout_rate': [0.0, 0.5],
    'model__regularize_terms': [None, l1(0.01), l2(0.01)]
}

#Neural Network 2 Function
def neural_net_2(input_shape, optimizer, learning_rate, activation_func, neurons_n, dropout_rate, regularize_terms, loss_func='mse', metrics_func=['mae']):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Dense(neurons_n[0], activation=activation_func, kernel_regularizer=regularize_terms))
    if dropout_rate > 0:
        model.add(Dropout(dropout_rate))
    model.add(Dense(neurons_n[1], activation=activation_func, kernel_regularizer=regularize_terms))
    if dropout_rate > 0:
        model.add(Dropout(dropout_rate))
    model.add(Dense(1))
    optimizer_instance = tf.keras.optimizers.get(optimizer)
    optimizer_instance.learning_rate = learning_rate
    model.compile(optimizer=optimizer_instance, loss=loss_func, metrics=metrics_func)
    return model
val_scores_dict_2 = defaultdict(list)

for train_indices, test_indices in tqdm(rolling_window_index_generator(X_train=X_train, step_size=n_train_window), file=sys.stdout):
    X_train_cv, y_train_cv = X_train[train_indices], y_train[train_indices]
    X_val_cv, y_val_cv = X_train[test_indices], y_train[test_indices]

    current_fold = np.zeros(X_train_cv.shape[0] + X_val_cv.shape[0])
    current_fold[:X_train_cv.shape[0]] = -1
    current_fold[X_train_cv.shape[0]:] = 0
    ps = skl.model_selection.PredefinedSplit(current_fold)

    X_combined = np.vstack((X_train_cv, X_val_cv))
    y_combined = np.concatenate((y_train_cv, y_val_cv))

    NN_2 = KerasRegressor(model=neural_net_2, input_shape=(X_combined.shape[1],), verbose=2)
    grid_2 = skl.model_selection.GridSearchCV(estimator=NN_2, param_grid=simple_param_grid_nn_2, scoring='neg_mean_squared_error', cv=ps, error_score='raise')
    grid_2.fit(X_combined, y_combined)

    for params, mean_score in zip(grid_2.cv_results_['params'], grid_2.cv_results_['mean_test_score']):
        hashable_params = make_hashable(params)
        val_scores_dict_2[tuple(hashable_params.items())].append(mean_score)

mean_val_scores_2 = {params: np.mean(scores) for params, scores in val_scores_dict_2.items()}
best_params_2 = min(mean_val_scores_2, key=mean_val_scores_2.get)
optimised_params_2 = dict(best_params_2)
print(f'Optimised hyperparameters for 2 layers: {optimised_params_2} & Average validation score: {mean_val_scores_2[best_params_2]}')

#Train models with optimised hyperparameters
NN_2 = neural_net_2(input_shape=(X_train.shape[1],),
                    optimizer=optimised_params_2['model__optimizer'],
                    learning_rate=optimised_params_2['model__learning_rate'],
                    activation_func=optimised_params_2['model__activation_func'],
                    dropout_rate=optimised_params_2['model__dropout_rate'],
                    neurons_n=optimised_params_2['model__neurons_n'],
                    regularize_terms=optimised_params_2['model__regularize_terms'])

X_NN = original_data.drop(['ret','excess_ret','rfree','permno','date'], axis=1)
y_NN = original_data['excess_ret']
date = original_data['date']

def expanding_window_indices(start_date, end_date, step_size='1Y', val_size='4Y', test_size='1Y'):
    date_range = pd.date_range(start=start_date, end=end_date, freq=step_size)
    indices = []
    for end_val in date_range:
        start = pd.Timestamp(start_date)
        end_train = end_val - pd.DateOffset(years=int(test_size[:-1])) - pd.DateOffset(years=int(val_size[:-1]))
        if end_train < start:
            continue
        train_mask = (date >= start) & (date <= end_train)
        val_mask = (date > end_train) & (date <= (end_train + pd.DateOffset(years=int(val_size[:-1]))))
        test_mask = (date > (end_train + pd.DateOffset(years=int(val_size[:-1])))) & (date <= end_val)
        if test_mask.any():
            indices.append((train_mask, val_mask, test_mask))
    return indices

# Generate the expanding window indices
expanding_indices = expanding_window_indices('1986-02-01', '2016-12-01')

r2_oos_2 = []

for train_mask, val_mask, test_mask in tqdm(expanding_indices):
    X_train, y_train = X_NN.loc[train_mask].values, y_NN.loc[train_mask].values
    X_val, y_val = X_NN.loc[val_mask].values, y_NN.loc[val_mask].values
    X_test, y_test = X_NN.loc[test_mask].values, y_NN.loc[test_mask].values

    NN2_history = NN_2.fit(X_train, y_train, epochs=100, batch_size=10000,
                           validation_data=(X_val, y_val), verbose=0,
                           callbacks=[EarlyStopping(patience=2, restore_best_weights=True)])

    predictions = NN_2.predict(X_test)
    ss_res = np.sum((y_test - predictions.T) ** 2)
    ss_tot = np.sum((y_test) ** 2)
    r2_out_of_sample = 1 - (ss_res / ss_tot)
    r2_oos_2.append(r2_out_of_sample)

r2_oos_2_mean = np.mean(r2_oos_2)
print(r2_oos_2_mean)

