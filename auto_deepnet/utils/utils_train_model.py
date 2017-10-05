import datetime
import pdb
import random

from keras.models import load_model, Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard, TerminateOnNaN
import numpy as np
from sklearn.model_selection import train_test_split

def np_train_test_split_inplace(X, y, test_size=0.1):

    np.random.shuffle(X)

    split_idx = int(test_size * X.shape[0])
    X_test = X[:split_idx]
    X_train = X[split_idx:]
    y_test = y[:split_idx]
    y_train = y[split_idx:]

    return X_train, X_test, y_train, y_test


def train_deep_learning_model(model, X, y, X_test=None, y_test=None, batch_size=32, epochs=200, verbose=1, shuffle=True):


    print('Preparing to train one deep learning model')
    if X_test is None:

        # GPUs get pretty messy with memory management stuff.
        # make our X_test only 1000 rows max
        if X.shape[0] > 10000:
            test_size = float(1000.00 / X.shape[0])
        else:
            test_size = 0.1
        X_train, X_test, y_train, y_test = np_train_test_split_inplace(X, y, test_size=test_size)
        print('X_train.shape')
        print(X_train.shape)
        print('X_test.shape')
        print(X_test.shape)

    else:
        X_train = X
        y_train = y

    leftover = X_train.shape[0] % batch_size
    X_train = X_train[leftover:]
    y_train = y_train[leftover:]

    leftover = X_test.shape[0] % batch_size
    X_test = X_test[leftover:]
    y_test = y_test[leftover:]

    print('Successfully created test data to avoid overfitting while training')
    print('Shape of training data:')
    print(X_train.shape)
    print('Shape of test data:')
    print(X_test.shape)

    # early_stopping = EarlyStopping(monitor='loss', min_delta=0, patience=25, verbose=verbose)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=25, verbose=verbose)

    now_time = datetime.datetime.now()
    time_string = str(now_time.year) + '_' + str(now_time.month) + '_' + str(now_time.day) + '_' + str(now_time.hour) + '_' + str(now_time.minute)
    temp_file_name = '_tmp_dl_model_checkpoint_' + time_string + str(random.random()) + '.h5'
    model_checkpoint = ModelCheckpoint(temp_file_name, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True, mode='auto', period=1)
    # model_checkpoint = ModelCheckpoint(temp_file_name, monitor='loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)

    # TODO: look at increasing the learning rate, and keeping each local minima as a snapshot, per
    # https://openreview.net/forum?id=BJYwwY9ll
    # https://arxiv.org/abs/1704.00109
    # reduce_lr_plateau = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=10, verbose=verbose, cooldown=2, min_lr=0.001)
    reduce_lr_plateau = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, verbose=verbose, cooldown=2, min_lr=0.0001)

    terminate_on_nan = TerminateOnNaN()
    callbacks = [early_stopping, reduce_lr_plateau, terminate_on_nan, model_checkpoint]


    print('Crated all callbacks. About to train the model now.')

    model_config = model.get_config()
    try:
        hist = model.fit(X_train, y_train, validation_data=(X_test, y_test), callbacks=callbacks, batch_size=batch_size, epochs=epochs, verbose=verbose, shuffle=shuffle)
    except KeyboardInterrupt as e:
        print('Caught the following exception')
        print(e)

        print('Now trying to load the best model checkpoint, and returning that, so you have the best of your progress')
        new_model = Sequential.from_config(model_config)
        new_model.load_weights(temp_file_name, by_name=False)
        model = new_model
        hist = None

    except RuntimeError as e:
        print('Caught the following exception')
        print(e)

        print('Now trying to load the best model checkpoint, and returning that, so you have the best of your progress')

        new_model = Sequential.from_config(model_config)
        new_model.load_weights(temp_file_name, by_name=False)
        model = new_model
        hist = None

    print('type(model)')
    print(type(model))
    print('model')
    print(model)

    return model, hist




