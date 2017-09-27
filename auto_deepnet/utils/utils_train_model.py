import datetime
import random

from keras.models import load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard, TerminateOnNaN
from sklearn.model_selection import train_test_split


def train_deep_learning_model(model, X, y, X_test=None, y_test=None, batch_size=32, epochs=200, verbose=1, shuffle=True):

    if X_test is None:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
        leftover = X_test.shape[0] % batch_size
        X_test = X_test[leftover:]
        y_test = y_test[leftover:]
    else:
        X_train = X
        y_train = y

    leftover = X_train.shape[0] % batch_size
    X_train = X_train[leftover:]
    y_train = y_train[leftover:]

    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=25, verbose=verbose)

    now_time = datetime.datetime.now()
    time_string = str(now_time.year) + '_' + str(now_time.month) + '_' + str(now_time.day) + '_' + str(now_time.hour) + '_' + str(now_time.minute)
    temp_file_name = '_tmp_dl_model_checkpoint_' + time_string + str(random.random()) + '.h5'
    model_checkpoint = ModelCheckpoint(temp_file_name, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)

    # TODO: look at increasing the learning rate, and keeping each local minima as a snapshot, per
    # https://openreview.net/forum?id=BJYwwY9ll
    # https://arxiv.org/abs/1704.00109
    reduce_lr_plateau = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, verbose=verbose, cooldown=2, min_lr=0.001)

    # Might have to pass in batch_size here
    tensor_board = TensorBoard()

    terminate_on_nan = TerminateOnNaN()

    callbacks = [early_stopping, model_checkpoint, reduce_lr_plateau, tensor_board, terminate_on_nan]

    try:
        hist = model.fit(X_train, y_train, validation_data=(X_test, y_test), callbacks=callbacks, batch_size=batch_size, epochs=epochs, verbose=verbose, shuffle=shuffle)
    except KeyboardInterrupt as e:
        print('Caught the following exception')
        print(e)

        print('Now trying to load the best model checkpoint, and returning that, so you have the best of your progress')

        model = load_model(temp_file_name)
        hist = None

    return model, hist




