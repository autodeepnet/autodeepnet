import datetime
import json
from math import exp, log
import os
import pdb
import random
import shutil
import zipfile

from auto_ml import Predictor
from auto_ml.utils_models import load_ml_model
import dill
import keras.backend as K
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, LSTM
from keras.optimizers import Adam
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot
import numpy as np
from tensorflow.python.client import timeline

from auto_deepnet.utils.utils_train_model import train_deep_learning_model

class RNNTimeSeriesPredictor(object):


    def __init__(self,
                 column_descriptions,
                 type_of_estimator='classifier',
                 output_column_name=None,
                 lookback=1000,
                 batch_size=512,
                 epochs=50,
                 verbose=True,
                 model_params=None,
                 build_fn=None
                 ):

        # Tensorflow does an overwhelming volume of super annoying logging- suppress it
        os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

        self.type_of_estimator = type_of_estimator
        self.column_descriptions = column_descriptions.copy()
        self.output_column_name = output_column_name
        self.lookback = lookback
        self.batch_size = batch_size
        self.epochs = epochs
        self.verbose=True
        self.prediction_big_batch_size = 128
        self.prediction_small_batch_size = 16
        self.aml_transformation_pipeline = None

        self.column_descriptions[str(self.output_column_name)] = 'output'

        self.model_params = {
            'learning_rate': 0.01
            , 'batch_size': 64
            , 'epochs': 200
            , 'lookback': 1000
            , 'optimizer': 'Adam'
        }

        if model_params is None:
            self.user_model_params = {}
        else:
            self.user_model_params = model_params

        if build_fn is not None:
            self.user_build_fn = build_fn


    def fit_transformation_pipeline(self, X, y):
        # Use auto_ml (http://auto.ml) for it's feature transformation.
        # This transforms a pandas DataFrame into a scipy.sparse matrix,
        # handling a bunch of the one-hot-encoding and missing values and such along the way

        if self.aml_transformation_pipeline is None:

            ml_predictor = Predictor(
                                     type_of_estimator=self.type_of_estimator,
                                     column_descriptions=self.column_descriptions
                                     )

            X[str(self.output_column_name)] = y

            if self.type_of_estimator == 'classifier':
                model_name = 'LogisticRegression'
            elif self.type_of_estimator == 'regressor':
                model_name = 'LinearRegression'
            ml_predictor.train(
                               X,
                               perform_feature_scaling=True,
                               perform_feature_selection=False,
                               model_names=model_name
                               )

            t_pipeline_name = '__transformation_pipeline.dill'
            ml_predictor.save(t_pipeline_name)
            self.aml_transformation_pipeline = load_ml_model(t_pipeline_name)
            os.remove(t_pipeline_name)

        print('Performing initial data transform using auto_ml\'s transformation pipeline now.')
        transformed_X = self.aml_transformation_pipeline.transform_only(X)
        transformed_X = transformed_X.todense()

        return transformed_X


    # Future: save number of epochs that we trained for, which we can pass in as a param for keras when refitting or warm-starting
    def refit(self, X, y, epochs=None):
        print('Now continuing to train the already fitted model')
        if epochs is None:
            epochs = self.epochs

        reformatted_X, reformatted_y = self.transform(X, y)

        # Fit the RNN model!
        model = self.model
        # TODO: eventually pass in how many epochs we've already trained for
        model, history = train_deep_learning_model(model, reformatted_X, reformatted_y, X_test=None, y_test=None, batch_size=self.batch_size, epochs=epochs, verbose=1, shuffle=True)

        print('model after training')
        print(model)
        print('type(model) after training')
        print(type(model))

        self.model = model

        # Keras assumes that we will have the same batch_size at training and
        # prediction time. This is not particularly useful. Frequently we will
        # want to train in batch, but predict just one at a time.
        # The following is a hack to train using batches, but predict using
        # single predictions (or whatever size the user specifies)
        self._make_prediction_models()

        # pyplot.plot(history.history['loss'], label='train')
        # pyplot.legend()
        # pyplot.savefig('training_charts.png', bbox_inches='tight')
        return self




    def transform(self, X, y, inplace=True):
        if inplace == False:
            X = X.copy()
            y = y.copy()

        y = np.array(y)
        y = y.astype(np.float32)

        transformed_X = self.fit_transformation_pipeline(X, y)
        transformed_X = transformed_X.astype(np.float32)
        print('Now turning that data into the lookback window format we expect')

        # Keras expects input in the format of (num_rows, lookback_window, num_cols)
        # Right now, we have it in the format (num_rows, num_cols)
        reformatted_X = []
        reformatted_y = []
        for i in range(self.lookback + 1, transformed_X.shape[0]):
            reformatted_X.append(transformed_X[i - self.lookback + 1: i + 1])
            reformatted_y.append(y[i])
        reformatted_X = np.array(reformatted_X)
        reformatted_y = np.array(reformatted_y)

        # Now we have to reshape X so that it's evenly divisible by our batch_size
        remainder = reformatted_X.shape[0] % self.batch_size
        if self.verbose:
            print('Found {} rows of training data that will not be used due to batch size '
                'constraints. '.format(remainder))
            print('This is calculated by X.shape[0] % batch_size')
            print('If it is important to use more of the training data, consider using '
                'a batch size that leaves a smaller remainder')
        reformatted_X = reformatted_X[remainder:]
        reformatted_y = reformatted_y[remainder:]

        return reformatted_X, reformatted_y


    def fit(self, X, y, inplace=True):

        reformatted_X, reformatted_y = self.transform(X, y)

        print('Successfully ran feature engineering on the input data. You will likely have noticed a large increase in memory usage during this phase')
        print('X.shape after formatting data:')
        print(reformatted_X.shape)

            # Fit the RNN model!
        self.X_shape = reformatted_X.shape
        print('Constructing the model now')
        model = self.construct_model(use_dropout=True)
        print('Model constructed. Training model now.')

        model, history = train_deep_learning_model(model, reformatted_X, reformatted_y, X_test=None, y_test=None, batch_size=self.batch_size, epochs=self.epochs, verbose=1, shuffle=True)

        self.model = model

        del reformatted_X
        del reformatted_y

        # Keras assumes that we will have the same batch_size at training and
        # prediction time. This is not particularly useful. Frequently we will
        # want to train in batch, but predict just one at a time.
        # The following is a hack to train using batches, but predict using
        # single predictions (or whatever size the user specifies)
        self._make_prediction_models()

        # pyplot.plot(history.history['loss'], label='train')
        # pyplot.legend()
        # pyplot.savefig('training_charts.png', bbox_inches='tight')
        return self


    def create_test_set(self, X, y, test_size=0.1):

        test_rows = int(test_size * X.shape[0])
        print('We are going to split off the last {} rows as test data, to make sure we do not overfit at training time.'.format(test_rows))
        print('Note that this is NOT a random split. We are using a time-based split, so that we can maintain a history for each row.')
        print('If a random split is important to you, please either pass in your own X_test and y_test, or use .fit() instead of .fit_generator()')

        split_idx = test_rows + self.lookback
        X_test = X[-split_idx:]
        y_test = y[-split_idx:]

        # Note that we are intentionally splitting on a different idx here
        # When we create our batches inside the generator, we do it from the idx point onwards (in other words, we assume the lookback window is included in the dataset itself that's passed into adn_generator, but we don't pass any of the rows in the lookback as a result of adn_generator, other than as part of the lookback)
        # so the lookback overlaps between the two- they are training rows for X_train, and lookback rows for X_test
        X = X[:-test_rows]
        y = y[:-test_rows]

        print('Size of training data:')
        print(X.shape)
        print('Size of test data:')
        print(X_test.shape)

        return X, X_test, y, y_test


    def fit_generator(self, X, y):

        y = np.array(y)
        y = y.astype(np.float32)

        transformed_X = self.fit_transformation_pipeline(X, y)
        transformed_X = transformed_X.astype(np.float32)
        self.X_shape = (transformed_X.shape[0], self.lookback, transformed_X.shape[1])
        print('Successfully ran basic data transformation over the input data. We have not yet transformed it into lookback windows (which will be done with a generator)')

        print('Constructing the model now')
        model = self.construct_model(use_dropout=True)
        print('Model constructed. Training model now.')

        print('transformed_X.shape before create_test_set')
        print(transformed_X.shape)
        transformed_X, X_test, y, y_test = self.create_test_set(transformed_X, y)

        model, history = train_deep_learning_model(model, transformed_X, y, X_test=X_test, y_test=y_test, batch_size=self.batch_size, epochs=self.epochs, verbose=1, shuffle=True, fit_generator=adn_generator, generator_lookback=self.lookback)

        self.model = model

        self._make_prediction_models()

        print('Finished training the model!')

        return self


    def _make_prediction_models(self):
        # TODO: do an isinstance check on self.model to see if it is already weights, or a trained model
        # trained_weights = self.model
        trained_weights = self.model.get_weights()

        new_model = self.construct_model(batch_size=self.prediction_big_batch_size)
        new_model.set_weights(trained_weights)
        self.trained_big_batch_model = new_model

        new_model = self.construct_model(batch_size=self.prediction_small_batch_size)
        new_model.set_weights(trained_weights)
        self.trained_small_batch_model = new_model

        new_model = self.construct_model(batch_size=1)
        new_model.set_weights(trained_weights)
        self.trained_model = new_model

    # TODO: offer to take in a custom model definition from the user, or at least model_params that makes all of the below configurable
    def construct_model(self, batch_size=None, use_dropout=False):
        if batch_size is None:
            batch_size = self.batch_size

        model = Sequential()
        model.add(LSTM(40, batch_input_shape=(batch_size, self.X_shape[1], self.X_shape[2]), return_sequences=True))
        # model.add(Dropout(0.15))
        model.add(LSTM(20, batch_input_shape=(batch_size, self.X_shape[1], self.X_shape[2]), return_sequences=False))
        model.add(Dense(1))
        model.compile(loss='mse', optimizer=Adam(lr=0.004))
        return model


    def _make_prediction_groups(self, X, num_rows_to_predict):
        remainder = num_rows_to_predict % self.prediction_small_batch_size
        big_remainder = num_rows_to_predict % self.prediction_big_batch_size
        small_remainder = big_remainder % self.prediction_small_batch_size

        big_remainder_idx = X.shape[0] - big_remainder - 1
        small_remainder_idx = X.shape[0] - small_remainder - 1
        remainder_idx = X.shape[0] - remainder - 1

        reformatted_X_big_batch = []
        reformatted_X_small_batch = []
        reformatted_X_individuals = []
        for i in range(X.shape[0] - num_rows_to_predict, X.shape[0]):
            pred_window = X[i - self.lookback + 1: i + 1]
            if i > small_remainder_idx:
                reformatted_X_individuals.append([pred_window])
            elif i > big_remainder_idx:
                reformatted_X_small_batch.append(pred_window)
            else:
                reformatted_X_big_batch.append(pred_window)

        reformatted_X_big_batch = np.array(reformatted_X_big_batch)
        reformatted_X_small_batch = np.array(reformatted_X_small_batch)
        reformatted_X_individuals = np.array(reformatted_X_individuals)

        return reformatted_X_big_batch, reformatted_X_small_batch, reformatted_X_individuals


    # NOTE: you must pass in at least (num_rows_to_predict + lookback) rows so we can do the feature engineering for your prediction rows
    # We assume the input here is in ascending order, and that you want predictions on the most recent num_rows_to_predict. That is to say, the rows with the highest index location.
    def predict(self, X, num_rows_to_predict=1):

        X_predict = X.copy()
        # TODO: sure seems like we want to use self.transform() here
        X_transformed = self.aml_transformation_pipeline.transform_only(X_predict)
        X_transformed = X_transformed.todense()

        # Try to get predictions as rapidly as possible by using the biggest batch size reasonable
        # We have 3 different predictors, all with the same weights, but with different batch_sizes
        reformatted_X_big_batch, reformatted_X_small_batch, reformatted_X_individuals = self._make_prediction_groups(X_transformed, num_rows_to_predict)


        print('shapes of groups:')
        print(reformatted_X_big_batch.shape)
        print(reformatted_X_small_batch.shape)
        print(reformatted_X_individuals.shape)

        results = []
        if reformatted_X_big_batch.shape[0] > 0:
            big_batch_predictions = self.trained_big_batch_model.predict(reformatted_X_big_batch, batch_size=self.prediction_big_batch_size)
            results.append(big_batch_predictions)
            del reformatted_X_big_batch

        if reformatted_X_small_batch.shape[0] > 0:
            small_batch_predictions = self.trained_small_batch_model.predict(reformatted_X_small_batch, batch_size=self.prediction_small_batch_size)
            results.append(small_batch_predictions)
            del reformatted_X_small_batch

        individual_predictions = []
        for row in reformatted_X_individuals:
            pred = self.trained_model.predict(row, batch_size=1)
            individual_predictions.append(pred[0])
        individual_predictions = np.array(individual_predictions)
        del reformatted_X_individuals

        if individual_predictions.shape[0] > 0:
            print('Appending individual predictions after big batch and small batch')
            results.append(individual_predictions)

        if len(results) > 1:
            raw_predictions = np.vstack(results)
        else:
            raw_predictions = results[0]


        predictions = raw_predictions
        cleaned_predictions = [pred[0] for pred in predictions]

        return cleaned_predictions


    def save(self, file_name=None):
        if file_name is None:
            now_time = datetime.datetime.now()
            time_string = str(now_time.year) + '_' + str(now_time.month) + '_' + str(now_time.day) + '_' + str(now_time.hour) + '_' + str(now_time.minute)
            file_name = 'trained_auto_deepnet_model_{}.zip'.format(time_string)

        # TODO:

        last_dot_index = file_name.rfind('.')
        if last_dot_index == -1:
            folder_name = file_name
        else:
            folder_name = file_name[:last_dot_index]
        # create a new directory
        os.mkdir(folder_name)
        # 1. write transformation pipeline to disk
        with open(os.path.join(folder_name, 'transformation_pipeline.dill') , 'wb') as open_file_name:
            dill.dump(self.aml_transformation_pipeline, open_file_name)

        # 2. write trained model to disk
            # we should just need one for it's weights, then we'll load it into the different batched predictors ourselves on load
        self.trained_model.save(os.path.join(folder_name, 'trained_model.h5'))

        config_to_save = {
            'X_shape': self.X_shape
            , 'prediction_big_batch_size': self.prediction_big_batch_size
            , 'prediction_small_batch_size': self.prediction_small_batch_size
            , 'batch_size': self.batch_size
        }

        # 3. write metadata to disk?
            # number of trained epochs is the only thing that jumps to mind right now
            # but we'll probably have stuff like hyperparameter search info over time, so hopefully you can have a hyperparameter search interrupted, and reload it again
        with open(os.path.join(folder_name, 'config_info.txt'), 'w') as open_file_name:
            json.dump(config_to_save, open_file_name)

        # 4. zip up that directory
        shutil.make_archive(file_name, 'zip', folder_name)

        # delete the directory
        shutil.rmtree(folder_name)
        # os.rmdir(folder_name)


    def _load(self, file_name):
        last_dot_index = file_name.rfind('.')
        if last_dot_index == -1:
            folder_name = file_name
        else:
            folder_name = file_name[:last_dot_index]

        with zipfile.ZipFile(file_name, 'rb') as zip_ref:
            zip_ref.extractall(folder_name)

        self.aml_transformation_pipeline = load_ml_model(os.path.join(folder_name, 'transformation_pipeline.dill'))

        self.trained_model = load_model(os.path.join(folder_name, 'trained_model.h5'))
        self.model = self.trained_model

        with open(os.path.join(folder_name, 'config_info.txt')) as json_file:
            self.saved_configs = json.load(json_file)

        self.X_shape = self.saved_configs['X_shape']
        self.prediction_big_batch_size = self.saved_configs['prediction_big_batch_size']
        self.prediction_small_batch_size = self.saved_configs['prediction_small_batch_size']
        self.batch_size = self.saved_configs['batch_size']

        os.rmdir(folder_name)

        print('Successfully loaded the transformation pipeline, and the trained model.')

        # TODO: make prediction models

        self._make_prediction_models()


def adn_generator(X, y, batch_size, lookback, verbose=True):

    remainder = (X.shape[0] - lookback - 2) % batch_size
    if verbose and remainder > 0:
        print('Found {} rows of training data that will not be used due to batch size '
            'constraints. '.format(remainder))
        print('This is calculated by X.shape[0] % batch_size')
        print('If it is important to use more of the training data, consider using '
            'a batch size that leaves a smaller remainder')
    X = X[remainder:]
    y = y[remainder:]
    print('X.shape after making sure it conforms to batch_size')
    print(X.shape)

    while True:

        batch_X = []
        batch_y = []
        batch_idx = 0
        if lookback + 1 > X.shape[0]:
            raise(ValueError('X is too small after subtracting out the lookback window'))
        # TODO: shuffle
        # first, get a list of indices
        # then, shuffle them
        # then, iterate through the shuffled list
        # Note that this does not shuffle the data itself, so our lookback windows are still accurate
        indices = range(lookback + 1, X.shape[0])
        random.shuffle(indices)
        for i in indices:

            if ((batch_idx % batch_size) == 0) and (batch_idx > 0):
                yield np.array(batch_X).astype(np.float32), np.array(batch_y).astype(np.float32)
                batch_X = []
                batch_y = []
                batch_idx = 0

            batch_X.append(X[i - lookback + 1: i + 1])
            batch_y.append(y[i])
            batch_idx += 1


def load_rnn_model(file_name):
    pass
    # TODO: instantiate a new RNNTimeSeriesPredictor
    # load up the things we need from disk
    # pass those into model._load()


