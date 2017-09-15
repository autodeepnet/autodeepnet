from math import exp, log
import os

from auto_ml import Predictor
from auto_ml.utils_models import load_ml_model
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class RNNTimeSeriesPredictor(object):


    def __init__(self,
                 column_descriptions,
                 type_of_estimator='classifier',
                 output_column_name=None,
                 lookback=1000,
                 batch_size=512,
                 epochs=50,
                 prediction_batch_size=16,
                 verbose=True
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
        self.prediction_batch_size = prediction_batch_size
        self.verbose=True
        self.prediction_big_batch_size = 512
        self.prediction_small_batch_size = 16

        self.column_descriptions[str(self.output_column_name)] = 'output'


    def _pandas_to_matrix(self, X, y):
        # Use auto_ml (http://auto.ml) for it's feature transformation.
        # This transforms a pandas DataFrame into a scipy.sparse matrix,
        # handling a bunch of the one-hot-encoding and missing values and such along the way
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

        transformed_X = self.aml_transformation_pipeline.transform_only(X)
        transformed_X = transformed_X.todense()

        return transformed_X


    def fit(self, X, y):
        X_train = X.copy()
        y_train = y.copy()

        # if np.ptp(y) > 1000:
        # scaled_y = [log(val) for val in y]
        self.y_scaler = MinMaxScaler(feature_range=(0, 1))
        y_train = np.array(y_train)
        y_train = y_train.reshape(-1, 1)
        scaled_y = self.y_scaler.fit_transform(y_train)

        transformed_X = self._pandas_to_matrix(X, scaled_y)

        # Keras expects input in the format of (num_rows, lookback_window, num_cols)
        # Right now, we have it in the format (num_rows, num_cols)
        reformatted_X = []
        reformatted_y = []
        for i in range(self.lookback + 1, transformed_X.shape[0]):
            reformatted_X.append(transformed_X[i - self.lookback + 1: i + 1])
            reformatted_y.append(scaled_y[i])
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

        # Fit the RNN model!
        model = self.construct_model(reformatted_X)
        self.model = model
        history = self.model.fit(
                                 reformatted_X,
                                 reformatted_y,
                                 epochs=self.epochs,
                                 batch_size=self.batch_size,
                                 verbose=1,
                                 shuffle=True
                                 )

        # Keras assumes that we will have the same batch_size at training and
        # prediction time. This is not particularly useful. Frequently we will
        # want to train in batch, but predict just one at a time.
        # The following is a hack to train using batches, but predict using
        # single predictions (or whatever size the user specifies)
        self._make_prediction_models(reformatted_X)

        pyplot.plot(history.history['loss'], label='train')
        pyplot.legend()
        pyplot.savefig('training_charts.png', bbox_inches='tight')
        return self


    def _make_prediction_models(self, X):
        trained_weights = self.model.get_weights()

        new_model = self.construct_model(X, batch_size=self.prediction_big_batch_size)
        new_model.set_weights(trained_weights)
        self.trained_big_batch_model = new_model

        new_model = self.construct_model(X, batch_size=self.prediction_small_batch_size)
        new_model.set_weights(trained_weights)
        self.trained_small_batch_model = new_model

        new_model = self.construct_model(X, batch_size=1)
        new_model.set_weights(trained_weights)
        self.trained_model = new_model

    # TODO: offer to take in a custom model definition from the user, or at least model_params that makes all of the below configurable
    def construct_model(self, X, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        model = Sequential()
        model.add(LSTM(50, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), return_sequences=True))
        model.add(LSTM(50))
        model.add(Dense(1))
        model.compile(loss='mse', optimizer='adam')
        return model


    # NOTE: you must pass in at least (num_rows_to_predict + lookback) rows so we can do the feature engineering for your prediction rows
    # We assume the input here is in ascending order, and that you want predictions on the most recent num_rows_to_predict. That is to say, the rows with the highest index location.
    def predict(self, X, num_rows_to_predict=1):
        # TODO: we could dynamically try to set the batch_size here if the number of predictions we're trying to get is large
        # TODO: certainly look into parallelizing this over time. but i think the batch predictions idea is better
        # TODO: yeah, definitely the batch stuff
        # TODO: figure out how to send as many predictions as possible through the batch predictor, then the rest through the individual predictor
        X_predict = X.copy()
        X_transformed = self.aml_transformation_pipeline.transform_only(X_predict)
        X_transformed = X_transformed.todense()

        remainder = num_rows_to_predict % self.prediction_small_batch_size
        big_remainder = num_rows_to_predict % self.prediction_big_batch_size
        small_remainder = big_remainder % self.prediction_small_batch_size

        remainder_idx = X.shape[0] - remainder - 1

        reformatted_X_batch = []
        reformatted_X_individuals = []
        for i in range(X_transformed.shape[0] - num_rows_to_predict, X_transformed.shape[0]):
            pred_window = X_transformed[i - self.lookback + 1: i + 1]
            if i > remainder_idx:
                reformatted_X_individuals.append([pred_window])
            else:
                reformatted_X_batch.append(pred_window)
        reformatted_X_batch = np.array(reformatted_X_batch)
        reformatted_X_individuals = np.array(reformatted_X_individuals)

        batch_predictions = self.trained_small_batch_model.predict(reformatted_X_batch, batch_size=self.prediction_batch_size)

        individual_predictions = []
        for row in reformatted_X_individuals:
            pred = self.trained_model.predict(row, batch_size=1)
            individual_predictions.append(pred[0])

        raw_predictions = np.vstack((batch_predictions, individual_predictions))

        print(raw_predictions[:100])

        rescaled_predictions = self.y_scaler.inverse_transform(raw_predictions)
        # rescaled_predictions = [exp(val) for val in raw_predictions]
        print(rescaled_predictions[:100])
        # cleaned_predictions = [pred[0] for pred in rescaled_predictions]

        return rescaled_predictions






# 1. take in properties to set on self that takes in most of hte params (sklearn api)
# 2. have a fit property (sklearn api) that takes in very few params, along with both X and y inputs
# 3. clean data
    # by default, change to be pct_change_over_prev
    # future: utility function to enable different look_forward distances
# 4. transform data from pandas dataframe to 2d matrix using auto_ml's transform_only method
# 5. reshape into being a 3d matrix
# 6. train the predictor! for now, just a single keras predictor- no hyperparameter search yet
# 7. have predict and predict_proba methods
# later- figure out how to save this


