import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn import metrics

pd.set_option('display.max_columns', 15)
pd.set_option('display.width', 1000)


def encode_data(encoder, data, features):
    return data[features].apply(encoder.fit_transform)


def scale_data(scaler, X):
    return scaler.fit_transform(X)


def print_metrics(actual, pred):
    print('Mean Absolute Error:', round(metrics.mean_absolute_error(actual, pred), 2))
    print('Mean Squared Error:', round(metrics.mean_squared_error(actual, pred), 2))
    print('Root Mean Squared Error:', round(np.sqrt(metrics.mean_squared_error(actual, pred)), 2))


def print_model_score(model, X, y):
    print('Model score: ', model.score(X, y))


def print_prediction_accuracy(test_or_train, actual, pred, acc_range):
    # Find predictions within x% of the true value for given data set.
    pred_error = abs(actual - pred)/actual*100
    print('Predictions within given error for {} dataset:'.format(test_or_train))
    print('{:<8} {:<10}'.format('Error', 'Proportion'))
    for mismatch in list(range(acc_range[0], acc_range[1]+1, 5)):
        n_pred_accurate = sum(x < mismatch for x in pred_error)
        print('{:>3}%: {:>10.2f}%'.format(mismatch, n_pred_accurate/len(actual)*100))


def print_feature_importance(model, feature_list):
    # Check feature importance using the built in function.
    try:
        importances = list(model.feature_importances_)
    except AttributeError:
        print('No feature importance included in this model.')
    else:
        feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
        feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)
        [print('Feature: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];


def plot_actual_vs_predicted(actual, pred, filename=None):
    plt.hexbin(actual, pred, gridsize=100, bins='log', cmap=plt.cm.Greens)
    # If the predictions are 100% accurate they should all map to the line x=y,
    # so we plot this line for visual reference.
    plt.plot(actual, actual, color='red')
    plt.colorbar(label='log10(N)')
    plt.clim(1, 1e3)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Actual vs Predicted values')
    plt.grid()
    plt.axis([actual.min(), actual.max(), actual.min(), actual.max()])
    if filename:
        plt.savefig(filename)
    else:
        plt.show()
    plt.cla()


if __name__ == '__main__':
    # The input data here must already be cleaned and sanitized.
    data = pd.read_csv('TrainAndValid_clean.csv')
    print(data.head(10))

    # List of categorical features.
    features_cat = ['ageAtSaletime', 'YearMade', 'saledate', 'fiModelDesc', 'fiProductClassDesc', 'state',
                    'Enclosure', 'ProductGroup', 'Hydraulics', 'Track_Type', 'auctioneerID', 'UsageBand',
                    'MachineHoursCurrentMeter', 'ProductSize', 'MachineID',
                    'Drive_System', 'Transmission']
    # List of continuous features that don't need encoding.
    features_cont = ['ModelID', 'ageAtSaletimeInMonths', 'ageAtSaletimeInYears']
    # List of labels.
    label_list = 'SalePrice'
    # Encode the categorical features using LabelEncoder().
    le = LabelEncoder()
    data_input = encode_data(le, data, features_cat)
    # Combine all features with the labels.
    data_input = pd.concat([data[label_list], data[features_cont], data_input], axis=1)

    # Check correlation between features.
    #corr_list = ['SalePrice', 'YearMade', 'saledate']
    sns.heatmap(data_input.corr(), annot=True, cmap=plt.cm.Reds)
    plt.show()

    # Remove features that have been found to not contribute significantly to the model,
    # or that are redundant due to other features. Determined by trial and error and feature importance metrics.
    droppable_list = ['Hydraulics', 'auctioneerID', 'state', 'Track_Type', 'MachineHoursCurrentMeter',
                      'UsageBand', 'ageAtSaletime', 'ageAtSaletimeInMonths', 'ageAtSaletimeInYears',
                      'Transmission', 'Drive_System', 'MachineID']
    data_input = data_input.drop(droppable_list, axis=1)

    # Save labels to a separate vector and drop from our feature matrix.
    y = data_input[label_list]
    X = data_input.drop([label_list], axis=1)
    X.info()

    # Split the data into a training set and a test set. Choose to train the model on 80% of the data.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12345)
    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)

    # Scale the feature set to values between 0 and 1 using the MinMaxScaler().
    mms = MinMaxScaler()
    X_train = scale_data(mms, X_train)
    X_test = scale_data(mms, X_test)

    # RandomForestRegressor()
    reg = RandomForestRegressor(n_estimators=200, verbose=1, random_state=0,
                                n_jobs=-1)

#    reg = RandomForestRegressor(n_estimators=200, verbose=1, random_state=0,
#                                max_features=5, max_depth=12, min_samples_leaf=1,
#                                min_samples_split=5,
#                                n_jobs=-1)

    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)

#    # MLPRegressor
#    reg = MLPRegressor(solver='adam', max_iter=300, activation='relu', random_state=8,
#                       learning_rate_init=0.6, alpha=1.5,
#                       hidden_layer_sizes=[4, 2], verbose=1)
#    #learning_rate_init = 0.001, batch_size = X_train.shape[0],
#    reg.fit(X_train, y_train)
#    y_pred = reg.predict(X_test)
#    print(y_pred)

    # Run a prediction on the training set so we can compare performance on the training vs test data.
    y_pred_train = reg.predict(X_train)

    # Check how well our model has done.
    print_metrics(y_test, y_pred)
    print_model_score(reg, X_test, y_test)
    print_model_score(reg, X_train, y_train)
    # Find predictions within x% of the true value for both training and test sets.
    acc_range = [5, 25]
    print_prediction_accuracy('testing', y_test, y_pred, acc_range)
    print_prediction_accuracy('training', y_train, y_pred_train, acc_range)

    # Check feature importance.
    print_feature_importance(reg, list(X.columns))

    # Plot actual vs predicted values.
    plot_actual_vs_predicted(y_test, y_pred)
    plot_actual_vs_predicted(y_train, y_pred_train)
    # Add a filename to save the figure.
    #fig_name = 'actual_vs_predicted_testing.png'
    #plot_actual_vs_predicted(y_test, y_pred, fig_name)

