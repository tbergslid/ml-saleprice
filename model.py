import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn.metrics import r2_score
#from sklearn.feature_extraction import FeatureHasher
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.tree import export_graphviz
from sklearn import metrics
import pydot
from tensorflow import keras
from tensorflow.keras import layers, losses

pd.set_option('display.max_columns', 15)
pd.set_option('display.width', 1000)

def main():
    #data = pd.read_csv('TrainMini.csv')
    #data = pd.read_csv('TrainAndValid.csv')
    data = pd.read_csv('TrainAndValid_clean.csv')
    print(data.head(10))
    data.info()
    #print(data)

    # Encode the categorical features using LabelEncoder().
    le = LabelEncoder()
    data[['YearMade', 'saledate', 'fiModelDesc', 'fiProductClassDesc', 'state',
          'ProductGroup', 'Enclosure', 'Hydraulics', 'Track_Type']] = \
    data[['YearMade', 'saledate', 'fiModelDesc', 'fiProductClassDesc', 'state',
          'ProductGroup', 'Enclosure', 'Hydraulics', 'Track_Type']].apply(le.fit_transform)
#    data[['fiModelDesc', 'fiProductClassDesc', 'state', 'ProductGroup',
#          'Enclosure', 'Hydraulics', 'Track_Type']] = \
#        data[['fiModelDesc', 'fiProductClassDesc', 'state', 'ProductGroup',
#              'Enclosure', 'Hydraulics', 'Track_Type']].apply(le.fit_transform)
    #data[['Forks']] = data[['Forks']].apply(le.fit_transform)
    #data[['Coupler']] = data[['Coupler']].apply(le.fit_transform)

    # Extract only the years from YearMade and saledate.
    #data['modelYear'] = pd.DatetimeIndex(data['YearMade']).year
    #data['saleYear'] = pd.DatetimeIndex(data['saledate']).year

    # Remove features that have been found to not contribute significantly to the model,
    # or that are redundant due to other features.
    data = data.drop(['Track_Type', 'Hydraulics', 'auctioneerID', 'state', 'MachineID'], axis=1)
    data = data.drop(['MachineHoursCurrentMeter', 'UsageBand'], axis=1)
    #data = data.drop(['YearMade', 'saledate'], axis=1)
    #data = data.drop(['YearMade', 'saledate', 'ageAtSaletime', 'ageAtSaletimeInMonths'], axis=1)
    data = data.drop(['ageAtSaletime', 'ageAtSaletimeInMonths', 'ageAtSaletimeInYears'], axis=1)
    data.info()
    print(data.head(10))

    # Save SalePrice to a separate vector and drop it from our feature matrix.
    y = data['SalePrice']
    X = data.drop(['SalePrice'], axis=1)
    X.info()

    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9, random_state=12345)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12345)
    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)

    # Scale the feature set to values between 0 and 1.
    mms = MinMaxScaler()
    print(X_train)
    X_train = mms.fit_transform(X_train)
    #X_test = mms.transform(X_test)
    X_test = mms.fit_transform(X_test)
    print(X_train)

    # Hyperparameter optimization.
    n_estimators = [5, 30, 50, 100, 200, 400]
    max_features = ['auto', 'sqrt']
    #max_depth = ['None']
    min_samples_split = [2, 5, 10]
    min_samples_leaf = [1, 2, 4]
    bootstrap = ['True', 'False']
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap
                   }

    #reg = RandomForestRegressor(n_estimators=200, verbose=1, random_state=0, n_jobs=-1)
    #reg = RandomForestRegressor()

    reg_random = RandomizedSearchCV(estimator=RandomForestRegressor(), param_distributions=random_grid, n_iter=100, cv=3,
                                    verbose=2, random_state=0, n_jobs=-1)
    reg_random.fit(X_train, y_train)
    print(reg_random.best_params_)

    # Create estimator using the best parameters.
    #reg = RandomForestRegressor(**reg_random.best_params_)
    #print(reg.get_params())

    #reg.fit(X_train, y_train)
    #y_pred = reg.predict(X_test)
    y_pred = reg_random.predict(X_test)

    print('Mean Absolute Error:', round(metrics.mean_absolute_error(y_test, y_pred), 2))
    print('Mean Squared Error:', round(metrics.mean_squared_error(y_test, y_pred), 2))
    print('Root Mean Squared Error:', round(np.sqrt(metrics.mean_squared_error(y_test, y_pred)), 2))

    R = r2_score(y_test, y_pred)
    print("R^2: ", R)

    features = X
    feature_list = list(features.columns)
    features = np.array(features)

#    rf_small = RandomForestRegressor(n_estimators=10, max_depth=3)
#    rf_small.fit(X_train, y_train)
#    tree_small = rf_small.estimators_[5]
#    export_graphviz(tree_small, out_file='small_tree.dot', feature_names=feature_list,
#                    rounded=True, precision=1, proportion=False, filled=True)
#    (graph, ) = pydot.graph_from_dot_file('small_tree.dot')
#    graph.write_png('small_tree.png')

    #importances = list(reg.feature_importances_)
    importances = list(reg_random.best_estimator_.feature_importances_)
    feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
    feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)

    [print('Feature: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];

    df_pred = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    #print(df_pred.head(25)['Actual'].tolist())
    #print(df_pred.index[:25].tolist())

    # Plot an overlapping bar chart of actual vs predicted values
    # to get a visual impression of the prediction accuracy.
    n_points = 20000
    #plt.bar(np.arange(n_points), df_pred.head(n_points)['Actual'].tolist(), color='b', width=0.8, label='Actual')
    #plt.bar(np.arange(n_points), df_pred.head(n_points)['Predicted'].tolist(), color='r', width=0.6, label='Predicted')
    #plt.scatter(np.arange(n_points), df_pred.head(n_points)['Actual'].tolist(), color='b', label='Actual')
    #plt.scatter(np.arange(n_points), df_pred.head(n_points)['Predicted'].tolist(), color='r', label='Predicted')
    #plt.scatter(df_pred.head(n_points)['Actual'], df_pred.head(n_points)['Predicted'].tolist(),
    #            color='black', marker='x')
    plt.hexbin(df_pred['Actual'], df_pred['Predicted'], gridsize=100, bins='log', cmap=plt.cm.Greens)
    plt.plot(df_pred['Actual'], df_pred['Actual'], color='red')
    plt.colorbar(label='log10(N)')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Actual vs Predicted values')
    plt.grid()
    plt.axis([df_pred['Actual'].min(), df_pred['Actual'].max(), df_pred['Actual'].min(), df_pred['Actual'].max()])
    #plt.bar(df_pred.index.tolist())
    #plt.legend()
    #plt.tight_layout()
    plt.show()

#    plt.style.use('fivethirtyeight')
#    x_values = list(range(len(importances)))
#    plt.bar(x_values, importances, orientation='vertical')
#    plt.xticks(x_values, feature_list, rotation='vertical')
#    plt.ylabel('Importance')
#    plt.xlabel('Feature')
#    plt.title('Feature Importances')
#    plt.show()

#    model = keras.Sequential()
#    model.add(layers.Dense(8, input_dim=7, activation='relu'))
#    #model.add(layers.ReLU(alpha=1.0))
#    #model.add(layers.ReLU())
#    model.add(layers.Dense(4, activation='relu'))
#    #model.add(layers.ReLU(alpha=1.0))
#    #model.add(layers.ReLU())
#    #model.add(layers.Dense(25, activation='softmax'))
#    model.add(layers.Dense(1, activation='linear'))
#
#    model.summary()
#
#    #model.compile(loss=losses.MeanSquaredError(), optimizer=keras.optimizers.Adam(lr=0.001))
#    model.compile(loss=losses.MeanAbsolutePercentageError(), optimizer=keras.optimizers.Adam(lr=1e-3, decay=1e-3/200))
#
#    X_train = np.asarray(X_train)
#    X_test = np.asarray(X_test)
#    y_train = np.asarray(y_train)
#    y_test = np.asarray(y_test)
#
#    values = model.fit(x=X_train, y=y_train, batch_size=8, epochs=50, verbose=1,
#                       validation_data=(X_test, y_test))
#
#    plt.plot(values.history['loss'])
#    plt.plot(values.history['val_loss'])
#    plt.title('Model loss')
#    plt.ylabel('loss')
#    plt.xlabel('epoch')
#    plt.legend(['train', 'test'], loc='upper right')
#    plt.show()
#
#    price_pred = model.predict(X_test).flatten()
#    df_pred_seq = pd.DataFrame({'Actual': y_test, 'Predicted': price_pred})
#    print(df_pred_seq.head(25))
#
#    print('Keras MLP results:')
#    print('Mean Absolute Error:', round(metrics.mean_absolute_error(y_test, price_pred), 2))
#    print('Mean Squared Error:', round(metrics.mean_squared_error(y_test, price_pred), 2))
#    print('Root Mean Squared Error:', round(np.sqrt(metrics.mean_squared_error(y_test, price_pred)), 2))
#
#    R_mlp = r2_score(y_test, price_pred)
#    print("R^2: ", R_mlp)

'''
    # X = data[['MachineID', 'ModelID', 'YearMade', 'MachineHoursCurrentMeter']]
    #filterYear = np.where(data['YearMade'] > 1800)[0]
    # X = data[['YearMade']].iloc[filterYear]
    # Y = data['SalePrice'].iloc[filterYear]
    # X = data[['ProductGroup']]
    Y = data['SalePrice']
    # Transform categorical variables into numeric representations.
    ProductGroup_le = LabelEncoder()
    ProductGroup_labels = ProductGroup_le.fit_transform(data['ProductGroup'])
    data['ProductGroup_Label'] = ProductGroup_labels
    # Pick out variables to use in the model.
    data_sub = data[['SalePrice', 'ProductGroup', 'ProductGroup_Label']]
    print(data_sub.iloc[0:10])

    # Encode the numeric representations using the one-hot encoding scheme.
    ProductGroup_ohe = OneHotEncoder(handle_unknown='ignore')
    ProductGroup_feature_arr = ProductGroup_ohe.fit_transform(data[['ProductGroup_Label']]).toarray()
    ProductGroup_feature_labels = list(ProductGroup_le.classes_)
    # print(ProductGroup_feature_labels)
    ProductGroup_features = pd.DataFrame(ProductGroup_feature_arr, columns=ProductGroup_feature_labels)
    print(ProductGroup_features)
    data_ohe = pd.concat([data_sub, ProductGroup_features], axis=1)
    print(data_ohe.iloc[0:10])
    # enc.transform(X)
    # X_hash = ce.HashingEncoder(cols=['ProductGroup'])
    # X_hash.fit_transform(X, Y)
    # print(X_hash)

    # train = data.iloc[filterYear]
    # test  = data.iloc[filterYear]
    # train = train[:(int((len(data)*0.8)))]
    # test  = test[(int((len(data)*0.8))):]
    # train = data[:(int((len(data)*0.8)))]
    # test  = data[(int((len(data)*0.8))):]
    train = data_ohe[:(int((len(data) * 0.8)))]
    test = data_ohe[(int((len(data) * 0.8))):]

    regr = linear_model.LinearRegression()

    # train_x = np.array(train[['MachineID', 'ModelID', 'YearMade', 'MachineHoursCurrentMeter']])
    # train_x = np.array(train[['YearMade']])
    # train_x = np.array(train[['ProductGroup']])
    train_x = np.array(train[['BL', 'MG', 'SSL', 'TEX', 'TTT', 'WL']])
    train_y = np.array(train['SalePrice'])

    # test_x = np.array(test[['MachineID', 'ModelID', 'YearMade', 'MachineHoursCurrentMeter']])
    # test_x = np.array(test[['YearMade']])
    # test_x = np.array(test[['ProductGroup']])
    test_x = np.array(test[['BL', 'MG', 'SSL', 'TEX', 'TTT', 'WL']])
    test_y = np.array(test['SalePrice'])

    regr.fit(train_x, train_y)

    coeff_data = pd.DataFrame(regr.coef_, data_ohe[['BL', 'MG', 'SSL', 'TEX', 'TTT', 'WL']].columns,
                              columns=["Coefficients"])
    print(coeff_data)

    Y_pred = regr.predict(test_x)

    R = r2_score(test_y, Y_pred)
    print("R^2: ", R)
    # print(Y_pred)
    # print(test_y)

    X = data_ohe[['BL', 'MG', 'SSL', 'TEX', 'TTT', 'WL']]
    print(X)
    y = data['SalePrice']
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    print(y_train)

    #gsc = GridSearchCV(estimator=RandomForestRegressor(),
    #                   param_grid={'max_depth': range(3, 7),
    #                               'n_estimators': (10, 50, 100, 1000)},
    #                   cv=5, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1)
    #grid_result = gsc.fit(x_train, y_train)
    #best_params = grid_result.best_params_

    #rfr = RandomForestRegressor(max_depth=best_params["max_depth"], n_estimators=best_params["n_estimators"],
    #                            random_state=False, verbose=False)
    rfr = RandomForestRegressor(max_depth=5, n_estimators=100,
                                random_state=False, verbose=False)

    scores = cross_val_score(rfr, x_train, y_train, cv=10, scoring='neg_mean_absolute_error')
    print(scores)

    rfr.fit(x_train, y_train)

    y_pred = rfr.predict(x_test)
    R = r2_score(y_test, y_pred)
    print("R^2: ", R)

#    forest_clf = RandomForestClassifier(n_estimators=100)
#    forest_clf.fit(x_train, y_train)
#
#    y_pred = forest_clf.predict(x_test)
#
#    print('Accuracy:', metrics.accuracy_score(y_test, y_pred))
#
#    RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
#                           max_depth=None, max_features='auto', max_leaf_nodes=None,
#                           min_impurity_decrease=0.0, min_impurity_split=None,
#                           min_samples_leaf=1, min_samples_split=2,
#                           min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=1,
#                           oob_score=False, random_state=None, verbose=0,
#                           warm_start=False)
#
#    feature_imp = pd.Series(forest_clf.feature_importances_,
#                            index=ProductGroup_feature_labels).sort_values(ascending=False)
    feature_imp = pd.Series(rfr.feature_importances_,
                            index=ProductGroup_feature_labels).sort_values(ascending=False)
    print(feature_imp)

    # plt.scatter(test_x, test_y)
    # plt.show()
'''

if __name__ == '__main__':
    main()
