import pandas as pd
import statsmodels.genmod.generalized_linear_model as glm
import statsmodels.api as sm
from statsmodels.miscmodels.count import PoissonOffsetGMLE
import numpy as np
from patsy import dmatrices
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import mean_squared_error, r2_score,  mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None


def read_sample_data(file):
    data = pd.read_csv(file)
    return data


def ridge_regression(data, y, x):
    Y, X = dmatrices(y + ' ~ ' + ' + '.join(x), data=data, return_type='dataframe')
    # 默认添加常数列
    # X = sm.add_constant(X)
    # 初始化Ridge回归模型
    ridge = RidgeCV(alphas=[1e-2, 5e-2, 1e-1, 5e-1, 1, 2], fit_intercept=False, cv=None)
    res = ridge.fit(X, Y)
    error = abs(res.predict(X) - Y)
    score = res.score(X, Y)
    # print("Coefficient: ", res.coef_)
    print("Ridge Regression Error: ", np.mean(error))
    print("Ridge Regression Score: ", score)
    return res


def random_forest_regression(data, y, x):
    y, X = dmatrices(y + ' ~ ' + ' + '.join(x), data=data, return_type='dataframe')
    regr = RandomForestRegressor(n_estimators=100, max_depth=7, min_samples_split=10)  # random_state=0
    res = regr.fit(X, y)
    score = regr.score(X, y)
    # print(res.get_params())
    # print("Random Forest Score: ", score)
    return res


def regression_grid_search(regressor, param_grid, cv, input_data, y, x, figure):
    grid_search = GridSearchCV(regressor, param_grid, cv=cv, scoring='neg_mean_squared_error')
    xtrain, xtest, ytrain, ytest = train_test_split(input_data[x], input_data[y], test_size=0.3)  # random_state=42
    grid_search.fit(xtrain, ytrain)
    y_hat = grid_search.predict(xtest)

    metrics = {
        'Best Params:': grid_search.best_params_,
        'neg_mean_squared_error': grid_search.best_score_,
        'R2:': r2_score(ytest, y_hat),
        'MAE:': mean_absolute_error(ytest, y_hat),
        'MSE:': mean_squared_error(ytest, y_hat),
        'RMSE:': np.sqrt(mean_squared_error(ytest, y_hat))
               }
    if figure:
        plt.figure(figsize=(10,6))
        t = np.arange(len(xtest))
        plt.plot(t, ytest, color='red', label="Actual")
        plt.plot(t, y_hat, color='blue', label="Predicted")
        plt.legend()
        plt.interactive(True)
        plt.show()
        print(metrics)
    return grid_search


def regression_prediction(model, data, y, x):
    y_label, X = dmatrices(y + ' ~ ' + ' + '.join(x), data=data, return_type='dataframe')
    pred = model.predict(X)  # pred is a array
    y_real = y_label[y]  # y_label is a dataframe, y_real is a series
    error = abs(pred - y_real.array)
    print('Prediction ERROR: ', np.mean(error))
    return pred, error


# X[offset] is a dataframe while X[offset[0]] is a series
def poisson_regression(data, y, x, offset):
    y, X = dmatrices(y + ' ~ ' + ' + '.join(x + offset), data=data, return_type='dataframe')
    # X['Intercept'] = X['Intercept']*data['Y_Estimation']
    feature_cols = ['Intercept'] + x
    poisson = PoissonOffsetGMLE(y, X[feature_cols], offset=X[offset[0]].array)
    # poisson = glm.GLM(y, X[feature_cols], family=sm.families.Poisson(), offset=X[offset[0]].tolist())
    res = poisson.fit(start_params=[1, -0.02, -0.005, -0.005, -0.005, -0.005], method='nm', maxfun=5000, xtol=1e-5, ftol=1e-5)
    print("Poisson Regression: ", res.summary())
    return res


def k_fold_regression(data, k, random_state):
    kfold = KFold(n_splits=k, shuffle=True, random_state=random_state)
    train, valid = [], []
    for train_idx, valid_idx in kfold.split(data):
        train.append(train_idx)
        valid.append(valid_idx)
    return train, valid


def parameter_metrics(para_real, para_est, metric):
    if metric == 'MAE':
        return np.mean(abs(para_real - para_est))
    if metric == 'MSE':
        return np.mean((para_real - para_est) ** 2)
    if metric == 'RMSE':
        return np.sqrt(np.mean((para_real - para_est) ** 2))


def coef_estimation(data, k, random_state):
    theta_real = [-0.02, -0.005, -0.005, -0.005, -0.005]
    theta_est_kfold = []

    train, valid = k_fold_regression(data, k, random_state)
    X = ['X_0', 'X_1', 'X_2', 'X_3', 'X_4', 'X_5', 'X_6', 'X_7', 'X_8', 'X_9']
    X_n = ['price_residual',
           'price_residual_cross_X_1',
           'price_residual_cross_X_2',
           'price_residual_cross_X_3',
           'price_residual_cross_X_4']
    Y1 = 'price'
    Y2 = 'poisson_sample'

    for t, v in zip(train, valid):
        train_data, valid_data = data.iloc[t],  data.iloc[v]
        # the first stage
        res_p = ridge_regression(train_data, Y1, X)
        res_s = random_forest_regression(train_data, Y2, X)
        # the second stage
        pred_p, err_p = regression_prediction(res_p, valid_data, Y1, X)
        pred_s, err_s = regression_prediction(res_s, valid_data, Y2, X)

        valid_data.loc[:, 'est_price'] = pred_p
        valid_data.loc[:, 'price_residual'] = valid_data['price'] - valid_data['est_price']
        valid_data.loc[:, 'price_residual_cross_X_1'] = valid_data['price_residual']*valid_data['X_1']
        valid_data.loc[:, 'price_residual_cross_X_2'] = valid_data['price_residual']*valid_data['X_2']
        valid_data.loc[:, 'price_residual_cross_X_3'] = valid_data['price_residual']*valid_data['X_3']
        valid_data.loc[:, 'price_residual_cross_X_4'] = valid_data['price_residual']*valid_data['X_4']
        valid_data.loc[:, 'Y_Estimation'] = pred_s
        valid_data.loc[:, 'offset'] = np.log(pred_s)
        valid_data.to_csv('/Users/huiqiangmao/Downloads/poisson_samples_process_data.csv')
        res = poisson_regression(valid_data, Y2, X_n, ['offset'])
        # print(res.params)
        theta_est_kfold.append(res.params[:])
    theta_est = np.mean(theta_est_kfold, axis=0)
    # print("THETA ESTIMATION: ", theta_est)
    theta_est_error = parameter_metrics(theta_real, theta_est[1:], metric='MAE')
    # print("THETA ESTIMATION ERROR: ", theta_est_error)
    return theta_est, theta_est_error


def coef_estimation_n_times(data, k, n):
    param_list = []
    param_err_list = []
    for j in range(n):
        res, error = coef_estimation(data, k, j*1000+j*j)
        param_list.append(res)
        param_err_list.append(error)
    print("PARAMETER ESTIMATION: ", np.mean(param_list, axis=0))
    print("PARAMETER ESTIMATION ERROR: ", np.mean(param_err_list, axis=0))
    return np.mean(param_list, axis=0), np.mean(param_err_list, axis=0)


if __name__ == "__main__":
    data = read_sample_data('/Users/huiqiangmao/Downloads/poisson_samples_test.csv')
    k = 5
    n = 5

    # res = coef_estimation(data, k)
    param, param_err = coef_estimation_n_times(data, k, n)

    X = ['X_0', 'X_1', 'X_2', 'X_3', 'X_4', 'X_5', 'X_6', 'X_7', 'X_8', 'X_9']
    Y1 = 'price'
    Y2 = 'poisson_sample'

    # res = ridge_regression(data, Y1, X)

    param_grid_RF = {"n_estimators": [300],
                       "max_depth": [6, 7],
                       "min_samples_split": [9, 10]}
    # grid_search_res = regression_grid_search(RandomForestRegressor(),
    #                                          param_grid_RF,
    #                                          cv=5,
    #                                          input_data=data,
    #                                          y=Y2,
    #                                          x=X,
    #                                          figure=True)

    param_grid_ridge = {'alpha':[.0001, .0005, 0.001, 0.005, 0.01, 0.05, 0.01, 0.05, 1],
                        'tol': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1],
                        'solver': ['auto']}
    # res = ridge_regression(data, Y1, X)
    # grid_search_res = regression_grid_search(Ridge(),
    #                                          param_grid_ridge,
    #                                          cv=5,
    #                                          input_data=data,
    #                                          y=Y1,
    #                                          x=X,
    #                                          figure=True)

