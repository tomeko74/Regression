import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model as linm

# Reggression models
# http://scikit-learn.org/stable/modules/linear_model.html

# Load the diabetes dataset
boston = datasets.load_boston()
# print description
print(boston.DESCR)
# get the data
boston_X = boston.data
boston_Y = boston.target
# Split the data into training/testing sets
boston_X_train = boston_X[:-50]
boston_X_test = boston_X[-50:]

# Split the targets into training/testing sets
boston_y_train = boston_Y[:-50]
boston_y_test = boston_Y[-50:]

regressors = {}
regressors['LinReg'] = linm.LinearRegression()
regressors['Ridge'] = linm.Ridge(alpha=.5)
regressors['Lasso'] = linm.Lasso(alpha=5.1)
regressors['ElNet'] = linm.ElasticNet(alpha=.5, l1_ratio=0.5)

fit_results = {}

for key in regressors:
    # Train the model using the training sets
    regr = regressors[key]
    regr.fit(boston_X_train, boston_y_train)
    # mean square error
    mse = np.mean((regr.predict(boston_X_test) - boston_y_test) ** 2)
    w = regr.coef_
    # l1 norm
    wl1 = np.sum(np.abs(w))
    # l2 norm
    wl2 = np.sqrt(np.sum(w ** 2))
    fit_results[key] = {'mse': mse, 'wl2': wl2, 'wl1': wl1, 'w': w}
    print("{}\n----------\n  mse={}\n  wl1={}\n  wl2={}\n  w={}\n ".format(key, mse, wl1, wl2, w))

groups = 3
index = np.arange(groups)
bar_width = .2
opacity = 0.4

fig, ax = plt.subplots(figsize=(15, 9))

t = 0
for key in regressors:
    results = fit_results[key]
    res_val = (results['mse'], results['wl1'], results['wl2'])
    plt.bar(index + bar_width * t, res_val, bar_width,
            alpha=opacity,
#            color=np.random.rand(3, 1),
            label=key)
    t += 1

# plt.xlabel('Modele regresji')
plt.title('Porownanie modeli regresji: MSE, wl1, wl2')
plt.xticks(index + (t - 2) * bar_width, ('MSE', 'wl1', 'wl2'))
plt.legend()

plt.tight_layout()
plt.show()