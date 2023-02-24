import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
from pprint import pprint as pprint
from sklearn.linear_model import LinearRegression

df = pd.read_csv('./austin_final.csv')
X = df.drop(['PrecipitationSumInches'], axis=1)
X = X.filter(['TempAvgF', 'DewPointAvgF', 'HumidityAvgPercent', 'SeaLevelPressureAvgInches', 'VisibilityAvgMiles', 'WindAvgMPH'], axis=1)
X["augment"] = [1 for _ in range(len(X))]

Y = df["PrecipitationSumInches"]
Y = Y.values.reshape(-1, 1)

print("Correlation")
pprint(X.corr())

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(X[X.columns.values[0]], X[X.columns.values[2]], Y)
# plt.show()


def train_val_test_split(X, Y):

	# pandas version
	X_train = X.sample(frac=0.7)
	X_validate = X.drop(X_train.index)

	X_test = X_validate.sample(frac=0.66)
	X_validate = X_validate.drop(X_test.index)

	'''print("Train, Validate, Test")
	pprint(X_train)
	pprint(X_validate)
	pprint(X_test)'''

	# numpy version
	p = np.random.permutation(len(Y))
	tr = np.floor(len(Y) * 0.7).astype('int')
	te = np.floor(len(Y) * 0.8).astype('int')
	Y_train = Y[p[:tr]]
	Y_validate = Y[p[tr + 1:te]]
	Y_test = Y[p[te + 1:]]

	return X_train, Y_train, X_validate, Y_validate, X_test, Y_test


X_train, Y_train, X_validate, Y_validate, X_test, Y_test = train_val_test_split(X, Y)
pprint(X_train)

# X_train["augment"] = [1 for _ in range(len(X_train))]
# X_validate["augment"] = [1 for _ in range(len(X_validate))]

LR = LinearRegression()
LR.fit(X_train, Y_train)
print(LR.intercept_, LR.coef_)

Yhat_skl_val = LR.predict(X_validate)

theta = np.matmul(np.matmul(np.linalg.inv(np.matmul(X_train.to_numpy().transpose(), X_train.to_numpy())), X_train.to_numpy().transpose()), Y_train)
print(theta)

'''E_in = np.mean(Y_train - LR.predict(X_train)**2)
E_out = np.mean(Y_validate-LR.predict(X_validate)**2)
print(E_in, E_out)

E_in = np.mean(Y_train-np.matmul(X_train, theta)**2)
E_out = np.mean(Y_validate-np.matmul(X_validate, theta)**2)
print(E_in, E_out)'''

X_aug_train = np.c_[X_train, np.ones(len(Y_train))]
X_aug_validate = np.c_[X_validate, np.ones(len(Y_validate))]

theta = np.matmul(np.matmul(np.linalg.inv(np.matmul(X_aug_train.to_numpy().transpose(), X_aug_train.to_numpy())), X_aug_train.to_numpy().transpose()), Y_train)
hatY_aug_val = np.matmul(X_aug_validate, theta)
print(theta)


def init(X, zeros=True):
	n = X.shape[1]
	if zeros:
		theta = np.zeros((n, 1))
	else:
		theta = np.random.rand(n, 1) - 0.5
		theta[-1] = 0
	return theta


theta = init(X_train, zeros=False)


def predict(X, theta):
	return np.matmul(X, theta)


predict(X_train, theta)


def update_weights(X_train, Y_train, theta):

	Y_pred = predict(X, theta)
	m = X.shape[0]
	dtheta = (2 * (X.T).dor(Y - Y_pred)) / m
	return


learning_rate = 0.00001
theta = init(X_train)
for _ in range(1000):
	dtheta = update_weights(X_train, Y_train, theta)
	print(dtheta)
	theta = theta - (learning_rate*dtheta)