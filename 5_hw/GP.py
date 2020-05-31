#! /usr/local/bin/python3

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.optimize import minimize


def readData():
	X = []
	Y = []
	with open("input.data", 'r', encoding='utf-8') as file:
		for line in file:
			x, y = line.split(' ')
			X.append(float(x))
			Y.append(float(y))
	X = np.array(X, dtype=np.float64).reshape(-1, 1)
	Y = np.array(Y, dtype=np.float64).reshape(-1, 1)
	return X, Y

# rational quadratic kernel
def kernel(X1, X2, sigma, alpha, length_scale):
	return (sigma ** 2) * (1 + (cdist(X1, X2, 'sqeuclidean') / (2 * alpha * (length_scale ** 2)))) ** (-alpha)


def GaussianProcess(X, Y, beta, sigma, alpha, length_scale, optimized):
	mu = np.zeros(X.shape)
	cov = kernel(X, X, sigma, alpha, length_scale) + 1 / beta * np.identity(X.shape[0])
	cov_inv = np.linalg.inv(cov)

	# test
	X_test = np.linspace(-60, 60, 120).reshape(-1, 1)

	cov_test = kernel(X_test, X_test, sigma, alpha, length_scale) + 1 / beta
	cov_train_test = kernel(X, X_test, sigma, alpha, length_scale)

	Y_test = cov_train_test.T @ cov_inv @ Y
	std = np.sqrt(cov_test - cov_train_test.T @ cov_inv @ cov_train_test)
	# 95% confidence interval: 2 * std
	Y_test_plus = Y_test + 2 * (np.diag(std).reshape(-1, 1))
	Y_test_minus = Y_test - 2 * (np.diag(std).reshape(-1, 1))

	#print(Y_test_plus)
	#print(Y_test_minus)
	
	# graph
	if optimized:
		plt.title("Optimized Gaussian Process\nlength-scale : {0:.2f} sigma : {1:.2f} alpha : {2:.2f}".format(length_scale, sigma, alpha))
	else:
		plt.title("Gaussian Process\nlength-scale : {0:.2f} sigma : {1:.2f} alpha : {2:.2f}".format(length_scale, sigma, alpha))
	plt.plot(X_test, Y_test, color='b')
	plt.plot(X_test, Y_test_plus, color='r')
	plt.plot(X_test, Y_test_minus, color='r')
	plt.fill_between(X_test.ravel(), Y_test_plus.ravel(), Y_test_minus.ravel(), facecolor='pink')
	plt.scatter(X, Y, color='k')

	plt.xlim(-60, 60)
	plt.show()

def negLogLikelihood(pars, X, Y, beta):
	kern = kernel(X, X, pars[0], pars[1], pars[2])
	kern += np.identity(len(X), dtype=np.float64) * (1 / beta)

	nll = np.sum(np.log(np.diagonal(np.linalg.cholesky(kern))))
	nll += 0.5 * Y.T @ np.linalg.inv(kern) @ Y
	nll += 0.5 * len(X) * np.log(2 * np.pi)

	return nll


if __name__ == "__main__":

	X, Y = readData()

	beta = 5
	sigma = 1
	alpha = 1
	length_scale = 1

	GaussianProcess(X, Y, beta, sigma, alpha, length_scale, 0)

	# hyperparameter optimization
	sigma = 1
	alpha = 1
	length_scale = 1

	opt = minimize(negLogLikelihood, [sigma, alpha, length_scale], bounds=((1e-8, 1e6), (1e-8, 1e6), (1e-8, 1e6)), args=(X, Y, beta))
	sigma_opt = opt.x[0]
	alpha_opt = opt.x[1]
	length_scale_opt = opt.x[2]

	GaussianProcess(X, Y, beta, sigma_opt, alpha_opt, length_scale_opt, 1)

