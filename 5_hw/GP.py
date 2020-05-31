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


def GaussainProcess(X, Y, beta, sigma, alpha, length_scale):
	mu = np.zeros(X.shape)
	cov = kernel(X, X, sigma, alpha, length_scale) + 1 / beta * np.identity(X.shape[0])
	cov_inv = np.linalg.inv(cov)

	# test
	X_test = np.linspace(-60, 60, 1000).reshape(-1, 1)

	cov_test = kernel(X_test, X_test, sigma, alpha, length_scale) + 1 / beta
	cov_train_test = kernel(X, X_test, sigma, alpha, length_scale)

	Y_test = np.linalg.multi_dot([cov_train_test.T, cov_inv, Y])
	std = np.sqrt(cov_test - np.linalg.multi_dot([cov_train_test.T, cov_inv, cov_train_test]))
	Y_test_plus = Y_test + 2 * (np.diag(std).reshape(-1, 1))
	Y_test_minus = Y_test - 2 * (np.diag(std).reshape(-1, 1))

	print(Y_test_plus)
	print(Y_test_minus)
	
	# graph
	plt.title("Gaussian Process with length-scale : {0:.2f} sigma : {1:.2f} alpha : {2:.2f}".format(length_scale, sigma, alpha))
	plt.scatter(X, Y, color='black')
	plt.plot(X_test, Y_test, color='blue')
	plt.plot(X_test, Y_test_plus, color='red')
	plt.plot(X_test, Y_test_minus, color='red')

	plt.xlim(-60, 60)
	plt.show()



if __name__ == "__main__":

	X, Y = readData()

	'''
	beta = 5
	sigma = 1
	alpha = 1
	length_scale = 1
	'''
	GaussainProcess(X, Y, 5, 1, 1, 1)
