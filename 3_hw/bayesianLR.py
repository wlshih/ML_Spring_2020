#! /usr/local/bin/python3

import numpy as np
import math
import matplotlib.pyplot as plt
import pprint

from seqEstimator import univariate_data_generator

# polynomial regression modle data generator
def PR_model_generator(weight, var):
	x = np.random.uniform(-1.0, 1.0, 1)
	y = 0.0
	for i in range(len(weight)):
		y += weight[i] * (x**i)
	error = univariate_data_generator(0, var)

	return x, y + error 


def read_input():
	print("precision(b): ", end = '')
	b = float(input())
	print("polynomial bases(n): ", end = '')
	n = int(input())
	print("variation(a): ", end = '')
	a = float(input())
	print("weight matirx(W): ", end = '')
	W = list(map(float, input().split()))

	return b, n, a, W

# linear regression matrix form
def build_matrix(x, n):
	A = []
	for i in range(n):
		A.append(x ** i)

	return np.array(A).reshape(1, -1)

def print_iteration(cnt, x, y, mean_post, var_post, mean_predict, var_predict):
	print("Count = {}".format(cnt))
	print("Add data point ({}, {}):".format(x, y))
	print("")
	print("Posterior mean:")
	print(mean_post)
	print("")
	print("Posterior covariance:")
	print(var_post)
	print("")
	print("Predictive distribution ~ N({}, {})".format(mean_predict, var_predict))
	print("---------------------------------------------")
	return



if __name__ == "__main__":
	precision, poly_bases, var, weight = read_input()
	
	
	# linear regression
	# first iteration
	cnt = 1
	x, y = PR_model_generator(weight, var)
	# data record
	data_x = [x]
	data_y = [y]

	A = build_matrix(x, poly_bases)
	
	var_post = np.linalg.inv(var * np.matmul(A.T, A) + precision * np.eye(poly_bases))
	mean_post = var * np.matmul(var_post, A.T) * y

	var_predict = 1 / var + np.matmul(np.matmul(A, var_post), A.T)
	mean_predict = np.matmul(A, mean_post)

	print_iteration(cnt, x, y, mean_post, var_post, mean_predict, var_predict)

	while(True):
		cnt += 1
		
		x, y = PR_model_generator(weight, var)
		data_x.append(x)
		data_y.append(y)

		A = build_matrix(x, poly_bases)
		
		var_prior = var_post.copy()
		var_prior_inv = np.linalg.inv(var_prior)
		mean_prior = mean_post.copy()
		
		var_post = np.linalg.inv(var * np.matmul(A.T, A) + np.linalg.inv(var_prior))
		mean_post = np.matmul(var_post, (var * A.T * y + np.matmul(var_prior_inv, mean_prior)))

		var_predictive = 1 / var + np.matmul(np.matmul(A, var_post), A.T)
		mean_predictive = np.matmul(A, mean_post)

		print_iteration(cnt, x, y, mean_post, var_post, mean_predict, var_predict)

		if ( abs(np.sum(mean_prior - mean_post)) < 1e-6 ) and ( abs(np.sum(var_prior - var_post)) < 1e-6 ):
			break

		# data recording
		if cnt == 10:
			mean_10 = mean_post.copy()
			S_inv_10 = np.linalg.inv(var_post)
			data_x_10 = data_x.copy()
			data_y_10 = data_y.copy()

		if cnt == 50:
			mean_50 = mean_post.copy()
			S_inv_50 = np.linalg.inv(var_post)
			data_x_50 = data_x.copy()
			data_y_50 = data_y.copy()
