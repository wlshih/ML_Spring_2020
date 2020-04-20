#! /usr/local/bin/python3

import numpy as np
import math
import matplotlib.pyplot as plt
import pprint

from seqEstimator import univariate_data_generator

# Part 1.b: Polynomial regression model data generator
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


# Part 3: Bayesian linear regression
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
	
	var_post_inv = var * np.matmul(A.T, A) + precision * np.eye(poly_bases)
	var_post = np.linalg.inv(var_post_inv)
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
		
		var_prior_inv = var_post_inv.copy()
		var_prior = var_post.copy()
		mean_prior = mean_post.copy()
		
		var_post_inv = var * np.matmul(A.T, A) + var_prior_inv
		var_post = np.linalg.inv(var_post_inv)
		mean_post = np.matmul(var_post, (var * A.T * y + np.matmul(var_prior_inv, mean_prior)))

		var_predict = 1 / var + np.matmul(np.matmul(A, var_post), A.T)
		mean_predict = np.matmul(A, mean_post)

		print_iteration(cnt, x, y, mean_post, var_post, mean_predict, var_predict)

		if ( abs(np.sum(mean_prior - mean_post)) < 1e-6 ) and ( abs(np.sum(var_prior - var_post)) < 1e-6 ):
			break

		# data recording
		if cnt == 10:
			mean_10 = mean_post.copy()
			var_10 = var_post.copy()
			data_x_10 = data_x.copy()
			data_y_10 = data_y.copy()

		if cnt == 50:
			mean_50 = mean_post.copy()
			var_50 = var_post.copy()
			data_x_50 = data_x.copy()
			data_y_50 = data_y.copy()




	# display plot
	# ground truth
	plt.subplot(221)
	plt.xlim(-2.0, 2.0)
	plt.ylim(-15.0, 25)
	plt.title("Ground Truth")

	ground_func = np.poly1d(np.flip(weight))
	ground_x = np.linspace(-2.0, 2.0, 30)
	ground_y = ground_func(ground_x)
	plt.plot(ground_x, ground_y, color = 'black')
	
	ground_y += var
	plt.plot(ground_x, ground_y, color = 'red')

	ground_y -= 2 * var
	plt.plot(ground_x, ground_y, color = 'red')


	# predice result
	plt.subplot(222)
	plt.xlim(-2.0, 2.0)
	plt.ylim(-15.0, 25)
	plt.title("Predict result")

	predict_x = np.linspace(-2.0, 2.0, 30)
	predict_func = np.poly1d(np.flip(mean_post.flatten()))
	predict_y = predict_func(predict_x)
	predict_y_plus = predict_func(predict_x)
	predict_y_minus = predict_func(predict_x)

	for i in range(len(predict_x)):
		predict_A = build_matrix(predict_x[i], poly_bases)
		predict_var_predict = 1 / var + np.matmul(np.matmul(predict_A, var_post), predict_A.T)
		predict_y_plus[i] += predict_var_predict[0]
		predict_y_minus[i] -= predict_var_predict[0]

	plt.plot(predict_x, predict_y, color = 'black')
	plt.plot(predict_x, predict_y_plus, color = 'red')
	plt.plot(predict_x, predict_y_minus, color = 'red')
	plt.scatter(data_x, data_y)

	# after 10 incomes
	plt.subplot(223)
	plt.xlim(-2.0, 2.0)
	plt.ylim(-15.0, 25)
	plt.title("Predict result")

	predict_x = np.linspace(-2.0, 2.0, 30)
	predict_func = np.poly1d(np.flip(mean_10.flatten()))
	predict_y = predict_func(predict_x)
	predict_y_plus = predict_func(predict_x)
	predict_y_minus = predict_func(predict_x)

	for i in range(len(predict_x)):
		predict_A = build_matrix(predict_x[i], poly_bases)
		predict_var_predict = 1 / var + np.matmul(np.matmul(predict_A, var_10), predict_A.T)
		predict_y_plus[i] += predict_var_predict[0]
		predict_y_minus[i] -= predict_var_predict[0]

	plt.plot(predict_x, predict_y, color = 'black')
	plt.plot(predict_x, predict_y_plus, color = 'red')
	plt.plot(predict_x, predict_y_minus, color = 'red')
	plt.scatter(data_x_10, data_y_10)


	# after 50 incomes
	plt.subplot(224)
	plt.xlim(-2.0, 2.0)
	plt.ylim(-15.0, 25)
	plt.title("Predict result")

	predict_x = np.linspace(-2.0, 2.0, 30)
	predict_func = np.poly1d(np.flip(mean_50.flatten()))
	predict_y = predict_func(predict_x)
	predict_y_plus = predict_func(predict_x)
	predict_y_minus = predict_func(predict_x)

	for i in range(len(predict_x)):
		predict_A = build_matrix(predict_x[i], poly_bases)
		predict_var_predict = 1 / var + np.matmul(np.matmul(predict_A, var_50), predict_A.T)
		predict_y_plus[i] += predict_var_predict[0]
		predict_y_minus[i] -= predict_var_predict[0]

	plt.plot(predict_x, predict_y, color = 'black')
	plt.plot(predict_x, predict_y_plus, color = 'red')
	plt.plot(predict_x, predict_y_minus, color = 'red')
	plt.scatter(data_x_50, data_y_50)

	plt.tight_layout()
	plt.show()






