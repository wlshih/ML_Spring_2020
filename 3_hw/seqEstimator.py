#! /usr/local/bin/python3

import numpy as np
import math
import time
import sys

# Part 1.a: univariate Gaussian data generator
def univariate_data_generator(mean, var):
	# rely on central limit theorem, we take 12 U(0, 1) deviates, WHY?
	standard_normal = np.sum(np.random.uniform(0.0, 1.0, 12)) - 6
	sigma = math.sqrt(var)

	return mean + standard_normal * sigma

if __name__ == "__main__":

	mean = float(sys.argv[1])
	var = float(sys.argv[2])


	print("Data point source funtion: N(", mean, ", ", var, ")\n", sep = '')

	mean_prev = 0.0
	mean_mle = 0.0
	var_prev = 0.0
	var_mle = 0.0

	n = 0
	while(True):
		n += 1
		data = univariate_data_generator(mean, var)
		print("Add data point:", data)

		# total mean of all means
		mean_mle = (mean_prev * (n-1) + data) / n
		# total mean of all variables
		var_mle = (var_prev * (n-1) + (data - mean_mle)**2) / n

		print("Mean =", mean_mle, " Variance =", var_mle)

		# break until the estimates converge
		if(abs(mean_prev - mean_mle) < 0.00001 and abs(var_prev - var_mle) < 0.00001):
			break

		mean_prev = mean_mle
		var_prev = var_mle
