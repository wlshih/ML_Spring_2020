#! /usr/local/bin/python3

import numpy as np
import math
import sys

def univariate_data_generator(mean, var):
	standard_normal = np.sum(np.random.uniform(0.0, 1.0, 12)) - 6
	sigma = math.sqrt(var)
	return mean + sigma * standard_normal

def get_data():

	# read input
	'''
	n = int(input("Number of data points: "))
	mx1 = float(input("mx1: "))
	my1 = float(input("my1: "))
	mx2 = float(input("mx2: "))
	my2 = float(input("my2: "))
	vx1 = float(input("vx1: "))
	vy1 = float(input("vy1: "))
	vx2 = float(input("vx2: "))
	vy2 = float(input("vy2: "))
	'''
	n = 50
	mx1 = my1 = 1
	mx2 = my2 = 3
	vx1 = vy1 = 2
	vx2 = vy2 = 4

	# generate data point
	# design matrix X
	X = []
	y = []

	# generate D1, label = 0
	for i in range(n):
		x1 = univariate_data_generator(mx1, math.sqrt(vx1))
		y1 = univariate_data_generator(my1, math.sqrt(vy1))
		X.append([1.0, x1, y1])
		y.append([0])
		
	
	# generate D2, label = 1
	for i in range(n):
		x2 = univariate_data_generator(mx2, math.sqrt(vx2))
		y2 = univariate_data_generator(my2, math.sqrt(vy2))
		X.append([1.0, x2, y2])
		y.append([1])

	return np.array(X), np.array(y)

# return sigmoid funtion value for gradient descent
def sigmoid(X, w):
	result = []
	z = np.matmul(X, w)
	for i in range(X.shape[0]):
		z = np.matmul(X[i], w)
		result.append(1 / (1 + np.exp((-1) * z)))

	return np.array(result)


if __name__ == "__main__":

	X, y = get_data()
	#print(X)
	#print(y)

	# Gradient descent
	w_0 = np.array([[0.0], [0.0], [0.0]])
	while(True):
		gradient = np.matmul(X.T, (y - sigmoid(X, w_0)))
		w_new = w_0 + gradient * 0.01

		#print(np.sum(gradient))
		if(abs(np.sum(gradient)) < 1e-4):
			break
		
		w_0 = np.copy(w_new)
	
	predict = np.matmul(X, w_0)
	print(predict)
	predict[predict > 0.5] = 1
	predict[predict <= 0.5] = 0
	print(predict)

	print("Gradient descent:")
	print("w:")
	np.savetxt(sys.stdout, w_0, fmt="  %.10f")

	# confusion matrix

	print("Confusion matrix:")
	print("\t\tPredict cluster 0\tPredict cluster 1")
	print("Real cluster 0\t\t{}\t\t\t{}".format(TN,FP))
	print("Real cluster 1\t\t{}\t\t\t{}".format(FN,TP))
	print("Sensitivity (Successfully predict cluster 0): {}".format(TN / (TN + FP)))
	print("Specificity (Successfully predict cluster 1): {}".format(TP / (TP + FN)))

	
	# Newton's method

