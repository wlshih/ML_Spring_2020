#! /usr/local/bin/python3

import numpy as np
from numba import jit
from warnings import filterwarnings

fnImage = "train-images-idx3-ubyte"
fnLabel = "train-labels-idx1-ubyte"

def read_data():
	data_type = np.dtype("int32").newbyteorder('>')  # big endian

	data = np.fromfile(fnImage, dtype = 'ubyte').astype("float64")
	data = data[4 * data_type.itemsize : ].reshape(60000, 28 * 28).transpose()  # ignore 16-byte headers
	data = np.divide(data, 128).astype("int")

	label = np.fromfile(fnLabel, dtype = 'ubyte').astype('int')
	label = label[2 * data_type.itemsize : ].reshape(60000)  # ignore 8-byte headers
	
	return data, label

@jit
def E_step(data, lamb, P):
	# k labels, n training data sets
	Z = np.zeros((10, 60000), dtype=np.float64)
	for n in range(60000):
		#print(n)
		marginal = 0  # denominator
		for k in range(10):
			p = lamb[k]
			for i in range(28*28):
				if data[n][i]:
					p *= P[i][k]
				else:
					p *= (1 - P[i][k])
			Z[k][n] = p 
			marginal += p

		if marginal == 0:
			marginal = 1
		for k in range(10):
			Z[k][n] /= marginal
	
	return Z


@jit
def M_step(data, lamb, P, Z):
	for k in range(10):
		N = np.sum(Z[k])  # N = sigma(Zm) = total 1's occurs in class m
		lamb[k] = N / 60000
		if N == 0:
			N = 1
		for i in range(28 * 28):
			P[i][k] = np.sum(np.matmul(Z[k], data[:][i])) / N

	return lamb, P
			

# print imagine of number
def print_imagin(P):
	for k in range(10):
		print("\nclass: ", k)
		for i in range(28*28):
			if i % 28 == 0 and i != 0:
				print("")
			if P[i][k] >= 0.5:
				print(" ", end=' ')
			else:
				print("0", end=' ')
		print("")


#@jit
def EM_algorithm():

	print("---(read data)---")
	data, label = read_data()
	print("\n  train data size: {}\n".format(data.shape))


	P = np.random.rand(28 * 28, 10).astype(np.float64)    # probability of each bit of each class
	Z = np.full((10, 60000), 0.1, dtype=np.float64)       # responsibility
	lamb = np.full(10, 0.1, dtype=np.float64)             # lambda, mean MLE of each class
	lamb_prev = np.zeros(10, dtype=np.float64)            # lambda from last iteration, for convergence test


	cond = 0
	it = 0 
	while(True):
		it += 1

		# E step, get Z
		print("---(E step {})---".format(it))
		Z = E_step(data, lamb, P)

		# M step, get lamb, P
		print("---(M step {})---".format(it))
		lamb, P = M_step(data, lamb, P, Z)

		# condition check
		if 0 in lamb:
			cond = 0
			# initialize lambda and P if lambda == 0 
			P = np.random.rand(28 * 28, 10).astype(np.float64)
			lamb = np.full(10, 0.1, dtype=np.float64)
			print("---(restart)---")
		else:
			cond += 1

		diff = np.sum(abs(lamb - lamb_prev))
		if diff < 5e-3 and cond >= 8 and np.sum(P) > 0.95:
			break
		
		lamb_prev = np.copy(lamb)

		print_imagin(P)
		print("No. of Iteration: {}, Difference: {}\n".format(it, diff))
		print("---------------------------------------------------------------------------\n")
	
	print("---------------------------------------------------------------\n")






if __name__ == "__main__":
	
	filterwarnings("ignore")

	EM_algorithm()
