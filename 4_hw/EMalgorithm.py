#! /usr/local/bin/python3

import numpy as np
from numba import jit
from warnings import filterwarnings

fnImage = "train-images-idx3-ubyte"
fnLabel = "train-labels-idx1-ubyte"

def read_data():
	data_type = np.dtype("int32").newbyteorder('>')  # big endian

	data = np.fromfile(fnImage, dtype = 'ubyte')
	img_bin = data[4 * data_type.itemsize:].astype("float64").reshape(60000, 28 * 28).transpose()
	img_bin = np.divide(img_bin, 128).astype("int")

	label = np.fromfile(fnLabel, dtype = 'ubyte').astype('int')
	label = label[2 * data_type.itemsize : ].reshape(60000)
	
	return img_bin, label

@jit
def E_step(img_bin, MU, PI, Z):
	for n in range(60000):
		tmp = np.zeros(shape=(10), dtype=np.float64)
		for k in range(10):
			tmp1 = np.float64(1.0)
			for i in range(28*28):
				if img_bin[i][n]:
					tmp1 *= MU[i][k]
				else:
					tmp1 *= (1 - MU[i][k])
			tmp[k] = PI[k][0] * tmp1
		tmp2 = np.sum(tmp)
		if tmp2 == 0:
			tmp2 = 1
		for k in range(10):
			Z[k][n] = tmp[k] / tmp2
	
	return Z

@jit
def M_step(img_bin, MU, PI, Z):
	N = np.sum(Z, axis=1)
	for j in range(28*28):
		for m in range(10):
			tmp = np.dot(img_bin[j], Z[m])
			tmp1 = N[m]
			if tmp1 == 0:
				tmp1 = 1
			MU[j][m] = tmp / tmp1
	
	for i in range(10):
		PI[i][0] = N[i] / 60000
	
	return MU, PI
	

# print image
def print_MU(MU):
	print("MU size", len(MU), len(MU[0]))
	for i in range(10):
		print("\nclass: ", i)
		for j in range(28*28):
			if j % 28 == 0 and j != 0:
				print("")
			if MU.T[i][j] >= 0.5:
				print(" ", end=' ')
			else:
				print("0", end=' ')
		print("")


@jit
def EM_algorithm():
	img_bin, label = read_data()

	PI = np.full((10, 1), 0.1, dtype=np.float64)
	MU = np.random.rand(28 * 28, 10).astype(np.float64)
	MU_prev = np.zeros((28 * 28, 10), dtype=np.float64)
	Z = np.full((10, 60000), 0.1, dtype=np.float64)

	cond = 0
	it = 1
	while(True):
		it += 1

		# E step, get Z
		print("---(E step)---")
		Z = E_step(img_bin, MU, PI, Z)

		# M step, get MU, PI
		print("---(M step)---")
		MU, PI = M_step(img_bin, MU, PI, Z)

		# condition check
		flag = 0
		for i in range(10):
			if PI[i][0] == 0:
				cond = 0
				flag = 1
				PI = np.full((10, 1), 0.1, dtype=np.float64)
				MU = np.random.rand(28 * 28, 10).astype(np.float64)
				Z = np.full((10, 60000), 0.1, dtype=np.float64)
				break
		
		if flag == 0:
			cond += 1

		if np.sum(abs(MU - MU_prev)) < 20 and cond >= 8 and np.sum(PI) > 0.95:
			break
			
		MU_prev = MU

		print_MU(MU)
		print("No. of Iteration: {}, Difference: {}\n".format(it, np.sum(abs(MU - MU_prev))))
		print("---------------------------------------------------------------------------\n")
	
	print("---------------------------------------------------------------\n")






if __name__ == "__main__":
	
	filterwarnings("ignore")

	EM_algorithm()
