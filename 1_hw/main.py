import sys
import csv
import math
import numpy as np 
import matplotlib.pyplot as plot 
import pprint

def read_file(filename):
	x = []
	y = []
	with open(filename, 'r') as f:
		read = csv.reader(f, delimiter = ',')
		for row in read:
			x.append(float(row[0]))
			y.append(float(row[1]))
	return x, y

def matrix_A(x, poly_bases):
	A = []
	for i in range(len(x)):
		tmp = []
		for j in range(poly_bases - 1, -1, -1):
			tmp.append(x[i]**j)
		A.append(tmp)
	return A

# convert 1d array into a matrix
def matrix_b(y):
	b = []
	for i in range(len(y)):
		tmp = [y[i]]
		b.append(tmp)
	return b

def matrix_transpose(A):
	A_t = []
	for col in range(len(A[0])):
		tmp = []
		for row in range(len(A)):
			tmp.append(A[row][col])
		A_t.append(tmp)
	return A_t

# C = A x B
def matrix_mult(A, B):
	if len(A[0]) != len(B):
		print("matrix_mult: invalid matrix size")
		sys.exit()

	C = [[0 for j in range(len(B[0]))] for i in range(len(A))]
	for i in range(len(A)):
		for j in range(len(B[0])):
			for k in range(len(B)):
				C[i][j] += A[i][k] * B[k][j]
	return C

# return a identity matrix with scalar
def matrix_eye(size, scalar):
	A = [[0 for j in range(size)] for i in range(size)]
	for i in range(size):
		A[i][i] = scalar
	return A

def matrix_add(A, B):
	if len(A) != len(B) or len(A[0]) != len(B[0]):
		print("matrix_add: input matrices have different size")
		sys.exit()

	C = [row[:] for row in A]
	for i in range(len(A)):
		for j in range(len(A[0])):
			C[i][j] += B[i][j]
	return C

def matrix_sub(A, B):
	if len(A) != len(B) or len(A[0]) != len(B[0]):
		print("matrix_add: input matrices have different size")
		sys.exit()

	C = [row[:] for row in A]
	for i in range(len(A)):
		for j in range(len(A[0])):
			C[i][j] -= B[i][j]
	return C


# derive inverse with LU decomposition
# A = LU -> A_i = U_i * L_i
def matrix_inverse(A):	
	# LU decomposition, gives L_inverse and U
	Li = matrix_eye(len(A), 1)
	U = [row[:] for row in A]
	for i in range(len(A)-1): # col
		L_tmp = matrix_eye(len(A), 1) # accumulate elementary matrices
		for j in range(i+1, len(A)): # row
			scale = U[j][i] / U[i][i]
			L_tmp[j][i] -= scale
			for k in range(len(A[0])): # row addition
				U[j][k] -= U[i][k] * scale
		Li = matrix_mult(L_tmp, Li)
	
	# calculate U inverse
	Ui = matrix_eye(len(U), 1)
	for i in range(len(U[0])-1, -1, -1): # col
		for j in range(i): # row
			scale = U[j][i] / U[i][i]
			U[j][i] -= U[i][i] * scale
			# row addition + multiplication
			for k in range(i, len(U[0])):
				Ui[j][k] -= Ui[i][k] * scale 
	# divide by diagonal element
	for i in range(len(U)):
		for j in range(len(U[0])):
			Ui[i][j] /= U[i][i]

	# A inverse
	# pprint.pprint(U)
	# pprint.pprint(Ui)
	# pprint.pprint(Li)
	Ai = matrix_mult(Ui, Li)	
	return Ai

# B = cA
def matrix_scale(A, scalar):
	B = [[0 for j in range(len(A[0]))] for i in range(len(A))]
	for i in range(len(A)):
		for j in range(len(A[0])):
			B[i][j] = A[i][j] * scalar
	return B

def difference(A):
	tmp = 0
	for i in range(len(A)):
		for j in range(len(A[0])):
			tmp += (A[i][j])**2
	return math.sqrt(tmp)

# return formula and print at the same time
def formula(A, x_hat, b, title):
	print(title + ":")
	print("Fitting line:", end = "")

	formula = ""
	i = 0
	for j in range(len(x_hat)-1, -1, -1):
		print(" ", end = "")
		formula += str(abs(x_hat[i][0]))
		print(abs(x_hat[i][0]), end = " ")
		i += 1
		if j != 0:
			print("X^", end = "")
			print(j, end = " ")
			formula += ("*x**" + str(j))

			sign = "+" if x_hat[i][0] >= 0 else "-"  
			formula += sign
			print(sign, end = "")

	print("")

	predict = matrix_mult(A, x_hat)
	error = 0.0
	for i in range(len(y)):
		error += (b[i][0] - predict[i][0]) ** 2
	print("Total error: ", error)

	print("")
	return formula



def display(result_lse, result_newton, X, Y):
	plot.subplot(2, 1, 1)
	plot.title("LSE")
	plot.scatter(X, Y, c = 'red')
	x = np.array(range(math.floor(min(X)) - 1, math.ceil(max(X)) + 2))
	if 'x' in result_lse:
		y = np.array(eval(result_lse))
		plot.plot(x, y)  
	else:
		plot.axhline(y = float(result_lse), color='b', linestyle='-') 
	plot.subplot(2,1,2)
	plot.tight_layout()
	plot.title("Newton's Method")
	plot.scatter(X, Y, c = 'red')
	if 'x' in result_newton:
		y = np.array(eval(result_newton))
		plot.plot(x, y)  
		plot.show()
	else:
		plot.axhline(y = float(result_newton), color='b', linestyle='-') 
		plot.show()


if __name__ == '__main__':

	filename = sys.argv[1]
	poly_bases = int(sys.argv[2])
	lse_lambda = float(sys.argv[3])

	x, y = read_file(filename)
	A = matrix_A(x, poly_bases)
	b = matrix_b(y)

	A_transpose = matrix_transpose(A)
	A_transpose_A = matrix_mult(A_transpose, A)
	lambda_I = matrix_eye(poly_bases, lse_lambda)
	A_transpose_b = matrix_mult(A_transpose, b)
	x_hat = matrix_mult(matrix_inverse(matrix_add(A_transpose_A, lambda_I)), A_transpose_b)

	result_lse = formula(A, x_hat, b, "LSE")
	

	# Newton
	x_new = []
	for i in range(poly_bases):
		x_new.append([100])

	hessian_inverse = matrix_inverse(matrix_scale(A_transpose_A, 2))
	A_transpose_b_2 = matrix_scale(matrix_mult(A_transpose, b), 2)

	while(1):
		tmp = x_new
		gradient = matrix_sub(matrix_scale(matrix_mult(A_transpose_A, x_new), 2), A_transpose_b_2)
		x_new = matrix_sub(x_new, matrix_mult(hessian_inverse, gradient))

		if difference(matrix_sub(x_new, tmp)) < poly_bases:
			break

	result_newton = formula(A, x_new, b, "Newton")



	# display graph
	display(result_lse, result_newton, x, y)