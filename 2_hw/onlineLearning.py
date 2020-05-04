import sys
import numpy
import math

# mode 0: discrete
# mode 1: continuous

def factorial(n):
	if(n > 2):
		return factorial(n-1)*n
	else:
		return n
	
# Part 2: online learning
if __name__ == "__main__":

	filename = sys.argv[1]
	a = int(sys.argv[2])
	b = int(sys.argv[3])

	with open(filename, 'r') as f:
		data = f.read().splitlines()
		for case in range(len(data)):
			line = data[case]
			case += 1
			print("case ", case, ": ", line)

			# N = a+b, m = a
			m = 0
			N = len(line)
			for i in range(N):
				if line[i] == '1':
					m += 1
			likelihood = ( factorial(N) / ( factorial(m) * factorial(N - m) ) ) * ((m / N) ** m) * ((1 - m / N) ** (N - m))
			print("Likelihood: ", likelihood)
			print("Beta prior:\ta = ", a, "\tb = ", b)

			# Beta(p, a+m, b+N-m)
			a += m
			b += (N - m)
			print("Beta posterior:\ta = ", a, "\tb = ", b)
			print("")
