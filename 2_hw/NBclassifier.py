import sys
import math
import numpy as np

fnTrainImage = "train-images-idx3-ubyte"
fnTrainLabel = "train-labels-idx1-ubyte"
fnTestImage = "t10k-images-idx3-ubyte"
fnTestLabel = "t10k-labels-idx1-ubyte"

def readTrainDiscrete():
	with open(fnTrainImage, 'rb') as img, open(fnTrainLabel, 'rb') as lab:
		img.read(16) # file headers
		lab.read(8)

		prior = np.zeros(10, dtype = int)
		imgBin = np.zeros((10, 28*28, 32), dtype = int)

		for i in range(60000):
			label = int.from_bytes(lab.read(1), 'big')
			prior[label] += 1
			for j in range(28*28):
				pixel = int.from_bytes(img.read(1), 'big')
				imgBin[label][j][int(pixel/8)] += 1
		
		return prior, imgBin

def testDiscrete(prior, imgBin):

	# calculate the number of each label 
	sumImgBin = np.zeros(10, dtype = int)
	for i in range(10):
		for j in range(32):
			sumImgBin[i] += imgBin[i][0][j]
		# print(sumImgBin[i])

	with open(fnTestImage, 'rb') as img, open(fnTestLabel, 'rb') as lab:
		img.read(16) # ignore file headers
		lab.read(8)

		sumError = 0
		for i in range(10000):
			testPix = np.zeros(28*28, dtype = int)
			prob = np.zeros(10, dtype = float)

			for j in range(28*28):
				testPix[j] = int.from_bytes(img.read(1), 'big') / 8

			for j in range(10):
				prob[j] += np.log(float(prior[j]/60000))
				for k in range(28*28):
					tmp = imgBin[j][k][testPix[k]]
					if tmp == 0:
						prob[j] += np.log(float(0.001 / sumImgBin[j]))
					else:
						prob[j] += np.log(float(tmp / sumImgBin[j]))

			normalization(prob)
			ans = int.from_bytes(lab.read(1), 'big')
			sumError += result(prob, ans)

		return (sumError / 10000)


def printImagineDiscrete(imgBin):
	for n in range(10):
		print(n, ":")
		for i in range(28):
			for j in range(28):
				tmp = 0
				for k in range(32):
					if k < 16:
						tmp += imgBin[n][i * 28 + j][k]
					else:
						tmp -= imgBin[n][i * 28 + j][k]

				if tmp > 0:
					print("0", end = " ")
				else:
					print(" ", end = " ")
			print("")
		print("")

####################################################

def readTrainContinue():
	with open(fnTrainImage, 'rb') as img, open(fnTrainLabel, 'rb') as lab:
		img.read(16) # file headers
		lab.read(8)

		prior = np.zeros(10, dtype = int)
		pixelSqr = np.zeros((10, 28*28), dtype = float)
		mean = np.zeros((10, 28*28), dtype = float)
		var = np.zeros((10, 28*28), dtype = float)

		for i in range(60000):
			label = int.from_bytes(lab.read(1), 'big')
			prior[label] += 1
			for j in range(28*28):
				pixel = int.from_bytes(img.read(1), 'big')
				mean[label][j] += pixel
				pixelSqr[label][j] += (pixel**2)

		for i in range(10):
			for j in range(28*28):
				mean[i][j] = float(mean[i][j] / prior[i])
				pixelSqr[i][j] = float(pixelSqr[i][j] / prior[i])
				var[i][j] = pixelSqr[i][j] - mean[i][j]**2

				if var[i][j] == 0:
					var[i][j] = 0.0001
				# print(pixelSqr[i][j])

		return mean, var, prior

def testContinue(mean, var, prior):
	with open(fnTestImage, 'rb') as img, open(fnTestLabel, 'rb') as lab:
		img.read(16) # ignore file headers
		lab.read(8) 

		sumError = 0
		for i in range(10000):
			prob = np.zeros(10, dtype = float)
			testPix = np.zeros(28*28, dtype = float)
			for j in range(28*28):
				testPix[j] = int.from_bytes(img.read(1), 'big')
			for j in range(10):
				prob[j] += np.log(float(prior[j] / 60000))
				for k in range(28*28):
					prob[j] += np.log(float(1.0 / math.sqrt(2.0 * math.pi * var[j][k])))
					prob[j] -=  float(((testPix[k] - mean[j][k]) ** 2) / (2 * var[j][k]))

			normalization(prob)
			ans = int.from_bytes(lab.read(1), 'big')
			sumError += result(prob, ans)

		return (sumError / 10000)

def printImagineContinue(mean):
	for i in range(10):
		print(i, ":")
		for j in range(28):
			for k in range(28):
				if mean[i][j * 28 + k] < 128:
					print("0", end = " ")
				else:
					print("1", end = " ")
			print("")
		print("")

#####################################################

def normalization(prob):
	tmp = 0
	for i in range(10):
		tmp += prob[i]
	for i in range(10):
		prob[i] /= tmp

# return the result of prediction
def result(prob, ans):
	print("Posterior (in log scale):")
	for j in range(10):
		print(j, ": ", prob[j])
	predict = np.argmin(prob)
	print("Prediction: ", predict, ", Ans: ", ans)
	print("")

	return 0 if predict == ans else 1



# Part 1: Naive Bayes classifier
if __name__ == '__main__':
	if sys.argv[1] == '0':
		prior, imgBin = readTrainDiscrete()
		error = testDiscrete(prior, imgBin)
		printImagineDiscrete(imgBin)
		print("Error rate: ", error)
	else:
		mean, var, prior = readTrainContinue()
		error = testContinue(mean, var, prior)
		printImagineContinue(mean)
		print("Error rate: ", error)