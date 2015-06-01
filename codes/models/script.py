import sys
import os
import matplotlib.pyplot as plt
import numpy as np
def read_file(file_name):
	fp = open(filename, 'r');
	output = []
	for line in fp:
		words = line.split(' ');
		if words[0] == 'Cost':
			output.append(float(words[-1]))
	return output;

def epoch():
	#filename = 'rnn2_30_30.txt';
	filename = 'rnn_wv30.txt';
	acc = read_file(filename);
	n = len(acc);
	train = [acc[2*i] for i in range(n/2)];
	dev = [acc[2*i+1] for i in range(n/2)];
		
	devv = np.array(dev);
	x = np.argmax(devv);
	print x, devv[x];
	plt.plot(range(1,len(train)+1), [1-t for t in train], color = 'red')
	plt.plot(range(1,len(dev)+1), [1-d for d in dev], color = 'blue')
	
	plt.scatter(range(1,len(train)+1), [1-t for t in train], color = 'red')
	plt.scatter(range(1,len(dev)+1), [1-d for d in dev], color = 'blue')
	
	plt.axis([0,30, 0.05, .3])
	plt.show()

def problem1():
	wvdim = np.array([5, 15, 25, 30,  35, 45])
	acc_dev = np.array([0.757329, 0.756074, 0.773060, 0.798657, 0.801047, 0.800010])
	plt.plot(wvdim, acc_dev, color = 'blue')
	plt.scatter(wvdim, acc_dev, color = 'blue')
	plt.show()

def problem2():
	mdim = np.array([5, 15, 25, 30,  35, 45])
	acc_dev = np.array([0.798541, 0.801227, 0.808216, 0.8092121,  0.807972, 0.802326])
	plt.plot(mdim, acc_dev, color = 'blue')
	plt.scatter(mdim, acc_dev, color = 'blue')
	plt.show()



if __name__ == "__main__":
	#epoch()
	problem1();
	problem2()
