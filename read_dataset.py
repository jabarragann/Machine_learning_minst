#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
import struct
import pickle
from minst_data_set import minst_data_set

labels_test_set_file = "./data/t10k-labels.idx1-ubyte"
labels_train_set_file = "./data/train-labels.idx1-ubyte"
images_test_set_file = "./data/t10k-images.idx3-ubyte"
images_train_set_file = "./data/train-images.idx3-ubyte"


def get_images(images_file,labels_file):
	#Get Images 
	with open(images_file,'rb') as file:
		magic_nr, size,rows,cols = struct.unpack(">IIII", file.read(16))
		images = np.zeros((size,rows*cols))

		for k in range(size):
			for i in range(rows*cols):
				pixel, = struct.unpack(">B", file.read(1))
				images[k,i] = pixel

	#Get Labels
	with open(labels_file,'rb') as file:
		magic_nr, size = struct.unpack(">II", file.read(8))
		labels = np.zeros(size)

		for k in range(size):
			labels[k], = struct.unpack(">B", file.read(1))
			
	return images,labels


if __name__ == "__main__":

	train_data = minst_data_set( *get_images(images_train_set_file,labels_train_set_file) )
	test_data  = minst_data_set( *get_images(images_test_set_file,labels_test_set_file) )

	with open('./saved_models/training.pickle', 'wb') as f:
	 	pickle.dump(train_data, f, pickle.HIGHEST_PROTOCOL)

	with open('./saved_models/test.pickle', 'wb') as f:
		pickle.dump(test_data, f, pickle.HIGHEST_PROTOCOL)

	x_test =test_data.x
	y_test =test_data.y

	fig, axes = plt.subplots(4)
	axes[0].imshow(x_test[8].reshape((28,28)),cmap='gray_r')
	axes[1].imshow(x_test[38].reshape((28,28)),cmap='gray_r')
	axes[2].imshow(x_test[63].reshape((28,28)),cmap='gray_r')
	axes[3].imshow(x_test[97].reshape((28,28)),cmap='gray_r')
	

	plt.show()
	