# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

from random import shuffle
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import csv
import tensorflow as tf

np.set_printoptions(threshold=np.nan) #Always print the whole matrix

# ----- FUNCTIONS START -----

# Count how many bits differ between pattern x and y
def count_errors(x, y):
	count = 0
	for i in range(0, len(x)):
		if (x[i] != y[i]):
			count = count + 1
	return(count)

def calculate_average_error(x, y):
	error = 0
	for i in range(0, len(x)):
		error += abs(x[i] - y[i])
	return(error)

# Change targets of form 0-9 to something like 0, 0, 0, 1, 0, 0, 0, 0, 0, 0
def binary_target(input) :
	output = [0]*10
	output[input] = 1
	return output

# ----- FUNCTIONS END -----

# ----- DATAGEN PART -----

# Array with 10000 (8000 trn and 2000 tst) of 784-dim vectors representing matrices of 28x28

os.chdir( os.path.dirname(os.path.abspath(__file__)) )

t_trn_f = open("binMNIST_data/bindigit_trn.csv", "r")
reader = csv.reader(t_trn_f)

pics = [ [int(row[i]) for i in range( 0, len(row) ) ] for row in reader]

t_trn_f.close()

t_tst_f = open("binMNIST_data/bindigit_tst.csv", "r")
reader = csv.reader(t_tst_f)

pics += [ [int(row[i]) for i in range( 0, len(row) ) ] for row in reader]

t_tst_f.close()


# Matrix of target classifications

trn_f = open("binMNIST_data/targetdigit_trn.csv", "r")
reader = csv.reader(trn_f)

cls = [ int(row[0]) for row in reader]

trn_f.close()

trn_f = open("binMNIST_data/targetdigit_tst.csv", "r")
reader = csv.reader(trn_f)

cls += [ int(row[0]) for row in reader]

trn_f.close()


# Use pics for pictures and cls for the targeted classification

# ----- DATA READY -----

# Split data
training_input = np.array(pics[0:8000])
test_input = np.array(pics[8000:10000])

tmp_training_target = np.array(cls[0:8000])
tmp_test_target = np.array(cls[8000:10000])

training_target = []
test_target = []

for i in range(0, 8000):
	training_target.append(binary_target(tmp_training_target[i]))
training_target = np.array(training_target)

for i in range(0, 2000):
	test_target.append(binary_target(tmp_test_target[i]))
test_target = np.array(test_target)

columns = 10

order = [18, 3, 7, 0, 2, 1, 15, 8, 6, 5] # order used to find one of each number to reconstruct

reconstruction_pic = np.empty((28 * 5, 28 * columns))
# Display original images
for c in range(columns):
	# Draw the original digits
	reconstruction_pic[0 * 28:(0 + 1) * 28, c * 28:(c + 1) * 28] = \
		test_input[order[c]].reshape([28, 28])

# Training Parameters
learning_rate = 100
num_steps = 1500

display_step = 10000
nodes = [25, 50, 100, 150]
learning_curves = []
for run in range(1, 5):

	# Network Parameters
	num_hidden_1 = nodes[run-1]
	num_input = 784 # MNIST data input (img shape: 28*28)

	print("Run: ", run - 1, "nodes: ", num_hidden_1)
	learning_curve = []

	

	# tf Graph input (only pictures)
	X = tf.placeholder("float", [None, num_input])


	# ----- SINGLE LAYER START -----

	weights = {
		'encoder_h1': tf.Variable(tf.random_normal([num_input, num_hidden_1])),
		'decoder_h1': tf.Variable(tf.random_normal([num_hidden_1, num_input])),
	}

	biases = {
		'encoder_b1': tf.Variable([float(0)]*num_hidden_1),
		'decoder_b1': tf.Variable([float(0)]*num_input),
	}

	# Building the encoder
	def encoder(x):
		# Encoder Hidden layer with sigmoid activation #1
		layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']), biases['encoder_b1']))
		return layer_1

	# Building the decoder
	def decoder(x):
		# Decoder Hidden layer with sigmoid activation #1
		layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']), biases['decoder_b1']))
		return layer_1

	# ----- SINGLE LAYER END -----

	# Construct model
	encoder_op = encoder(X)
	decoder_op = decoder(encoder_op)

	# Prediction
	y_pred = decoder_op
	# Targets (Labels) are the input data.
	y_true = X

	# Define loss and optimizer, minimize the squared error

	loss = tf.reduce_mean(abs(y_true - y_pred))

	optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)

	# Initialize the variables (i.e. assign their default value)
	init = tf.global_variables_initializer()

	# Start Training
	# Start a new TF session
	with tf.Session() as sess:


		# Run the initializer
		sess.run(init)
	
		#for v in tf.trainable_variables(): # Print all tf variables
			#print(v)
		#print(tf.trainable_variables()[1].eval(sess)) # Get session variable	
		#print(tf.trainable_variables()[2].eval(sess))
	
		# Training
		for i in range(0, num_steps+1):
			# Run optimization op (backprop) and cost op (to get loss value)
			training_target, l = sess.run([optimizer, loss], feed_dict={X: training_input})
			learning_curve.append(l)
			
			# Display logs per step
			#if i % display_step == 0 or i == 1:
				#print('Step %i: Loss: %f' % (i, l))

		learning_curves.append(learning_curve)
	
		# Encode and decode the digit image
		test_output = sess.run(decoder_op, feed_dict={X: test_input})

		# Display reconstructed images
		for c in range(columns):
			# Draw the reconstructed digits
			reconstruction_pic[run * 28:(run + 1) * 28, c * 28:(c + 1) * 28] = \
				test_output[order[c]].reshape([28, 28])

	tf.reset_default_graph()

for i in range(0, len(learning_curves)):
	#print(learning_curves[i])
	plt.plot(learning_curves[i])

plt.legend(['50 nodes', '75 nodes', '100 nodes', '150 nodes'])
plt.ylabel('Mean error')
plt.xlabel('Epochs')
plt.show()

plt.figure(figsize=(5, columns))
plt.imshow(reconstruction_pic, origin="upper", cmap="gray")
plt.show()

