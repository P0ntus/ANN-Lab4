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

# Training Parameters
learning_rate = 10
num_steps = 100

display_step = 1

# Network Parameters
num_input = 784 # MNIST data input (img shape: 28*28)
num_hidden_1 = 100 #256 # 1st layer num features
num_hidden_2 = 50 #128 # 2nd layer num features (the latent dim)







# ----- SINGLE LAYER START -----

# tf Graph input (only pictures)
X = tf.placeholder("float", [None, num_input])

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

# Construct model
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

# Prediction
y_pred = decoder_op
# Targets (Labels) are the input data.
y_true = X

print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
print(y_true)
print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
print(y_pred)
print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")

# Define loss and optimizer, minimize the squared error

loss = tf.reduce_mean(abs(y_true - y_pred))

optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

enc_h1 = 0
enc_b1 = 0

# Start Training
# Start a new TF session
with tf.Session() as sess:


	# Run the initializer
	sess.run(init)
	
	for v in tf.trainable_variables(): # Print all tf variables
		print(v)
	#print(tf.trainable_variables()[2].eval(sess))
	
	# Training
	for i in range(1, num_steps+1):
		# Run optimization op (backprop) and cost op (to get loss value)
		training_target, l = sess.run([optimizer, loss], feed_dict={X: training_input})
		# Display logs per step
		if i % display_step == 0 or i == 1:
			print('Step %i: Loss: %f' % (i, l))

	enc_h1 = tf.trainable_variables()[0].eval(sess)
	enc_b1 = tf.trainable_variables()[2].eval(sess)
	
	# Testing
	# Encode and decode images from test set and visualize their reconstruction.
	columns = 10
	reconstruction_pic = np.empty((28 * 2, 28 * columns))
	
	# Encode and decode the digit image
	test_output = sess.run(decoder_op, feed_dict={X: test_input})

	order = [18, 3, 7, 0, 2, 1, 15, 8, 6, 5] # order used to find one of each number to reconstruct
	
	# Display original images
	for c in range(columns):
		# Draw the original digits
		reconstruction_pic[0 * 28:(0 + 1) * 28, c * 28:(c + 1) * 28] = \
			test_input[order[c]].reshape([28, 28])
	# Display reconstructed images
	for c in range(columns):
		# Draw the reconstructed digits
		reconstruction_pic[1 * 28:(1 + 1) * 28, c * 28:(c + 1) * 28] = \
			test_output[order[c]].reshape([28, 28])

tf.reset_default_graph()

# ----- SINGLE LAYER END -----





# ----- TWO LAYER START -----

# tf Graph input (only pictures)
X = tf.placeholder("float", [None, num_input])

weights = {
	'old_layer': tf.Variable(enc_h1),
	'encoder_h2': tf.Variable(tf.random_normal([num_hidden_1, num_hidden_2])),
	'decoder_h1': tf.Variable(tf.random_normal([num_hidden_2, num_hidden_1])),
}

biases = {
	'old_bias': tf.Variable(enc_b1),
	'encoder_b2': tf.Variable([float(0)]*num_hidden_2),
	'decoder_b1': tf.Variable([float(0)]*num_hidden_1),
}

# Building the encoder
def old_input(x):
	layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['old_layer']), biases['old_bias']))
	return layer_1

# Building the encoder
def encoder(x):
	layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h2']), biases['encoder_b2']))
	return layer_1

# Building the decoder
def decoder(x):
	layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']), biases['decoder_b1']))
	return layer_1

# Construct model
old_op = old_input(X)
encoder_op = encoder(old_op)
decoder_op = decoder(encoder_op)

# Prediction
y_pred = decoder_op
# Targets (Labels) are the input data.
y_true = old_op

# Define loss and optimizer, minimize the squared error

loss = tf.reduce_mean(abs(y_true - y_pred))

L2 = [tf.trainable_variables()[1], tf.trainable_variables()[2], tf.trainable_variables()[4], tf.trainable_variables()[5]]

optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss, var_list = L2)

#optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

for v in tf.trainable_variables(): # Print all tf variables
	print(v)

# Start Training
# Start a new TF session
with tf.Session() as sess:


	# Run the initializer
	sess.run(init)
	
	for v in tf.trainable_variables(): # Print all tf variables
		print(v)
	#print(tf.trainable_variables()[0].eval(sess)[0][0])
	
	# Training
	for i in range(1, num_steps+1):
		# Run optimization op (backprop) and cost op (to get loss value)
		training_target, l = sess.run([optimizer, loss], feed_dict={X: training_input})
		# Display logs per step
		if i % display_step == 0 or i == 1:
			print('Step %i: Loss: %f' % (i, l))

	#print(tf.trainable_variables()[0].eval(sess)[0][0])

tf.reset_default_graph()

# ----- TWO LAYER END -----


