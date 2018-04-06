# -*- coding: utf-8 -*-

"""
From Aymeric Damien code on github : https://github.com/aymericdamien/TensorFlow-Examples/
"""

from __future__ import division
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import csv
import tensorflow as tf

# Additionnal ANN packages

np.random.seed(1337)  # for reproducibility
from sklearn.model_selection import train_test_split


# ----- PARAMETERS -----

# Basic

learning_rate = 0.01
training_epochs = 1000
batch_size = 100
display_step = 1

# Network Parameters

n_hidden_1 = 150 # 1st layer number of neurons
n_hidden_2 = 100 # 2nd layer number of neurons
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)


# ----- CLASSES -----
class picture:
	""" Class to handle automatically the plot management.
	By using the defined classes you can automatically add vectors the general plot.
	Create multiple instances to have multiple plot frames
	NOT TESTED YET
	"""

	def __init__(self, x_dim, y_dim, total) :
		fig = plt.figure()
		x = x_dim
		y = y_dim
		rows = int( math.sqrt( total ) )
		if ( total % rows ) == 0 :
			columns = rows
		else :
			columns = rows + 1

		counter = 1

	def add_vector(self, vector) : 
		if total > counter and i * j == len( vector ) :
			# We start by transforming the vector to a matrix
			matrix = [ [ int( vector[ i + j * x] ) for i in range( 0, x ) ] for j in range( 0, y ) ]
			counter += 1
			fig.add_subplot(rows, columns, counter)
			plt.imshow( matrix )
		else :
			print( "WARNING : a vector have been ommited due to overtacked number of graphs or unapropriate vector length." )

	def show_pictures(self, vector) :
		plt.show()

# MLP Model

def multilayer_perceptron(x):
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer
		


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

# Convert to numpy arrays
pics = np.array( pics )
raw_cls = np.array( cls )

# Rewriting the cls : 3 -> 0010000000 (Useful for MLP classification)
cls = []
for j in raw_cls :
	temp = [ 0 for k in range( n_classes ) ]
	temp[j] = 1
	cls.append( temp )


# Use pics for pictures and cls to create data parts

X_train, X_test, Y_train, Y_test = train_test_split(pics, cls, test_size=0.2, random_state=0) # As we have 8000 for trn and 2000 for tst


# ----- DATA READY -----

# ----- MODELING -----

# tf Graph input
X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_classes])

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}

biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
logits = multilayer_perceptron(X)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

train_op = optimizer.minimize(loss_op)

# Initializing the variables
init = tf.global_variables_initializer()


# ----- LEARNING -----
with tf.Session() as sess:
	sess.run(init)

	# Training cycle
	for epoch in range(training_epochs):
		avg_cost = 0.
		total_batch = int( len( X_train ) / batch_size )

	# Loop over all batches
	for i in range(total_batch):
		X_train_part = X_train[ i * 100 : (i+1) * 100 ]
		Y_train_part = Y_train[ i * 100 : (i+1) * 100 ]

		# Run optimization op (backprop) and cost op (to get loss value)
		_, c = sess.run([train_op, loss_op], feed_dict={X: X_train_part,
				                            Y: Y_train_part})
		# Compute average loss
		avg_cost += c / total_batch

	# For printing purposes
	if epoch % display_step == 0:
		print("Epoch:", '%04d' % (epoch+1), "cost={:.9f}".format(avg_cost))

	print("Optimization Finished!")

	pred = tf.nn.softmax(logits)  # Apply softmax to logits
	correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))

	# Calculate accuracy
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# ----- TEST -----

	print("Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))

# ----- PLOT -----











