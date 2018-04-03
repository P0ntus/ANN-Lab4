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

# mnist = input_data.read_data_sets(pics)

# ----- DATA READY -----

# Split data
training_input = np.array(pics[0:8000])
test_input = np.array(pics[8000:10000])

training_target = np.array(cls[0:8000])
test_target = np.array(cls[8000:10000])

# Training Parameters
learning_rate = 10
num_steps = 100

display_step = 1

# Network Parameters
num_input = 784
num_hidden_1 = 256
num_hidden_2 = 128
num_hidden_3 = 64





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

for v in tf.trainable_variables(): # Print all tf variables
	print(v)

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

# Define loss and optimizer, minimize the squared error

loss = tf.reduce_mean(abs(y_true - y_pred))

optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

training_encoded_output = []
test_encoded_output = []

# Start Training
# Start a new TF session
with tf.Session() as sess:

	# Run the initializer
	sess.run(init)
	
	# Training
	for i in range(1, num_steps+1):
		# Run optimization op (backprop) and cost op (to get loss value)
		training_target, l = sess.run([optimizer, loss], feed_dict={X: training_input})
		# Display logs per step
		if i % display_step == 0 or i == 1:
			print('Step %i: Loss: %f' % (i, l))

	training_encoded_output = sess.run(encoder_op, feed_dict={X: training_input})
	test_encoded_output = sess.run(encoder_op, feed_dict={X: test_input})
	
tf.reset_default_graph()

# ----- SINGLE LAYER END -----





# ----- CLASSIFICATION START -----
training_target = np.array(cls[0:8000])
test_target = np.array(cls[8000:10000])

feature_columns = [tf.feature_column.numeric_column("x", shape=[16, 16])]

classifier = tf.estimator.DNNClassifier(
	feature_columns=feature_columns,
	hidden_units=[256],
	optimizer=tf.train.AdamOptimizer(),
	n_classes=10,
	dropout=0.1,
	model_dir="./tmp/mnist_model3"
)

train_input_fn = tf.estimator.inputs.numpy_input_fn(
	x={"x": training_encoded_output},
	y=training_target,
	num_epochs=None,
	batch_size=50,
	shuffle=True
)

classifier.train(input_fn=train_input_fn, steps=10000)

test_input_fn = tf.estimator.inputs.numpy_input_fn(
	x={"x": test_encoded_output},
	y=test_target,
	num_epochs=1,
	shuffle=False
)

accuracy_score = classifier.evaluate(input_fn=test_input_fn)["accuracy"]
print("\nTest Accuracy: {0:f}%\n".format(accuracy_score*100))
# ----- CLASSIFICATION END -----
