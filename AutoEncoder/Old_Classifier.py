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

tmp_training_target = np.array(cls[0:8000])
tmp_test_target = np.array(cls[8000:10000])

training_target = np.array(cls[0:8000])#[]
test_target = np.array(cls[8000:10000])#[]

#print(training_target)

#test_target = []

#print(training_target)


'''
for i in range(0, 8000):
	training_target.append(binary_target(tmp_training_target[i]))
training_target = np.array(training_target)

for i in range(0, 2000):
	test_target.append(binary_target(tmp_test_target[i]))
test_target = np.array(test_target)

training_combined = pics[0:8000]
training_combined.append(training_target)
training_combined = np.array(training_combined)

test_combined = pics[8000:10000]
test_combined.append(test_target)
test_combined = np.array(training_combined)
'''
#print(training_combined)
#print(training_combined[0:len(training_combined)-1])
#print(training_combined[len(training_combined)-1])

print(training_input[0].shape)
print(test_input[0].shape)

# Training Parameters
learning_rate = 10
num_steps = 1

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
	'classifier_h1': tf.Variable(tf.random_normal([num_input, 1])),
}

biases = {
	'encoder_b1': tf.Variable([float(0)]*num_hidden_1),
	'decoder_b1': tf.Variable([float(0)]*num_input),
	'classifier_b1': tf.Variable([float(0)]*1),
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

# Building the decoder
def classifier(x):
	# Decoder Hidden layer with sigmoid activation #1
	layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['classifier_h1']), biases['classifier_b1']))
	return layer_1

# Construct model
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)
classifier_op = classifier(decoder_op)

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

	#print(tf.trainable_variables()[2].eval(sess))
	
	# Training
	for i in range(1, num_steps+1):
		# Run optimization op (backprop) and cost op (to get loss value)
		training_target, l = sess.run([optimizer, loss], feed_dict={X: training_input})
		# Display logs per step
		if i % display_step == 0 or i == 1:
			print('Step %i: Loss: %f' % (i, l))

	#layer_1_h = tf.trainable_variables()[0].eval(sess)
	
	#print(training_input[0:2])
	
	training_encoded_output = sess.run(encoder_op, feed_dict={X: training_input})
	test_encoded_output = sess.run(encoder_op, feed_dict={X: test_input})
	
	#print(test_output[0:2])
	
	print(training_encoded_output[0].shape)
	print(test_encoded_output[0].shape)
	
	'''
	# Encode and decode the digit image
	test_output = sess.run(decoder_op, feed_dict={X: test_input})

	order = [18, 3, 7, 0, 2, 1, 15, 8, 6, 5] # order used to find one of each number to reconstruct
	
	for i in range(rows):
		# Display original images
		for j in range(columns):
			# Draw the original digits
			canvas_orig[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = \
				test_input[order[i*columns + j]].reshape([28, 28])
		# Display reconstructed images
		for j in range(columns):
			# Draw the reconstructed digits
			canvas_recon[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = \
				test_output[order[i*columns + j]].reshape([28, 28])

	weights = tf.trainable_variables()[1].eval(sess) # Get session variable	
	'''
	
tf.reset_default_graph()

# ----- SINGLE LAYER END -----

training_target = np.array(cls[0:8000])
test_target = np.array(cls[8000:10000])

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("/tmp/data/")
#mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

feature_columns = [tf.feature_column.numeric_column("x", shape=[16, 16])]
#feature_columns = [tf.feature_column.numeric_column("x", shape=[28, 28])]

classifier = tf.estimator.DNNClassifier(
	feature_columns=feature_columns,
	hidden_units=[128, 32],
	optimizer=tf.train.AdamOptimizer(1e-4),
	n_classes=10,
	dropout=0.1,
	model_dir="./tmp/mnist_model3"
)

train_input_fn = tf.estimator.inputs.numpy_input_fn(
	x={"x": training_encoded_output},
	#x={"x": training_input},
	y=training_target,
	num_epochs=None,
	batch_size=50,
	shuffle=True
)

classifier.train(input_fn=train_input_fn, steps=1000)

test_input_fn = tf.estimator.inputs.numpy_input_fn(
	x={"x": test_encoded_output},
	#x={"x": test_input},
	y=test_target,
	num_epochs=1,
	shuffle=False
)

accuracy_score = classifier.evaluate(input_fn=test_input_fn)["accuracy"]
print("\nTest Accuracy: {0:f}%\n".format(accuracy_score*100))
