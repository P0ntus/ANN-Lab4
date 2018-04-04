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

def show_weights(weights):
	# Calculate number of rows and columns needed to display all weight matrices
	columns = math.ceil(math.sqrt(len(weights)))
	rows = 1
	count = rows * columns
	while(count < len(weights)):
		rows += 1
		count = rows * columns

	# Add all weight matrices
	weight_pics = np.empty((int(math.sqrt(len(weights[0]))) * rows, int(math.sqrt(len(weights[0]))) * columns))
	for r in range(0, rows):
		for c in range(0, columns):
			if(r*columns+c >= len(weights)):
				break
			weight_pics[r * int(math.sqrt(len(weights[0]))):(r + 1) * int(math.sqrt(len(weights[0]))), c * int(math.sqrt(len(weights[0]))):(c + 1) * int(math.sqrt(len(weights[0])))] = \
					weights[r*columns + c].reshape([int(math.sqrt(len(weights[0]))), int(math.sqrt(len(weights[0])))])

	#print("Weight Images using", 50 * run, "nodes")
	plt.figure(figsize=(rows, columns))
	plt.imshow(weight_pics, origin="upper", cmap="gray")
	plt.show()

def show_reconstruction(input, output, dimension):
	order = [18, 3, 7, 0, 2, 1, 15, 8, 6, 5] # order used to find one of each number to reconstruct
	reconstruction_pic = np.empty((dimension * 2, dimension * len(order)))
	# Display original images
	for c in range(len(order)):
		# Draw the original digits
		reconstruction_pic[0 * dimension:(0 + 1) * dimension, c * dimension:(c + 1) * dimension] = \
			input[order[c]].reshape([dimension, dimension])

	# Display reconstructed images
	for c in range(len(order)):
		# Draw the reconstructed digits
		reconstruction_pic[1 * dimension:(1 + 1) * dimension, c * dimension:(c + 1) * dimension] = \
			output[order[c]].reshape([dimension, dimension])
			
	plt.figure(figsize=(2, len(order)))
	plt.imshow(reconstruction_pic, origin="upper", cmap="gray")
	plt.show()
	
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
learning_rate = 100
#num_steps = 250
num_steps1 = 1500
num_steps2 = 50
num_steps3 = 30
classifier_steps = 10000

display_step = 25

# Network Parameters
num_input = 784
num_hidden_1 = 144
num_hidden_2 = 121
num_hidden_3 = 100

final_weights = []
reconstruction_pic = []
learning_curves = []

# ----- CLASSIFICATION START -----
training_target = np.array(cls[0:8000])
test_target = np.array(cls[8000:10000])

feature_columns = [tf.feature_column.numeric_column("x", shape=[int(math.sqrt(num_input)), int(math.sqrt(num_input))])]

classifier = tf.estimator.DNNClassifier(
	feature_columns=feature_columns,
	hidden_units=[num_input],
	optimizer=tf.train.AdamOptimizer(),
	n_classes=10,
	dropout=0.1,
	model_dir="./tmp/mnist_model0"
)

train_input_fn = tf.estimator.inputs.numpy_input_fn(
	x={"x": training_input},
	y=training_target,
	num_epochs=None,
	batch_size=50,
	shuffle=True
)

classifier.train(input_fn=train_input_fn, steps=classifier_steps)

test_input_fn = tf.estimator.inputs.numpy_input_fn(
	x={"x": test_input},
	y=test_target,
	num_epochs=1,
	shuffle=False
)

accuracy_score = classifier.evaluate(input_fn=test_input_fn)["accuracy"]
print("\nTest Accuracy: {0:f}%\n".format(accuracy_score*100))
# ----- CLASSIFICATION END -----


# ----- SINGLE LAYER START -----
learning_curve = []
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

# Define loss and optimizer, minimize the squared error

loss = tf.reduce_mean(abs(y_true - y_pred))

optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

layer_1_h = 0
layer_1_b = 0

training_encoded_output = []
test_encoded_output = []

# Start Training
# Start a new TF session
with tf.Session() as sess:

	# Run the initializer
	sess.run(init)
	
	for v in tf.trainable_variables(): # Print all tf variables
		print(v)
	#print(tf.trainable_variables()[2].eval(sess))
	
	# Training
	for i in range(1, num_steps1+1):
		# Run optimization op (backprop) and cost op (to get loss value)
		training_target, l = sess.run([optimizer, loss], feed_dict={X: training_input})
		learning_curve.append(l)

		# Display logs per step
		if i % display_step == 0 or i == 1:
			print('Step %i: Loss: %f' % (i, l))
			
	learning_curves.append(learning_curve)

	layer_1_h = tf.trainable_variables()[0].eval(sess)
	layer_1_b = tf.trainable_variables()[2].eval(sess)
	
	training_encoded_output = sess.run(encoder_op, feed_dict={X: training_input})
	test_encoded_output = sess.run(encoder_op, feed_dict={X: test_input})
	
	final_weights.append(tf.trainable_variables()[1].eval(sess))
	
	enc1 = np.copy(test_input)
	dec1 = sess.run(decoder_op, feed_dict={X: test_input})

tf.reset_default_graph()

# ----- CLASSIFICATION START -----
training_target = np.array(cls[0:8000])
test_target = np.array(cls[8000:10000])

feature_columns = [tf.feature_column.numeric_column("x", shape=[int(math.sqrt(num_hidden_1)), int(math.sqrt(num_hidden_1))])]

classifier = tf.estimator.DNNClassifier(
	feature_columns=feature_columns,
	hidden_units=[num_hidden_1],
	optimizer=tf.train.AdamOptimizer(),
	n_classes=10,
	dropout=0.1,
	model_dir="./tmp/mnist_model1"
)

train_input_fn = tf.estimator.inputs.numpy_input_fn(
	x={"x": training_encoded_output},
	y=training_target,
	num_epochs=None,
	batch_size=50,
	shuffle=True
)

classifier.train(input_fn=train_input_fn, steps=classifier_steps)

test_input_fn = tf.estimator.inputs.numpy_input_fn(
	x={"x": test_encoded_output},
	y=test_target,
	num_epochs=1,
	shuffle=False
)

accuracy_score = classifier.evaluate(input_fn=test_input_fn)["accuracy"]
print("\nTest Accuracy: {0:f}%\n".format(accuracy_score*100))
# ----- CLASSIFICATION END -----

# ----- SINGLE LAYER END -----





# ----- TWO LAYER START -----
learning_curve = []

# tf Graph input (only pictures)
X = tf.placeholder("float", [None, num_input])

weights = {
	'encoder_h1': tf.Variable(layer_1_h),
	'encoder_h2': tf.Variable(tf.random_normal([num_hidden_1, num_hidden_2])),
	'decoder_h1': tf.Variable(tf.random_normal([num_hidden_2, num_hidden_1])),
}

biases = {
	'encoder_b1': tf.Variable(layer_1_b),
	'encoder_b2': tf.Variable([float(0)]*num_hidden_2),
	'decoder_b1': tf.Variable([float(0)]*num_hidden_1),
}

# Building the encoder
def old_input(x):
	layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']), biases['encoder_b1']))
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

layer_2_h = 0
layer_2_b = 0

training_encoded_output = []
test_encoded_output = []

# Start Training
# Start a new TF session
with tf.Session() as sess:

	# Run the initializer
	sess.run(init)
	
	for v in tf.trainable_variables(): # Print all tf variables
		print(v)
	
	# Training
	for i in range(1, num_steps2+1):
		# Run optimization op (backprop) and cost op (to get loss value)
		training_target, l = sess.run([optimizer, loss], feed_dict={X: training_input})
		learning_curve.append(l)	
		# Display logs per step
		if i % display_step == 0 or i == 1:
			print('Step %i: Loss: %f' % (i, l))

	learning_curves.append(learning_curve)
			
	layer_2_h = tf.trainable_variables()[1].eval(sess)
	layer_2_b = tf.trainable_variables()[4].eval(sess)
	
	training_encoded_output = sess.run(encoder_op, feed_dict={X: training_input})
	test_encoded_output = sess.run(encoder_op, feed_dict={X: test_input})

	final_weights.append(tf.trainable_variables()[2].eval(sess))
	enc2 = sess.run(old_op, feed_dict={X: test_input})
	dec2 = sess.run(decoder_op, feed_dict={X: test_input})
	
tf.reset_default_graph()

# ----- CLASSIFICATION START -----
training_target = np.array(cls[0:8000])
test_target = np.array(cls[8000:10000])

feature_columns = [tf.feature_column.numeric_column("x", shape=[int(math.sqrt(num_hidden_2)), int(math.sqrt(num_hidden_2))])]

classifier = tf.estimator.DNNClassifier(
	feature_columns=feature_columns,
	hidden_units=[num_hidden_2],
	optimizer=tf.train.AdamOptimizer(),
	n_classes=10,
	dropout=0.1,
	model_dir="./tmp/mnist_model2"
)

train_input_fn = tf.estimator.inputs.numpy_input_fn(
	x={"x": training_encoded_output},
	y=training_target,
	num_epochs=None,
	batch_size=50,
	shuffle=True
)

classifier.train(input_fn=train_input_fn, steps=classifier_steps)

test_input_fn = tf.estimator.inputs.numpy_input_fn(
	x={"x": test_encoded_output},
	y=test_target,
	num_epochs=1,
	shuffle=False
)

accuracy_score = classifier.evaluate(input_fn=test_input_fn)["accuracy"]
print("\nTest Accuracy: {0:f}%\n".format(accuracy_score*100))
# ----- CLASSIFICATION END -----

# ----- TWO LAYER END -----





# ----- THREE LAYER START -----
learning_curve = []
# tf Graph input (only pictures)
X = tf.placeholder("float", [None, num_input])

weights = {
	'encoder_h1': tf.Variable(layer_1_h),
	'encoder_h2': tf.Variable(layer_2_h),
	'encoder_h3': tf.Variable(tf.random_normal([num_hidden_2, num_hidden_3])),
	'decoder_h1': tf.Variable(tf.random_normal([num_hidden_3, num_hidden_2])),
}

biases = {
	'encoder_b1': tf.Variable(layer_1_b),
	'encoder_b2': tf.Variable(layer_2_b),
	'encoder_b3': tf.Variable([float(0)]*num_hidden_3),
	'decoder_b1': tf.Variable([float(0)]*num_hidden_2),
}

# Building the encoder
def old_input(x):
	layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']), biases['encoder_b1']))
	layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']), biases['encoder_b2']))
	return layer_2

# Building the encoder
def encoder(x):
	layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h3']), biases['encoder_b3']))
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

L3 = [tf.trainable_variables()[2], tf.trainable_variables()[3], tf.trainable_variables()[6], tf.trainable_variables()[7]]

optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss, var_list = L3)

#optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

training_encoded_output = []
test_encoded_output = []

# Start Training
# Start a new TF session
with tf.Session() as sess:


	# Run the initializer
	sess.run(init)
	
	for v in tf.trainable_variables(): # Print all tf variables
		print(v)
	#print(tf.trainable_variables()[0].eval(sess)[0][0])
	
	# Training
	for i in range(1, num_steps3+1):
		# Run optimization op (backprop) and cost op (to get loss value)
		training_target, l = sess.run([optimizer, loss], feed_dict={X: training_input})
		learning_curve.append(l)	
		# Display logs per step
		if i % display_step == 0 or i == 1:
			print('Step %i: Loss: %f' % (i, l))
			
	learning_curves.append(learning_curve)

	training_encoded_output = sess.run(encoder_op, feed_dict={X: training_input})
	test_encoded_output = sess.run(encoder_op, feed_dict={X: test_input})

	final_weights.append(tf.trainable_variables()[3].eval(sess))
	enc3 = sess.run(old_op, feed_dict={X: test_input})
	dec3 = sess.run(decoder_op, feed_dict={X: test_input})
	
tf.reset_default_graph()

# ----- CLASSIFICATION START -----
training_target = np.array(cls[0:8000])
test_target = np.array(cls[8000:10000])

feature_columns = [tf.feature_column.numeric_column("x", shape=[int(math.sqrt(num_hidden_3)), int(math.sqrt(num_hidden_3))])]

classifier = tf.estimator.DNNClassifier(
	feature_columns=feature_columns,
	hidden_units=[num_hidden_3],
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

classifier.train(input_fn=train_input_fn, steps=classifier_steps)

test_input_fn = tf.estimator.inputs.numpy_input_fn(
	x={"x": test_encoded_output},
	y=test_target,
	num_epochs=1,
	shuffle=False
)

accuracy_score = classifier.evaluate(input_fn=test_input_fn)["accuracy"]
print("\nTest Accuracy: {0:f}%\n".format(accuracy_score*100))
# ----- CLASSIFICATION END -----

# ----- THREE LAYER END -----

for i in range(0, len(learning_curves)):
	plt.plot(learning_curves[i])

plt.legend(['1 layer', '2 layer', '3 layer'])
plt.ylabel('Reconstruction error')
plt.xlabel('Epochs')
plt.show()

show_reconstruction(enc1, dec1, int(math.sqrt(num_input)))
show_reconstruction(enc2, dec2, int(math.sqrt(num_hidden_1)))
show_reconstruction(enc3, dec3, int(math.sqrt(num_hidden_2)))

#show_weights(final_weights[0])
#show_weights(final_weights[1])
#show_weights(final_weights[2])