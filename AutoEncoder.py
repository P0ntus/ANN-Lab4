# -*- coding: utf-8 -*-
from __future__ import division
from random import shuffle
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import csv
import tensorflow as tf

# ----- FUNCTIONS START -----

# Change targets of form 0-9 to something like 0, 0, 0, 1, 0, 0, 0, 0, 0, 0
def binary_target(input) :
	output = [0]*10
	output[input] = 1
	return output

# ----- FUNCTIONS END -----

np.set_printoptions(threshold=np.nan) #Always print the whole matrix

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

print(training_target.shape, test_target.shape)

# Training Parameters
learning_rate = 0.01
num_steps = 10000
#batch_size = 256

display_step = 50
examples_to_show = 10

# Network Parameters
num_hidden_1 = 256 # 1st layer num features
num_hidden_2 = 128 # 2nd layer num features (the latent dim)
num_input = 784 # MNIST data input (img shape: 28*28)

# tf Graph input (only pictures)
X = tf.placeholder("float", [None, num_input])

weights = {
    'encoder_h1': tf.Variable(tf.random_normal([num_input, num_hidden_1])),
    'encoder_h2': tf.Variable(tf.random_normal([num_hidden_1, num_hidden_2])),
    'decoder_h1': tf.Variable(tf.random_normal([num_hidden_2, num_hidden_1])),
    'decoder_h2': tf.Variable(tf.random_normal([num_hidden_1, num_input])),
}
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([num_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([num_hidden_2])),
    'decoder_b1': tf.Variable(tf.random_normal([num_hidden_1])),
    'decoder_b2': tf.Variable(tf.random_normal([num_input])),
}

# Building the encoder
def encoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                   biases['encoder_b1']))
    # Encoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                   biases['encoder_b2']))
    return layer_2

# Building the decoder
def decoder(x):
    # Decoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                                   biases['decoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                   biases['decoder_b2']))
    return layer_2

# Construct model
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

# Prediction
y_pred = decoder_op
# Targets (Labels) are the input data.
y_true = X

# Define loss and optimizer, minimize the squared error
loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start Training
# Start a new TF session
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)
			
	# Training
    for i in range(1, num_steps+1):
        # Prepare Data
        # Get the next batch of MNIST data (only images are needed, not labels)
        batch_x = training_input
        _ = training_target

        # Run optimization op (backprop) and cost op (to get loss value)
        _, l = sess.run([optimizer, loss], feed_dict={X: batch_x})
        # Display logs per step
        if i % display_step == 0 or i == 1:
            print('Step %i: Minibatch Loss: %f' % (i, l))


    # Testing
    # Encode and decode images from test set and visualize their reconstruction.
    n = 4
    canvas_orig = np.empty((28 * n, 28 * n))
    canvas_recon = np.empty((28 * n, 28 * n))
    for i in range(n):
        # MNIST test set
        batch_x = test_input
        _ = test_target
        # Encode and decode the digit image
        g = sess.run(decoder_op, feed_dict={X: batch_x})

        # Display original images
        for j in range(n):
            # Draw the original digits
            canvas_orig[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = \
                batch_x[j].reshape([28, 28])
        # Display reconstructed images
        for j in range(n):
            # Draw the reconstructed digits
            canvas_recon[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = \
                g[j].reshape([28, 28])
    '''
    print("Original Images")
    plt.figure(figsize=(n, n))
    plt.imshow(canvas_orig, origin="upper", cmap="gray")
    plt.show()
    '''
    print("Reconstructed Images")
    plt.figure(figsize=(n, n))
    plt.imshow(canvas_recon, origin="upper", cmap="gray")
plt.show()
