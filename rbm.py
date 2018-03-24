# -*- coding: utf-8 -*-
from __future__ import division
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import csv

# Additionnal ANN packages

np.random.seed(1337)  # for reproducibility
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics.classification import accuracy_score

from dbn import SupervisedDBNClassification


# ----- CLASSES -----
class picture:
	""" Class to handle automatically the plot management.
	By using the defined classes you can automatically add vectors the general plot.
	Create multiple instances to have multiple plot frames
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
cls = np.array( cls )


# Use pics for pictures and cls to create data parts

X_train, X_test, Y_train, Y_test = train_test_split(pics, cls, test_size=0.2, random_state=0) # As we have 8000 for trn and 2000 for tst

# ----- DATA READY -----

# ----- TRAINING -----

classifier = SupervisedDBNClassification(hidden_layers_structure=[50],
                                         learning_rate_rbm=0.05,
                                         learning_rate=0.1,
                                         n_epochs_rbm=10,
                                         n_iter_backprop=50,
                                         batch_size=32,
                                         activation_function='relu',
                                         dropout_p=0.2)
classifier.fit(X_train, Y_train)

# Save the model
classifier.save('model.pkl')

# Restore it
classifier = SupervisedDBNClassification.load('model.pkl')

# ----- TEST -----

Y_pred = classifier.predict(X_test)
print('Done.\nAccuracy: %f' % accuracy_score(Y_test, Y_pred))

# ----- PLOT -----











