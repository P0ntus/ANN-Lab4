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

#from dbn.tensorflow import SupervisedDBNClassification
#from dbn import BinaryRBM
from rbm_config import RBMConfig
from dbn_config import DBNConfig


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
		
def read_data():
  # ----- DATAGEN PART -----

  # Array with 10000 (8000 trn and 2000 tst) of 784-dim vectors representing matrices of 28x28
  os.chdir( os.path.dirname(os.path.abspath(__file__)) )
  t_trn_f = open("binMNIST_data/bindigit_trn.csv", "r")
  reader = csv.reader(t_trn_f)
  X_train = np.array([ [int(row[i]) for i in range( 0, len(row) ) ] for row in reader])
  t_trn_f.close()

  t_tst_f = open("binMNIST_data/bindigit_tst.csv", "r")
  reader = csv.reader(t_tst_f)
  X_test = np.array([ [int(row[i]) for i in range( 0, len(row) ) ] for row in reader])
  t_tst_f.close()

  # Matrix of target classifications
  trn_f = open("binMNIST_data/targetdigit_trn.csv", "r")
  reader = csv.reader(trn_f)
  Y_train = np.array([ int(row[0]) for row in reader])
  trn_f.close()

  trn_f = open("binMNIST_data/targetdigit_tst.csv", "r")
  reader = csv.reader(trn_f)
  Y_test = np.array([ int(row[0]) for row in reader])
  trn_f.close()

  return X_train, X_test, Y_train, Y_test


X_train, X_test, Y_train, Y_test = read_data()


task = 3

if task == 1:
  # plotting reconstruction error
  config = 'units'
  rbm_configs = []
  if config == 'learning_rate':
    rbm_configs = [
      RBMConfig(X_train, X_test, n_hidden_units=150, learning_rate=0.2, plot_test=True, plot_label="0.2 learning rate"),
      RBMConfig(X_train, X_test, n_hidden_units=150, learning_rate=0.4, plot_test=True, plot_label="0.4 learning rate"),
      RBMConfig(X_train, X_test, n_hidden_units=150, learning_rate=0.8, plot_test=True, plot_label="0.8 learning rate"),
      RBMConfig(X_train, X_test, n_hidden_units=150, learning_rate=1.6, plot_test=True, plot_label="1.6 learning rate")
    ]
  elif config == 'units':
    rbm_configs = [
      RBMConfig(X_train, X_test, n_hidden_units=50, learning_rate=0.4, plot_test=True, plot_label="50 hidden units"),
      RBMConfig(X_train, X_test, n_hidden_units=75, learning_rate=0.4, plot_test=True, plot_label="75 hidden units"),
      RBMConfig(X_train, X_test, n_hidden_units=100, learning_rate=0.4, plot_test=True, plot_label="100 hidden units"),
      RBMConfig(X_train, X_test, n_hidden_units=150, learning_rate=0.4, plot_test=True, plot_label="150 hidden units")
    ]

  if rbm_configs:
    for rbm_config in rbm_configs:
      rbm_config.run()
    plt.legend(loc='best', fancybox=True, framealpha=0.5)
    plt.show()

elif task == 2:
  # plotting reconstructed digits
  hidden_units = 150
  config = RBMConfig(X_train, X_test, learning_rate=0.4, n_hidden_units=hidden_units, load=True)
  config.plot_digits()

elif task == 3:
  # plotting reconstructed weights
  hidden_units = 100
  config = RBMConfig(X_train, X_test, learning_rate=0.4, n_hidden_units=hidden_units, load=True)
  config.plot_weights(10)

elif task == 4:
  # training dbn's and getting classification accuracy
  configs = [
    #DBNConfig(X_train, Y_train, X_test, Y_test, hidden_layers_structure=[], load=True),
    DBNConfig(X_train, Y_train, X_test, Y_test, hidden_layers_structure=[150], load=False),
    DBNConfig(X_train, Y_train, X_test, Y_test, hidden_layers_structure=[150, 150], load=False),
    DBNConfig(X_train, Y_train, X_test, Y_test, hidden_layers_structure=[150, 125], load=False),
    DBNConfig(X_train, Y_train, X_test, Y_test, hidden_layers_structure=[150, 100], load=False),
    DBNConfig(X_train, Y_train, X_test, Y_test, hidden_layers_structure=[150, 150, 150], load=False),
    DBNConfig(X_train, Y_train, X_test, Y_test, hidden_layers_structure=[150, 150, 100], load=False),
    DBNConfig(X_train, Y_train, X_test, Y_test, hidden_layers_structure=[150, 125, 125], load=False),
    DBNConfig(X_train, Y_train, X_test, Y_test, hidden_layers_structure=[150, 125, 100], load=False),
    DBNConfig(X_train, Y_train, X_test, Y_test, hidden_layers_structure=[150, 100, 50], load=False)
  ]

  for config in configs:
    config.run()

elif task == 5:
  # plotting reconstructed weights
  config = DBNConfig(X_train, Y_train, X_test, Y_test, hidden_layers_structure=[150, 125, 125], load=True)
  config.run()
  config.plot_weights(0, 25)

