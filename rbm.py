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
from dbn.cool_models import CoolBinaryRBM


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

class RBMConfig:
  #def __init__(self, errors, n_hidden_units, training=True):
  def __init__(self,
               X_train,
               X_test,
               n_hidden_units=150,
               learning_rate=0.1,
               batch_size=32,
               plot_training=False,
               plot_test=False,
               plot_label="Please fill in this yourself"):

    self.X_train = X_train
    self.X_test = X_test
    self.n_hidden_units = n_hidden_units
    self.learning_rate = learning_rate
    self.batch_size = batch_size
    self.plot_training = plot_training
    self.plot_test = plot_test
    self.plot_label = plot_label

    self.model = CoolBinaryRBM(n_hidden_units=n_hidden_units,
                               learning_rate=learning_rate,
                               n_epochs=2,
                               contrastive_divergence_iter=1,
                               batch_size=batch_size,
                               verbose=True,
                               X_test=X_test)

  def plot(self):
    if self.plot_training:
      plt.plot(self.model.training_errors, label=self.plot_label + " (training)")
    if self.plot_test:
      plt.plot(self.model.test_errors, label=self.plot_label + " (test)")

  def run(self):
    self.model.fit(self.X_train)
    self.plot()

  def save(self, filename):
    self.model.save(filename)

config = 'units'
rbm_configs = []

if config == 'units':
  rbm_configs = [
    RBMConfig(X_train, X_test, n_hidden_units=50, plot_test=True, plot_label="50 hidden units"),
    RBMConfig(X_train, X_test, n_hidden_units=75, plot_test=True, plot_label="75 hidden units"),
    RBMConfig(X_train, X_test, n_hidden_units=100, plot_test=True, plot_label="100 hidden units"),
    RBMConfig(X_train, X_test, n_hidden_units=150, plot_test=True, plot_label="150 hidden units")
  ]
elif config == 'learning rate':
  rbm_configs = [
    RBMConfig(X_train, X_test, learning_rate=0.2, plot_test=True, plot_label="0.2 learning rate"),
    RBMConfig(X_train, X_test, learning_rate=0.4, plot_test=True, plot_label="0.4 learning rate"),
    RBMConfig(X_train, X_test, learning_rate=0.8, plot_test=True, plot_label="0.8 learning rate"),
    RBMConfig(X_train, X_test, learning_rate=1.6, plot_test=True, plot_label="1.6 learning rate")
  ]

if rbm_configs:
  for rbm_config in rbm_configs:
    rbm_config.run()
  plt.legend(loc='best', fancybox=True, framealpha=0.5)
  plt.show()

'''
final_config = RBMConfig(X_train, X_test, n_hidden_units=?, learning_rate=?)
final_config.run()
final_config.save('coolmodel.pkl')
model = CoolBinaryRBM.load('coolmodel.pkl')
model.plot_digits(X_test)
model.plot_weights()
'''



'''
classifier = SupervisedDBNClassification(hidden_layers_structure=[50],
                                         learning_rate_rbm=0.05,
                                         learning_rate=0.1,
                                         n_epochs_rbm=10,
                                         n_iter_backprop=50,
                                         batch_size=32,
                                         activation_function='sigmoid',
                                         dropout_p=0.2)
classifier.fit(X_train, Y_train)

# Save the model
classifier.save('model.pkl')

# Restore it
classifier = SupervisedDBNClassification.load('model.pkl')

# ----- TEST -----

Y_pred = classifier.predict(X_test)
print('Done.\nAccuracy: %f' % accuracy_score(Y_test, Y_pred))
'''
# ----- PLOT -----


