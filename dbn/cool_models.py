import numpy as np
#from .tensorflow.models import BinaryRBM, SupervisedDBNClassification
from .models import BinaryRBM, SupervisedDBNClassification
import matplotlib.pyplot as plt
import math

class CoolBinaryRBM(BinaryRBM):

  def __init__(self,
               n_hidden_units=100,
               activation_function='sigmoid',
               optimization_algorithm='sgd',
               learning_rate=0.1,
               n_epochs=20,
               contrastive_divergence_iter=1,
               batch_size=32,
               verbose=True,
               X_test=[]):

    super().__init__(n_hidden_units,
                     activation_function,
                     optimization_algorithm,
                     learning_rate,
                     n_epochs,
                     contrastive_divergence_iter,
                     batch_size,
                     verbose)

    self.training_errors = []
    self.X_test = X_test
    self.test_errors = []

  def _reconstruction_error_helper(self, data):
    data_transformed = self.transform(data)
    data_reconstructed = self._reconstruct(data_transformed)
    return np.mean(np.mean(np.absolute(data_reconstructed - data), 1))

  def _compute_reconstruction_error(self, data):
    training_error = self._reconstruction_error_helper(data)
    test_error = self._reconstruction_error_helper(self.X_test)
    self.training_errors.append(training_error)
    self.test_errors.append(test_error)
    return training_error


  def plot_digits(self, data):
    columns = 10
    reconstruction_pic = np.empty((28 * 2, 28 * columns))
    order = [18, 3, 7, 0, 2, 1, 15, 8, 6, 5] # order used to find one of each number to reconstruct
    
    # Display original images
    for c in range(columns):
      reconstruction_pic[0 * 28:(0 + 1) * 28, c * 28:(c + 1) * 28] = \
        data[order[c]].reshape([28, 28])

    data_transformed = self.transform(data)
    data_reconstructed = self._reconstruct(data_transformed)

    # Display reconstructed images
    for c in range(columns):
      reconstruction_pic[1 * 28:(1 + 1) * 28, c * 28:(c + 1) * 28] = \
        data_reconstructed[order[c]].reshape([28, 28])

    plt.figure(figsize=(2, columns))
    plt.imshow(reconstruction_pic, origin="upper", cmap="gray")
    plt.show()

  def plot_weights(self, columns=0):
    size = self.n_hidden_units

    # Calculate number of rows and columns needed to display all weight matrices
    if columns == 0:
      columns = math.ceil(math.sqrt(size))
    rows = 1
    count = rows * columns
    while(count < size):
      rows += 1
      count = rows * columns

    # Add all weight matrices
    #weights ?
    weight_pics = np.empty((28 * rows, 28 * columns))
    for r in range(0, rows):
      for c in range(0, columns):
        if(r*columns+c >= size):
          break
        #print(r*columns+c)
        weight_pics[r * 28:(r + 1) * 28, c * 28:(c + 1) * 28] = \
            self.W[r*columns + c].reshape([28, 28])

    #print(rows)
    #print("Weight Images")
    plt.figure(figsize=(rows, columns))
    plt.imshow(weight_pics, origin="upper", cmap="gray")
    plt.show()

class CoolSupervisedDBNClassification(SupervisedDBNClassification):

  def __init__(self,
               hidden_layers_structure=[100, 100],
               learning_rate=0.1,
               learning_rate_rbm=0.1,
               n_epochs_rbm=20,
               batch_size=32):

    self.hidden_layers_structure = hidden_layers_structure

    super().__init__(hidden_layers_structure=hidden_layers_structure,
                     learning_rate=learning_rate,
                     learning_rate_rbm=learning_rate_rbm,
                     n_epochs_rbm=n_epochs_rbm,
                     batch_size=batch_size,
                     n_iter_backprop=100,
                     dropout_p=0.2)

  def plot_weights(self, layer, columns=0):
    W = self.unsupervised_dbn.rbm_layers[layer].W

    size = self.hidden_layers_structure[layer]

    # Calculate number of rows and columns needed to display all weight matrices
    if columns == 0:
      columns = math.ceil(math.sqrt(size))
    rows = 1
    count = rows * columns
    while(count < size):
      rows += 1
      count = rows * columns

    # Add all weight matrices
    #weights ?
    weight_pics = np.empty((28 * rows, 28 * columns))
    for r in range(0, rows):
      for c in range(0, columns):
        if(r*columns+c >= size):
          break
        #print(r*columns+c)
        weight_pics[r * 28:(r + 1) * 28, c * 28:(c + 1) * 28] = \
            W[r*columns + c].reshape([28, 28])

    #print(rows)
    #print("Weight Images")
    plt.figure(figsize=(rows, columns))
    plt.imshow(weight_pics, origin="upper", cmap="gray")
    plt.show()
