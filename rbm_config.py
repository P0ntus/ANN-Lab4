from dbn.cool_models import CoolBinaryRBM
import matplotlib.pyplot as plt
from timeit import default_timer as timer
import os.path

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
               plot_label="Please fill in this yourself",
               load=False):

    self.X_train = X_train
    self.X_test = X_test
    self.n_hidden_units = n_hidden_units
    self.learning_rate = learning_rate
    self.batch_size = batch_size
    self.plot_training = plot_training
    self.plot_test = plot_test
    self.plot_label = plot_label
    self.load = load

    self.n_epochs = 20

    if self.load and os.path.isfile(self.filename()):
      self.model = CoolBinaryRBM.load(self.filename())
    else:
      self.load = False
      self.model = CoolBinaryRBM(n_hidden_units=n_hidden_units,
                                 learning_rate=learning_rate,
                                 n_epochs=self.n_epochs,
                                 contrastive_divergence_iter=1,
                                 batch_size=batch_size,
                                 verbose=True,
                                 X_test=X_test)

  def filename(self):
    return ''.join([
      'models/',
      'rbm_n-', str(self.n_hidden_units),
      '_rate-', str(self.learning_rate),
      '_size-', str(self.batch_size),
      '_epochs-', str(self.n_epochs),
      '.pkl'])

  def plot(self):
    if self.plot_training:
      plt.plot(self.model.training_errors, label=self.plot_label + " (training)")
    if self.plot_test:
      plt.plot(self.model.test_errors, label=self.plot_label + " (test)")

  def run(self):
    self.execution_time = 0
    if not self.load:
      start = timer()
      self.model.fit(self.X_train)
      end = timer()
      self.execution_time = end - start;
      print(self.filename() + " learning time (s): " + str(self.execution_time))
      self.model.save(self.filename())
      self.plot()

  def plot_digits(self):
    self.model.plot_digits(self.X_test)

  def plot_weights(self, columns=0):
    self.model.plot_weights(columns)

