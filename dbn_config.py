from dbn.cool_models import CoolSupervisedDBNClassification
import matplotlib.pyplot as plt
from timeit import default_timer as timer
import os.path
from sklearn.metrics.classification import accuracy_score

class DBNConfig:
  def __init__(self,
               X_train,
               Y_train,
               X_test,
               Y_test,
               hidden_layers_structure=[100, 100],
               learning_rate=0.4,
               learning_rate_rbm=0.4,
               batch_size=32,
               load=False):

    self.X_train = X_train
    self.Y_train = Y_train
    self.X_test = X_test
    self.Y_test = Y_test
    self.hidden_layers_structure = hidden_layers_structure
    self.learning_rate = learning_rate
    self.learning_rate_rbm = learning_rate_rbm
    self.batch_size = batch_size
    self.load = load

    self.n_epochs_rbm = 20

    if self.load and os.path.isfile(self.filename()):
      self.model = CoolSupervisedDBNClassification.load(self.filename())
    else:
      self.load = False
      self.model = CoolSupervisedDBNClassification(hidden_layers_structure=hidden_layers_structure,
                                                   learning_rate=learning_rate,
                                                   learning_rate_rbm=learning_rate_rbm,
                                                   n_epochs_rbm=self.n_epochs_rbm,
                                                   batch_size=batch_size)

  def filename(self):
    return ''.join([
      'models/',
      'dbn_n-', ','.join(map(lambda x: str(x), self.hidden_layers_structure)),
      '_rate-', str(self.learning_rate),
      '_rbmrate-', str(self.learning_rate_rbm),
      '_size-', str(self.batch_size),
      '_epochs-', str(self.n_epochs_rbm),
      '.pkl'])

  def run(self):
    self.execution_time = 0
    if not self.load:
      start = timer()
      self.model.fit(self.X_train, self.Y_train)
      end = timer()
      self.execution_time = end - start;
      print(self.filename() + " learning time (s): " + str(self.execution_time))
      self.model.save(self.filename())

    Y_pred = self.model.predict(self.X_test)
    print(self.filename() + 'Done.\nAccuracy: %f' % accuracy_score(self.Y_test, Y_pred))

  def plot_weights(self, layer, columns=0):
    self.model.plot_weights(layer, columns)

