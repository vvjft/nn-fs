#### Imports ####
import numpy as np
import optuna
import argparse
import configparser
import logging
from datetime import datetime
from data_loader import mnist_loader

#### Utility functions ####
'''Reads parameters from the config file'''
def read_config(config_path, section='hyperparameters'):
    config = configparser.ConfigParser()
    config.read(config_path)
    if section not in config:
        raise ValueError(f"Section '{section}' not found in the config file.")
    
    elif section == 'hyperparameters':
        config_values = {
            'layers': config.get(section, 'layers'),
            'epochs': config.getint(section, 'epochs'),
            'batch_size': config.getint(section, 'batch_size'),
            'eta': config.getfloat(section, 'eta'),
            'lmbda': config.getfloat(section, 'lmbda'),
            'cost': config.get(section, 'cost'),
            'activation': config.get(section, 'activation'),

        }
    
    elif section == 'options':
        config_values = {
            'download_data': config.getboolean(section, 'download_data'),
            'show_history': config.getboolean(section, 'show_history'),
            'n_trials': config.getint(section, 'n_trials'),
            'n_augment': config.getint(section, 'n_augment')
        }
    
    return config_values

"""Activation functions and derivatives with respect to z"""
def sigmoid(z):
    return 1/(1+np.exp(-z))

def sigmoid_derivative(z):
    return sigmoid(z)*(1-sigmoid(z))

def relu(z):
    return np.maximum(0,z)

def relu_derivative(z):
    return np.where(z > 0, 1.0, 0.0)

"""Cost functions and derivatives with respect to activated neuron (a)"""
def cross_entropy(y,a):
    return np.sum(np.nan_to_num(-y*np.log(a) - (1-y)*np.log(1-a)))

def cross_entropy_derivative(y,a):
    return np.nan_to_num((a-y) / (a*(1-a)))

def quadratic(y,a):
    return np.sum(0.5*(y-a)**2)

def quadratic_derivative(y,a):
    return a-y

activation_functions = {'sigmoid': (sigmoid, sigmoid_derivative), 'relu': (relu, relu_derivative)}
cost_functions = {'quadratic': (quadratic, quadratic_derivative), 'cross_entropy': (cross_entropy, cross_entropy_derivative)}

#### MLP class ####
class MLP:
    def __init__(self, layers, cost_function = 'cross_entropy', activation_function = 'sigmoid', dropout_rate=0.5):
        self.L = layers
        self.W = [np.random.randn(x,y)/np.sqrt(y) for x,y in zip(self.L[1:],self.L[0:-1])] # divide by stadard deviation to avoid saturation
        self.B = [np.random.randn(x,1) for x in self.L[1:]]

        if cost_function in cost_functions:
            self.cost_function, self.cost_function_derivative = cost_functions[cost_function]
        else:
            raise ValueError(f"Invalid cost function: {cost_function}")
        if activation_function in activation_functions:
            self.activation_function, self.activation_derivative = activation_functions[activation_function]
        else:
            raise ValueError(f"Invalid activation function: {activation_function}")
        
        self.dropout_rate=dropout_rate

    def feedforward(self, X):
        A = X
        for w, b in zip(self.W, self.B):
            Z = np.dot(w, A) + b
            A = self.activation_function(Z)
        return A

    def fit(self, train_set, batch_size, epochs, eta, lmbda, patience=10, valid_set=None, show_history=True, trial=None):
        # Set-up
        X_train, Y_train = train_set
        num_training_examples = X_train.shape[1]
        if valid_set is not None:
            X_valid, Y_valid = valid_set
            print("Tracking progress on the validation set")
        else:
            print("Tracking progress on the training set")
        best_acc, no_progress_count, best_epoch = 0.0, np.Inf, 0

        # Training
        for epoch in range(epochs):
            X_batches = np.array_split(X_train, X_train.shape[1] // batch_size, axis=1)
            Y_batches = np.array_split(Y_train, Y_train.shape[1] // batch_size, axis=1)
            for X_batch, Y_batch in zip(X_batches, Y_batches):
                
                self.__SGD(X_batch, Y_batch, eta, lmbda, num_training_examples)

            # Tracking progress
            best_W, best_B = self.W.copy(), self.B.copy() # intialize best parameters as current ones
            if valid_set is not  None:
                acc, cost = self.__track_progress(X_valid, Y_valid)
                acc_train, cost_train = self.__track_progress(X_train, Y_train)
            else:
                acc, cost = self.__track_progress(X_train, Y_train)

            if acc > best_acc:
                best_W, best_B = self.W.copy(), self.B.copy()
                best_acc = acc
                best_epoch = epoch
                no_progress_count = 0
            else:
                no_progress_count += 1
            
            if no_progress_count > patience:
                self.W, self.B = best_W, best_B
                print(f"Early stopping: no improvement on validation set for {patience} epochs. Saving parameters from epoch {best_epoch}.")
                break
            elif show_history: 
                if valid_set is not None:
                    now = datetime.now().strftime("%Y-%m-%d %H:%M")
                    print(f"[{now}] epoch: {epoch}, ACC_val: {acc}, cost_val: {cost}, ACC_train: {round(acc_train,4)}, cost_train: {round(cost_train,4)}, no_progress_count: {no_progress_count}")
                else:    
                    print(f"epoch: {epoch}, ACC: {acc}, cost: {cost}")
        
            # Prune unpromising trial (only for hyperparameters tuning) 
            if trial:
                trial.report(acc, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()
                
        self.best_acc = best_acc 
        self.__track_progress(X_train, Y_train)

    def __SGD(self, X_batch, Y_batch, eta, lmbda, num_training_examples):
        nabla_B = [np.zeros(b.shape) for b in self.B]
        nabla_W = [np.zeros(w.shape) for w in self.W]
        for i in range(X_batch.shape[1]):
            a = X_batch[:,i].reshape(-1,1)
            y = Y_batch[:,i].reshape(-1,1)
            delta_nabla_W, delta_nabla_B = self.__backprop(a, y)
            nabla_B = [nb+dnb for nb, dnb in zip(nabla_B, delta_nabla_B)] 
            nabla_W = [nw+dnw for nw, dnw in zip(nabla_W, delta_nabla_W)]
        self.W = [w-eta*nw/X_batch.shape[1] - eta*lmbda*w/num_training_examples for w, nw in zip(self.W, nabla_W)] # L2 regularization
        #self.W = [w-eta*nw/X_batch.shape[1] - eta*w/num_training_examples for w, nw in zip(self.W, nabla_W)]
        self.B = [b-eta*nb/X_batch.shape[1] for b, nb in zip(self.B, nabla_B)]
  
    def __backprop(self, a, y):
        """ Updates network's weights and biases by applying backpropagation. """
        Z=[]
        A=[a]
        # forward pass
        for i in range(len(self.W)):
            z = np.dot(self.W[i],A[-1])+self.B[i]
            a=self.activation_function(z)
            Z.append(z)
            A.append(a)
        A, masks = self.dropout(A, 0.5)
        (delta_nabla_W, delta_nabla_B) = self.__get_gradients(y, A, Z, masks)
        return (delta_nabla_W, delta_nabla_B)
    
    def dropout(self, A, rate=0.5):      
            masks = [(np.random.random(np.shape(a)) > rate).astype(np.float32) for a in A[1:len(self.W)]]
            for i, mask in zip(range(1, len(self.W)), masks):
                A[i]=A[i]*mask/(1-rate)
            return A, masks
    
    def __get_gradients(self, y, A, Z, masks):
        def delta(y,x,z):
            return self.cost_function_derivative(y,x)*self.activation_derivative(z) 
                
        D = [delta(y,A[-1],Z[-1])] # eq. (1)
        for i in range(1,len(Z)):
            D.insert(0, np.dot(self.W[-i].T,D[0])*self.activation_derivative(Z[-i-1])) # eq. (2)
        for i, mask in zip(range(len(masks)),masks):
            D[i]*=mask
        B_grad = D # eq. (3)
        W_grad = []
        for a,d in zip(A[0:-1],D):
            W_grad.append(np.dot(d,np.transpose(a))) # eq. (4)
        return (W_grad, B_grad)

    def __track_progress(self, X, Y):
        """Evaluates accuracy and cost and the end of each epoch."""
        acc = self.evaluate(X, Y)
        cost = round(self.cost_function(Y, self.feedforward(X))/X.shape[1], 4)
        return acc, cost

    def evaluate(self, X, Y):
        correct_predictions = 0
        predictions = self.feedforward(X)
        for prediction, y in zip(predictions.T, Y.T):
            if np.argmax(prediction) == np.argmax(y):
                correct_predictions += 1
        return correct_predictions/(X.shape[1])
    
    def save(self, path): 
        np.savez(f'{path}/weights.npz', *self.W)
        np.savez(f'{path}/biases.npz', *self.B)

    def load(self, path):
        weights_data = np.load(f'{path}/weights.npz')
        biases_data = np.load(f'{path}/biases.npz')
        self.W = [weights_data[key] for key in weights_data.files]
        self.B = [biases_data[key] for key in biases_data.files]

#### Main section ####
def learn(net, data, epochs, batch_size, eta, lmbda, show_history, trial=None):
    train, valid, test = data
    net = net
    net.fit(train_set=train, batch_size=batch_size, epochs=epochs, eta=eta, lmbda=lmbda, patience=10, valid_set=valid, show_history=show_history, trial=trial)
    net.save('./data/')
    acc = net.evaluate(*test)
    print(f'Accuracy: {acc}')
    return acc

def load_weights_and_biases(net, data, path='./data'):
    train, valid, test = data
    net = net
    net.load(path)

    acc = net.evaluate(*test)
    print(f'Accuracy: {acc}')
    return acc

def tune_hyperparameters(n_trials, data, epochs): 
    def objective(trial):
        n_neurons = trial.suggest_int('n_neurons', 1, 1500)
        eta = trial.suggest_float('eta', 1e-3, 0.5)
        lmbda = trial.suggest_float('lmbda', 1e-3, 10)
        batch_size = trial.suggest_int('batch_size', 10, 100)
        dropout_rate = trial.suggest_float('dropout_rate', 0, 0.8)
        #cost_function = trial.suggest_categorical('cost_function', ['cross_entropy', 'quadratic'])
        #activation_function = trial.suggest_categorical('activation_function', ['sigmoid'])

        net = MLP(layers=[784, n_neurons, 10], cost_function='cross_entropy', activation_function='sigmoid', dropout_rate=dropout_rate)
        acc_test = learn(net, data, epochs=epochs, batch_size=batch_size, eta=eta, lmbda=lmbda, show_history=True, trial=trial)
        acc_valid = net.best_acc
        
        return acc_valid
    
    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.SuccessiveHalvingPruner())
    study.optimize(objective, n_trials=n_trials)
    print("Number of finished trials: ", len(study.trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

if __name__ == '__main__':
    # TO DO: add cupy, rename show_history
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train a neural network on MNIST dataset.')
    parser.add_argument('command', choices=['learn', 'load', 'tune'], help='What to do.')
    parser.add_argument('--layers', nargs='+', type=int, help='Number of neurons in each layer.')
    parser.add_argument('--epochs', type=int, help='Number of epochs to train.')
    parser.add_argument('--batch_size', type=int, help='Batch size for training.')
    parser.add_argument('--eta', type=float, help='Learning rate.')
    parser.add_argument('--lmbda', type=float, help='Regularization parameter.')
    parser.add_argument('--cost', choices=['cross_entropy', 'quadratic'], help='Cost function.')
    parser.add_argument('--activation', choices=['sigmoid', 'relu'], help='Activation function.')
    parser.add_argument('--show_history', type=bool, help='Show training history.')
    parser.add_argument('--download_data', type=bool, help='Download the dataset.')
    parser.add_argument('--n_trials', type=int, help='Number of trials for hyperparameter tuning.')
    parser.add_argument('--n_augment', type=int, help='Number of augmentations for the training set (rotations and shifting).')
    args = parser.parse_args()

    # Load default values from config file
    config_hyperparameters = read_config('config.ini', section='hyperparameters')
    layers_str = config_hyperparameters['layers']
    default_layers = [int(x) for x in layers_str.split(',')] # parse string to list of integers
    default_epochs = config_hyperparameters['epochs']
    default_batch_size = config_hyperparameters['batch_size']
    default_eta = config_hyperparameters['eta']
    default_lmbda = config_hyperparameters['lmbda']
    default_cost = config_hyperparameters['cost']
    default_activation = config_hyperparameters['activation']

    config_options = read_config('config.ini', section='options')
    show_history = config_options['show_history']
    download_data = config_options['download_data']
    default_n_trials = config_options['n_trials']
    default_n_augment = config_options['n_augment']

    # Override defaults with command-line arguments if provided
    layers = args.layers if args.layers is not None else default_layers
    epochs = args.epochs if args.epochs is not None else default_epochs
    batch_size = args.batch_size if args.batch_size is not None else default_batch_size
    eta = args.eta if args.eta is not None else default_eta
    lmbda = args.lmbda if args.lmbda is not None else default_lmbda
    activation = args.activation if args.activation is not None else default_activation
    cost = args.cost if args.cost is not None else default_cost
    show_history = args.show_history if args.show_history is not None else show_history
    download_data = args.download_data if args.download_data is not None else download_data
    n_trials = args.n_trials if args.n_trials is not None else default_n_trials
    n_augment = args.n_augment if args.n_augment is not None else default_n_augment
    
    # Load the data
    data_loader = mnist_loader(download=download_data, path='./data', n_augment=n_augment)
    data = data_loader.train, data_loader.valid, data_loader.test

    # Initialize the network
    net = MLP(layers=layers, cost_function=cost, activation_function=activation)

    if args.command == 'learn':
        learn(net=net,
              data=data, 
              epochs=epochs, 
              batch_size=batch_size, 
              eta=eta, 
              lmbda=lmbda, 
              show_history=show_history)
    elif args.command == 'load':
        load_weights_and_biases(net=net, data=data)
    elif args.command == 'tune':
        tune_hyperparameters(n_trials=n_trials, data=data, epochs=epochs)