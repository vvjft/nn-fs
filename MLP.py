#### Imports ####
import numpy as np
import optuna 
import matplotlib.pyplot as plt

import argparse
import logging
import sys

from data_loader import mnist_loader

#### Utility functions ####
"""Activation functions and derivatives with respect to z"""
def sigmoid(z):
    return 1/(1+np.exp(-z))

def sigmoid_derivative(z):
    return sigmoid(z)*(1-sigmoid(z))

def ReLu(z):
    return np.maximum(0,z)

def ReLu_derivative(z):
    return np.where(z > 0, 1.0, 0.001)

"""Cost functions and derivatives with respect to a"""
def cross_entropy(y,a):
    return np.sum(np.nan_to_num(-y*np.log(a) - (1-y)*np.log(1-a)))

def cross_entropy_derivative(y,a):
    return np.nan_to_num((a-y) / (a*(1-a)))

def quadratic(y,a):
    return np.sum(0.5*(y-a)**2)

def quadratic_derivative(y,a):
    return a-y

activation_functions = {'sigmoid': (sigmoid, sigmoid_derivative), 'relu': (ReLu, ReLu_derivative)}
cost_functions = {'quadratic': (quadratic, quadratic_derivative), 'cross_entropy': (cross_entropy, cross_entropy_derivative)}

#### MLP class ####
class Network:
    def __init__(self, layers, cost_function = 'cross_entropy', activation_function = 'sigmoid'):
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

    def feedforward(self, X):
        A = X
        for w, b in zip(self.W, self.B):
            Z = np.dot(w, A) + b
            A = self.activation_function(Z)
        return A

    def fit(self, train_set, batch_size, epochs, eta, lmbda, patience=10, valid_set=None, show_history=True, visualize=False):
        # Set-up
        X_train, Y_train = train_set
        num_training_examples = X_train.shape[1]
        if valid_set is not None:
            X_valid, Y_valid = valid_set
            print("Tracking progress on the validation set:")
        else:
            print("Tracking progress on the training set:")
        best_acc, best_cost, no_progress_count = 0.0, np.Inf, 0

        # Training
        accs = {'valid': [], 'train': []}
        costs = {'valid': [], 'train': []}
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
                accs['valid'].append(acc)
                costs['valid'].append(cost)
                accs['train'].append(acc_train)
                costs['train'].append(cost_train)  
            else:
                acc, cost = self.__track_progress(X_train, Y_train)
                accs['train'].append(acc)
                costs['train'].append(cost)

            best_W, best_B = self.__update_best_parameters(acc, cost, best_acc, best_cost)
            no_progress_count = self.__check_no_progress(acc, cost, best_acc, best_cost, no_progress_count)

            if no_progress_count > patience or eta<1e-6:
                self.W, self.B = best_W, best_B
                print(f"Early stopping: no improvement on validation set for {patience} epochs. Saving parameters from epoch {epoch-patience}.")
                #break
            elif show_history:
                if valid_set is not None:
                    print(f"epoch: {epoch}, ACC_val: {acc}, cost_val: {cost}, ACC_train: {acc_train}, cost_train: {cost_train}, no_progress_count: {no_progress_count}")
                else:    
                    print(f"epoch: {epoch}, ACC: {acc}, cost: {cost}")

        self.__track_progress(X_train, Y_train, visualize=visualize, accs=accs, costs=costs)

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
        self.B = [b-eta*nb/X_batch.shape[1] for b, nb in zip(self.B, nabla_B)]

    def __backprop(self, a, y):
        """ Updates network's weights and biases by applying backpropagation. """
        Z=[]
        A=[a]
        for w,b in zip(self.W,self.B):
            z = np.dot(w,A[-1])+b
            a=self.activation_function(z)
            Z.append(z)
            A.append(a)
        (delta_nabla_W, delta_nabla_B) = self.__get_gradients(y, A, Z)
        return (delta_nabla_W, delta_nabla_B)

    def __get_gradients(self, y, A, Z):
        def delta(y,x,z):
            return self.cost_function_derivative(y,x)*self.activation_derivative(z)
            
        D = [delta(y,A[-1],Z[-1])] # eq. (1)
        for i in range(1,len(Z)):
            D.insert(0, np.dot(self.W[-i].T,D[0])*self.activation_derivative(Z[-i-1])) # eq. (2)
        B_grad = D # eq. (3)
        W_grad = []
        for a,d in zip(A[0:-1],D):
            W_grad.append(np.dot(d,a.T)) # eq. (4)
        return (W_grad, B_grad)

    def __update_best_parameters(self, acc, cost, best_acc, best_cost):
        if acc > best_acc or cost < best_cost:
            best_W, best_B = self.W.copy(), self.B.copy()
            if acc > best_acc:
                best_acc = acc
            if cost < best_cost:
                best_cost = cost
        return best_W, best_B
    
    def __check_no_progress(self, acc, cost, best_acc, best_cost, no_progress_count):
        if acc > best_acc or cost < best_cost:
            no_progress_count = 0
        else:
            no_progress_count += 1
        return no_progress_count

    def __track_progress(self, X, Y, visualize=False, accs={}, costs={}):
        """ Evaluates accuracy and cost and the end of each epoch. """
        acc = self.evaluate(X, Y)[1]
        cost = round(self.cost_function(Y, self.feedforward(X))/X.shape[1], 4)
        if visualize:
            self.visualize_progress(accs, costs)
        return acc, cost

    def visualize_progress(self, accs, costs):
        fig, axs = plt.subplots(1, 2, figsize=(12, 6)) 
        fig.suptitle('Training Progress')
        
        axs[0].plot(range(len(costs['valid'])), costs['valid'], 'r-', label='Validation Cost')
        axs[0].plot(range(len(costs['train'])), costs['train'], 'b-', label='Training Cost')
        axs[0].set_title('Cost Over Epochs')
        axs[0].set_xlabel('Epochs')
        axs[0].set_ylabel('Cost')
        axs[0].set_yscale('log')
        axs[0].legend()

        axs[1].plot(range(len(accs['valid'])), accs['valid'], 'r-', label='Validation ACC')
        axs[1].plot(range(len(accs['train'])), accs['train'], 'b-', label='Training ACC')
        axs[1].set_title('Accuracy Over Epochs')
        axs[1].set_xlabel('Epochs')
        axs[1].set_ylabel('Accuracy')
        axs[1].legend()

        plt.show()

    def evaluate(self, X, Y):
        correct_predictions = 0
        predictions = self.feedforward(X)
        for prediction, y in zip(predictions.T, Y.T):
            if np.argmax(prediction) == np.argmax(y):
                correct_predictions += 1
        return correct_predictions, correct_predictions/(X.shape[1])
    
    def save(self, path):
        np.savez(path, W=self.W)


#### Main ####
def main(layers=[784, 30, 10], epochs=10, batch_size=34, eta=0.5, lmbda=1.132):
    
    data_loader = mnist_loader(download=False, path='./data', num_augmentations=0)
    train, valid, test = data_loader.train, data_loader.valid, data_loader.test

    net = Network(layers=layers, cost_function='cross_entropy', activation_function='sigmoid')
    net.fit(train_set=train, batch_size=batch_size, epochs=epochs, eta=eta,  lmbda=lmbda, patience=10, valid_set=valid, show_history=True, visualize=True)
    _, acc = net.evaluate(*test)
    return acc

def tune_hyperparameters():
    def objective(trial):
        n_neurons = trial.suggest_int('n_neurons', 1, 1000)
        eta = trial.suggest_float('eta', 1e-3, 0.5)
        lmbda = trial.suggest_float('lmbda', 1e-3, 10)
        batch_size = trial.suggest_int('batch_size', 10, 100)
        #cost_function = trial.suggest_categorical('cost_function', ['cross_entropy', 'quadratic'])
        #activation_function = trial.suggest_categorical('activation_function', ['sigmoid'])

        acc = main(layers=[784, n_neurons, 10], batch_size=batch_size, eta=eta, lmbda=lmbda)
        #trial.report(intermediate_acc, 15)

        #if trial.should_prune():
            #raise optuna.TrialPruned()
        
        return acc
    #optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    #study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=25)
    print("Number of finished trials: ", len(study.trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a neural network on MNIST dataset.')
    parser.add_argument('command', choices=['tune_hyperparameters', 'main'])
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train.')
    parser.add_argument('--batch_size', type=int, default=34, help='Batch size for training.')
    parser.add_argument('--eta', type=float, default=0.4933969709233855, help='Learning rate.')
    parser.add_argument('--lmbda', type=float, default=1.131779, help='Regularization parameter.')
    
    args = parser.parse_args()
    
    if args.command == 'tune_hyperparameters':
        tune_hyperparameters()
    elif args.command == 'main':
        main(epochs=args.epochs, batch_size=args.batch_size, eta=args.eta, lmbda=args.lmbda)