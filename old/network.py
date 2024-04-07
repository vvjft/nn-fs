import numpy as np
import random

class Network(object):
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) 
                        for x, y in zip(sizes[:-1], sizes[1:])]
        
    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w,a)+b)
        return a
    
    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        for i in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[0:mini_batch_size]]
            for mini_batch in mini_batches:
                self.learn(mini_batch, eta)
    
        
    def learn(self, train, eta):
        nabla_b = [np.zeros(b.shape) for b in net.biases]
        nabla_w = [np.zeros(w.shape) for w in net.weights]
        for x, y in train:
            nabla_b_shift, nabla_w_shift = self.backprob(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, nabla_b_shift)] # O ile przesunac b i w
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, nabla_w_shift)]
        self.weights = [w-(eta/len(train))*nw for w, nw in zip(self.weights, nabla_w)] # przesuwanie b i w zgodnie z sgd
        self.biases = [b-(eta/len(train))*nb 
                       for b, nb in zip(self.biases, nabla_b)]

    def backprob(self, x, y):
        nabla_b_shift = [np.zeros(b.shape) for b in self.biases]
        nabla_w_shift = [np.zeros(w.shape) for w in self.weights]
        activation = x
        activations = [x]
        zs = []
        for l in range(len(self.weights)):
            z = np.dot(self.weights[l], activation) + self.biases[l]
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])

        nabla_b_shift[-1] = delta
        nabla_w_shift[-1] = np.dot(delta, activations[-2].transpose()) # output error

        for l in range(2, self.num_layers): # backpropagate
            z = zs[-l]
            delta = np.dot((self.weights[-l+1]).transpose(), delta)*sigmoid_prime(z)

            nabla_b_shift[-l] = delta
            nabla_w_shift[-l] = np.dot(delta, activations[-l-1].transpose())
        return nabla_b_shift, nabla_w_shift

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return test_results, sum(int(y[x] == 1) for (x, y) in test_results) 

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y) 


def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))


import pandas as pd
import numpy as np
training = pd.read_csv('mnist_train.csv').to_numpy()
X=training[:,1:]
Y_val=training[:,0]
Y = np.zeros((Y_val.size, 10))
Y[np.arange(len(Y)), Y_val]=1
train = [(X[0], Y[0])]
train = []
for x, y in [(X[i], Y[i]) for i in range(100)]: 
    x=x.reshape(-1,1)
    y=y.transpose()
    y=y.reshape(-1,1)
    train.append((x,y))
net=Network([784, 30, 10])
net.SGD(train, 40, 1, 3.0)


y_vector = np.zeros(10,10)
y_vector[y]=1
test_results = [(np.argmax(net.feedforward(x)), y_vector)
                for (x, y) in train]

for x,y in train:
    nabla_b_shift = [np.zeros(b.shape) for b in net.biases]
    nabla_w_shift = [np.zeros(w.shape) for w in net.weights]
    activation = x
    activations = [x]
    zs = []
    for l in range(len(net.weights)):
        z = np.dot(net.weights[l], activation) + net.biases[l]
        zs.append(z)
        activation = sigmoid(z)
        activations.append(activation)
    delta = net.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
   
   
   
nabla_b = [np.zeros(b.shape) for b in net.biases]
nabla_w = [np.zeros(w.shape) for w in net.weights]
for x, y in one_train:
    delta, nabla_b_shift, nabla_w_shift = net.backprob(x, y)
    nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, nabla_b_shift)] # O ile przesunac b i w
    nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, nabla_w_shift)]