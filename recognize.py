import numpy as np

class Network(object):
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) 
                        for x, y in zip(sizes[:-1], sizes[1:])]
        
    def learn(self, train, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in train:
            nabla_b_shift, nabla_w_shift = self.backprob(x, y)
            print(nabla_b_shift)
            print(nabla_w_shift)
            nabla_b = nabla_b + nabla_b_shift # O ile przesunac b i w
            nabla_w = nabla_w + nabla_w_shift
            print(nabla_b)
            print(nabla_w)
        """self.weights = [w-(eta/len(train))*nw # przesuwanie b i w zgodnie z sgd
                        for w, nw in (self.weights, nabla_w)]
        self.biases = [b-(eta/len(train))*nb 
                       for b, nb in (self.biases, nabla_b)]"""

    def backprob(self, x, y):
        nabla_b_shift = [np.zeros(b.shape) for b in self.biases]
        nabla_w_shift = [np.zeros(w.shape) for w in self.weights]
        activation = x
        activations = [x]
        zs = []
        for l in range(len(net.weights)):
            z = np.dot(net.weights[l], activation) + net.biases[l]
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


net = Network([2, 3, 1])