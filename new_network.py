import numpy as np
def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x)*(1.0-sigmoid(x))

def cost(y,x):
    return np.sum(0.5*(x-y)**2)

def cost_prime(y,x):
    return x-y

class Network:
    def __init__(self, layers):
        self.L = layers
        self.W = [np.random.randn(x,y) for x,y in zip(self.L[1:],self.L[0:-1])]
        self.B = [np.random.randn(x,1) for x in self.L[1:]]

    def feedforward(self,a):
        for w,b in zip(self.W,self.B):
            a = sigmoid(np.dot(w,a)+b)
        return a

    def learn(self, a, y):
        Z=[]
        A=[]
        A.append(a)
        for w,b in zip(self.W,self.B):
            z = np.dot(w,A[-1])+b
            a=sigmoid(z)
            Z.append(z)
            A.append(a)

        self.__backprob(y, A, Z)

    def __backprob(self,y,A,Z):
        D = []
        D.append(cost_prime(y,A[-1])*sigmoid_prime(Z[-1]))
        for i in range(1,len(Z)):
            D.insert(0, np.dot(self.W[-i].T,D[0])*sigmoid_prime(Z[-i-1]))
        B_shifts = D
        W_shifts = []
        for a,d in zip(A[0:-1],D):
            W_shifts.append(np.dot(d,a.T))

        for i in range(len(self.W)):
            self.W[i] -= 0.5 * W_shifts[i]
            self.B[i] -= 0.5 * B_shifts[i] 

L = [1,3,2,3]
a=np.array([[1]])
y=np.array([[0],[1],[0]])

net = Network(L)
net.feedforward(a)
for i in range(50):
    net.learn(a,y)
net.feedforward(a)