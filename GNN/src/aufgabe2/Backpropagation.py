import numpy as np
import math
import matplotlib.pyplot as plt

def calculate_target(x, y):
    if( math.sqrt(x * x + y * y) <= 1):
        return 0.8 
    else:
        return 0.0  

class Neural_Network(object):
    def __init__(self):
        #parameters
        self.inputSize = 2
        self.outputSize = 1
        self.hiddenSize = 4
        self.n = 0.05

        #weights
        self.Wji = np.random.randn(self.inputSize +1, self.hiddenSize )
        self.Wkj = np.random.randn(self.hiddenSize +1, self.outputSize ) 

    def forward(self, x,y):
        bias = 1
        self.input = np.array([x,y,bias])
        self.wji_oi = np.array(np.matmul(self.input, self.Wji))
        self.wji_oi = np.append(self.wji_oi, bias)
        self.oj = np.array(self.sigmoid(self.wji_oi))
        self.wkj_oj = np.matmul(self.oj, self.Wkj) 
        ok = self.sigmoid(self.wkj_oj)
        return ok 

    def sigmoid(self, activation):
        return 1/ (1 + np.exp(-activation))

    def sigmoid_derivative(self, x):
        return self.sigmoid(x) * (1.0 - self.sigmoid(x)) 

    def backward(self,target, o):
        #Weights from hidden to output
        ek = (o - target) * self.sigmoid_derivative(self.wkj_oj)
        delta_weight_kj = - self.n * ek * self.oj
        for x in range(len(self.Wkj)):
            #print("WKJ", self.Wkj[x])
            self.Wkj[x] = self.Wkj[x][0] + delta_weight_kj[x] 
        self.input = np.array([self.input])
        
        #Weights from input to hidden
        ej = ek * np.transpose(self.Wkj) * self.sigmoid_derivative(self.wji_oi)
        delta_weight_wji = -self.n * ej * np.transpose(self.input)
        delta_weight_wji = np.delete(delta_weight_wji, 4,1)
        self.Wji += delta_weight_wji

    def train (self, x,y):
        o = self.forward(x,y)
        target = calculate_target(x,y)
        #print("target",target)
        self.backward(target, o)
        return o

X = list()
Y = list()
O = list()
NN = Neural_Network()
for i in range(1000000): # trains the NN 1,000 times
    x = np.random.uniform(-1.2,1.2)
    y = np.random.uniform(-1.2,1.2)
    o = NN.train(x, y)
    X.append(x)
    Y.append(y)
    O.append(o[0])
    '''if(i == 100000):
        sc = plt.scatter(X, Y, c=O)
        plt.colorbar(sc)
        plt.show()'''

print("Prediction:")
print("Expect:",calculate_target(0,0.5), NN.forward(0,0.5))
print("Expect:",calculate_target(0.5,0), NN.forward(0.5,0))
print("Expect:",calculate_target(0,0), NN.forward(0,0))
print("Expect:",calculate_target(1,1), NN.forward(1,1))
print("Expect:",calculate_target(1.2,1.2), NN.forward(1.2,1.2))

sc = plt.scatter(X, Y, c=O)
plt.colorbar(sc)
plt.show()