import numpy as np
import matplotlib.pyplot as plt

class Rekurrentes_Netz(object):
    def __init__(self):
        self.inputs = np.array([0.770719, 0.00978897, 1])
        self.weights = np.array([[6, 10],
                                 [-10, 0],
                                 [-4,4]])
        
    def activate(self):
        self.out = np.array(np.matmul(self.inputs, self.weights))
        self.inputs = np.append(self.out, 1)
        #print("inputs", self.inputs)
        return self.out

RN = Rekurrentes_Netz()
output1 = list()
output2 = list()
for i in range(4):
    out = RN.activate()
    print(out)
    output1 = np.append(output1, out[0])
    output2 = np.append(output2, out[1])
    
plt.plot(output1)
plt.plot(output2)
plt.show()