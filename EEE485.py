from NeuralNetwork import NeuralNetwork as nn
from Layers import *
from Optimization import *
from data_layer import *
import matplotlib.pyplot as plt

opt = Optimizers.Adam(0.01, 0.9, 0.8)
init_w = Initializers.He()
init_b = Initializers.He()

net = nn(opt, init_w, init_b)
data = Dataset('temp.csv', 1, 'label')
loss = Loss.L2Loss()
net.data = data
net.loss_layer = loss

layer1 = FullyConnected.FullyConnected(4, 4)
act1 = ReLU.ReLU()
layer2 = FullyConnected.FullyConnected(4, 1)

net.append_layer(layer1)
net.append_layer(act1)
net.append_layer(layer2)

epochs = 10000
loss = net.train(epochs)

print(min(loss))

plt.plot(loss)
plt.show()