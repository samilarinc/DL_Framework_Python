from NeuralNetwork import NeuralNetwork as nn
from Layers import *
from Optimization import *
from data_layer import *

opt = Optimizers.Sgd(0.001)
init_w = Initializers.He()
init_b = Initializers.He()

net = nn(opt, init_w, init_b)
data = Dataset('temp.csv', 1, 'label')
loss = Loss.L2Loss()
net.data = data
net.loss_layer = loss

layer1 = FullyConnected.FullyConnected(4, 4)
act1 = ReLU.ReLU()
layer2 = FullyConnected.FullyConnected(4, 4)
act2 = ReLU.ReLU()
layer3 = FullyConnected.FullyConnected(4, 2)
act3 = ReLU.ReLU()
layer4 = FullyConnected.FullyConnected(2, 1)

net.append_layer(layer1)
net.append_layer(act1)
net.append_layer(layer2)
net.append_layer(act2)
net.append_layer(layer3)
net.append_layer(act3)
net.append_layer(layer4)

epochs = 1000
loss = net.train(epochs)

print(loss)