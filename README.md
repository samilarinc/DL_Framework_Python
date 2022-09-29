# DL_Framework_Python

## Usage:

```
from NeuralNetwork import *
from Layers import *
from Optimization import *

weight_initializer = Initializers.Xavier() # Constant, He, Xavier
bias_initializer = Initializers.Xavier() # Constant, He, Xavier
optimizer = Optimizers.SGD(learning_rate) # SGD, SGDWithMomentum, Adam

net = NeuralNetwork(optimizer, weight_initializer, bias_initializer)
net.data_layer = Helpers.IrisData() # or Helpers.MNISTData(batch_size)

net.append_layer(Conv.Conv(stride_shape, convolution_shape, num_kernels))
net.append_layer(FullyConnected.FullyConnected(input_size, output_size))

net.loss_layer = Loss.CrossEntropyLoss()

net.train(num_epochs)

if you_want_to_save_the_result:
    NeuralNetwork.save(dir, net)

plt.figure('Loss function for training LeNet on the given dataset')
plt.plot(net.loss, '-x')
plt.show()

data, labels = net.data_layer.get_test_set()

results = net.test(data)
accuracy = Helpers.calculate_accuracy(results, labels)

print('\nOn the given dataset, we achieve an accuracy of: ' + str(accuracy * 100) + '%')
```
