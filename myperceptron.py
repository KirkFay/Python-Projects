#
#   Kirk Fay
#   Artificial Intelligence
#
import numpy as np
import matplotlib.pyplot as plt
import neurolab as nl

text = np.loadtxt('perceptron_data.txt')

# Separate datapoints and labels
data = text[:, :2]
labels = text[:, 2].reshape((text.shape[0], 1))

print(data)
print(labels)

plt.figure()
plt.scatter(data[:,0], data[:,1])
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.title('Input data')
plt.show()

# Define minimum and maximum values for each dimension
dim1_min, dim1_max, dim2_min, dim2_max = 0, 1, 0, 1
n_output = labels.shape[1]
print(n_output)

D1 = [dim1_min, dim1_max]
D2 = [dim2_min, dim2_max]
perceptron = nl.net.newp([D1, D2], n_output)

error_progress = perceptron.train(data, labels, epochs=4, lr=0.01)

plt.figure()
plt.plot(error_progress)
plt.xlabel('epochs')
plt.ylabel('Training error')
plt.title('Training progress')
plt.grid()
plt.show()
