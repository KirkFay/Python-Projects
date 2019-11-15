#
# Kirk Fay
# Artificial Intelligence
#
import numpy as np
import matplotlib.pyplot as plt
import neurolab as nl

text = np.loadtxt('first_nn_data.txt')
data = text[:, 0:2]
labels = text[:, 2:]
print(data)
print(labels)

D1 = [data[:,0].min(), data[:,0].max()]
D2 = [data[:,1].min(), data[:,1].max()]
n_output = labels.shape[1]
print(n_output)

nn = nl.net.newp([D1, D2], n_output)
error_progress = nn.train(data, labels, epochs=50, lr=0.02)
plt.figure()
plt.plot(error_progress)
plt.xlabel('Epochs')
plt.ylabel('Training error')
plt.title('Training progress')
plt.grid()
plt.show()

test = [[0.5, 4.5], [4.0, 0.5], [4.5, 8.2]]
for t in test:
    print(t, '-->', nn.sim([t])[0])
