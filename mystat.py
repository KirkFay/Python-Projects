#
#   Kirk Fay
#   Artificial Intelligence
#
import numpy as np
from sklearn import preprocessing
input_data = np.array([[5, -2, 3], [-1, 7, -6],[3, 0, 2],[7, -9, -4]])
print(input_data)

data_binarized = preprocessing.Binarizer(threshold=2.2).transform(input_data)
print("\nBinarized data:\n", data_binarized)

print("axis=0")
print("Mean =", input_data.mean(axis=0))
print("variance =", input_data.var(axis=0))
print("Std deviation =", input_data.std(axis=0))
print("axis=1")
print("Mean =", input_data.mean(axis=1))
print("variance =", input_data.var(axis=1))
print("Std deviation =", input_data.std(axis=1))
