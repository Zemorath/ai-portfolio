import numpy as np

array_1d=np.array([1,2,3,4,5])
print("1D Array:", array_1d)

array_2d=np.array([[1,2,3], [4,5,6]])
print("2D Array:\n", array_2d)

print("Mean:", np.mean(array_1d))
print("Sum:", np.sum(array_2d))
print("Reshaped 1D to 3x2:\n", array_2d.reshape(3,2))