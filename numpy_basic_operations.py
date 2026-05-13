import numpy as np 

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print(f"A + B = \n{A + B}") #summ
print(f"A * 2 = \n{A * 2}") #multiply by 2
print(f"A . B ={np.dot(A, B)}") # gives a single value
print(f"A.T = \n {A.T}") # transposal
print(f"Shape of A = {A.shape}")