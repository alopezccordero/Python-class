import numpy as np

x = np.array([2.2, 1.3, 4.2, 5.8, 3.4, 8.7])
y = np.array([6.14, 4.72, 11.17, 14.23, 9.55, 22.49])

a, b = 1.5, 5.0 # this is the values that we decided to use for model
y_pred = a * x + b 
mse = np.mean((y_pred - y) ** 2)
print(f"MSE for a = {a} and b = {b}\n mse = {mse:.4f}")

##calculating r2 to calculate variability

ss_res = np.sum((y - y_pred) ** 2) ##ss res is y value - y prediction
ss_tot = np.sum((y - y.mean()) ** 2)## ss tot is y - mean of y
r2 = 1 - ss_res / ss_tot ##r2 = 1 - ss_res / ss_tot.
print(f"R^2 = {r2:.4f}")