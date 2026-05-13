##from celsius to farenheit.
import numpy as np 
cvalues = [20.1, 20.6, 22.5]
#we create a nomnal array
C = np.array(cvalues)
#we convert normal array to numpy array
print(C * 9 / 5 + 32)
#when performing operations with the numpy array
#linear algebra is done

##from celsius to farenheit

f_values = [32.0, 69.0, 77.0, 100.0]

f_val = np.array(f_values)
C = (f_val - 32) * 5/9
print(f"Celsius: {np.round(C, 2)}")
# this formula converts to farenheit
#:.2f doesnt work with numpy arrays
#np.round(value, number of decimals) needs to be used.
