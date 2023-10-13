import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
excel_file = 'C:/Users/ak172/Downloads/STM_data_100pts.xlsx'
df = pd.read_excel(excel_file)

voltage = df['Voltage (V)']
current = df['Current (nA)']
def model_function(V, a1, a2, b1):
    p = 5
    return ((a1 * V)  + (a2 * V**3)) * np.exp(-b1 * np.sqrt(p) * 0.0000000005)
initial_params = [1, 1, 1]
fit_params, _ = curve_fit(model_function, voltage, current, p0=initial_params)
plt.figure(figsize=(10, 6))
plt.plot(voltage, current, marker='o', linestyle='-', label='Data')
plt.plot(voltage, model_function(voltage, *fit_params), 'r-', label='Fitted Curve')
plt.xlabel('Voltage (V)')
plt.ylabel('Current (nA)')
plt.title('I-V Curve and Fitted Curve with New Equation')
plt.grid(True)
plt.legend()
plt.show()

print("Fitted parameters:")
print("a1:", fit_params[0])
print("a2:", fit_params[1])
print("b1:", fit_params[2])
