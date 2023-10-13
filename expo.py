import numpy as np
import matplotlib.pyplot as plt
hbar = 1.0 
mass = 1.0  
omega = 1.0  
num_steps = 500  
total_time = 10.0  
dt = total_time / num_steps  
x0 = -2.0
k0 = 5.0   
sigma = 0.5
def potential(x, t):
    return 0.5 * mass * omega**2 * (x - np.sin(omega * t))**2
def kinetic_operator(p):
    return np.exp(-1j * p**2 * dt / (2 * mass * hbar))
def potential_operator(x, t):
    return np.exp(-1j * potential(x, t) * dt / (2*hbar))
def initial_wave_function(x):
    return np.exp(-(x - x0)*2 / (2 * sigma**2)) * np.exp(1j * k0 * x / hbar)
x_values = np.linspace(-5, 5, 100)
psi = initial_wave_function(x_values)
for step in range(num_steps):
    t = step * dt  
    psi_momentum = np.fft.fft(psi)
    psi_momentum *= kinetic_operator(2 * np.pi * hbar * np.fft.fftfreq(len(x_values), d=(x_values[1] - x_values[0])))
    psi = np.fft.ifft(psi_momentum)
    psi *= potential_operator(x_values, t)
    psi_momentum = np.fft.fft(psi)
    psi_momentum *= kinetic_operator(2 * np.pi * hbar * np.fft.fftfreq(len(x_values), d=(x_values[1] - x_values[0])))
plt.plot(x_values, np.abs(psi)**2)
plt.xlabel('Position')
plt.ylabel('Probability Density')
plt.title('Time Evolution of Quantum Wave Function')
plt.show()

