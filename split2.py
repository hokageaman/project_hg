"""import numpy as np
import matplotlib.pyplot as plt
import math
hbar=1   
num_steps = 500  
total_time = 10.0  
dt = total_time / num_steps  
x0 = -2.0
k0 = 5.0   
sigma = 0.5
xmin=-100
xmax=100
dx=0.001 
mass = 1.0  
omega = 1.0 
x=np.arange(xmin,xmax,dx)
y=np.exp(-(x - xmin)**2 / (2 * sigma**2)) * np.exp(1j * k0 * x / hbar)
N=len(x)
k=2*np.pi*hbar*np.fft.fftfreq(N,dx)
z=np.fft.fft(y)
def dft(signal):
    N = len(signal)
    dft_result = [0] * N
    
    for k in range(N):
        sum_real = 0
        sum_imag = 0
        for n in range(N):
            angle = 2 * np.pi * k * n / N
            sum_real += signal[n].real * np.cos(angle) + signal[n].imag * np.sin(angle)
            sum_imag += -signal[n].real * np.sin(angle) + signal[n].imag * np.cos(angle)
        dft_result[k] = complex(sum_real, sum_imag)
    
    return dft_result

# Generate a sample function
sampling_rate = 100  # Number of samples per second
duration = 1.0  # Duration of the function in seconds
x_min = -10.0
x_max = 10.0
num_samples = int(sampling_rate * duration)
x = [x_min + (x_max - x_min) * i / (num_samples - 1) for i in range(num_samples)]  # Position points

# Create the function
x0 = 0.0
sigma = 1.0
function_values = [np.exp(-(xi - x0)**2 / (2 * sigma**2)) for xi in x]

# Calculate the Discrete Fourier Transform
dft_result = dft(function_values)

# Frequency values corresponding to the DFT components
frequencies = [k * sampling_rate / len(dft_result) for k in range(len(dft_result))]
def idft(dft_result):
    N = len(dft_result)
    signal = [0] * N
    
    for n in range(N):
        sum_real = 0
        sum_imag = 0
        for k in range(N):
            angle = 2 * math.pi * k * n / N
            sum_real += dft_result[k].real * math.cos(angle) - dft_result[k].imag * math.sin(angle)
            sum_imag += dft_result[k].real * math.sin(angle) + dft_result[k].imag * math.cos(angle)
        signal[n] = complex(sum_real / N, sum_imag / N)
    
    return signal

# Generate a sample function
sampling_rate = 100  # Number of samples per second
duration = 1.0  # Duration of the function in seconds
x_min = -10.0
x_max = 10.0
num_samples = int(sampling_rate * duration)
x = [x_min + (x_max - x_min) * i / (num_samples - 1) for i in range(num_samples)]  # Position points

# Create the function
x0 = 0.0
sigma = 1.0
function_values = [math.exp(-(xi - x0)**2 / (2 * sigma**2)) for xi in x]

# Calculate the Discrete Fourier Transform
dft_result = dft(function_values)

# Calculate the Inverse Discrete Fourier Transform
reconstructed_signal = idft(dft_result)

def potential(x, t):
    return 0.5 * mass * omega**2 * (x - np.sin(omega * t))**2
def kinetic_operator(p):
    return np.exp(-1j * p**2 * dt / (2 * mass * hbar))
def potential_operator(x, t):
    return np.exp(-1j * potential(x, t) * dt / hbar)
def initial_wave_function(x):
    return np.exp(-(x - x0)**2 / (2 * sigma**2)) * np.exp(1j * k0 * x / hbar)
x_values = np.linspace(-5, 5, 100)
psi = initial_wave_function(x_values)
for step in range(num_steps):
    t = step * dt  
    psi_momentum = dft(psi)
    psi_momentum *= kinetic_operator(k)
    psi =idft(psi_momentum)
    psi *= potential_operator(x_values, t)
    psi_momentum = dft(psi)
    psi_momentum *= kinetic_operator(k)
plt.plot(x_values, np.abs(psi)**2)
plt.xlabel('Position')
plt.ylabel('Probability Density')
plt.title('Time Evolution of Quantum Wave Function')
plt.show()"""
import numpy as np
import matplotlib.pyplot as plt
import math

hbar = 1
num_steps = 500
total_time = 10.0
dt = total_time / num_steps
x0 = -2.0
k0 = 5.0
sigma = 0.5
xmin = -100
xmax = 100
dx = 0.001
mass = 1.0
omega = 1.0

x_values = np.linspace(-5, 5, 100)
x = np.arange(xmin, xmax, dx)
k = 2 * np.pi * np.fft.fftfreq(len(x), dx)

def dft(signal):
    N = len(signal)
    dft_result = [0] * N
    
    for k_idx in range(N):
        sum_real = 0
        sum_imag = 0
        for n in range(N):
            angle = 2 * np.pi * k_idx * n / N
            sum_real += signal[n].real * np.cos(angle) + signal[n].imag * np.sin(angle)
            sum_imag += -signal[n].real * np.sin(angle) + signal[n].imag * np.cos(angle)
        dft_result[k_idx] = complex(sum_real, sum_imag)
    
    return dft_result

def idft(dft_result):
    N = len(dft_result)
    signal = [0] * N
    
    for n in range(N):
        sum_real = 0
        sum_imag = 0
        for k_idx in range(N):
            angle = 2 * np.pi * k_idx * n / N
            sum_real += dft_result[k_idx].real * np.cos(angle) - dft_result[k_idx].imag * np.sin(angle)
            sum_imag += dft_result[k_idx].real * np.sin(angle) + dft_result[k_idx].imag * np.cos(angle)
        signal[n] = complex(sum_real / N, sum_imag / N)
    
    return signal

def potential(x, t):
    return 0.5 * mass * omega**2 * (x - np.sin(omega * t))**2

def kinetic_operator(p):
    return np.exp(-1j * p**2 * dt / (2 * mass * hbar))

def potential_operator(x, t):
    return np.exp(-1j * potential(x, t) * dt / hbar)

def initial_wave_function(x):
    return np.exp(-(x - x0)**2 / (2 * sigma**2)) * np.exp(1j * k0 * x / hbar)

psi = initial_wave_function(x)

for step in range(num_steps):
    t = step * dt
    psi_momentum = dft(psi)
    psi_momentum *= kinetic_operator(k)
    psi = idft(psi_momentum)
    psi *= potential_operator(x, t)
    psi_momentum = dft(psi)
    psi_momentum *= kinetic_operator(k)

plt.plot(x, np.abs(psi)**2)
plt.xlabel('Position')
plt.ylabel('Probability Density')
plt.title('Time Evolution of Quantum Wave Function')
plt.show()
