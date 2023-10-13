"""import math
import matplotlib.pyplot as plt

def dft(signal):
    N = len(signal)
    dft_result = [0] * N
    
    for k in range(N):
        sum_real = 0
        sum_imag = 0
        for n in range(N):
            angle = 2 * math.pi * k * n / N
            sum_real += signal[n].real * math.cos(angle) + signal[n].imag * math.sin(angle)
            sum_imag += -signal[n].real * math.sin(angle) + signal[n].imag * math.cos(angle)
        dft_result[k] = complex(sum_real, sum_imag)
    
    return dft_result

# Generate a sample wave function
sampling_rate = 1000  # Number of samples per second
duration = 1.0  # Duration of the wave in seconds
frequency = 5  # Frequency of the wave in Hz
t = [i / sampling_rate for i in range(int(sampling_rate * duration))]  # Time points

# Create a wave function (sine wave)
wave_function = [math.sin(2 * math.pi * frequency * i) for i in t]

# Calculate the Discrete Fourier Transform
dft_result = dft(wave_function)

# Frequency values corresponding to the DFT components
frequencies = [k * sampling_rate / len(dft_result) for k in range(len(dft_result))]

# Plot the original wave function and its DFT magnitude
plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.plot(t, wave_function)
plt.title("Original Wave Function")
plt.xlabel("Time")
plt.ylabel("Amplitude")

plt.subplot(2, 1, 2)
plt.plot(frequencies, [abs(x) for x in dft_result])
plt.title("DFT Magnitude")
plt.xlabel("Frequency")
plt.ylabel("Magnitude")
plt.xlim(0, sampling_rate / 2)  # Show positive frequencies only

plt.tight_layout()
plt.show()"""
import math
import matplotlib.pyplot as plt

def dft(signal):
    N = len(signal)
    dft_result = [0] * N
    
    for k in range(N):
        sum_real = 0
        sum_imag = 0
        for n in range(N):
            angle = 2 * math.pi * k * n / N
            sum_real += signal[n].real * math.cos(angle) + signal[n].imag * math.sin(angle)
            sum_imag += -signal[n].real * math.sin(angle) + signal[n].imag * math.cos(angle)
        dft_result[k] = complex(sum_real, sum_imag)
    
    return dft_result

# Generate a sample function
sampling_rate = 1000  # Number of samples per second
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

# Frequency values corresponding to the DFT components
frequencies = [k * sampling_rate / len(dft_result) for k in range(len(dft_result))]

# Plot the original function and its DFT magnitude
plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.plot(x, function_values)
plt.title("Original Function")
plt.xlabel("Position (x)")
plt.ylabel("Amplitude")

plt.subplot(2, 1, 2)
plt.plot(frequencies, [abs(x) for x in dft_result])
plt.title("DFT Magnitude")
plt.xlabel("Frequency")
plt.ylabel("Magnitude")
plt.xlim(0, sampling_rate / 2)  # Show positive frequencies only

plt.tight_layout()
plt.show()
