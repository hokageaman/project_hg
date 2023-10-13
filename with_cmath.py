import cmath






# Define the function and parameters
def my_function(x):
    return cmath.exp(-(x +2)*2 / (2 * 0.5*2)) * cmath.exp(1j * 5 * x)

# Create a time vector
sampling_rate = 1000  # Number of samples per second
duration = 2  # Duration of the signal in seconds
num_samples = int(sampling_rate * duration)
t = [i / sampling_rate for i in range(num_samples)]

# Evaluate the function
signal = [my_function(ti) for ti in t]

# Compute the Discrete Fourier Transform
dft_result = [sum(signal[k] * cmath.exp(-2j * cmath.pi * k * n / num_samples) for k in range(num_samples)) for n in range(num_samples)]
freqs = [n * sampling_rate / num_samples for n in range(num_samples)]

# Plot the original signal and its Discrete Fourier Transform
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.plot(t, [signal_i.real for signal_i in signal])
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Original Signal')

plt.subplot(2, 1, 2)
plt.plot(freqs, [abs(dft_i) for dft_i in dft_result])
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.title('Discrete Fourier Transform')

plt.tight_layout()
plt.show()
