import numpy as np
import matplotlib.pyplot as plt

# Parameters
N = 100000  # Number of spatial grid points
L = 10.0  # Spatial range [-L, L]
dx = 2.0 * L / N
x = np.linspace(-L, L+1-dx, N)
V0 = 1.0  # Amplitude of potential
omega = 5  # Frequency of potential
T = 20.0  # Total simulation time
dt = 0.05  # Time step
m = 1

# Create potential function
def potential(x, t):
    return 0.5 * m * omega ** 2 * x ** 2 + (x - np.sin(omega * t)) ** 2

# Create the kinetic energy operator
def kinetic_operator(N, dx):
    kx = np.fft.fftfreq(N, dx) * 2 * np.pi
    return np.exp(-1j * kx ** 2 * dt / 2.0)

# Create the potential operator
def potential_operator(potential_values):
    return np.exp(-1j * potential_values * dt)

# Initialize wavefunction
psi = np.exp(-(x ** 2) / 2.0) / np.pi ** (1 / 4)
psi /= np.sqrt(np.trapz(np.abs(psi) ** 2, x=x))

# Time evolution using split-operator method
num_steps = int(T / dt)
plot_interval = 50  # Plot every 50 steps
for step in range(num_steps):
    # Calculate potential at current time step
    V = potential(x, step * dt)
    
    # Calculate kinetic operator and apply it in momentum space
    psi = np.fft.fft(psi)
    psi *= kinetic_operator(N, dx)
    psi = np.fft.ifft(psi)
   
    # Calculate potential operator and applying it in position space
    psi *= potential_operator(V)
   
    # Calculate kinetic operator and applying it in momentum space
    psi = np.fft.fft(psi)
    psi *= kinetic_operator(N, dx)
    psi = np.fft.ifft(psi)
   
    # Plotting the wavefunction at specified intervals
    if step % plot_interval == 0:
        plt.plot(x, np.abs(psi) ** 2, label=f'Time step {step}')
        #plt.plot(x, np.abs(psi) ** 2, label=f'Time step {step}')


plt.title("Evolution of Probability Density")
plt.xlabel("Position")
plt.ylabel("|Psi|^2")
plt.legend()
plt.show()
