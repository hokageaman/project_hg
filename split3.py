import numpy as np
import matplotlib.pyplot as plt

# Constants
hbar = 1.0   # Reduced Planck's constant
mass = 1.0   # Particle mass
omega = 1.0  # Angular frequency
L = 10.0     # Length of the spatial domain
N = 50     # Number of spatial points
dx = L / (N - 1)   # Spatial step
dt = 0.01    # Time step
T = 10.0     # Total simulation time

# Initial wave function parameters
x0 = 0.0
sigma = 0.5

# Initialize spatial grid and wave function
x = np.linspace(0, L, N)
psi = np.exp(-0.5 * ((x - x0) / sigma)**2) / np.sqrt(np.pi * sigma)

# Calculate momentum values
momentum = 2 * np.pi * np.fft.fftshift(np.arange(-N // 2, N // 2)) / (N * dx)

# Construct the potential operator matrix
def potential_operator(potential):
    return np.diag(np.exp(-1j * dt * potential / hbar))

# Construct the kinetic operator matrix
kinetic_operator = np.exp(-0.5j * hbar * dt / (2 * mass) * momentum**2)

# Simulation loop
timesteps = int(T / dt)
for step in range(timesteps):
    t = step * dt
    
    # Construct the potential operator matrix
    potential = 0.5 * mass * omega**2 * (x - np.sin(omega * t))**2
    
    # Apply kinetic operator in momentum space
    psi_momentum = np.dot(kinetic_operator, psi)
    
    # Apply potential operator in position space
    psi_position = np.dot(potential_operator(potential), psi_momentum)
    
    # Apply kinetic operator in momentum space again
    psi = np.dot(kinetic_operator, psi_position)

# Plot the final wave function
plt.plot(x, np.abs(psi)**2)
plt.xlabel('Position')
plt.ylabel('Probability Density')
plt.title('Final Wave Function with Time-Dependent Potential (Matrix Method without FFT/IFFT)')
plt.show()
