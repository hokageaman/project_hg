import numpy as np

def gaussian_freq(omega, sigma):
    return np.exp(-omega**2 * sigma**2 / 2) / (sigma * np.sqrt(2 * np.pi))

def inverse_fourier_transform_simpson_3_8(sigma, t_max, num_points):
    a = -10 / (sigma * t_max)  # Lower limit of integration
    b = 10 / (sigma * t_max)   # Upper limit of integration
    h = (b - a) / num_points

    integral_approx = 0
    for k in range(num_points + 1):
        tk = a + k * h
        fk = gaussian_freq(tk, sigma)
        
        if k == 0 or k == num_points:
            weight = 1
        elif k % 3 == 0:
            weight = 2
        else:
            weight = 3
        
        integral_approx += weight * fk
    
    integral_approx *= 3 * h / 8
    return integral_approx

# Parameters
sigma = 1.0
t_max = 10.0
num_points = 1000

# Calculate Inverse Fourier Transform using Simpson's 3/8 rule
approx_inverse_ft = inverse_fourier_transform_simpson_3_8(sigma, t_max, num_points)

print("Approximated Inverse Fourier Transform:", approx_inverse_ft)
