import numpy as np
import scipy as sp
import matplotlib as plt
xmin=-100
xmax=100
dx=0.001
hbar = 1.0 
mass = 1.0  
omega = 1.0
sigma=1.0
k0=5.0  
x=np.arange(xmin,xmax,dx)
print(x)
y= np.exp(-(x**2) / 2.0) / np.pi**(1/4)
N=len(x)
k=2*np.pi*np.fft.fftfreq(N,dx)
#z=np.fft.fft(y)
print(k)
