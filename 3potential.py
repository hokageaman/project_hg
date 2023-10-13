import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
m = 1.0  
w = 1  
t = np.arange(0, 10, 0.01)  
x = np.linspace(-5, 5, 100)  
X, T = np.meshgrid(x, t)
V = 0.5 * m * w**2 * (X - np.sin(w * T))**2
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, T, V, cmap='viridis')
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('V(x, t)')
ax.set_title('3D Surface Plot of Potential Energy')
plt.show()

