import matplotlib.pyplot as plt
import numpy as np
e=1
r=np.arange(10**(-10),10**(-8),10)
z=1
l=0
"""plt.plot(e**2/(4*np.pi*r))           
plt.xlabel('r')
plt.ylabel('potential ')
plt.title('potential plot')
plt.show()"""
plt.plot((-z*e**2/4*np.pi*r)+(l*(l+1)/2*(r**2)))
plt.xlabel('r')
plt.ylabel('potential ')
plt.title('potential plot')
plt.show()
