

import numpy as np
import matplotlib.pyplot as plt

def rho(x):
    return (x**3)

def f(x):
    return (-x**5)/20+(2.05*x)+3

def g(x):
    return (-x**5+x)

n=100
x=np.linspace(0,1,n)
dx=x[1]-x[0]
A=np.zeros((n,n))
B=np.zeros(n)

for i in range(0,n):
    for j in range(0,n):
        if (i==j):
            A[i][j]=-2
        elif abs(i-j)==1:
            A[i][j]=1
        else:
            A[i][j]=0

print(A)

for i in range(0,n):
    B[i]=-(dx**2)*rho(x[i])
    
sol=np.linalg.solve(A,B)
print(sol)

plt.figure(1)
plt.plot(x,sol,label='Finite Difference Method')
plt.figure(2)
plt.plot(x,g(x),label='Analytical Solution',ls='--')
plt.figure(3)
plt.plot(x,rho(x),label='Charge Distribution')
plt.legend()
plt.show()


        