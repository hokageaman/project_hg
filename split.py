import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
a=sp.zeros((3,3))
print(a)
"""a[0,0]=2
a[0,1]=2
a[0,2]=3
a[1,0]=4
a[1,1]=5
a[1,2]=6
a[2,0]=6
a[2,1]=7
a[2,2]=8
print(a)
E,V=np.linalg.eig(a)
print(E,V)
print(V[:,0]@V[:,0])
print(V[:,1]@V[:,1])
print(V[:,0]@V[:,2])
print(V[:,2]@V[:,1])"""
for i in range(0,3,1):
    a[i,i]=2
for i in range(0,2,1):
    a[i,i+1]=-1
    a[i+1,i]=-1
    print(a)
    """ psi_mom = []
    for i in range(5,11):
        psi_mom.append(kinetic_operator(i))

    psi = np.fft.ifft(psi_momentum)
"""



