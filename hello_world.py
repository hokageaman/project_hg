# print("Hello World",6,7,sep="+",end="-\n" )
"""print("aman is the greatest leader\"of all time\"\tand he is going to, be a legend")
str="itachi"
str1="shishui"
str2='jiraya'
print(str[3])
for character in str2:
    print(character) """ 

"""india="sacrifice"
print(india[0:5])
print(india[:8])
print(india[0:-3])
print(india[0:len(india)-2])
print(india[-3:-2])"""
import numpy as np
import matplotlib.pyplot as plt


hbar = 1.0
m = 1.0
omega = 1.0
num_points = 512
x_max = 10.0
t_max = 5.0
num_time_steps = 500
dt = t_max / num_time_steps


x_values = np.linspace(-x_max, x_max, num_points)
dx = x_values[1] - x_values[0]


V_matrix = 0.5 * m * omega**2 * (x_values - np.sin(omega*np.linspace(0, t_max, num_time_steps)))**2

T_matrix = -hbar**2 / (2 * m) * (np.diag(np.ones(num_points - 1), k=-1) - 2 * np.diag(np.ones(num_points)) + np.diag(np.ones(num_points - 1), k=1)) / dx**2


U_potential = np.exp(-1j * V_matrix * dt / hbar)
U_kinetic = np.exp(-1j * T_matrix * dt / hbar)

psi = np.exp(-x_values**2 / 2.0) / np.sqrt(np.sqrt(np.pi))

wave_packet_data = []
for step in range(num_time_steps):
    psi = np.dot(U_potential[step], psi)
    psi = np.dot(U_kinetic, psi)
    psi = np.dot(U_potential[step], psi)
    
    wave_packet_data.append(np.abs(psi)**2)


wave_packet_data = np.array(wave_packet_data)
plt.figure(figsize=(10, 6))

for t_index, t in enumerate(np.linspace(0, t_max, 6)):
    time_step_index = int(t / t_max * num_time_steps)
    plt.plot(x_values, wave_packet_data[time_step_index], label=f'Time = {t:.2f}')
    
plt.xlabel('Position')
plt.ylabel('Probability Density')
plt.title('Wave Packet Evolution using Finite Difference Matrix')
plt.legend()
plt.grid()
plt.show()
