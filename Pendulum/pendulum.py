import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import time
import os
from matplotlib.animation import PillowWriter
import matplotlib.font_manager as font_manager
import vector_operations as vc 
from progress import progress_bar

plt.rcParams.update({
    'font.family': 'serif',
    'mathtext.fontset': 'cm',
    'axes.unicode_minus': False,
    'axes.formatter.use_mathtext': True,
    'font.size': 12
})
cmfont = font_manager.FontProperties(fname=mpl.get_data_path() + '/fonts/ttf/cmr10.ttf')

import numpy as np
import matplotlib.pyplot as plt

# Pendulum parameters
length = 0.5                # Length of the pendulum
gravity = 9.8               # Acceleration due to gravity (m/s^2)
initial_angle = np.pi / 4  # Initial angle of the pendulum (radians)
dt = 0.01                   # Time step for simulation
max_time = 10.0             # Total simulation time
damping_coefficient = 0.0

# Arrays to store simulation results
t = np.arange(0, max_time, dt)
theta = np.zeros_like(t)    # Angular displacement
omega = np.zeros_like(t)    # Angular velocity

# Initial conditions
theta[0] = initial_angle
omega[0] = 0.0

# Perform simulation using Euler's method
for i in range(1, len(t)):
    # Calculate angular acceleration using the equation of motion
    alpha = - gravity / length * np.sin(theta[i-1]) - damping_coefficient * omega[i-1]
    
    # Update angular velocity and displacement using Euler's method
    omega[i] = omega[i-1] + alpha * dt
    theta[i] = theta[i-1] + omega[i] * dt

# Convert angular displacement to Cartesian coordinates
r = np.column_stack((length * np.sin(theta), -length * np.cos(theta)))

# Plotting the pendulum motion
plt.plot(r[:,0], r[:,1])
plt.title('Simple Pendulum Simulation')
plt.xlabel('Horizontal Displacement (m)')
plt.ylabel('Vertical Displacement (m)')

plt.scatter(r[0][0],r[0][1],marker='.',s=60,color='red',zorder=2)
plt.scatter(0,0,marker='.',s=60,color='blue',zorder=2)
plt.plot([0,r[0][0]],[0,r[0][1]])

plt.gca().set_aspect('equal', adjustable='box')
plt.grid(True)
plt.show()
plt.close()

for i in range(len(t)):

    plt.plot(r[:,0], r[:,1])
    plt.scatter(r[i][0],r[i][1],marker='.',s=60,color='red',zorder=2)

    # plt.plot(theta , omega)
    # plt.scatter(theta[i],omega[i],marker='.',s=60,color='red',zorder=2)

    plt.title(f'Simple Pendulum Simulation ; t = {t[i]:.2f}')
    plt.xlabel('Horizontal Displacement (m)')
    plt.ylabel('Vertical Displacement (m)')

    
   

    plt.scatter(0,0,marker='.',s=60,color='blue',zorder=2)
    plt.plot([0,r[i][0]],[0,r[i][1]])

    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True)
    plt.pause(0.001)
    plt.clf()