import numpy as np
import matplotlib.pyplot as plt
import vector_operations as vc
from ksp_planets import *
from progress import progress_bar
import time

def drag_acc(m, v, area, C_d, rho):
    return -(0.5/m)*area*C_d*rho* (vc.mag(v))**2 * vc.normalize(v)
def rho(h):
    return 1.225*np.exp( -0.00012305 * h)
    
dt = 0.05
t = np.arange(0,500+dt,dt)

h = 1000
v = 100

m   = 20
A   = 0.1
C_d = 0.5

v_i = np.array([v, 0])
p_i = np.array([0, 1000])

p = np.zeros((len(t), 2))
v = np.zeros((len(t), 2))
a = np.zeros((len(t), 2))

p_d = np.zeros((len(t), 2))
v_d = np.zeros((len(t), 2))
a_d = np.zeros((len(t), 2))

def a_g(h):
    h = float(h)
    a = (kerbin.standard_gravitational_parameter) / ((h+kerbin.equatorial_radius)**2)
    return np.array([0,-a])


v[0]   = v_i
p[0]   = p_i
v_d[0] = v_i
p_d[0] = p_i

a[0] = a_g(p_i[1])
a[0] = a_g(p_i[1]) + drag_acc(m,v_i,A,C_d,rho(p_i[1]))
valid_data_points = 0

for i in range(1,len(t)):
    # Store positions and velocities without drag
    a[i] = a_g(p[i-1,1])
    v[i] = v[i-1] + a[i] * dt
    p[i] = p[i-1] + v[i] * dt
    

    # Store positions and velocities with drag
    a_d[i] = a_g(p_d[i-1,1]) + drag_acc(m,v_d[i-1],A,C_d,rho(p_d[i-1,1]))
    v_d[i] = v_d[i-1] + a_d[i] * dt
    p_d[i] = p_d[i-1] + v_d[i] * dt

    valid_data_points += 1
    
    if p_d[i,1] < 0:
        break

p = p[:valid_data_points]
v = v[:valid_data_points]
a = a[:valid_data_points]
p_d = p_d[:valid_data_points]
v_d = v_d[:valid_data_points]
a_d = a_d[:valid_data_points]

plt.plot(p[:,0] , p[:,1] , marker = '' )
plt.plot(v[:,0] , v[:,1] , marker = '' )
plt.plot(a[:,0] , a[:,1] , marker = '' )

plt.plot(p_d[:,0], p_d[:,1], marker = '' )
plt.plot(v_d[:,0], v_d[:,1], marker = '' )
plt.plot(a_d[:,0], a_d[:,1], marker = '' )

plt.xlabel('Horizontal Position (m)')
plt.ylabel('Vertical Position (m)')

ax = plt.gca()
ax.set_axisbelow(True)
ax.set_aspect('equal', adjustable='box')
plt.grid()
plt.show()


# We have officially coded and demonstrated the trajectory of a bomb drop under air resistance. 
# The next step would be to figure out the physical characteristics of the bomb
# Some notes - it arives late. It may have something to do with the time, however, since everything is distance based, maybe distance is truly enough. We need to experiment to know for sure
# We also have to experimentally determine the parameters for the bomb. Mass and rho is easy, C_D and A are to be experimentally determined. We can do a data analysis by dropping them from 1 km to 100 m, then analyze the trajectory.
# We can do two for loops, one determining the area from 0 to one, and one determining C_d, each spaced from 0 to one with 0.01 spacing, running the simulation each and every time. It then returns the result closest to experimental data
# We can then iterate within that 0.01 to 0.001 to have 3 sigfig accuracy. So now we have it, nice!
# Majority of what I have been doing now is just solving problems that have been bugging me for the past few years

x_i = 0
v_i = 10

dt = 0.01
t = np.arange(0,100+dt,dt)

x = np.zeros_like(t)
v = np.zeros_like(t)

rh = 1.225
C_d = 0.5
A = 0.1
m = 0.5

x[0] = x_i
v[0] = v_i
def acc(m, A, rh, C_d, v):
    return -(0.5/m) * A * rh * C_d * v**2 

for i in range(1,len(t)):
    a = acc(m, A, rh, C_d, v[i-1])
    v[i] = v[i-1] + a * dt
    x[i] = x[i-1] + v[i] * dt  

plt.plot(t,x)
plt.plot(t,v)
plt.grid()
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Initializations
dt = 0.001
t = np.arange(0,100+dt,dt)

x_i = np.array([0,0])
v_i = np.array([5,5])

# Array preallocation
x = np.zeros((len(t),2))
v = np.zeros((len(t),2))
x_d = np.zeros((len(t),2))
v_d = np.zeros((len(t),2))

# Drag equation parameters
rh = 1.225
C_d = 0.5
A = 0.1
m = 1
def acc(m, A, rh, C_d, v):
    return -(0.5/m) * A * rh * C_d * v**2 * np.linalg.norm(v)

# Initial values
x[0] = x_i
v[0] = v_i
x_d[0] = x_i
v_d[0] = v_i
a_gr = np.array([0,-9.8])
# Simulation Loop
for i in range(1,len(t)):
    v[i] = v[i-1] + a_gr * dt           # No drag
    x[i] = x[i-1] + v[i] * dt  

    a_d = acc(m, A, rh, C_d, v_d[i-1])  # With drag
    v_d[i] = v_d[i-1] + (a_gr + a_d )* dt
    x_d[i] = x_d[i-1] + v_d[i] * dt  

    if x_d[i,1]<0: # Check if it hits the ground
        break

# Cutting breakpoints
x = x[:i+1]
v = v[:i+1]
x_d= x_d[:i+1]
v_d = v_d[:i+1]

plt.plot(x[:,0],x[:,1], label = 'No Drag')
plt.plot(x_d[:,0],x_d[:,1], label = 'With Drag')
plt.legend()
plt.grid()
plt.show()
    


