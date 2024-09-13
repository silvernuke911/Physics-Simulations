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

h = 5000
v = 100

m   = 20
A   = 0.1
C_d = 0.5

v_i = np.array([v, 100, 0])
p_i = np.array([0, 0, h])

p = np.zeros((len(t), 3))
v = np.zeros((len(t), 3))

p_d = np.zeros((len(t), 3))
v_d = np.zeros((len(t), 3))

def a_g(h):
    h = float(h)
    a = (kerbin.standard_gravitational_parameter) / ((h+kerbin.equatorial_radius)**2)
    return np.array([0,0,-a])


v[0]   = v_i
p[0]   = p_i
v_d[0] = v_i
p_d[0] = p_i

valid_data_points = 0

for i in range(1,len(t)):
    # Store positions and velocities without drag
    v[i] = v[i-1] + a_g(p[i-1,2]) * dt
    p[i] = p[i-1] + v[i] * dt

    # Store positions and velocities with drag
    a_d = drag_acc(m,v_d[i-1],A,C_d,rho(p_d[i-1,2]))
    v_d[i] = v_d[i-1] + a_d * dt + a_g(p_d[i-1,2]) * dt
    p_d[i] = p_d[i-1] + v_d[i] * dt

    valid_data_points += 1
    
    if p_d[i,2] < 0:
        break

p = p[:valid_data_points]
v = v[:valid_data_points]

p_d = p_d[:valid_data_points]
v_d = v_d[:valid_data_points]

def plot():
    def set_axes_equal(x,y,z,ax):
        x_limits = [np.min(x),np.max(x)]
        y_limits = [np.min(y),np.max(y)]
        z_limits = [np.min(z),np.max(z)]
        x_range = abs(x_limits[1] - x_limits[0])
        y_range = abs(y_limits[1] - y_limits[0])
        z_range = abs(z_limits[1] - z_limits[0])
        x_middle = np.mean(x_limits)
        y_middle = np.mean(y_limits)
        z_middle = np.mean(z_limits)
        plot_radius = 0.6*max([x_range, y_range, z_range])
        ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
        ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
        ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
    print('Plotting')

    fig = plt.figure()
    ax =  plt.axes(projection='3d')

    set_axes_equal(p[:, 0], p[:, 1], p[:, 2],ax)
    ax.set_aspect('equal')

    ax.plot(p[:, 0], p[:, 1], p[:, 2],zorder=2,marker='')
    ax.plot(p_d[:, 0], p_d[:, 1], p_d[:, 2],zorder=2,marker='')
    ax.plot(v[:, 0], v[:, 1], v[:, 2],zorder=2)
    ax.plot(v_d[:, 0], v_d[:, 1], v_d[:, 2],zorder=2)

    ax.set_xlabel('Horizontal Position (m)')
    ax.set_ylabel('Depth Position (m)')
    ax.set_zlabel('Vertical Position (m)')

    plt.show()
    print('Plotted')
plot()