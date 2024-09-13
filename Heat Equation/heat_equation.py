import numpy as np
import matplotlib.pyplot as plt
import time
# import matplotlib.animation as animation

# # def heat_equation_1d_step(u, alpha, dx, dt):
# #     """
# #     Perform a single time step update for the heat equation in 1D.
    
# #     Parameters:
# #     u (np.ndarray): Current temperature distribution.
# #     alpha (float): Thermal diffusivity constant.
# #     dx (float): Spatial step size.
# #     dt (float): Time step size.
    
# #     Returns:
# #     np.ndarray: Updated temperature distribution after one time step.
# #     """
# #     # Create a new array to store the next time step
# #     u_new = np.copy(u)
    
# #     # Update temperature values based on the finite difference method
# #     for i in range(1, len(u) - 1):
# #         u_new[i] = u[i] + alpha * dt / dx**2 * (u[i+1] - 2*u[i] + u[i-1])
    
# #     return u_new

# # def animate_heat_equation(x, func, t_end, alpha=1, dx=0.01, dt=0.01):
# #     """
# #     Animate the heat equation in one dimension.
    
# #     Parameters:
# #     x (np.ndarray): Array of position values.
# #     func (np.ndarray): Initial temperature distribution at time t=0.
# #     t_end (float): End time for the simulation.
# #     alpha (float): Thermal diffusivity constant. Default is 1.
# #     dx (float): Spatial step size. Default is 0.01.
# #     dt (float): Time step size. Default is 0.01.
# #     """
# #     # Number of time steps to iterate
# #     n_steps = int(t_end / dt)
    
# #     # Initial temperature distribution
# #     u = np.copy(func)
    
# #     # Set up the figure, axis, and plot element for animation
# #     fig, ax = plt.subplots()
# #     line, = ax.plot(x, u)
# #     ax.set_xlim(x.min(), x.max())
# #     ax.set_ylim(-1.5, 1.5)  # Adjust based on expected temperature range
# #     ax.set_xlabel('Position')
# #     ax.set_ylabel('Temperature')
# #     ax.set_title('Heat Equation in 1D')

# #     def update(frame):
# #         nonlocal u
# #         u = heat_equation_1d_step(u, alpha, dx, dt)
# #         line.set_ydata(u)
# #         return line,

# #     ani = animation.FuncAnimation(fig, update, frames=n_steps, blit=True, interval=50, repeat=False)
# #     plt.show()

# # # Example usage:
# # x = np.linspace(0, 1, 100)  # Spatial positions
# # initial_func = np.sin(np.pi * x)  # Initial temperature distribution
# # t_end = 1.0  # End time for the simulation
# # alpha = 1  # Thermal diffusivity constant

# # animate_heat_equation(x, initial_func, t_end, alpha)

# # import numpy as np

# # def heat_equation_1d(x, func, t, alpha=1, dx=0.01, dt=0.01):
# #     """
# #     Simulate the heat equation in one dimension.
    
# #     Parameters:
# #     x (np.ndarray): Array of position values.
# #     func (np.ndarray): Initial temperature distribution at time t=0.
# #     t (float): Current time.
# #     alpha (float): Thermal diffusivity constant. Default is 1.
# #     dx (float): Spatial step size. Default is 0.01.
# #     dt (float): Time step size. Default is 0.01.
    
# #     Returns:
# #     np.ndarray: Updated temperature distribution at time t.
# #     """
# #     # Number of time steps to iterate
# #     n_steps = int(t / dt)
    
# #     # Copy the initial function to avoid modifying the original
# #     u = np.copy(func(x))
    
# #     for _ in range(n_steps):
# #         # Create a new array to store the next time step
# #         u_new = np.copy(u)
        
# #         # Update temperature values based on the finite difference method
# #         for i in range(1, len(x) - 1):
# #             u_new[i] = u[i] + alpha * dt / dx**2 * (u[i+1] - 2*u[i] - u[i-1])
        
# #         # Update the temperature array
# #         u = u_new
    
# #     return u

# # # Example usage:
# # x = np.arange(0, 1, 0.01)  # Spatial positions
# # def initial_func(x):
# #     return np.sin(np.pi * x)  # Initial temperature distribution
# # t = 0.1  # Current time
# # alpha = 1  # Thermal diffusivity constant

# # temperature_distribution = heat_equation_1d(x, initial_func, t, alpha)

# # print(temperature_distribution)

# # def zlims(x,func):
# #     z=func(x)
# #     yran=(np.max(z)-np.min(z))
# #     yave=(np.max(z)+np.min(z))/2
# #     return [yave-0.75*yran,yave+0.75*yran]
# # z_min = zlims(x,initial_func)[0]
# # z_max = zlims(x,initial_func)[1]

# # def plot_2d():
# #     print('plotting')
# #     z1=initial_func(x)
# #     z2=heat_equation_1d(x, initial_func, t, alpha)
# #     plt.axis([0,1, z_min, z_max])
# #     plt.grid()
# #     plt.plot(x,z1,color='red')
# #     plt.plot(x,z2,color='magenta')
# #     plt.show()

# # plot_2d()

# import numpy as np
# import matplotlib.pyplot as plt

# def heat_equation_1d(x, func, t, alpha=1, dx=0.01, dt=0.01):
#     """
#     Simulate the heat equation in one dimension.
    
#     Parameters:
#     x (np.ndarray): Array of position values.
#     func (np.ndarray): Initial temperature distribution at time t=0.
#     t (float): Current time.
#     alpha (float): Thermal diffusivity constant. Default is 1.
#     dx (float): Spatial step size. Default is 0.01.
#     dt (float): Time step size. Default is 0.01.
    
#     Returns:
#     np.ndarray: Updated temperature distribution at time t.
#     """
#     # Number of time steps to iterate
#     n_steps = int(t / dt)
    
#     # Copy the initial function to avoid modifying the original
#     u = np.copy(func(x))
    
#     for _ in range(n_steps):
#         # Create a new array to store the next time step
#         u_new = np.copy(u)
        
#         # Update temperature values based on the finite difference method
#         for i in range(1, len(x) - 1):
#             u_new[i] = u[i] + alpha * dt / dx**2 * (u[i+1] - 2*u[i] + u[i-1])
        
#         # Update the temperature array
#         u = u_new
    
#     return u

# # Example usage:
# x = np.arange(0, 1, 0.01)  # Spatial positions
# def initial_func(x):
#     return np.sin(np.pi * x)  # Initial temperature distribution
# t = 0.1  # Current time
# alpha = 1  # Thermal diffusivity constant

# temperature_distribution = heat_equation_1d(x, initial_func, t, alpha)
# print(temperature_distribution)

# def zlims(x, func):
#     z = func(x)
#     yran = (np.max(z) - np.min(z))
#     yave = (np.max(z) + np.min(z)) / 2
#     return [yave - 0.75 * yran, yave + 0.75 * yran]

# z_min = zlims(x, initial_func)[0]
# z_max = zlims(x, initial_func)[1]

# def plot_2d():
#     print('plotting')
#     z1 = initial_func(x)
#     z2 = heat_equation_1d(x, initial_func, t, alpha)
#     plt.axis([0, 1, z_min, z_max])
#     plt.grid()
#     plt.plot(x, z1, color='red', label='Initial')
#     plt.plot(x, z2, color='magenta', label='After t = {}'.format(t))
#     plt.legend()
#     plt.show()

# plot_2d()

import numpy as np

def execution_time(func):
    """
    Decorator to measure and print the time taken by a function to execute.

    Parameters:
        func (callable): The function to be decorated.

    Returns:
        callable: Decorated function.
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()  # Record start time
        result = func(*args, **kwargs)  # Call the original function
        end_time = time.time()  # Record end time
        execution_time = end_time - start_time  # Calculate execution time
        print(f"Execution time of '{func.__name__}': {execution_time:.6f} seconds")
        return result
    return wrapper

def numerical_second_derivative(x, y):
    """
    Returns the array of the second derivative of an array of function values y and independent variable x

    Parameters:
    x (np.ndarray): An array of numbers representing values on the x-axis
    y (np.ndarray): An array of numbers of the function value dependent on x

    Returns:
    np.ndarray: An array of second derivative values
    """
    # Initialize the output array
    second_deriv = np.zeros_like(y)
    
    # Compute the second derivative using central difference method
    for i in range(1, len(x) - 1):
        second_deriv[i] = (y[i+1] - 2*y[i] + y[i-1]) / (0.5*(x[i+1] - x[i-1]))**2

    # Handle boundaries with forward and backward difference
    second_deriv[0] = (y[2] - 2*y[1] + y[0]) / (x[1] - x[0])**2
    second_deriv[-1] = (y[-1] - 2*y[-2] + y[-3]) / (x[-1] - x[-2])**2

    return second_deriv

def numerical_differentiator_array(x,y):

    # Returns the array of the numerical derivative of an array of function values y and independent variable x
    
    # Inputs:           x       : an array of numbers representing values on the x axis
    #                   y       : an array of numbers of the function value dependent on x

    if len(x)!=len(y):                                          # Checking array length compatibility
        raise Exception("Variable and function do not match")
    output=np.zeros_like(y)                                     # Output array initialization
    for i in range(len(x)):                                     # Differentiation loop
        if i-1==-1:
            m=(y[i+1]-y[i])/(x[i+1]-x[i])
        elif i+1==len(x):
            m=(y[i]-y[i-1])/(x[i]-x[i-1])
        else:
            m=(y[i+1]-y[i-1])/(x[i+1]-x[i-1])
        output[i]=m
    return output

def numerical_second_derivative2(x,y):
    
    # Returns the array of the second derivative of an array of function values y and independent variable x

    # Inputs:           x       : an array of numbers representing values on the x axis
    #                   y       : an array of numbers of the function value dependent on x

    first_deriv=numerical_differentiator_array(x,y)     # Calling first function derivative
    second_deriv=numerical_differentiator_array(x,first_deriv)  # Calling second function derivative
    return second_deriv

@execution_time
def numerical_integrator_array(x, y, c=0):
    """
    Returns the array of the numerical integral of an array of function values y and independent variable x.

    Parameters:
        x (np.ndarray): Array of numbers representing values on the x axis.
        y (np.ndarray): Array of numbers representing function values dependent on x.
        c (float): Constant of integration (default is 0).

    Returns:
        np.ndarray: Array of numerical integral values.
    """
    if len(x) != len(y):  # Check array length compatibility
        raise ValueError("Lengths of x and y must match")
    
    output = np.zeros_like(y)  # Output array initialization
    s = 0  # Sum initialization
    
    for i in range(len(x)):  # Integration loop
        if i == 0:
            s += c
        else:
            s += 0.5 * (y[i] + y[i - 1]) * (x[i] - x[i - 1])  # Trapezoidal rule
        output[i] = s
        
    return output

@execution_time
def numerical_integrator_array2(x,y,c=0):
    # Returns the array of the numerical integral of an array of function values y and independent variable x
    
    # Inputs:           x       : an array of numbers representing values on the x axis
    #                   y       : an array of numbers of the function value dependent on x

    if len(x)!=len(y):                                          # Checking array length compatibility
        raise Exception("Variable and function do not match")   
    output=np.zeros_like(y)                                     # Output array initialization
    s=0                                                         # Sum initialization
    def mean(*nums):                                            # Average function
        return sum(nums)/len(nums)
    for i in range(len(x)):                                     # Integration loop
        if i==0:
            s+=c
        elif i+1==len(x):
            s+=(y[i])*(x[i]-x[i-1])
        else:
            s+=mean(y[i],y[i-1])*(x[i]-x[i-1])
        output[i]=s
    return output
@execution_time
def numerical_integrator1(x,func,c=0):

    # Returns the array of the numerical integral of a function y(x), int y(x) dx
    
    # Inputs:           x       : an array of numbers representing values on the x axis
    #                   func    : a function taking x as an input
    #                   c       : a constant of integration, default is 0

    y=func(x)                                       # Creating the function values
    output=np.zeros_like(y)                         # Initializing an array          
    s=0                                             # Initializing sum

    def mean(*nums):
        return sum(nums)/len(nums)
    for i in range(len(x)):                         # Integration loop
        if i==0:
            s+=c
        elif i+1==len(x):
            s+=(y[i])*(x[i]-x[i-1])
        else:
            s+=mean(y[i],y[i-1])*(x[i]-x[i-1])
        output[i]=s
    return output
@execution_time
def numerical_integrator(x, func, c=0):
    """
    Returns the array of the numerical integral of a function y(x), ∫ y(x) dx.

    Parameters:
        x (np.ndarray): Array of numbers representing values on the x axis.
        func (callable): A function taking x as an input.
        c (float): A constant of integration, default is 0.

    Returns:
        np.ndarray: Array of numerical integral values.
    """
    y = func(x)  # Creating the function values
    output = np.zeros_like(y)  # Initializing an array

    # Using the trapezoidal rule for integration
    dx = np.diff(x)
    mid_y = (y[:-1] + y[1:]) / 2  # Midpoints of y values for trapezoidal rule

    # Calculate cumulative sum of areas of trapezoids
    cumulative_sum = np.cumsum(mid_y * dx)
    
    output[1:] = cumulative_sum
    output[0] = c  # Initial value with the constant of integration
    
    output += c  # Add constant of integration to all values
    
    return output

# Example usage:
x = np.arange(0, 10, 0.000001)
def func(x):
    return np.sin(x)
# FFy= numerical_second_derivative(x, y)
F1y=numerical_integrator1(x,func)
Fy=numerical_integrator(x,func)


@execution_time
def plot_2d():
    print('plotting')
    # plt.axis([0, 1, z_min, z_max])
    plt.grid()
    plt.plot(x, func(x), color='red', label='y(x)')
    plt.plot(x, Fy, color='pink',label='y\'(x)')
    # plt.plot(x, second_derivative_values, color='magenta',label='y\'\'(x)')
    plt.legend()
    plt.show()

plot_2d()
