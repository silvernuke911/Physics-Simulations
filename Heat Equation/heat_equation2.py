import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import time
import os
from matplotlib.animation import PillowWriter
import matplotlib.font_manager as font_manager
mpl.rcParams['font.family']='serif'
cmfont = font_manager.FontProperties(fname=mpl.get_data_path() + '/fonts/ttf/cmr10.ttf')
mpl.rcParams['font.serif']=cmfont.get_name()
mpl.rcParams['mathtext.fontset']='cm'
mpl.rcParams['axes.unicode_minus']=False
mpl.rcParams['axes.formatter.use_mathtext']=True
mpl.rcParams.update({'font.size': 12})

def progress_bar(progress, total, start_time, scale=0.50):
    # Creates a progress bar on the command line, input is progress, total, and a present start time
    # progress and total can be any number, and this can be placed in a for or with loop

    percent = 100 * (float(progress) / float(total))                        # Calculate the percentage of progress
    bar = '█' * round(percent*scale) + '-' * round((100-percent)*scale)     # Create the progress bar string
    elapsed_time = time.time() - start_time                                 # Calculate elapsed time
    if progress > 0:                                                        # Estimate total time and remaining time
        estimated_total_time = elapsed_time * total / progress
        remaining_time = estimated_total_time - elapsed_time
        remaining_seconds = int(remaining_time)
        remaining_milliseconds = int((remaining_time - remaining_seconds) * 1_000)
        remaining_str = time.strftime("%H:%M:%S", time.gmtime(remaining_seconds))
        remaining_str = f"{remaining_str}.{remaining_milliseconds:03d}"
    else:
        remaining_str = '...'
    print(f'|{bar}| {percent:.2f}% Time remaining: {remaining_str}  ', end='\r')    # Print the progress bar with the remaining time
    if progress == total: 
        elapsed_seconds = int(elapsed_time)
        elapsed_ms=int((elapsed_time-elapsed_seconds)*1000)                         # Print elapsed time when complete
        elapsed_seconds =  time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
        print('\n'+f'Elapsed time : {elapsed_seconds}.{elapsed_ms:03d}')

def numerical_differentiator_array(x, y):
    """
    Returns the array of the numerical derivative of an array of function values y and independent variable x.

    Parameters:
        x (np.ndarray): Array of numbers representing values on the x axis.
        y (np.ndarray): Array of numbers representing function values dependent on x.

    Returns:
        np.ndarray: Array of numerical derivative values.
    """
    if len(x) != len(y):  # Check array length compatibility
        raise ValueError("Lengths of x and y must match")
    
    output = np.zeros_like(y)  # Output array initialization
    
    # Using central differences for the interior points
    output[1:-1] = (y[2:] - y[:-2]) / (x[2:] - x[:-2])
    
    # Using forward difference for the first point
    output[0] = (y[1] - y[0]) / (x[1] - x[0])
    
    # Using backward difference for the last point
    output[-1] = (y[-1] - y[-2]) / (x[-1] - x[-2])
    
    return output

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



def numerical_second_integral(x,y,c1=0,c2=0):
        
    # Returns the array of the second integral of an array of function values y and independent variable x

    # Inputs:           x       : an array of numbers representing values on the x axis
    #                   y       : an array of numbers of the function value dependent on x
    #                   c1      : constant of integration for the first integral, default is 0
    #                   c2      : constant of integration for the second integral, default is 0

    first_integral=numerical_integrator_array(x,y,c1)             # Calling first function integral
    second_integral=numerical_integrator_array(first_integral,c2) # Calling second function integral
    return second_integral

def numerical_second_derivative_3(x,y):
    if len(x)!=len(y):
        raise ValueError('Input Length not Equal')
    # z=np.zeros_like(y)
    # for i in range(len(y)):
    #     if i == 0:
    #         z[0]=(y[0]+y[1])/2
    #     elif i == (len(y)-1):
    #         z[(len(y)-1)]=(y[(len(y)-1)]+y[len(y)-2])/2
    #     else:
    #         diff_ave=((y[i]-y[i-1])-(y[i+1]-y[i]))
    #         z[i]=diff_ave
    # Initialize the output array
    second_deriv = np.zeros_like(y)
    
    # Compute the second derivative using central difference method
    for i in range(1, len(x) - 1):
        second_deriv[i] = (y[i+1] - 2*y[i] + y[i-1]) / (0.5*(x[i+1] - x[i-1]))**2

    # Handle boundaries with forward and backward difference
    second_deriv[0] = (y[1]+y[0])/2
    second_deriv[-1] = (y[-1]+y[-2])/2
    return second_deriv
def check_and_prompt_overwrite(filename):
    if os.path.isfile(filename):
        response = input(f"{filename} already exists, are you sure you want to overwrite it? (yes/no): ")
        if response.lower() == 'yes':
            print("Proceeding with overwrite...")
            return True
        elif response.lower() == 'no':
            print("Operation aborted.")
            return False
        else:
            print("Wrong input.")
            return False
    else:
        return True
def unit_step(x, a):
    """
    Returns a Heaviside unit step function array from 0 to 1 at the boundary value a.

    Parameters:
        x (np.ndarray): Array of x values.
        a (float): Boundary value.

    Returns:
        np.ndarray: Heaviside unit step function array.
    """
    output = np.zeros_like(x)
    output[x >= a] = 1
    return output
def dirac_delta_function(x):
    y=np.zeros_like(x)
    for i,el in enumerate(x):
        if np.isclose(el,0):
            y[i]=1
    return y
def square_func(x, a, b):
    # Initialize an array of zeros with the same shape as x
    result = np.zeros_like(x)
    
    # Set the elements to 1 where the condition a <= x <= b is met
    result[(x >= a) & (x <= b)] = 1
    
    return result
def numerical_second_derivative_4(x, y):
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
        dx = x[i+1] - x[i]  # Assuming uniform spacing
        if dx == 0:
            raise ValueError("Spacing (dx) cannot be zero.")
        second_deriv[i] = (y[i+1] - 2*y[i] + y[i-1]) / dx**2

    # Handle boundaries with forward and backward difference
    dx_forward = x[1] - x[0]
    dx_backward = x[-1] - x[-2]
    if dx_forward == 0 or dx_backward == 0:
        raise ValueError("Boundary spacing (dx) cannot be zero.")
    second_deriv[0] = (y[2] - 2*y[1] + y[0]) / dx_forward**2
    second_deriv[-1] = (y[-1] - 2*y[-2] + y[-3]) / dx_backward**2

    return second_deriv
def numerical_differentiator_array(x, y):
    """
    Returns the array of the numerical derivative of an array of function values y and independent variable x.

    Parameters:
        x (np.ndarray): Array of numbers representing values on the x axis.
        y (np.ndarray): Array of numbers representing function values dependent on x.

    Returns:
        np.ndarray: Array of numerical derivative values.
    """
    if len(x) != len(y):  # Check array length compatibility
        raise ValueError("Lengths of x and y must match")
    
    output = np.zeros_like(y)  # Output array initialization
    
    # Using central differences for the interior points
    output[1:-1] = (y[2:] - y[:-2]) / (x[2:] - x[:-2])

    # Using forward difference for the first point
    output[0] = (y[1] - y[0]) / (x[1] - x[0])
    
    # Using backward difference for the last point
    output[-1] = (y[-1] - y[-2]) / (x[-1] - x[-2])
    
    return output
def numerical_second_derivative_2(x,y):
    deriv1=numerical_differentiator_array(x,y)
    deriv1[0],deriv1[-1]=0,0
    deriv2=numerical_differentiator_array(x,deriv1)
    # fix the jaggedness on the unit step function. Why is it not smooth
    # the deriv1 has a plateau on top, it generates 2 points for a unit step where there should be one spike.
    # central differences is still the best method though

    return deriv2
dx = 0.05
dt = 0.001
x_min , x_max = 0,2*np.pi
t_min , t_max = 0,5
x = np.arange(x_min, x_max + dx, dx)
# y = dirac_delta_function(x)
# y = unit_step(x,0)
# y = np.exp(-16*(x-1)**2)+np.exp(-16*(x+1)**2)
y = np.exp(-x**2)
# y = square_func(x,-1.5,1.5)
# y=x
y = np.sin(x)
t = np.arange(t_min,t_max+dt,dt)

y_i=y
y_t_list = [0]*len(t)
dy_t_list = [0]*len(t)
ddy_t_list = [0]*len(t)
alpha = 1
for i , time_e in enumerate(t):
    if i==0:
        y_t_list[0]=y_i
        dy_t_list[0]=numerical_differentiator_array(x,y_i)
        # ddy_t_list[0]=numerical_second_derivative(x,y_i)
        ddy_t_list[0]=numerical_second_derivative_2(x,y_i)
    else: 
        # y_t_list[i]=y_i + alpha * numerical_second_derivative(x,y_i) * dt
        y_t_list[i]=y_i + alpha * numerical_second_derivative_2(x,y_i) * dt

        dy_t_list[i]=numerical_differentiator_array(x,y_i)
        # ddy_t_list[i]=numerical_second_derivative(x,y_i)
        ddy_t_list[i]=numerical_second_derivative_2(x,y_i)
        y_i = y_t_list[i]
    print(f'Generating plot at t = {time_e:.3f}, {100*time_e/t_max:.2f}% done', end='\r')
print()

# plt.style.use('dark_background')
def plot():
    print('Plotting ...')
    z = numerical_second_derivative_2(x,y)
    z2 = numerical_second_derivative(x,y)
    sz = numerical_differentiator_array(x,y)
    plt.plot(x,y,color='red')
    plt.plot(x,z,color='blue')
    plt.plot(x,sz,color='green')
    plt.plot(x,z2,color='cyan')
    plt.xlabel('x-axis')
    plt.ylabel(r'Temperature ($T$/$T_{\text{max}}$)')
    plt.xlim([x_min,x_max])
    plt.ylim([np.min(y)-0.25,np.max(y)+0.25])
    plt.title('1D Heat Equation Solver')
    plt.grid()
    plt.show()
    print('Plotted!')
plot()

# for accurate simulations of step functions, use the function numerical integration
# for accurate simuations of other functions with non-zero boundary slope, use numerical integration 2.
def animated_plot():
    metadata = dict(title='Movie', artist='silver')
    writer = PillowWriter(fps=20, metadata=metadata)
    fig = plt.figure()
    start_time=time.time()
    max_time=np.max(t)
    filename = "heat_eq14.gif"
    if not check_and_prompt_overwrite(filename):
        return
    print('Animating file ...')
    with writer.saving(fig, filename,100):
        for i in range(0,len(t),20):
            y = y_t_list[i]
            dy = dy_t_list[i]
            ddy = ddy_t_list[i]
            plt.plot(x,y,color='red',marker='')
            plt.plot(x,dy,color='blue',marker='')
            plt.plot(x,ddy,color='green',marker='')
            plt.xlabel('x-axis')
            plt.ylabel(r'Temperature ($T$/$T_{\text{max}}$)')
            plt.xlim([x_min,x_max])
            plt.ylim([np.min(y_t_list[0])-0.25,np.max(y_t_list[0])+0.25])
            plt.title(f'1D Heat Equation Solver : t={t[i]:.2f}')
            
            plt.grid()
            writer.grab_frame()
            # plt.pause(2)
            plt.pause(0.001)
            plt.clf()
            progress_bar(t[i],max_time,start_time)
        print("\n"+"File animated and saved!")

animated_plot()
