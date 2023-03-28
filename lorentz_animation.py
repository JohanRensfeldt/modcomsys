import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Runge-Kutta (RK4) Numerical Integration for System of First-Order Differential Equations


plt.style.use('dark_background')

def rk4(func, tk, _yk, _dt=0.01, **kwargs):
    """
    single-step fourth-order numerical integration (RK4) method
    func: system of first order ODEs
    tk: current time step
    _yk: current state vector [y1, y2, y3, ...]
    _dt: discrete time step size
    **kwargs: additional parameters for ODE system
    returns: y evaluated at time k+1
    """

    # evaluate derivative at several stages within time interval
    f1 = func(tk, _yk, **kwargs)
    f2 = func(tk + _dt / 2, _yk + (f1 * (_dt / 2)), **kwargs)
    f3 = func(tk + _dt / 2, _yk + (f2 * (_dt / 2)), **kwargs)
    f4 = func(tk + _dt, _yk + (f3 * _dt), **kwargs)

    # return an average of the derivative over tk, tk + dt
    return _yk + (_dt / 6) * (f1 + (2 * f2) + (2 * f3) + f4)

def lorenz(_t, _y, sigma=10, beta=(8 / 3), rho=28):
    """
    lorenz chaotic differential equation: dy/dt = f(t, y)
    _t: time tk to evaluate system
    _y: 3D state vector [x, y, z]
    sigma: constant related to Prandtl number
    beta: geomatric physical property of fluid layer
    rho: constant related to the Rayleigh number
    return: [x_dot, y_dot, z_dot]
    """
    return np.array([
        sigma * (_y[1] - _y[0]),
        _y[0] * (rho - _y[2]) - _y[1],
        (_y[0] * _y[1]) - (beta * _y[2]),
    ])


# ==============================================================
# simulation harness

# discrete time step size
dt = 0.01

# simulation time range
time = np.arange(0.0, 8.0, dt)

# lorenz initial conditions (x, y, z) at t = 0
y0 = np.array([-7, 8, 26])

# ==============================================================
# propagate state

# simulation results
state_history = []

# initialize yk
yk = y0

# intialize time
t = 0

# iterate over time
for t in time:
    # save current state
    state_history.append(yk)

    # update state variables yk to yk+1
    yk = rk4(lorenz, t, yk, dt)

# convert list to numpy array
state_history = np.array(state_history)

print(f'y evaluated at time t = {t} seconds: {yk[0]}')

# ==============================================================
# plot history


fig = plt.figure()  # figsize=(10, 8)
ax = plt.axes(projection='3d')
ax.set_xlim3d(min(state_history[:, 0]) - 0.05, max(state_history[:, 0]) + 0.05)
ax.set_ylim3d(min(state_history[:, 1]) - 0.05, max(state_history[:, 1]) + 0.05)
ax.set_zlim3d(min(state_history[:, 2]) - 0.05, max(state_history[:, 2]) + 0.05)

# trajectory data to plot
trajectory, = ax.plot([], [], [])

ax.set(xlabel='X', ylabel='Y', zlabel='Z', title='The Lorenz Equations - "Lorenz Attractor Simulation"')
ax.w_xaxis.set_pane_color((0.25, 0.25, 0.2, 0.4))
ax.w_yaxis.set_pane_color((0.25, 0.25, 0.2, 0.4))
ax.w_zaxis.set_pane_color((0.25, 0.25, 0.2, 0.4))
ax.grid()

# roate matplotlib axes
vertical_rotation_angles = np.linspace(0, 30, len(state_history[:, 0]) - 1)
horizontal_rotation_angles = np.linspace(0, 360, len(state_history[:, 0]) - 1)


def animate(i):
    # update axis view angle
    i -= len(time) * (i // len(time))
    ax.view_init(vertical_rotation_angles[i - 1], horizontal_rotation_angles[i - 1])

    # update trajectory for current time step
    trajectory.set_data(state_history[:i, 0], state_history[:i, 1])
    trajectory.set_3d_properties(state_history[:i, 2])
    return trajectory,


# show animation
anim = animation.FuncAnimation(fig, animate, interval=10, blit=False, save_count=len(time))
plt.show()
