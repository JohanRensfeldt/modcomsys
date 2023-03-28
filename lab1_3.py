import numpy as np
import matplotlib.pyplot as plt


# Runge-Kutta (RK4) Numerical Integration for System of First-Order Differential Equations


def ode_system(_t, _y, _sigma, _rho, _beta):
    """
    Lorenz system of first order differential equations
    _t: discrete time step value
    _y: state vector [y1, y2, y3]
    _sigma, _rho, _beta: Lorenz system parameters
    """
    y1, y2, y3 = _y
    return np.array([_sigma * (y2 - y1), y1 * (_rho - y3) - y2, y1 * y2 - _beta * y3])


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


# ==============================================================
# simulation harness

dt = 0.01
time = np.arange(0.0, 50.0 + dt, dt)

# Lorenz system parameters
sigma, rho, beta = 10, 28, 8/3

# third order system initial conditions [y1, y2, y3] at t = 0
y0 = np.array([0, 1, 0])

# ==============================================================
# propagate state

# simulation results
state_history = []

# initialize yk
yk = y0

# intialize time
t = 0

# approximate y at time t
for t in time:
    state_history.append(yk)
    yk = rk4(ode_system, t, yk, dt, _sigma=sigma, _rho=rho, _beta=beta)

# convert list to numpy array
state_history = np.array(state_history)

print(f'y evaluated at time t = {t} seconds: {yk[0]}')

# ==============================================================
# plot history

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

x = state_history[:, 0]
y = state_history[:, 1]
z = state_history[:, 2]

ax.plot(x, y, z)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title('Lorenz System Propagation')
plt.show()