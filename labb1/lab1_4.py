import numpy as np
from scipy.integrate import solve_ivp

def lorenz(t, y, sigma, rho, beta):
    dydt = [sigma*(y[1] - y[0]),
            y[0]*(rho - y[2]) - y[1],
            y[0]*y[1] - beta*y[2]]
    return dydt

def jacobian(t, y, sigma, rho, beta):
    J = np.array([[-sigma, sigma, 0],
                  [rho - y[2], -1, -y[0]],
                  [y[1], y[0], -beta]])
    return J

# Parameters
sigma = 10
rho = 28
beta = 8/3

# Initial condition
y0 = [1, 1, 1]

# Compute the Lyapunov exponent
N = 10000
lyapunov_sum = 0
y = np.array(y0)
for i in range(N):
    sol = solve_ivp(lorenz, [0, 0.01], y, method='LSODA', jac=jacobian, args=(sigma, rho, beta))
    y = sol.y[:, -1]
    J = jacobian(0, y, sigma, rho, beta)
    w, v = np.linalg.eig(J.T @ J)
    lyapunov_sum += np.log(np.sqrt(np.max(w)))
lyapunov = lyapunov_sum / N
print("Lyapunov exponent:", lyapunov)