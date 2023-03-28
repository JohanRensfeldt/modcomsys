import numpy as np
import matplotlib.pyplot as plt

interval = (2.8, 4)  # start, end
accuracy = 0.0001
reps = 600  # number of repetitions
numtoplot = 200
lims = np.zeros(reps)
a = 0.01

def sine_map(r,x):
    return r*np.sin(np.pi*x)

def logistic_map(r,x):
    return r*x*(1-x)

def perturbed_logistic_map(r, x, a):
    return r * x * (1 - x) + a * x**4

def find_bifurcations(rmin, rmax, n, iterations, a):
    r = np.linspace(rmin, rmax, 10000)
    x = np.random.rand(len(r))
    lastx = np.zeros(len(r))
    for i in range(iterations):
        x = perturbed_logistic_map(r, x, a)
        if i >= iterations-n:
            lastx = x
    return r[np.abs(x - lastx) < 1e-6]

r1 = find_bifurcations(*interval, 6, reps, a)
print(f"First 6 bifurcations: {r1}")

# calculate alpha and beta
alpha = np.mean([(r1[i+1]-r1[i])/(r1[i+2]-r1[i+1]) for i in range(len(r1)-2)])
print(f"Feigenbaum constant alpha: {alpha:.4f}")

beta = (r1[-1]-r1[-2])/(r1[-2]-r1[-3])
print(f"Scaling factor beta: {beta:.4f}")


fig, biax = plt.subplots()
fig.set_size_inches(16, 9)

lims[0] = np.random.rand()
for r in np.arange(interval[0], interval[1], accuracy):
    for i in range(reps-1):
        lims[i+1] = perturbed_logistic_map(r, lims[i],a)

    biax.plot([r]*numtoplot, lims[reps-numtoplot:], 'b.', markersize=.02)

biax.set(xlabel='r', ylabel='x', title='logistic map')
plt.show()