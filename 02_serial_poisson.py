# 02_serial_poisson.py

import numpy as np
import matplotlib.pyplot as plt
from paraview_helper import vtr
import argparse 
import sys 

parser = argparse.ArgumentParser()
parser.add_argument("--verbose", action = "store_true")
args = parser.parse_args(sys.argv[1:])

n:int = 16
h:float = 1.0 / n

# 创造包括 ghost 的坐标
x = h * np.arange(2 * n + 2) - 0.5 * h
y = h * np.arange(n + 2) - 0.5 * h

f  = np.zeros((2 * n + 2, n + 2))
u  = np.zeros((2 * n + 2, n + 2))
u0 = np.zeros((2 * n + 2, n + 2))

# 赋值 f
for i in range(2 * n + 2):
    for j in range(n + 2):
        f[i, j] = - 10 * np.exp(-((x[i] - 0.5)**2 + (y[j] - 0.5)**2) / 0.02)

# 边界条件
def g(x):
    return np.sin(5 * x)

def apply_bc():
    
    # 左、右 Dirichlet BC
    u[0, :] = - u[1, :]
    u[-1, :] = - u[-2, :]

    # 上、下 Neumann BC
    gx_h = g(x) * h
    u[:,  0] = u[:,  1] + gx_h
    u[:, -1] = u[:, -2] + gx_h

# 残差
def residual():
    return np.linalg.norm(u - u0) / (np.linalg.norm(u0) + 1e-10)

# 迭代
for iter in range(10000):

    # Jacobi sweep, slow
    # u[1:-1, 1:-1] = (u0[2:, 1:-1] + u0[0:-2, 1:-1] + u0[1:-1, 2:] + u0[1:-1, 0:-2] 
    #                  - f[1:-1, 1:-1] * h**2) / 4.
    
    # Gauss-Sediel sweep, faster
    for i in range(1, 2 * n + 1):
        for j in range(1, n + 1):
            u[i, j] = (u[i + 1, j] + u[i - 1, j] + 
                       u[i, j + 1] + u[i, j - 1]
                       - f[i, j] * h**2) / 4

    apply_bc()

    r = residual()
    u0[:, :] = u[:, :]
    if args.verbose:
        if iter % 500 == 0:
            print(f"iter = {iter}, residual = {r}")
    if r < 1e-5:
        print(f"Iteration is converged at {iter} step(s)")
        break

# 可视化：插值网格中心DOF至角点
x_interp = 0.5 * (x[0:-1] + x[1:])
y_interp = 0.5 * (y[0:-1] + y[1:])
u_interp = 0.25 * (u[0:-1, 0:-1] + u[1:, 0:-1] + u[0:-1, 1:] + u[1:, 1:])
field = {"solution": u_interp.reshape((1, 2 * n + 1, n + 1, 1))}
vtr("serial_poisson", x_interp, y_interp, np.array([0.0]), [1, 2 * n + 1], [1, n + 1], [1, 1], **field)