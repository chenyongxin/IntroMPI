# 03_parallel_poisson.py

from mpi4py import MPI
import numpy as np
from paraview_helper import vtr
import argparse 
import sys 

parser = argparse.ArgumentParser()
parser.add_argument("--verbose", action = "store_true")
args = parser.parse_args(sys.argv[1:])

# 得到MPI信息
comm = MPI.COMM_WORLD
size = comm.size
rank = comm.rank

# 1 维非周期性拓扑结构相邻进程信息，可拓展至 n 很大
# 1D non-periodic MPI topo where `n` can be extended to a very big number rather than 2
processes = np.arange(size)
left  = -1 if rank == 0 else np.roll(processes,  1)[rank]
right = -1 if rank == size - 1 else np.roll(processes, -1)[rank]

if size != 2:
    if rank == 0:
        print("Use 2 processes to run: mpirun -n 2 03_parallel_poisson.py")
    exit()

n:int = 16
h:float = 1.0 / n

# 创造包括 ghost 的坐标
x = h * np.arange(n + 2) - 0.5 * h + 1 * rank
y = h * np.arange(n + 2) - 0.5 * h

f  = np.zeros((n + 2, n + 2))
u  = np.zeros((n + 2, n + 2))
u0 = np.zeros((n + 2, n + 2))

# 赋值 f
for i in range(n + 2):
    for j in range(n + 2):
        f[i, j] = - 10 * np.exp(-((x[i] - 0.5)**2 + (y[j] - 0.5)**2) / 0.02)

# 边界条件
def g(x):
    return np.sin(5 * x)

def apply_bc():
    
    # 左、右 Dirichlet BC
    if left < 0:
        u[0, :] = - u[1, :]
    if right < size - 1:
        u[-1, :] = - u[-2, :]

    # 上、下 Neumann BC
    gx_h = g(x) * h
    u[:,  0] = u[:,  1] + gx_h
    u[:, -1] = u[:, -2] + gx_h

# 残差
def residual():
    diff = comm.allreduce(np.sqrt(np.sum((u[1:-1, 1:-1] - u0[1:-1, 1:-1])**2)), op = MPI.SUM)
    old  = comm.allreduce(np.sqrt(np.sum((u0[1:-1, 1:-1])**2)), op = MPI.SUM)
    return diff / (old + 1e-10)

def update_ghost():
    
    # 往右发，从左收
    send_buff = np.zeros(n + 2)
    recv_buff = np.zeros(n + 2)
    # 注意，这里需要内存连续
    # Make sure the packed data is contiguous in memory
    send_buff[:] = u[-2, :]        
    req1 = comm.Isend(send_buff, right, tag = 1)
    req2 = comm.Irecv(recv_buff, left, tag = 1)
    req1.wait()
    req2.wait()
    u[0, :] = recv_buff

    # 往左发，从右收
    send_buff = np.zeros(n + 2)
    recv_buff = np.zeros(n + 2)
    send_buff[:] = u[1, :]
    req1 = comm.Isend(send_buff, left, tag = 2)
    req2 = comm.Irecv(recv_buff, right, tag = 2)
    req1.wait()
    req2.wait()
    u[-1, :] = recv_buff

# 迭代
for iter in range(10000):

    # Gauss-Sediel sweep
    update_ghost()

    for i in range(1, n + 1):
        for j in range(1, n + 1):
            u[i, j] = (u[i + 1, j] + u[i - 1, j] + 
                       u[i, j + 1] + u[i, j - 1]
                       - f[i, j] * h**2) / 4

    apply_bc()

    r = residual()
    u0[:, :] = u[:, :]
    if args.verbose:
        if (iter % 500 == 0) & (rank == 0):
            print(f"iter = {iter}, residual = {r}", flush=True)
    if r < 1e-5:
        if rank == 0:
            print(f"Iteration is converged at {iter} step(s)")
        break

# 可视化：插值网格中心DOF至角点
update_ghost()
x_interp = 0.5 * (x[0:-1] + x[1:])
y_interp = 0.5 * (y[0:-1] + y[1:])
u_interp = 0.25 * (u[0:-1, 0:-1] + u[1:, 0:-1] + u[0:-1, 1:] + u[1:, 1:])
field = {"solution": u_interp.reshape((1, n + 1, n + 1, 1))}
vtr(f"parallel_poisson_{rank}", x_interp, y_interp, np.array([0.0]), [1, n + 1], [1, n + 1], [1, 1], **field)

# Coursework: Add a parallel paraview output with pvtr.