# 01_parallel_sum.py

from mpi4py import MPI
import argparse 
import sys 

# 得到MPI信息
comm = MPI.COMM_WORLD
size = comm.size
rank = comm.rank

# 1. 任务分配
# ----------
# 0号进程拿到总数，并将信息[分发]给其他进程
total_number = 0 
if rank == 0:
    parser = argparse.ArgumentParser()
    parser.add_argument("--total", type = int, 
                        default = 100, help = "Sum of total number")
    args = parser.parse_args(sys.argv[1:])
    total_number = args.total

total_number = comm.bcast(total_number, root = 0)

# 计算每个进程个数
portion = total_number // size
if rank == size - 1:
    portion = total_number - portion * (size - 1)

offset = total_number // size
start = offset * rank + 1
end = start + portion

# 2. 每个进程各自计算求和
# ---------------------
s = 0 
for i in range(start, end):
    s += i

# 3. 汇总各自结果
# --------------
s = comm.reduce(s, op = MPI.SUM, root = 0)
if rank == 0:
    print(f"{size} process(es) to compute 1 + 2 + ... + {total_number} = {s}") 