# 04_fenicsx_partition.py

from dolfinx import mesh, io, fem
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.rank

# 建立网格、函数空间和函数
_mesh = mesh.create_unit_square(comm, 32, 32)
Q = fem.functionspace(_mesh, ("DG", 0))
f = fem.Function(Q)
f.x.array[:] = rank

# 输出结果
with io.XDMFFile(comm, "FEniCSx_partition.xdmf", "w") as file:
    file.write_mesh(_mesh)
    file.write_function(f)