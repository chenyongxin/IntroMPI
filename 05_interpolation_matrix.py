import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import fem, geometry, cpp, mesh, io, __version__

assert __version__ == "0.8.0", f"Version of FEniCSx does not match. V0.8.0 is needed, while {__version__} is used."

comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size

meshf = mesh.create_unit_square(comm, 32, 32)
Qf = fem.functionspace(meshf, ("Lagrange", 1))

# 4x4 网格单元，一共 5x5=25 DOF
meshs = mesh.create_unit_square(comm, 4, 4)
meshs.geometry.x[:, :2] -= 0.5
meshs.geometry.x[:, :2] /= 2.0
meshs.geometry.x[:, :2] += 0.5
Qs = fem.functionspace(meshs, ("Lagrange", 1))

# 可视化
f = fem.Function(fem.functionspace(meshf, ("DG", 0)))
f.x.array[:] = rank

s = fem.Function(fem.functionspace(meshs, ("DG", 0)))
s.x.array[:] = rank

with io.XDMFFile(comm, "f.xdmf", "w") as file:
    file.write_mesh(meshf)
    file.write_function(f)

with io.XDMFFile(comm, "s.xdmf", "w") as file:
    file.write_mesh(meshs)
    file.write_function(s)

# 找到各个进程中DOF所落在单元的进程: dof_to_rank
num_dofs = Qs.dofmap.index_map.size_local
dof_coords = Qs.tabulate_dof_coordinates()[:num_dofs]
dof_to_rank = cpp.geometry.determine_point_ownership(meshf._cpp_object, dof_coords, 1e-6)[0]
global_dofs = Qs.dofmap.index_map.local_to_global(np.arange(num_dofs))

print(f"Rank: {rank}, before exchange DOF to rank: {dof_to_rank}", flush=True)

# Manipulate data and get ready for MPI
sorted_rank_indices = np.argsort(dof_to_rank)
sorted_dof_to_rank  = np.array(dof_to_rank, dtype=np.int32)[sorted_rank_indices]
sorted_global_dofs  = global_dofs.astype(np.int32)[sorted_rank_indices]
sorted_dof_coords   = dof_coords[sorted_rank_indices]

# Send content
send_counts = np.zeros(size, dtype=np.int32)
for procs_id in sorted_dof_to_rank:
    send_counts[procs_id] += 1

send_disp = np.array([0, *(np.cumsum(send_counts)[:-1])], dtype=np.int32)

# Recv content
recv_counts = np.zeros(size, dtype=np.int32)
comm.Alltoall(send_counts, recv_counts)
recv_length = np.sum(recv_counts)

# Exchange global DOFs: recv_global_dofs
recv_global_dofs = np.zeros(recv_length, dtype=np.int32)
recv_disp = np.array([0, *(np.cumsum(recv_counts)[:-1])], dtype=np.int32)
comm.Alltoallv([sorted_global_dofs, send_counts, send_disp, MPI.INT],
               [recv_global_dofs,   recv_counts, recv_disp, MPI.INT])

# Exchange coordinates of global DOFs: recv_global_dof_coords
recv_global_dof_coords = np.zeros((recv_length, 3), dtype=np.float64)
comm.Alltoallv([sorted_dof_coords,      send_counts*3, send_disp*3, MPI.DOUBLE],
               [recv_global_dof_coords, recv_counts*3, recv_disp*3, MPI.DOUBLE])


dof_to_rank = cpp.geometry.determine_point_ownership(meshf._cpp_object, recv_global_dof_coords, 1e-6)[0]
print(f"Rank: {rank}, after exchange DOF to rank: {dof_to_rank}", flush=True)
