import numpy as np
from sern import *

# simple MPI example

from mpi4py import MPI
from time import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

print('(%i/%i) load data from <nodes> and <links> ..' % (rank, size))
ids, lon, lat = np.loadtxt('nodes', unpack = True)
links = np.loadtxt('links', dtype = 'int')

print('(%i/%i) construct adjacency matrix and edge list ..' % (rank, size))
A, b = AdjacencyMatrix(ids, links)
lon, lat = lon[b], lat[b]
n = A.shape[0]
A[np.tril_indices(n)] = 0
edges = np.transpose(A.nonzero())
A = A[np.triu_indices(n, 1)]

print('(%i/%i) get all link distances ..' % (rank, size))
D, x = IntegerDistances(lat, lon)

print('(%i/%i) derive link probability ..' % (rank, size))
p = LinkProbability(A, D)

print('(%i/%i) original degree measure ..' % (rank, size))
v = np.bincount(edges.ravel())

nserns = 1024
sbatch = nserns // size
nserns = sbatch * size
vm = np.zeros((sbatch, n))

print('(%i/%i) measure on SERNs ..' % (rank, size))
start_time = time()
for i in range(sbatch):
    edges = SernEdges(D, p, n)
    vm[i,:] = np.bincount(edges.ravel())

stop_time = time()
print('(%i/%i) sern ensemble computation time:' % (rank, size), stop_time - start_time)

vmean = np.zeros((sbatch, n))
comm.Barrier()
print('(%i/%i) sum reduce ensemble to master ..' % (rank, size))
start_time = time()

for i in range(sbatch):
    comm.Reduce(vm[i,:], vmean[i,:], op = MPI.SUM, root = 0)

comm.Barrier()
vmean = vmean.sum(axis = 0)
vmean /= nserns
stop_time = time()
print('(%i/%i) reduction communication time:' % (rank, size), stop_time - start_time)
comm.Barrier()

# you should save final results to disk !
#if rank == 0:
#    np.save('vmean.npy', vmean) # e.g.

# in a real application exit the script here and do the visualization
# in an extra python script ..
if rank == 0:
    from matplotlib import pyplot as pl

    print('plot full example ..')
    fg, ax = pl.subplots(3, 1, sharex = True, sharey = True,
                         figsize = (19.2, 10.8))

    # original measure
    c = v
    im = ax[0].scatter(lon, lat, s = Scale(c), c = c,
               cmap = pl.cm.magma_r)
    cb = fg.colorbar(im, ax = ax[0])
    cb.set_label('Degree centrality')

    # sern ensemble mean
    c = vmean
    im = ax[1].scatter(lon, lat, s = Scale(c), c = c,
               cmap = pl.cm.magma_r)
    cb = fg.colorbar(im, ax = ax[1])
    cb.set_label('SERN Degree centrality')

    # corrected measure
    c = v - vmean
    im = ax[2].scatter(lon, lat, s = Scale(c), c = c,
               vmax = c.max(), vmin = -c.max(),
               cmap = pl.cm.seismic)
    cb = fg.colorbar(im, ax = ax[2])
    cb.set_label('Corrected Degree centrality (original - SERN)')

    pl.show()

