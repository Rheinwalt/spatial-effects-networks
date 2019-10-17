import numpy as np
from matplotlib import pyplot as pl
from scipy.stats import percentileofscore
import graph_tool.all as gt
from sern import *

print('load data from <nodes> and <links> ..')
ids, lon, lat = np.loadtxt('nodes', unpack = True)
links = np.loadtxt('links', dtype = 'int')

print('construct adjacency matrix and edge list ..')
A, b = AdjacencyMatrix(ids, links)
lon, lat = lon[b], lat[b]
n = A.shape[0]
A[np.tril_indices(n)] = 0
edges = np.transpose(A.nonzero())
A = A[np.triu_indices(n, 1)]

print('get all link distances ..')
D, x = IntegerDistances(lat, lon)

print('derive link probability ..')
p = LinkProbability(A, D)

print('original measure ..')
g = Graph(edges, n)
_, v = gt.eigenvector(g)
vo = np.array(v.a)

nserns = 1000
var = np.zeros((nserns, len(vo)))

print('measure on SERNs ..')
for i in range(var.shape[0]):
    e = SernEdges(D, p, n)
    g = Graph(e, n)
    _, v = gt.eigenvector(g)
    v = np.array(v.a)
    var[i] = v

print('plot full example ..')
fg, ax = pl.subplots(2, 2, figsize = (19.2, 10.8))

# original measure
c = vo
im = ax[0,0].scatter(lon, lat, s = Scale(c), c = c,
           cmap = pl.cm.magma_r)
cb = fg.colorbar(im, ax = ax[0,0])
cb.set_label('Eigenvector centrality')

# sern ensemble mean
c = var.mean(axis = 0)
im = ax[0,1].scatter(lon, lat, s = Scale(c), c = c,
           cmap = pl.cm.magma_r)
cb = fg.colorbar(im, ax = ax[0,1])
cb.set_label('SERN eigenvector centrality')

# corrected measure
c = vo - var.mean(axis = 0)
im = ax[1,0].scatter(lon, lat, s = Scale(c), c = c,
           vmax = c.max(), vmin = -c.max(),
           cmap = pl.cm.seismic)
cb = fg.colorbar(im, ax = ax[1,0])
cb.set_label('Corrected eigenvector centrality (original - SERN)')

# percentiles
c = np.array([percentileofscore(var[:,i], vo[i]) for i in range(len(vo))])
im = ax[1,1].scatter(lon, lat, s = Scale(c), c = c,
           vmax = 100, vmin = 0,
           cmap = pl.cm.seismic)
cb = fg.colorbar(im, ax = ax[1,1])
cb.set_label('Eigenvector centrality percentiles')

pl.tight_layout()
pl.savefig('sern_example.png')

