import sys
import numpy as np

def AdjacencyMatrix(ids, links):
    n = len(ids)
    a = np.zeros((n, n), dtype='int')
    for i in range(links.shape[0]):
        u = links[i, 0] == ids
        v = links[i, 1] == ids
        a[u, v] = a[v, u] = 1

    np.fill_diagonal(a, 0)
    b = a.sum(axis = 0) > 0
    return (a[b, :][:, b], b)

def GreatCircleDistance(u, v):
    """Great circle distance from (lat, lon) in degree in kilometers."""
    from math import radians, sqrt, sin, cos, atan2
    lat1 = radians(u[0])
    lon1 = radians(u[1])
    lat2 = radians(v[0])
    lon2 = radians(v[1])
    dlon = lon1 - lon2
    EARTH_R = 6372.8

    y = sqrt(
        (cos(lat2) * sin(dlon)) ** 2
        + (cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(dlon)) ** 2
        )
    x = sin(lat1) * sin(lat2) + cos(lat1) * cos(lat2) * cos(dlon)
    d = atan2(y, x)
    return EARTH_R * d

def IntegerDistances(lat, lon, scale = 50.0):
    """Link distances as indices for the binned statistics of link probabilities."""
    from scipy.spatial.distance import pdist, squareform

    # triu distances in km
    D = pdist(np.transpose((lat, lon)), GreatCircleDistance)
    Dm = D.min()

    # optional rescaling
    D = np.log10(D-Dm+1)

    # binning by rounding
    D = (scale * D).astype('int')

    # x axis for p, gives the lower distance of each bin
    x = 10**((np.arange(D.max() + 1) / scale) - 1)
    return (D, x)

def LinkProbability(A, D):
    m = D.max() + 1
    p = np.zeros(m)
    q = np.zeros(m)
    for i in range(len(D)):
        k = D[i]
        q[k] += 1
        if A[i]:
            p[k] += 1
    
    q[q == 0] = np.nan
    p /= q
    p[p == np.nan] = 0
    return p

def SernEdges(D, p, n):
    assert len(D) == n*(n-1)/2, 'n does not fit to D'
    a = np.zeros(D.shape, dtype = 'int')
    a[np.random.random(len(D)) <= p[D]] = 1
    A = np.zeros((n, n), dtype = 'int')
    A[np.triu_indices(n, 1)] = a
    edges = np.transpose(A.nonzero())
    return edges

def Scale(v):
    """Scatter plot dot scale."""
    s = v.copy()
    s = s.astype('float')
    s -= s.min()
    s /= s.max()
    s *= 10
    return s

from matplotlib import pyplot as pl
from scipy.stats import percentileofscore

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
v = np.bincount(edges.ravel())

nserns = 1000
var = np.zeros((nserns, n))

print('measure on SERNs ..')
for i in range(var.shape[0]):
    edges = SernEdges(D, p, n)
    var[i] = np.bincount(edges.ravel())

print('plot full example ..')
fg, ax = pl.subplots(2, 2, figsize = (19.2, 10.8))

# original measure
c = v
im = ax[0,0].scatter(lon, lat, s = Scale(c), c = c,
           cmap = pl.cm.magma_r)
cb = fg.colorbar(im, ax = ax[0,0])
cb.set_label('Degree centrality')

# sern ensemble mean
c = var.mean(axis = 0)
im = ax[0,1].scatter(lon, lat, s = Scale(c), c = c,
           cmap = pl.cm.magma_r)
cb = fg.colorbar(im, ax = ax[0,1])
cb.set_label('SERN Degree centrality')

# corrected measure
c = v - var.mean(axis = 0)
im = ax[1,0].scatter(lon, lat, s = Scale(c), c = c,
           vmax = c.max(), vmin = -c.max(),
           cmap = pl.cm.seismic)
cb = fg.colorbar(im, ax = ax[1,0])
cb.set_label('Corrected degree centrality (original - SERN)')

# percentiles
c = np.array([percentileofscore(var[:,i], v[i]) for i in range(n)])
im = ax[1,1].scatter(lon, lat, s = Scale(c), c = c,
           vmax = 100, vmin = 0,
           cmap = pl.cm.seismic)
cb = fg.colorbar(im, ax = ax[1,1])
cb.set_label('Eigenvector degree percentiles')

pl.tight_layout()
pl.savefig('sern_example_minimal.pdf')

