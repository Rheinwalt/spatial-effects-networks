import sys
import numpy as np
from sern import *

ids, lon, lat = np.loadtxt('nodes', unpack = True)
links = np.loadtxt('links', dtype = 'int')
A, b = AdjacencyMatrix(ids, links)
lon, lat = lon[b], lat[b]
n = A.shape[0]

# LinkProbability expects A as triu
A = A[np.triu_indices(n, 1)]

# play around with the scale, maybe you don't need log binning?
D, x = IntegerDistances(lat, lon, scale = 50)
p = LinkProbability(A, D)

from matplotlib import pyplot as pl
pl.plot(p, 'bo')
pl.ylabel('Link probability given distance')
pl.xlabel('Bin number')
pl.savefig('link_prob_bin.png')
pl.close('all')
pl.semilogx(x, p, 'bo')
pl.ylabel('Link probability given distance')
pl.xlabel('Distance [km]')
pl.savefig('link_prob_distance.png')
