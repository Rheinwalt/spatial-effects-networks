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
    from scipy.spatial.distance import pdist, squareform

    D = pdist(np.transpose((lat, lon)), GreatCircleDistance)
    Dm = D.min()
    D = np.log10(D-Dm+10**(1.1/scale))
    D = (scale * D).astype('int')
    m = D.max() + 1
    x = 10**((np.arange(m, dtype = 'float')-1) / scale)
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

def Graph(e, n):
    import graph_tool.all as gt

    g = gt.Graph(directed = False)
    g.add_vertex(n)
    g.add_edge_list(e)
    return g

def Scale(v):
    s = v.copy()
    s -= s.min()
    s /= s.max()
    s *= 10
    return s

