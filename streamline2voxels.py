import numpy as np


def intersection_one_voxel(x0, y0, z0,
                           x1, y1, z1,
                           a, b, c,
                           lx, ly, lz):
    """INCORRECT when x1<x0, y1<y0 or z1<z0 !!!
    """
    less_than = np.array([1.0,
                          (a + lx - x0) / (x1 - x0),
                          (b + ly - y0) / (y1 - y0),
                          (c + lz - z0) / (z1 - z0)])

    greater_than = np.array([0.0,
                             (a - x0) / (x1 - x0),
                             (b - y0) / (y1 - y0),
                             (c - z0) / (z1 - z0)])
    
    intersection = less_than.min(0) > greater_than.max(0)
    print("Intersection? %s" % intersection)
    return intersection


def intersection_voxels(p0, p1, v, l):
    """Compute which voxels v of size l are intersected by the segment [p0, p1].
    """
    less_than = (v + l - p0) / (p1 - p0)
    greater_than = (v - p0) / (p1 - p0)
    less_than[:, (p1 - p0) < 0], greater_than[:, (p1 - p0) < 0] = greater_than[:, (p1 - p0) < 0], less_than[:, (p1 - p0) < 0]
    intersection = greater_than.clip(min=0.0).max(1) < less_than.clip(max=1.0).min(1)
    return intersection

def intersection_segments_voxels(p0, p1, v, l):
    """Compute which voxels v of size l are intersected by the segmentS [p0, p1].
    """
    less_than = ((v + l)[:, :, None] + p0) / (p1 - p0)
    greater_than = (v[:, :, None] - p0) / (p1 - p0)
    less_than[:, (p1 - p0) < 0], greater_than[:, (p1 - p0) < 0] = greater_than[:, (p1 - p0) < 0], less_than[:, (p1 - p0) < 0]
    intersection = greater_than.clip(min=0.0).max(1) < less_than.clip(max=1.0).min(1)
    return intersection.any(1)


def streamlines2voxels(s, l):
    """Compute the voxels of size l of a streamline. ASSUMPTION: they are
    in the same reference system!
    """
    p0 = s[:-1].T
    p1 = s[1:].T
    v = ndim_grid(s.min(0), s.max(0))
    return v[intersection_segments_voxels(p0, p1, v, l)]


def intersection_segments_voxels_slow(p0, p1, v, l):
    """Non-vectorized code equivalent to intersection_segments_voxels().
    """
    voxels = []
    for i in range(len(p0)):
        intersection = intersection_voxels(p0[i], p1[i], v, l)
        # print(v[intersection])
        voxels += [tuple(vv) for vv in v[intersection]]

    voxels = set(voxels)
    return voxels


def ndim_grid(start,stop):
    """Generate all voxel coordinates between vector start, e.g. [0,0,0],
    and vector stop, e.g. [10,10,10].
    Taken from https://stackoverflow.com/questions/38170188/generate-a-n-dimensional-array-of-coordinates-in-numpy
    """
    # Set number of dimensions
    ndims = len(start)

    # List of ranges across all dimensions
    L = [np.arange(start[i],stop[i]) for i in range(ndims)]

    # Finally use meshgrid to form all combinations corresponding to all 
    # dimensions and stack them as M x ndims array
    return np.hstack((np.meshgrid(*L))).swapaxes(0,1).reshape(ndims,-1).T


if __name__ == '__main__':
    x0, y0, z0 = np.array([0.5, 0.4, 0.3])
    x1, y1, z1 = np.array([4.1, 5.2, 3.9])
    a, b, c = np.array([2, 2, 2])
    lx, ly, lz = np.array([1.0, 1.0, 1.0])
    intersection_one_voxel(x0, y0, z0, x1, y1, z1, a, b, c, lx, ly, lz)

    print("")
    
    # x0, y0, z0 = np.array([0.0, 0.0, 0.0])
    # x1, y1, z1 = np.array([3.5, 3.5, 3.5])
    x1, y1, z1 = np.array([3.5, 0.5, 0.5])
    x0, y0, z0 = np.array([0.5, 3.5, 3.5])

    a, b, c = ndim_grid([0, 0, 0], [5, 6, 7]).T

    # less_than = np.array([np.ones(a.size),
    #                       (a + lx - x0) / (x1 - x0),
    #                       (b + ly - y0) / (y1 - y0),
    #                       (c + lz - z0) / (z1 - z0)])

    # greater_than = np.array([np.zeros(a.size),
    #                          (a - x0) / (x1 - x0),
    #                          (b - y0) / (y1 - y0),
    #                          (c - z0) / (z1 - z0)])

    less_than = [np.ones(a.size)]
    greater_than = [np.zeros(a.size)]
    xl = (a + lx - x0) / (x1 - x0)
    xg = (a - x0) / (x1 - x0)
    yl = (b + ly - y0) / (y1 - y0)
    yg = (b - y0) / (y1 - y0)
    zl = (c + lz - z0) / (z1 - z0)
    zg = (c - z0) / (z1 - z0)
    if x1 > x0:
        less_than.append(xl)
        greater_than.append(xg)
    else:
        less_than.append(xg)
        greater_than.append(xl)

    if y1 > y0:
        less_than.append(yl)
        greater_than.append(yg)
    else:
        less_than.append(yg)
        greater_than.append(yl)
        
    if z1 > z0:
        less_than.append(zl)
        greater_than.append(zg)
    else:
        less_than.append(zg)
        greater_than.append(zl)

    less_than = np.array(less_than)
    greater_than = np.array(greater_than)
        
    intersection = less_than.min(0) > greater_than.max(0)
    print("Intersection? %s" % intersection)
    abc = np.vstack([a, b, c]).T
    print(abc[intersection])


    p0 = np.array([3.5, 0.5, 0.5])
    p1 = np.array([0.5, 3.5, 3.5])
    # v = np.array([0, 0, 0])
    v = ndim_grid([0, 0, 0], [5, 5, 5])
    l = np.array([1, 1, 1])
    less_than = (v + l - p0) / (p1 - p0)
    greater_than = (v - p0) / (p1 - p0)
    less_than[:, (p1 - p0) < 0], greater_than[:, (p1 - p0) < 0] = greater_than[:, (p1 - p0) < 0], less_than[:, (p1 - p0) < 0]
    intersection = greater_than.clip(min=0.0).max(1) < less_than.clip(max=1.0).min(1)
    print(v[intersection])

    voxels = []
    v = ndim_grid([0, 0, 0], [10, 10, 10])
    for i in range(2000):
        p0 = np.random.uniform(low=0.0, high=10.0, size=3)
        p1 = np.random.uniform(low=0.0, high=10.0, size=3)
        intersection = intersection_voxels(p0, p1, v, l)
        # print(v[intersection])
        voxels += [tuple(vv) for vv in v[intersection]]

    voxels = set(voxels)
    print(len(voxels))

    p0 = np.array([[0.5, 3.5],
                   [0.5, 3.5],
                   [0.5, 3.5]])

    p1 = np.array([[3.5, 5.5],
                   [3.5, 3.5],
                   [3.5, 4.5]])
    

    less_than = ((v + l)[:, :, None] + p0) / (p1 - p0)
    greater_than = (v[:, :, None] - p0) / (p1 - p0)
    less_than[:, (p1 - p0) < 0], greater_than[:, (p1 - p0) < 0] = greater_than[:, (p1 - p0) < 0], less_than[:, (p1 - p0) < 0]
    intersection = greater_than.clip(min=0.0).max(1) < less_than.clip(max=1.0).min(1)
    print(v[intersection.any(1)])

    
