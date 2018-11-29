"""Solution of the question whether a 3D segment with ending points p0
and p1 intersects or not a voxel with coordinates v and sides l.

This solution is obtained by reducing the system of equations and
inequalities given by the segment ((x,y,z = t(p1 - p0) + p, 0<=t<1)
and the boundary conditions of the voxel (v-l/2 <= (x,y,z) < v+l/2).

The assumption is that voxel and the segment are in the same reference
system. Moreover, the "origin" of the voxel is in its center. In order
to use one corner as origin, it is just a slight change to the code,
e.g. v <= (x,y,z) < v+l.

"""


import numpy as np


def intersection_segment_voxel_basic(p0, p1, v, l_2):
    """
    """
    x0, y0, z0 = p0
    x1, y1, z1 = p1
    a, b, c = v
    lx, ly, lz = l_2

    less_than = []
    greater_than = []

    bounds_less = [(a + lx - x0) / (x1 - x0),
                   (b + ly - y0) / (y1 - y0),
                   (c + lz - z0) / (z1 - z0)]
    bounds_greater = [(a - lx - x0) / (x1 - x0),
                      (b - ly - y0) / (y1 - y0),
                      (c - lz - z0) / (z1 - z0)]
    signs = [(x1 - x0),
             (y1 - y0),
             (z1 - z0)]

    for bl, bg, sign in zip(bounds_less, bounds_greater, signs):
        if sign >= 0:
            less_than.append(bl)
            greater_than.append(bg)
        else:
            less_than.append(bg)
            greater_than.append(bl)

    less_than = np.array(less_than)
    greater_than = np.array(greater_than)

    intersection = max(0.0, greater_than.max(0)) < min(1.0, less_than.min(0))
    return intersection


def intersection_segment_voxel(p0, p1, v, l_2):
    tmp = p1 - p0
    # tmp[tmp == 0.0] = 1.0e-9
    less_than = (v + l_2 - p0) / tmp
    greater_than = (v - l_2 - p0) / tmp
    lt = less_than.copy()
    gt = greater_than.copy()
    tmp = (p1 - p0) < 0
    lt[tmp] = greater_than[tmp]
    gt[tmp] = less_than[tmp]
    intersection = gt.clip(min=0.0).max() < lt.clip(max=1.0).min()
    return intersection


def intersection_segment_voxels(p0, p1, v, l_2):
    """Compute which voxels v of size l are intersected by the segment [p0, p1].
    """
    tmp = p1 - p0
    # tmp[tmp == 0.0] = 1.0e-9
    lt = (v + l_2 - p0) / tmp
    gt = (v - l_2 - p0) / tmp
    tmp = ((p1 - p0) < 0).T
    tmp1 = lt[:, tmp].copy()
    tmp2 = gt[:, tmp].copy()
    lt[:, tmp] = tmp2
    gt[:, tmp] = tmp1
    intersection = gt.clip(min=0.0).max(1) < lt.clip(max=1.0).min(1)
    return intersection


def intersection_segments_voxels(p0, p1, v, l_2):
    """Compute which voxels v of size l are intersected by the segmentS
    [p0, p1].

    """
    tmp = (p1 - p0).T
    lt = ((v + l_2)[:, :, None] - p0.T) / tmp
    gt = ((v - l_2)[:, :, None] - p0.T) / tmp
    tmp = ((p1 - p0) < 0).T
    tmp1 = lt[:, tmp].copy()
    tmp2 = gt[:, tmp].copy()
    lt[:, tmp] = tmp2
    gt[:, tmp] = tmp1
    intersection = gt.clip(min=0.0).max(1) < lt.clip(max=1.0).min(1)
    return intersection.any(1)


def intersection_segments_voxels_slow(p0, p1, v, l_2):
    """Non-vectorized code equivalent to intersection_segments_voxels().
    """
    intersection = np.zeros(v.shape[0], dtype=np.bool)
    for i in range(len(p0)):
        intersection += intersection_segment_voxels(p0[i], p1[i], v, l_2)

    return intersection


def streamline2voxels_basic(s, l_2):
    """Compute the voxels of size l of a streamline. ASSUMPTION: they are
    in the same reference system!
    """
    p0 = s[:-1]
    p1 = s[1:]
    v = ndim_grid(np.trunc(s.min(0)), np.trunc(s.max(0)))
    intersection = np.zeros(len(v), dtype=np.bool)
    for i in range(p0.shape[0]):
        for j, voxel in enumerate(v):
            intersection[j] += intersection_segment_voxel_basic(p0[i], p1[i], voxel, l_2)

    return v[intersection]


def streamline2voxels_slow(s, l_2):
    """Compute the voxels of size l of a streamline. ASSUMPTION: they are
    in the same reference system!
    """
    p0 = s[:-1]
    p1 = s[1:]
    v = ndim_grid(np.trunc(s.min(0)), np.trunc(s.max(0)))
    intersection = np.zeros(len(v), dtype=np.bool)
    for i in range(p0.shape[0]):
        for j, voxel in enumerate(v):
            intersection[j] += intersection_segment_voxel(p0[i], p1[i], voxel, l_2)

    return v[intersection]


def streamline2voxels(s, l_2):
    p0 = s[:-1]
    p1 = s[1:]
    v = ndim_grid(np.trunc(s.min(0)), np.trunc(s.max(0)))
    intersection = np.zeros(len(v), dtype=np.bool)
    for i in range(p0.shape[0]):
        intersection += intersection_segment_voxels(p0[i], p1[i], v, l_2)

    return v[intersection]


def streamline2voxels_fast(s, l_2):
    """Compute the voxels of size l of a streamline. ASSUMPTION: they are
    in the same reference system!
    """
    # p0 = np.atleast_2d(s[:-1]).T
    # p1 = np.atleast_2d(s[1:]).T
    p0 = s[:-1]
    p1 = s[1:]
    v = ndim_grid(np.trunc(s.min(0)), np.trunc(s.max(0)))
    return v[intersection_segments_voxels(p0, p1, v, l_2)]


def ndim_grid(start,stop):
    """Generate all voxel coordinates between vector start, e.g. [0,0,0],
    and vector stop, e.g. [10,10,10].
    adapted from https://stackoverflow.com/questions/38170188/generate-a-n-dimensional-array-of-coordinates-in-numpy
    """
    # Set number of dimensions
    ndims = len(start)

    # List of ranges across all dimensions
    L = [np.arange(start[i]-1,stop[i]+1) for i in range(ndims)]

    # Finally use meshgrid to form all combinations corresponding to all
    # dimensions and stack them as M x ndims array
    return np.hstack((np.meshgrid(*L))).swapaxes(0,1).reshape(ndims,-1).T


if __name__ == '__main__':
    p0 = np.array([0.5, 0.4, 0.3])
    p1 = np.array([4.1, 5.2, 3.9])
    v = np.array([2, 2, 2])
    l = np.array([1.0, 1.0, 1.0])
    print(intersection_segment_voxel_basic(p0, p1, v, l))

    print("")

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
        intersection = intersection_segment_voxels(p0, p1, v, l)
        # print(v[intersection])
        voxels += [tuple(vv) for vv in v[intersection]]

    voxels = set(voxels)
    print(len(voxels))

    p0 = np.atleast_2d(np.array([3.5, 0.5, 0.5]))
    p1 = np.atleast_2d(np.array([0.5, 3.5, 3.5]))
    v = ndim_grid([0, 0, 0], [5, 5, 5])
    l = np.array([1, 1, 1])
    intersection = intersection_segments_voxels(p0, p1, v, l)
    print(v[intersection])
