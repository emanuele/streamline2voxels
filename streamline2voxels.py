import numpy as np


def intersection_segment_voxel_basic(p0, p1, v, l):
    x0, y0, z0 = p0
    x1, y1, z1 = p1
    a, b, c = v
    lx, ly, lz = l

    less_than = []
    greater_than = []

    bounds_less = [(a + lx - x0) / (x1 - x0),
                   (b + ly - y0) / (y1 - y0),
                   (c + lz - z0) / (z1 - z0)]
    bounds_greater = [(a - x0) / (x1 - x0),
                      (b - y0) / (y1 - y0),
                      (c - z0) / (z1 - z0)]
    signs = [(x1 - x0),
             (y1 - y0),
             (z1 - z0)]

    for bl, bg, sign in zip(bounds_less, bounds_greater, signs):
        if sign > 0:
            less_than.append(bl)
            greater_than.append(bg)
        elif sign < 0:
            less_than.append(bg)
            greater_than.append(bl)
        else:
            pass

    less_than = np.array(less_than)
    greater_than = np.array(greater_than)

    intersection = max(0.0, greater_than.max(0)) < min(1.0, less_than.min(0))
    return intersection


def intersection_segment_voxel(p0, p1, v, l):
    less_than = (v + l - p0) / (p1 - p0)
    greater_than = (v - p0) / (p1 - p0)
    lt = less_than.copy()
    gt = greater_than.copy()
    tmp = (p1 - p0) < 0
    lt[tmp] = greater_than[tmp]
    gt[tmp] = less_than[tmp]
    intersection = gt.clip(min=0.0).max() < lt.clip(max=1.0).min()
    return intersection


def intersection_segment_voxels(p0, p1, v, l):
    """Compute which voxels v of size l are intersected by the segment [p0, p1].
    """
    less_than = (v + l - p0) / (p1 - p0)
    greater_than = (v - p0) / (p1 - p0)
    lt = less_than.copy()
    gt = greater_than.copy()
    tmp = (p1 - p0) < 0
    lt[:, tmp] = greater_than[:, tmp]
    gt[:, tmp] = less_than[:, tmp]
    intersection = gt.clip(min=0.0).max(1) < lt.clip(max=1.0).min(1)
    return intersection


def intersection_segments_voxels(p0, p1, v, l):
    """Compute which voxels v of size l are intersected by the segmentS [p0, p1].
    """
    less_than = ((v + l)[:, :, None] + p0) / (p1 - p0)
    greater_than = (v[:, :, None] - p0) / (p1 - p0)
    lt = less_than.copy()
    gt = greater_than.copy()
    tmp = (p1 - p0) < 0
    lt[:, tmp, :] = greater_than[:, tmp, :]
    gt[:, tmp, :] = less_than[:, tmp, :]
    intersection = greater_than.clip(min=0.0).max(1) < less_than.clip(max=1.0).min(1)
    return intersection.any(1)


def intersection_segments_voxels_slow(p0, p1, v, l):
    """Non-vectorized code equivalent to intersection_segments_voxels().
    """
    intersection = np.zeros(v.shape[0], dtype=np.bool)
    for i in range(len(p0)):
        intersection += intersection_segment_voxels(p0[i], p1[i], v, l)

    return intersection


def streamline2voxels_basic(s, l):
    """Compute the voxels of size l of a streamline. ASSUMPTION: they are
    in the same reference system!
    """
    p0 = s[:-1]
    p1 = s[1:]
    v = ndim_grid(np.trunc(s.min(0)), np.trunc(s.max(0)))
    intersection = np.zeros(len(v), dtype=np.bool)
    for i in range(p0.shape[0]):
        for j, voxel in enumerate(v):
            intersection[j] += intersection_segment_voxel_basic(p0[i], p1[i], voxel, l)

    return v[intersection]


def streamline2voxels_slow(s, l):
    """Compute the voxels of size l of a streamline. ASSUMPTION: they are
    in the same reference system!
    """
    p0 = s[:-1]
    p1 = s[1:]
    v = ndim_grid(np.trunc(s.min(0)), np.trunc(s.max(0)))
    intersection = np.zeros(len(v), dtype=np.bool)
    for i in range(p0.shape[0]):
        for j, voxel in enumerate(v):
            intersection[j] += intersection_segment_voxel(p0[i], p1[i], voxel, l)

    return v[intersection]


def streamline2voxels(s, l):
    p0 = s[:-1]
    p1 = s[1:]
    v = ndim_grid(np.trunc(s.min(0)), np.trunc(s.max(0)))
    intersection = np.zeros(len(v), dtype=np.bool)
    for i in range(p0.shape[0]):
        intersection += intersection_segment_voxels(p0[i], p1[i], v, l)

    return v[intersection]


def streamline2voxels_fast(s, l):
    """Compute the voxels of size l of a streamline. ASSUMPTION: they are
    in the same reference system!
    """
    # p0 = np.atleast_2d(s[:-1]).T
    # p1 = np.atleast_2d(s[1:]).T
    p0 = s[:-1]
    p1 = s[1:]
    v = ndim_grid(np.trunc(s.min(0)), np.trunc(s.max(0)))
    return v[intersection_segments_voxels_slow(p0, p1, v, l)]


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

    p0 = np.array([[0.5, 3.5],
                   [0.5, 3.5],
                   [0.5, 3.5]])

    p1 = np.array([[3.5, 5.5],
                   [3.5, 3.5],
                   [3.5, 4.5]])


    # p0 = np.atleast_2d(np.array([3.5, 0.5, 0.5])).T
    # p1 = np.atleast_2d(np.array([0.5, 3.5, 3.5])).T
    v = ndim_grid([0, 0, 0], [5, 5, 5])
    l = np.array([1, 1, 1])
    intersection = intersection_segments_voxels(p0, p1, v, l)
    print(v[intersection])
