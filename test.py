import numpy as np
from streamline2voxels import streamline2voxels, streamline2voxels_slow, streamline2voxels_fast, streamline2voxels_basic, streamline2voxels_faster
import nibabel as nib


def sort_rows(v):
    """sort rows of 2D matrix v

    see: https://stackoverflow.com/questions/38277143/sort-2d-numpy-array-lexicographically
    """
    return v[np.lexsort(np.rot90(v))]

if __name__ == '__main__':
    filename = 'sub-100206_var-FNAL_tract.trk'
    data = nib.streamlines.load(filename)
    streamlines = data.streamlines
    l = np.ones(3)
    l_2 = l / 2.0
    for i in range(100):
        s = streamlines[i]
        print(i, len(s))
        v0 = streamline2voxels_basic(s, l_2)
        v1 = streamline2voxels_slow(s, l_2)
        v2 = streamline2voxels(s, l_2)
        v3 = streamline2voxels_fast(s, l_2)
        v4 = streamline2voxels_faster(s, l_2)
        v0 = sort_rows(v3)
        v1 = sort_rows(v4)
        v2 = sort_rows(v3)
        v3 = sort_rows(v3)
        v4 = sort_rows(v4)
        assert((v0 == v1).all())
        assert((v1 == v2).all())
        assert((v2 == v3).all())
        assert((v3 == v4).all())
