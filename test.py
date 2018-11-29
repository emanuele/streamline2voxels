import numpy as np
from streamline2voxels import streamline2voxels, streamline2voxels_slow, streamline2voxels_fast, streamline2voxels_basic
import nibabel as nib


if __name__ == '__main__':
    filename = 'sub-100206_var-FNAL_tract.trk'
    data = nib.streamlines.load(filename)
    streamlines = data.streamlines
    for i in range(100, 200):
        s = streamlines[i]
        print(i, len(s))
        l = np.ones(3)
        v0 = streamline2voxels_basic(s, l)
        v1 = streamline2voxels_slow(s, l)
        v2 = streamline2voxels(s, l)
        v3 = streamline2voxels_fast(s, l)
        assert((v0 == v1).all())
        assert((v0 == v2).all())
        assert((v0 == v3).all())
        assert((v1 == v2).all())
        assert((v1 == v3).all())
        assert((v2 == v3).all())
