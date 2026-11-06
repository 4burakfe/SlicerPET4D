"""Tests of the resampling helper (needs the vtk package; skipped without it): python -m pytest Testing."""
import os
import sys

import numpy as np
import pytest

vtk = pytest.importorskip("vtk")
from vtk.util import numpy_support  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "DICOMGatedDynamicPET"))
from DICOMGatedDynamicPETLib import registration as reg  # noqa: E402
from DICOMGatedDynamicPETLib import resampling  # noqa: E402


def image(array):
    """vtkImageData of a (k, j, i) array, origin 0, spacing 1, as Slicer keeps the voxels."""
    data = vtk.vtkImageData()
    data.SetDimensions(array.shape[2], array.shape[1], array.shape[0])
    scalars = numpy_support.numpy_to_vtk(np.ascontiguousarray(array).ravel(), deep=True)
    data.GetPointData().SetScalars(scalars)
    return data


def array(data):
    dims = data.GetDimensions()
    return numpy_support.vtk_to_numpy(data.GetPointData().GetScalars()).reshape(dims[::-1])


def ramp(shape=(10, 12, 14)):
    k, j, i = np.indices(shape, dtype=np.float32)
    return 1.0 + i + 10.0 * j + 100.0 * k


def test_identity_keeps_voxels():
    source = ramp()
    ijk = np.diag([2.0, 2.0, 3.0, 1.0])
    out = resampling.Resampler(ijk, (14, 12, 10), ijk).resample(image(source))
    np.testing.assert_allclose(array(out), source, atol=1e-4)


def test_shift_of_whole_voxels():
    source = ramp()
    ijk = np.diag([2.0, 2.0, 3.0, 1.0])
    shift = vtk.vtkTransform()
    shift.Translate(4.0, 0.0, 0.0)           # world -> input: the input is 4 mm (2 voxels) further along R
    out = array(resampling.Resampler(ijk, (14, 12, 10), ijk, shift, linear=True).resample(image(source)))
    np.testing.assert_allclose(out[:, :, :-2], source[:, :, 2:], atol=1e-4)
    assert np.all(out[:, :, -2:] == 0)        # outside the input: 0


def test_linear_matrix_of_general_transform():
    general = vtk.vtkGeneralTransform()
    general.Translate(1.0, 2.0, 3.0)
    general.RotateZ(90.0)
    m = resampling.linearMatrix(general)
    np.testing.assert_allclose(m[:3, :3], [[0, -1, 0], [1, 0, 0], [0, 0, 1]], atol=1e-9)
    np.testing.assert_allclose(m[:3, 3], (1.0, 2.0, 3.0), atol=1e-9)


def test_nonlinear_path_matches_linear():
    """A transform given as non-linear (through the displacement grid) gives the linear result."""
    source = ramp()
    ijk = np.diag([2.0, 2.0, 3.0, 1.0])
    shift = vtk.vtkTransform()
    shift.Translate(3.0, -2.0, 1.5)
    general = vtk.vtkGeneralTransform()
    general.Concatenate(shift)
    linear = array(resampling.Resampler(ijk, (14, 12, 10), ijk, shift, linear=True).resample(image(source)))
    grid = array(resampling.Resampler(ijk, (14, 12, 10), ijk, general, linear=False).resample(image(source)))
    np.testing.assert_allclose(grid, linear, atol=1e-3)


def test_coarser_grid_and_nearest():
    source = ramp()
    ijk = np.diag([1.0, 1.0, 1.0, 1.0])
    dims, grid = reg.outputGrid((14, 12, 10), ijk, (2.0, 2.0, 2.0))
    assert dims == (7, 6, 5)
    out = array(resampling.Resampler(ijk, dims, grid, interpolation="Linear").resample(image(source)))
    # voxel centre (0.5, 0.5, 0.5) of the input: the mean of the 2x2x2 block
    assert out[0, 0, 0] == pytest.approx(source[:2, :2, :2].mean(), abs=1e-3)
    nearest = array(resampling.Resampler(ijk, dims, grid, interpolation="Nearest neighbour").resample(image(source)))
    assert set(np.unique(nearest)).issubset(set(np.unique(source)) | {0.0})
    with pytest.raises(ValueError):
        resampling.Resampler(ijk, dims, grid, interpolation="Sinc")


def test_grid_step_and_coarse_displacement_grid():
    assert resampling.gridStep((100, 100, 100)) == 1
    assert resampling.gridStep((512, 512, 300)) == 3
    source = ramp((20, 22, 24))
    ijk = np.eye(4)
    shift = vtk.vtkTransform()
    shift.Translate(2.5, 1.0, -1.0)
    general = vtk.vtkGeneralTransform()
    general.Concatenate(shift)
    linear = array(resampling.Resampler(ijk, (24, 22, 20), ijk, shift).resample(image(source)))
    old = resampling.MAX_GRID_POINTS
    try:
        resampling.MAX_GRID_POINTS = 500          # forces a step of 3 voxels (not a divisor of the dims)
        assert resampling.gridStep((24, 22, 20)) == 3
        coarse = resampling.Resampler(ijk, (24, 22, 20), ijk, general, linear=False)
    finally:
        resampling.MAX_GRID_POINTS = old
    np.testing.assert_allclose(array(coarse.resample(image(source))), linear, atol=1e-3)
