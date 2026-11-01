"""Tests of the Review/Edit spherical ROI helpers without Slicer: python -m pytest Testing."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "DICOMGatedDynamicPET"))
from DICOMGatedDynamicPETLib import roicurves as rc  # noqa: E402


def matrix(spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0)):
    out = np.diag([*spacing, 1.0])
    out[:3, 3] = origin
    return out


def brute(shape, ijkToRas, center, radius):
    """Voxels of the sphere, one by one."""
    k, j, i = np.indices(shape)
    ijk = np.stack([i.ravel(), j.ravel(), k.ravel(), np.ones(i.size)])
    ras = (ijkToRas @ ijk)[:3].T
    return (np.linalg.norm(ras - np.asarray(center), axis=1) <= radius).reshape(shape)


def full(shape, region):
    out = np.zeros(shape, dtype=bool)
    out[region[0]] = region[1]
    return out


def test_sphere_mask_voxels():
    slices, mask = rc.sphereMask((11, 11, 11), matrix(), (5, 5, 5), 1.0)
    assert slices == (slice(4, 7), slice(4, 7), slice(4, 7))
    assert mask.sum() == 7    # centre and its 6 neighbours
    region = rc.sphereMask((11, 11, 11), matrix(), (5, 5, 5), 3.0)
    assert region[1].sum() == brute((11, 11, 11), matrix(), (5, 5, 5), 3.0).sum()


def test_sphere_mask_uses_the_spacing_and_the_axis_order():
    ijkToRas = matrix((2.0, 3.0, 5.0), (-10.0, 4.0, 7.0))
    shape = (6, 9, 12)    # k, j, i
    center = (0.0, 13.0, 17.0)
    region = rc.sphereMask(shape, ijkToRas, center, 7.5)
    assert np.array_equal(full(shape, region), brute(shape, ijkToRas, center, 7.5))


def test_sphere_mask_oblique_negative_directions():
    # LPS-like directions (i, j point to -R, -A) and a rotation
    angle = np.radians(30)
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    ijkToRas = np.eye(4)
    ijkToRas[:3, :3] = rotation @ np.diag([-2.5, -2.5, 3.0])
    ijkToRas[:3, 3] = (100.0, 80.0, -40.0)
    shape = (20, 40, 40)
    center = ijkToRas @ np.array([17.3, 22.8, 9.4, 1.0])
    region = rc.sphereMask(shape, ijkToRas, center[:3], 9.0)
    assert np.array_equal(full(shape, region), brute(shape, ijkToRas, center[:3], 9.0))


def test_sphere_mask_is_cut_by_the_volume():
    shape = (8, 8, 8)
    region = rc.sphereMask(shape, matrix(), (0, 0, 0), 3.0)
    assert region[1].shape == (4, 4, 4)
    assert np.array_equal(full(shape, region), brute(shape, matrix(), (0, 0, 0), 3.0))


def test_sphere_mask_falls_back_to_the_nearest_voxel():
    slices, mask = rc.sphereMask((10, 10, 10), matrix((4.0, 4.0, 4.0)), (9.0, 13.0, 21.0), 0.5)
    assert slices == (slice(5, 6), slice(3, 4), slice(2, 3)) and mask.sum() == 1


def test_sphere_outside_the_volume_has_no_voxels():
    array = np.ones((5, 5, 5))
    assert rc.sphereMask(array.shape, matrix(), (50, 50, 50), 2.0)[1].size == 0
    assert rc.sphereMean(array, matrix(), (50, 50, 50), 2.0) is None
    assert rc.sphereMean(array, matrix(), (-3, 2, 2), 1.0) is None


def test_sphere_mean():
    array = np.zeros((9, 9, 9), dtype=np.int16)
    array[4, 4, 4] = 70
    assert rc.sphereMean(array, matrix(), (4, 4, 4), 1.0) == 10.0
    array = np.arange(1000, dtype=float).reshape(10, 10, 10)
    array[0, 0, 0] = np.nan
    assert rc.sphereMean(array, matrix(), (0, 0, 0), 0.4) is None    # not finite


def test_sphere_curve():
    frames = [(np.full((6, 6, 6), v), matrix()) for v in (1.0, 2.0)] + [None, (np.full((6, 6, 6), 5.0), matrix())]
    assert rc.sphereCurve(frames, (3, 3, 3), 2.0) == [1.0, 2.0, None, 5.0]
    moved = [(np.arange(216.0).reshape(6, 6, 6), matrix()), (np.arange(216.0).reshape(6, 6, 6), matrix(origin=(1, 0, 0)))]
    first, second = rc.sphereCurve(moved, (2, 2, 2), 0.1)
    assert (first, second) == (86.0, 85.0)    # the voxels follow each frame's own geometry
    assert rc.sphereCurve([], (0, 0, 0), 1.0) == []


def test_frame_midpoints():
    intervals = [{"start": 0.0, "end": 10.0}, {"start": 10.0, "end": 40.0}, {"start": 45.0, "end": 46.0}]
    assert rc.frameMidpoints(intervals) == [5.0, 25.0, 45.5]


def test_nice_axis():
    assert rc.niceAxis(0) == [0.0, 1.0] and rc.niceAxis(None) == [0.0, 1.0] and rc.niceAxis(-3) == [0.0, 1.0]
    assert rc.niceAxis(float("inf")) == [0.0, 1.0]
    assert rc.niceAxis(10) == [0.0, 2.5, 5.0, 7.5, 10.0]
    assert rc.niceAxis(0.3) == [0.0, 0.1, 0.2, 0.3]
    assert rc.niceAxis(12345) == [0.0, 5000.0, 10000.0, 15000.0]
    for value in (0.0123, 0.7, 1.0, 3.3, 99.9, 101, 7777, 2.5e6):
        ticks = rc.niceAxis(value)
        assert ticks[0] == 0 and ticks[-1] >= value and ticks[-2] < value and 2 <= len(ticks) <= 7


def test_format_value():
    assert rc.formatValue(None) == "?"
    assert rc.formatValue(12345.4) == "12,345"
    assert rc.formatValue(12.3456) == "12.35"
    assert rc.formatValue(0.125) == "0.125"
    assert rc.formatValue(0) == "0"


def test_colors():
    assert rc.nextColor([]) == rc.ROI_COLORS[0]
    assert rc.nextColor(rc.ROI_COLORS[:2]) == rc.ROI_COLORS[2]
    assert rc.nextColor([rc.ROI_COLORS[1]]) == rc.ROI_COLORS[0]
    assert rc.nextColor(list(rc.ROI_COLORS) + [rc.ROI_COLORS[0]]) == rc.ROI_COLORS[1]
    assert len(set(rc.ROI_COLORS)) == len(rc.ROI_COLORS)
    assert rc.colorRgb("#FF0080") == (1.0, 0.0, 128 / 255.0)
