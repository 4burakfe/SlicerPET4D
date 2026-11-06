"""Tests of the Dynamic PET to CT/MR Registration helpers without Slicer: python -m pytest Testing."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "DICOMGatedDynamicPET"))
from DICOMGatedDynamicPETLib import motioncorrection as mc  # noqa: E402
from DICOMGatedDynamicPETLib import registration as reg  # noqa: E402


def test_steps():
    assert reg.validateSteps(reg.DEFAULT_STEPS) == ["Rigid", "Affine"]
    assert reg.validateSteps(["Rigid", None, ""]) == ["Rigid"]
    assert reg.validateSteps(["Affine", "BSpline"]) == ["Affine", "BSpline"]
    assert reg.validateSteps(["Rigid", "Rigid"]) == ["Rigid", "Rigid"]
    for bad in ([], ["BSpline", "Rigid"], ["Affine", "Rigid"], ["Rigid", "Affine", "BSpline"], ["Foo"]):
        with pytest.raises(ValueError):
            reg.validateSteps(bad)
    assert reg.stepsText(["Rigid", "Affine"]) == "Rigid, then Affine"


def test_linear_summary_rigid():
    m = np.eye(4)
    m[:3, :3] = mc.rotationMatrix(4.0, -2.0, 10.0)
    center = np.array([10.0, -20.0, 30.0])
    m[:3, 3] = center - m[:3, :3] @ center + [1.0, 2.0, 3.0]     # rotation about the centre, then a shift
    s = reg.linearSummary(m, center)
    np.testing.assert_allclose(s["shift"], (1.0, 2.0, 3.0), atol=1e-9)
    np.testing.assert_allclose(s["angles"], (4.0, -2.0, 10.0), atol=1e-9)
    np.testing.assert_allclose(s["scales"], (100.0, 100.0, 100.0), atol=1e-9)
    assert s["rigid"]


def test_linear_summary_affine():
    rotation = mc.rotationMatrix(0.0, 0.0, 15.0)
    k = np.array([[1.1, 0.05, 0.0], [0.0, 0.9, 0.0], [0.0, 0.0, 1.0]])
    m = np.eye(4)
    m[:3, :3] = rotation @ k
    r, scales, shears = reg.decomposeAffine(m)
    np.testing.assert_allclose(r, rotation, atol=1e-9)
    np.testing.assert_allclose(scales, (1.1, 0.9, 1.0), atol=1e-9)
    assert shears[0] == pytest.approx(0.05 / 1.1)
    s = reg.linearSummary(m, (0, 0, 0))
    assert not s["rigid"]
    lines = reg.summaryLines(["Rigid", "Affine"], m, (0, 0, 0))
    assert any(line.startswith("Scaling") for line in lines)


def test_summary_lines_deformation_and_manual():
    manual = np.eye(4)
    manual[:3, 3] = (0.0, 5.0, 0.0)
    deformation = reg.displacementStats([[0, 0, 0], [1, 1, 1]], [[3, 4, 0], [1, 1, 1]])
    assert deformation["max"] == pytest.approx(5.0)
    assert deformation["mean"] == pytest.approx(2.5)
    lines = reg.summaryLines(["Rigid", "BSpline"], np.eye(4), (0, 0, 0), deformation, manual)
    assert lines[0] == "Steps: Rigid, then BSpline"
    assert any("B-spline deformation: mean 2.5 mm, max 5.0 mm" in line for line in lines)
    assert any(line.startswith("Manual: shift 0.0, 5.0, 0.0 mm") for line in lines)
    assert not any(line.startswith("Manual") for line in reg.summaryLines(["Rigid"], np.eye(4), (0, 0, 0),
                                                                           manual=np.eye(4)))
    assert reg.summaryLines([], None, (0, 0, 0)) == ["Not registered"]


def test_checkerboard():
    board = reg.checkerboard((4, 6, 8), (2.0, 2.0, 4.0), squareMm=4.0)
    assert board.shape == (4, 6, 8)
    assert board.dtype == np.uint8
    assert board[0, 0, 0] == 0 and board[0, 0, 2] == 1 and board[0, 2, 0] == 1 and board[1, 0, 0] == 1
    assert board[0, 0, 1] == 0                                # same 4 mm square along i
    assert 0.4 < board.mean() < 0.6
    with pytest.raises(ValueError):
        reg.checkerboard((2, 2, 2), (1, 1, 1), 0.0)


def test_output_grid_covers_fixed_field_of_view():
    fixed = np.diag([-0.5, -0.5, 1.0, 1.0])          # LPS-like flips, 0.5 mm in plane, 1 mm slices
    fixed[:3, 3] = (100.0, 120.0, -50.0)
    dims, grid = reg.outputGrid((512, 512, 100), fixed, (2.0, 2.0, 4.0))
    assert dims == (128, 128, 25)
    np.testing.assert_allclose(reg.gridSpacing(grid), (2.0, 2.0, 4.0))
    # the first corner and the far corner of both grids coincide
    corner = lambda m, d: m[:3, :3] @ (np.asarray(d, float) - 0.5) + m[:3, 3]   # noqa: E731
    np.testing.assert_allclose(corner(grid, (0, 0, 0)), corner(fixed, (0, 0, 0)))
    np.testing.assert_allclose(corner(grid, dims), corner(fixed, (512, 512, 100)))
    same, sameGrid = reg.outputGrid((512, 512, 100), fixed, (0.5, 0.5, 1.0))
    assert same == (512, 512, 100)
    np.testing.assert_allclose(sameGrid, fixed)


def test_output_spacing():
    assert reg.outputSpacing(reg.OUTPUT_PET_SPACING, (2, 2, 3), (1, 1, 1)) == (2.0, 2.0, 3.0)
    assert reg.outputSpacing(reg.OUTPUT_FIXED_GRID, (2, 2, 3), (1, 1, 1)) == (1.0, 1.0, 1.0)
    assert reg.outputSpacing(reg.OUTPUT_CUSTOM, (2, 2, 3), (1, 1, 1), (1.5, 1.5, 1.5)) == (1.5, 1.5, 1.5)
    for bad in (None, (1, 1), (1, 0, 1)):
        with pytest.raises(ValueError):
            reg.outputSpacing(reg.OUTPUT_CUSTOM, (2, 2, 3), (1, 1, 1), bad)


def test_layout():
    small = reg.layoutXml(False)
    big = reg.layoutXml(True)
    assert small.count("vtkMRMLSliceNode") == 6
    assert big.count("vtkMRMLSliceNode") == 8
    assert reg.viewName("Axial", "Checker") in big and reg.viewName("Axial", "Checker") not in small
    # checkerboard first, CT/MR last, axial row on top
    assert big.index("DynRegAxialChecker") < big.index("DynRegAxialPET") < big.index("DynRegAxialAnat") \
        < big.index("DynRegCoronalChecker")


def test_percentile_window():
    lower, upper = reg.percentileWindow(np.arange(1000.0))
    assert lower == pytest.approx(9.99, abs=0.1) and upper == pytest.approx(994.0, abs=1.0)
    assert reg.percentileWindow(np.array([])) == (0.0, 1.0)
    assert reg.percentileWindow(np.full(10, 3.0)) == (3.0, 4.0)


def test_working_spacing():
    # a 0.5 mm MR and a 2 mm PET: registered at 2 mm; a 3 mm slice axis stays 3 mm
    assert reg.workingSpacing((0.5, 0.5, 3.0), (2.0, 2.0, 2.0)) == (2.0, 2.0, 3.0)
    assert reg.workingSpacing((0.5, 0.5, 1.0), (2.0, 2.0, 2.0), 4.0) == (4.0, 4.0, 4.0)
    assert reg.workingSpacing((4.0, 4.0, 4.0), (2.0, 2.0, 2.0)) is None           # CT coarser than PET: as it is
    assert reg.smoothingSigmas((0.5, 0.5, 3.0), (2.0, 2.0, 3.0)) == (2.0, 2.0, 0.0)


def test_checkerboard_columns():
    board = reg.checkerboard((6, 8, 10), (2.0, 2.0, 2.0), 4.0, columnAxis=2)      # columns along k
    for k in range(6):
        np.testing.assert_array_equal(board[k], board[0])
    assert board[0, 0, 0] != board[0, 0, 2] and board[0, 0, 0] != board[0, 2, 0]
    coronal = reg.checkerboard((6, 8, 10), (2.0, 2.0, 2.0), 4.0, columnAxis=1)    # columns along j
    np.testing.assert_array_equal(coronal[:, 0, :], coronal[:, 7, :])


def test_closest_axis():
    lps = np.diag([-1.0, -1.0, 2.0, 1.0])                 # i -> -R, j -> -A, k -> S
    assert reg.closestAxis(lps, reg.SLICE_NORMALS["Axial"]) == 2
    assert reg.closestAxis(lps, reg.SLICE_NORMALS["Coronal"]) == 1
    coronalScan = np.array([[1, 0, 0, 0], [0, 0, 3, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=float)   # k -> A
    assert reg.closestAxis(coronalScan, reg.SLICE_NORMALS["Coronal"]) == 2
    assert reg.closestAxis(coronalScan, reg.SLICE_NORMALS["Axial"]) == 1


def test_resample_flags():
    assert reg.resampleFlags(reg.DEFAULT_STEPS) == [False, False]          # linear steps on the CT/MR as it is
    assert reg.resampleFlags(["Rigid", "BSpline"]) == [False, True]
    assert reg.resampleFlags(["Rigid", "BSpline"], [True, False]) == [True, False]
    assert reg.resampleFlags(["Affine", "BSpline"], [True]) == [True, True]
