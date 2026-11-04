"""Tests of the Motion Correction helpers without Slicer: python -m pytest Testing."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "DICOMGatedDynamicPET"))
from DICOMGatedDynamicPETLib import dynamicviews as dv  # noqa: E402
from DICOMGatedDynamicPETLib import motioncorrection as mc  # noqa: E402


def rigid(angles, translation):
    matrix = np.eye(4)
    matrix[:3, :3] = mc.rotationMatrix(*angles)
    matrix[:3, 3] = translation
    return matrix


def test_euler_angles_round_trip():
    for angles in [(0, 0, 0), (5, -3, 12), (-20, 10, -45), (1e-3, 0, 89)]:
        found = mc.eulerAngles(mc.rotationMatrix(*angles))
        np.testing.assert_allclose(found, angles, atol=1e-9)


def test_rotation_is_orthonormal():
    r = mc.rotationMatrix(7, -11, 23)
    np.testing.assert_allclose(r @ r.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(r) == pytest.approx(1.0)


def test_rigid_motion_shift_is_at_the_centre():
    # a pure rotation about the origin moves a centre away from the origin
    center = (100.0, 0.0, 0.0)
    shift, angles = mc.rigidMotion(rigid((0, 0, 90), (0, 0, 0)), center)
    np.testing.assert_allclose(shift, (-100.0, 100.0, 0.0), atol=1e-9)
    np.testing.assert_allclose(angles, (0, 0, 90), atol=1e-9)
    # a rotation about the centre itself gives no shift
    rotation = mc.rotationMatrix(3, 4, 5)
    c = np.array([10.0, -20.0, 30.0])
    matrix = rigid((3, 4, 5), c - rotation @ c + np.array([1.0, 2.0, -3.0]))
    shift, angles = mc.rigidMotion(matrix, c)
    np.testing.assert_allclose(shift, (1.0, 2.0, -3.0), atol=1e-9)
    np.testing.assert_allclose(angles, (3, 4, 5), atol=1e-9)


def test_low_activity_frames():
    activities = [0.0, 5.0, 999.0, 1000.0, 10.0, 9.99]
    assert mc.lowActivityFrames(activities, 1.0) == [True, True, False, False, False, True]
    assert mc.lowActivityFrames(activities, 0.1) == [True, False, False, False, False, False]
    assert mc.lowActivityFrames([0.0, 0.0]) == [False, False]
    assert mc.lowActivityFrames([]) == []
    assert mc.maxActivityFrame(activities) == 3


def test_frame_activity_skips_non_finite():
    assert mc.frameActivity(np.array([[1.0, np.nan], [np.inf, 2.0]])) == 3.0


def test_limit():
    points = mc.limitSamplePoints((-100, 100, -50, 50, 0, 200))
    assert points.shape == (27, 3)
    assert points[:, 0].min() == -50 and points[:, 0].max() == 50
    assert points[:, 2].min() == 50 and points[:, 2].max() == 150
    moved = points + np.array([0.0, -250.0, 3.0])
    displacement = mc.largestDisplacement(points, moved)
    np.testing.assert_allclose(displacement, (0.0, 250.0, 3.0))
    assert mc.exceedsLimit(displacement, 200.0)
    assert not mc.exceedsLimit(displacement, 300.0)
    assert not mc.exceedsLimit(displacement, None)
    assert not mc.exceedsLimit((-199.0, 0, 0), 200.0)
    assert mc.exceedsLimit((-201.0, 0, 0), 200.0)


def test_transform_types():
    assert mc.transformTypes("Rigid") == ["Rigid"]
    assert mc.transformTypes("Rigid", True) == ["Rigid", "Rigid"]
    assert mc.transformTypes("Rigid", True, "Affine") == ["Rigid", "Affine"]
    assert mc.transformTypes("Affine", True, "BSpline") == ["Affine", "BSpline"]
    assert mc.transformTypes("BSpline", True) == ["BSpline"]
    assert mc.transformTypes("Rigid", True, "Rigid") == ["Rigid", "Rigid"]
    assert mc.methodText("Rigid") == "Rigid"
    assert mc.methodText("Rigid", True) == "Rigid, then Rigid to the motion corrected sum"
    assert mc.rowsPass("Rigid", True) == 1
    assert mc.rowsPass("Rigid", True, "Rigid") == 1
    assert mc.rowsPass("Rigid", True, "Affine") == 0 and mc.rowsPass("Rigid") == 0
    with pytest.raises(ValueError):
        mc.transformTypes("Similarity")
    assert mc.showsMotionRows("Rigid")
    assert not mc.showsMotionRows("Affine") and not mc.showsMotionRows("BSpline")


def test_cell_text_fits_the_block():
    values = (1.234, -0.04, 12.0)
    assert mc.cellText(values, 200) == "1.2, 0.0, 12.0"
    assert mc.cellText(values, 40) == "12.0"
    assert mc.cellText((-7.0, 1.0, 2.0), 40) == "-7.0"
    assert mc.cellText(values, 10) == ""
    assert mc.cellText(None, 200) == ""
    assert mc.statusCellText(mc.LOW_ACTIVITY, 100) == "low"
    assert mc.statusCellText(mc.REGISTERED, 100) == ""
    assert mc.magnitudeLevel((0.5, -1.0, 0.2), 2.0, 5.0) == 0
    assert mc.magnitudeLevel((0.5, -3.0, 0.2), 2.0, 5.0) == 1
    assert mc.magnitudeLevel((9.0, 0, 0), 2.0, 5.0) == 2


def test_texts():
    tip = mc.frameMotionTooltip({"status": mc.REGISTERED, "shift": (1, 2, 3), "angles": (0.5, 0, 0)})
    assert "Shift R, A, S: 1.0, 2.0, 3.0 mm" in tip and "Angles" in tip
    assert mc.summaryText([mc.REGISTERED, mc.REGISTERED, mc.LOW_ACTIVITY, mc.REFERENCE]) == \
        "2 registered, 1 reference, 1 low activity"


def test_plot_axis():
    assert mc.plotLimit([], 2.0) == 2.0
    assert mc.plotLimit([-10.0, 3.0], 2.0) == pytest.approx(11.0)
    assert mc.plotTicks(11.0) == [-10.0, -5.0, 0.0, 5.0, 10.0]
    assert mc.plotTicks(2.0) == [-2.0, -1.0, 0.0, 1.0, 2.0]
    assert mc.plotTicks(0.0) == [0.0]


def test_pan_range():
    assert dv.panRange(100.0, 200.0, 1, 0.0, 1000.0) == (90.0, 190.0)
    assert dv.panRange(100.0, 200.0, -2, 0.0, 1000.0) == (120.0, 220.0)
    assert dv.panRange(5.0, 105.0, 1, 0.0, 1000.0) == (0.0, 100.0)
    assert dv.panRange(890.0, 990.0, -5, 0.0, 1000.0) == (900.0, 1000.0)


def test_layout_orientation_and_names():
    xml = dv.layoutXml(5, orientation="Coronal", mainView="DynMCMain", framePrefix="DynMCF",
                       emptyPrefix="DynMCEmpty", viewGroup=79)
    assert xml.count('action="default">Coronal<') == 7          # main + 5 frames + 1 empty
    assert 'singletontag="DynMCMain"' in xml and 'singletontag="DynMCF5"' in xml
    assert 'singletontag="DynMCEmpty1"' in xml and "DynPET" not in xml
    assert xml.count(">79<") == 7
    # Review/Edit keeps its axial layout and names
    assert dv.layoutXml(5).count('action="default">Axial<') == 7
    assert dv.frameViewName(0) == "DynPETF1" and dv.frameViewName(0, "DynMCF") == "DynMCF1"
