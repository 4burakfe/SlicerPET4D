"""Tests of the 3D-SSP helpers without Slicer: python -m pytest Testing."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "DICOMGatedDynamicPET"))
from DICOMGatedDynamicPETLib import neuroreview as nr, ssp  # noqa: E402


def test_offsets():
    assert ssp.offsets(13.5, 1.0)[-1] == pytest.approx(13.5)
    assert len(ssp.offsets(13.5, 1.0)) == 15
    assert ssp.offsets(4.0, 2.0).tolist() == [0.0, 2.0, 4.0]


def test_project_max_and_extreme():
    # 1 mm grid, origin 0: the map is 10 at depth 5 mm below the vertex (x = 20 - 5 = 15)
    array = np.zeros((10, 10, 40), dtype=np.float32)          # (k, j, i): i is x
    array[:, :, 15] = 10.0
    array[:, :, 30] = 99.0                                      # outside the vertex, never sampled
    ijkToRas = np.eye(4)
    points = np.array([[20.0, 5.0, 5.0]])
    outward = np.array([[2.0, 0.0, 0.0]])                       # not unit: normalized
    assert ssp.project(array, ijkToRas, points, outward)[0] == pytest.approx(10.0)
    assert ssp.project(array, ijkToRas, points, outward, depthMm=3.0)[0] == pytest.approx(0.0)
    signed = np.zeros_like(array)
    signed[:, :, 18] = 2.0
    signed[:, :, 16] = -3.0
    assert ssp.project(signed, ijkToRas, points, outward, mode="extreme")[0] == pytest.approx(-3.0)
    assert ssp.project(signed, ijkToRas, points, outward)[0] == pytest.approx(2.0)
    # a scaled, shifted grid
    matrix = np.diag([2.0, 2.0, 2.0, 1.0])
    matrix[:3, 3] = (-10.0, 0.0, 0.0)
    coarse = np.zeros((10, 10, 40), dtype=np.float32)
    coarse[:, :, 12] = 7.0                                       # x = -10 + 2 * 12 = 14
    assert ssp.project(coarse, matrix, np.array([[20.0, 4.0, 4.0]]), outward)[0] == pytest.approx(7.0)


def test_scores():
    z = ssp.scores([3.0, 5.0], [5.0, 5.0], [1.0, 1.0], kind=nr.SCORE_Z)
    assert z.tolist() == pytest.approx([-2.0, 0.0])
    t = ssp.scores([3.0], [5.0], [1.0], n=4, kind=nr.SCORE_T)
    assert t[0] == pytest.approx(-2.0 / np.sqrt(1.25))


def test_write_read_and_signature(tmp_path):
    surface = {"points": np.random.rand(5, 3), "normals": np.random.rand(5, 3), "triangles": np.array([[0, 1, 2]])}
    groups = {"master": {"mean": np.arange(5), "sd": np.ones(5), "n": 7, "name": "Master"}}
    path = os.path.join(tmp_path, "stats", "ssp.npz")
    ssp.write(path, surface, groups, {"depthMm": 13.5})
    read = ssp.read(path)
    assert read is not None
    back, backGroups, meta = read
    assert meta["depthMm"] == 13.5 and backGroups["master"]["n"] == 7
    assert backGroups["master"]["mean"].tolist() == list(range(5))
    assert ssp.sameSurface(surface, back)
    assert ssp.read(os.path.join(tmp_path, "none.npz")) is None
    a = ssp.signature(surface["points"], surface["normals"], 13.5, 1.0)
    assert a == ssp.signature(surface["points"], surface["normals"], 13.5, 1.0)
    assert a != ssp.signature(surface["points"], surface["normals"], 10.0, 1.0)
    assert ssp.sspPath(str(tmp_path), {"statistics": {"ssp": ssp.SSP_FILE}}) == path


def test_surface_colors():
    red = np.tile([[1.0, 0.0, 0.0]], (256, 1))
    assert ssp.surfaceColors(red, 0, 1, 4).shape == (4, 4)


def test_hemispheres_and_parts():
    ijkToRas = np.diag([1.0, 1.0, 1.0, 1.0])
    ijkToRas[0, 3] = -2.0                       # x = i - 2
    left, right = ssp.hemispheres((2, 3, 4), ijkToRas)
    assert left[0, 0].tolist() == [True, True, False, False]
    assert (left ^ right).all()
    tri = {"points": np.zeros((3, 3)), "normals": np.ones((3, 3)), "triangles": np.array([[0, 1, 2]])}
    whole = ssp.combine(tri, tri, None)
    assert whole["parts"].tolist() == [0, 0, 0, 1, 1, 1]
    assert whole["triangles"].tolist() == [[0, 1, 2], [3, 4, 5]]
    assert ssp.hasPart(whole, ssp.PART_LEFT) and not ssp.hasPart(whole, ssp.PART_RIGHT)
    piece, index = ssp.piece(whole, ssp.PART_LEFT)
    assert index.tolist() == [3, 4, 5] and piece["triangles"].tolist() == [[0, 1, 2]]
    assert not ssp.hasPart(tri, ssp.PART_LEFT) and ssp.parts(tri).tolist() == [0, 0, 0]
    assert ssp.viewPart("LeftMedial") == ssp.PART_LEFT and ssp.viewPart("Top") == ssp.PART_BRAIN


def test_parts_roundtrip(tmp_path):
    tri = {"points": np.zeros((3, 3)), "normals": np.ones((3, 3)), "triangles": np.array([[0, 1, 2]])}
    whole = ssp.combine(tri, tri, tri)
    path = str(tmp_path / "s.npz")
    ssp.write(path, whole, {}, {"maps": "suvr"})
    surface, _, _ = ssp.read(path)
    assert surface["parts"].tolist() == [0, 0, 0, 1, 1, 1, 2, 2, 2]


def test_pan_sync():
    center = np.zeros(3)
    # the top view (looking down, anterior up) panned 10 mm right, 5 mm up
    pan = ssp.panOffset(center, (10.0, 5.0, 0.0), (10.0, 5.0, 600.0), (0.0, 1.0, 0.0))
    assert np.allclose(pan, (10.0, 5.0))
    # the anterior view (looking back, top up) gets the same screen pan: right on its screen is -x... its own axes
    focal, position = ssp.pannedCamera(center, pan, (0.0, 0.0, 0.0), (0.0, 600.0, 0.0), (0.0, 0.0, 1.0))
    assert np.allclose(ssp.panOffset(center, focal, position, (0.0, 0.0, 1.0)), pan)
    assert np.isclose(np.linalg.norm(focal - position), 600.0)
    assert np.allclose((position - focal) / 600.0, (0.0, 1.0, 0.0))        # its direction kept
