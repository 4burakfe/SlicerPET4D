"""The iterative engine (sitkreg, needs SimpleITK and scipy) on a synthetic pair: python -m pytest Testing."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "DICOMGatedDynamicPET"))
from DICOMGatedDynamicPETLib import registration as reg  # noqa: E402

SR = pytest.importorskip("DICOMGatedDynamicPETLib.sitkreg")
pytest.importorskip("SimpleITK")
pytest.importorskip("scipy")


def phantom(shape, spacing, centre):
    k, j, i = np.meshgrid(*(np.arange(n) for n in shape), indexing="ij")
    x, y, z = i * spacing, j * spacing, k * spacing
    image = np.zeros(shape, np.float32)
    for (cx, cy, cz), r, value in (((0, 0, 0), 40, 1.0), ((15, 5, 0), 12, 3.0), ((-12, -10, 8), 8, 2.0)):
        inside = (x - centre[0] - cx) ** 2 + (y - centre[1] - cy) ** 2 + (z - centre[2] - cz) ** 2 < r ** 2
        image[inside] = value
    return image


def test_rigid_then_deformable_chain():
    spacing = 2.0
    shape = (48, 56, 56)
    matrix = np.diag([spacing, spacing, spacing, 1.0])
    fixed = phantom(shape, spacing, (56, 56, 48))
    moving = phantom(shape, spacing, (62, 52, 48))            # moved 6 mm R, -4 mm A in its own grid
    result = SR.register(SR.ImageInput(fixed, matrix), SR.ImageInput(moving, matrix, SR.KIND_FUNCTIONAL), None,
                         SR.RIGID, 2.0, "Fast", None, "current", deformable=True, gridSpacingMm=50.0, passes=2,
                         fieldGeometry=(shape, matrix))
    m = result["matrix"]
    point = np.array([56.0, 56.0, 48.0, 1.0])                  # fixed centre: its PET point is 6 mm R, 4 mm P
    np.testing.assert_allclose((np.linalg.inv(m) @ point)[:3], (62.0, 52.0, 48.0), atol=1.5)
    if result.get("displacement") is not None:
        residual = reg.residualField(result["displacement"], result["fieldIjkToWorld"], m)
        assert residual.shape == result["displacement"].shape
        assert reg.fieldStats(residual)["max"] < 10.0
