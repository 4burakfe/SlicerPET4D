"""Resampling of a volume's voxels onto another grid through a transform (VTK and numpy, no Slicer imports): used by
Harden of Dynamic PET to CT/MR Registration and, for a new voxel size, of Motion Correction.

A Resampler is made once for a geometry (input grid, output grid, transform) and resamples every frame of a series
with it. A non-linear transform (B-spline) is turned into a displacement grid on the output grid once, so the
frames are not each run through the (iterative) inverse of the B-spline.
"""

import numpy as np
import vtk

MAX_GRID_POINTS = 4000000
INTERPOLATION_MODES = {"Linear": "Linear", "Nearest neighbour": "NearestNeighbor", "Cubic": "Cubic"}


def vtkMatrix(array):
    m = vtk.vtkMatrix4x4()
    for r in range(4):
        for c in range(4):
            m.SetElement(r, c, float(array[r][c]))
    return m


def gridStep(dims, maxPoints=None):
    """Whole voxel step of the displacement grid over an output of dims voxels: 1 up to maxPoints (default
    MAX_GRID_POINTS) voxels."""
    maxPoints = maxPoints or MAX_GRID_POINTS
    total = float(np.prod([max(1, int(d)) for d in dims]))
    return max(1, int(np.ceil((total / maxPoints) ** (1.0 / 3.0) - 1e-9)))


def numpyMatrix(matrix):
    return np.array([[matrix.GetElement(r, c) for c in range(4)] for r in range(4)])


class Resampler:
    """Voxels of a grid (inputIjkToRas, 4x4) resampled onto another (outputDims (i, j, k), outputIjkToRas).
    worldToInput: None (both grids in the same space) or a vtkAbstractTransform from the output's RAS to the input's
    RAS (for a PET under a registration transform: from the world to the PET's own coordinates). linear: the
    transform is linear (its matrix is used directly)."""

    def __init__(self, inputIjkToRas, outputDims, outputIjkToRas, worldToInput=None, linear=True,
                 interpolation="Linear"):
        if interpolation not in INTERPOLATION_MODES:
            raise ValueError(f"Unknown interpolation {interpolation!r}")
        self.inputIjkToRas = np.asarray(inputIjkToRas, dtype=float)
        self.outputDims = tuple(int(v) for v in outputDims)
        self.outputIjkToRas = np.asarray(outputIjkToRas, dtype=float)
        self.interpolation = interpolation
        rasToInputIjk = np.linalg.inv(self.inputIjkToRas)
        if worldToInput is None or linear:
            middle = np.eye(4)
            if worldToInput is not None:
                middle = linearMatrix(worldToInput)
            transform = vtk.vtkTransform()
            transform.SetMatrix(vtkMatrix(rasToInputIjk @ middle @ self.outputIjkToRas))
            self.transform = transform
        else:
            composite = vtk.vtkGeneralTransform()
            composite.PostMultiply()
            composite.Concatenate(vtkMatrix(self.outputIjkToRas))
            composite.Concatenate(worldToInput)
            composite.Concatenate(vtkMatrix(rasToInputIjk))
            # the displacement on a grid of at most ~MAX_GRID_POINTS points over the output (a B-spline is smooth,
            # so a few voxels between the grid points change nothing visible), linearly interpolated in between
            step = gridStep(self.outputDims)
            extent = [int(np.ceil((d - 1) / float(step))) for d in self.outputDims]
            toGrid = vtk.vtkTransformToGrid()
            toGrid.SetInput(composite)
            toGrid.SetGridExtent(0, extent[0], 0, extent[1], 0, extent[2])
            toGrid.SetGridOrigin(0.0, 0.0, 0.0)
            toGrid.SetGridSpacing(float(step), float(step), float(step))
            toGrid.SetGridScalarTypeToFloat()
            toGrid.Update()
            grid = vtk.vtkGridTransform()
            grid.SetDisplacementGridData(toGrid.GetOutput())
            grid.SetInterpolationModeToLinear()
            self.transform = grid

    def resample(self, image, smoothSigmas=None):
        """New vtkImageData (float) on the output grid (origin 0, spacing 1, as Slicer keeps the voxels).
        smoothSigmas: Gaussian standard deviations (input voxels, i, j, k) applied first, against aliasing when the
        output is coarser."""
        info = vtk.vtkImageChangeInformation()
        info.SetInputData(image)
        info.SetOutputOrigin(0.0, 0.0, 0.0)
        info.SetOutputSpacing(1.0, 1.0, 1.0)
        cast = vtk.vtkImageCast()
        cast.SetInputConnection(info.GetOutputPort())
        cast.SetOutputScalarTypeToFloat()
        last = cast
        if smoothSigmas is not None and max(smoothSigmas) > 0:
            smooth = vtk.vtkImageGaussianSmooth()
            smooth.SetInputConnection(cast.GetOutputPort())
            smooth.SetDimensionality(3)
            smooth.SetStandardDeviations(*[float(v) for v in smoothSigmas])
            smooth.SetRadiusFactors(2.0, 2.0, 2.0)
            last = smooth
        reslice = vtk.vtkImageReslice()
        reslice.SetInputConnection(last.GetOutputPort())
        reslice.SetResliceTransform(self.transform)
        getattr(reslice, f"SetInterpolationModeTo{INTERPOLATION_MODES[self.interpolation]}")()
        reslice.SetOutputOrigin(0.0, 0.0, 0.0)
        reslice.SetOutputSpacing(1.0, 1.0, 1.0)
        reslice.SetOutputExtent(0, self.outputDims[0] - 1, 0, self.outputDims[1] - 1, 0, self.outputDims[2] - 1)
        reslice.SetBackgroundLevel(0.0)
        reslice.Update()
        out = vtk.vtkImageData()
        out.DeepCopy(reslice.GetOutput())
        return out


def linearMatrix(transform):
    """4x4 numpy matrix of a linear vtkAbstractTransform (e.g. a vtkGeneralTransform chaining linear transforms), from
    where it moves the origin and the unit points."""
    origin = np.array(transform.TransformPoint(0.0, 0.0, 0.0))
    matrix = np.eye(4)
    for c, e in enumerate(np.eye(3)):
        matrix[:3, c] = np.array(transform.TransformPoint(*[float(v) for v in e])) - origin
    matrix[:3, 3] = origin
    return matrix
