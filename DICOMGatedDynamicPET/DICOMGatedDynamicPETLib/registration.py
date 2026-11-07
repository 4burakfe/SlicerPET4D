"""Pure helpers of Dynamic PET to CT/MR Registration (numpy only, no Slicer imports): the registration steps, the
summary of the applied transform (shift, angles, scaling, shear, deformation), the checkerboard pattern, the output
grid of Harden and the layouts.

Conventions
- Transforms are 4x4 matrices in RAS (mm) that move the PET onto the CT/MR (Slicer's "to parent" direction: a PET
  volume under the transform lies on the CT/MR).
- The shift of a transform is the displacement of a centre point (the centre of the PET), not the matrix
  translation, which depends on where the RAS origin is.
- Angles are in degrees about R, A and S, as in Motion Correction (R = Rz(S) . Ry(A) . Rx(R)).
"""

import math

import numpy as np

from . import motioncorrection as mc

STEP_TYPES = ("Rigid", "Affine", "BSpline")
# the PET to the same patient's CT/MR: rigid only by default (step 2 "(none)"); affine/B-spline can be added
DEFAULT_STEPS = ("Rigid",)
MAX_STEPS = 2
# whether a step registers to the CT/MR resampled to the PET voxel size (workingSpacing), by default: every step (a
# sub-millimetre MR at full resolution takes tens of GB of memory; the PET resolution limits the match anyway)
RESAMPLE_BY_DEFAULT = {"Rigid": True, "Affine": True, "BSpline": True}
DEFAULT_SAMPLING = 0.05
DEFAULT_BSPLINE_GRID = 5
DEFAULT_INITIALIZATION = mc.DEFAULT_INITIALIZATION      # geometric centres, as in Motion Correction
INITIALIZATIONS = mc.INITIALIZATIONS
DEFAULT_CHECKER_MM = 20.0
DEFAULT_WORKING_VOXEL_MM = 0.0          # 0: register to the CT/MR at the PET voxel size
# output grid of Harden: the CT/MR directions and field of view (extended to the registered PET's where that is larger,
# except on the CT/MR grid itself), with one of these voxel sizes
OUTPUT_PET_SPACING = "petSpacing"       # keep the PET voxel size (default)
OUTPUT_FIXED_GRID = "fixedGrid"         # the CT/MR grid itself
OUTPUT_CUSTOM = "custom"                # a voxel size the user gives
OUTPUT_GRIDS = (("Keep the PET voxel size (CT/MR field of view, extended to the PET)", OUTPUT_PET_SPACING),
                ("Resample to the CT/MR grid", OUTPUT_FIXED_GRID),
                ("Resample to this voxel size:", OUTPUT_CUSTOM))
INTERPOLATIONS = ("Linear", "Nearest neighbour", "Cubic")
DEFAULT_INTERPOLATION = "Linear"
OVERWRITE = "overwrite"                 # Harden replaces the frames (and the summed image) of the series
SAVE_SEPARATELY = "separate"            # Harden makes a new series next to the original

# Registration engines: BRAINSFit in steps, or the iterative multi-resolution engine (sitkreg, as in EasyReg): a rigid
# or affine stage, then iterative B-spline passes (coarse to fine control points), at a working resolution of a few mm
ENGINE_BRAINSFIT = "brainsfit"
ENGINE_ITERATIVE = "iterative"
ENGINES = (("BRAINSFit (steps)", ENGINE_BRAINSFIT),
           ("Iterative multi-resolution (linear, then optional deformable)", ENGINE_ITERATIVE))
DEFAULT_ENGINE = ENGINE_BRAINSFIT
ITERATIVE_LINEAR_TYPES = ("Rigid", "Affine")
DEFAULT_ITERATIVE_LINEAR = "Rigid"
DEFAULT_ITERATIVE_DEFORMABLE = False   # same patient: rigid only; the Neuro normalization turns it on
DEFAULT_ITERATIVE_GRID_MM = 50.0        # control points of the last deformable pass
DEFAULT_ITERATIVE_SPACING_MM = 2.0      # final working resolution
DEFAULT_ITERATIVE_QUALITY = "Standard"
ITERATIVE_SUFFIX = " (iterative)"
DEFORMABLE_STEP = "Deformable"
FIELD_MOVED_MM = 0.05                   # field points that move less are left out of the mean deformation

# CT windows (title, lower, upper); "Auto" is the 1st - 99.5th percentile
CT_WINDOWS = (("Soft tissue (W 400, L 40)", -160.0, 240.0), ("Brain (W 80, L 40)", 0.0, 80.0),
              ("Bone (W 2000, L 400)", -600.0, 1400.0), ("Lung (W 1500, L -600)", -1350.0, 150.0))
AUTO_WINDOW = "Auto (1 - 99.5 %)"


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------
def validateSteps(steps):
    """The steps (BRAINSFit transform types, run one after the other, each starting from the previous) as a list.
    1 to MAX_STEPS steps; a B-spline only as the last step, and no rigid step after an affine one (it would undo the
    scaling). Raises ValueError otherwise."""
    steps = [s for s in steps if s]
    if not steps:
        raise ValueError("Pick at least one registration step.")
    if len(steps) > MAX_STEPS:
        raise ValueError(f"At most {MAX_STEPS} registration steps.")
    for s in steps:
        if s not in STEP_TYPES:
            raise ValueError(f"Unknown registration step {s!r}")
    if "BSpline" in steps[:-1]:
        raise ValueError("A B-spline step can only be the last step.")
    for previous, step in zip(steps, steps[1:]):
        if previous == "Affine" and step == "Rigid":
            raise ValueError("A rigid step cannot follow an affine step.")
    return steps


def resampleFlags(steps, flags=None):
    """Per step: register to the CT/MR resampled to the PET voxel size (True) or as it is. flags: given per step
    (missing ones default to RESAMPLE_BY_DEFAULT)."""
    flags = list(flags or [])
    return [bool(flags[i]) if i < len(flags) and flags[i] is not None else RESAMPLE_BY_DEFAULT[step]
            for i, step in enumerate(steps)]


def stepsText(steps):
    """'Rigid, then Affine, then BSpline'."""
    return ", then ".join(steps)


def isLinearStep(step):
    return step in ("Rigid", "Affine")


def iterativeSteps(linear=DEFAULT_ITERATIVE_LINEAR, deformable=DEFAULT_ITERATIVE_DEFORMABLE):
    """Step names of an iterative registration: ['Rigid (iterative)', 'Deformable (iterative)']."""
    if linear not in ITERATIVE_LINEAR_TYPES:
        raise ValueError(f"Unknown linear stage {linear!r}")
    return [linear + ITERATIVE_SUFFIX] + ([DEFORMABLE_STEP + ITERATIVE_SUFFIX] if deformable else [])


def iterativeInitialization(mode):
    """The iterative engine's start for a BRAINSFit initialization mode: the image centres (geometric centres and
    centres of mass), or the current position (Off)."""
    return "current" if mode == "Off" else "centres"


def residualField(displacement, fieldIjkToWorld, matrix):
    """The deformable part of an iterative result, for a chain PET -> linear (matrix) -> grid -> world. displacement
    [k, j, i, 3] (world RAS mm) maps each world point x of the field grid to the PET point x + d(x) (linear part
    included); the grid transform under the linear one must map x to matrix (x + d(x)), so its displacement (from
    parent) is matrix (x + d(x)) - x."""
    d = np.asarray(displacement, dtype=float)
    nk, nj, ni = d.shape[:3]
    k, j, i = np.meshgrid(np.arange(nk), np.arange(nj), np.arange(ni), indexing="ij")
    ijk = np.stack([i, j, k, np.ones_like(i)], axis=-1).reshape(-1, 4).astype(float)
    world = (ijk @ np.asarray(fieldIjkToWorld, dtype=float).T)[:, :3]
    moved = world + d.reshape(-1, 3)
    m = np.asarray(matrix, dtype=float)
    mapped = moved @ m[:3, :3].T + m[:3, 3]
    return (mapped - world).reshape(d.shape)


def fieldStats(residual, minimumMm=FIELD_MOVED_MM):
    """displacementStats-like {"mean", "max", "maxAxis"} of a residual field [k, j, i, 3]: the mean over the points that
    move (the field fades to 0 away from the body)."""
    d = np.asarray(residual, dtype=float).reshape(-1, 3)
    if d.size == 0:
        return {"mean": 0.0, "max": 0.0, "maxAxis": (0.0, 0.0, 0.0)}
    lengths = np.linalg.norm(d, axis=1)
    moving = lengths > minimumMm
    return {"mean": float(lengths[moving].mean()) if moving.any() else 0.0, "max": float(lengths.max()),
            "maxAxis": tuple(float(v) for v in np.abs(d).max(axis=0))}


# ---------------------------------------------------------------------------
# Transform summary
# ---------------------------------------------------------------------------
def decomposeAffine(matrix):
    """(rotation 3x3, scales (3,), shears (xy, xz, yz)) of the 3x3 part of a 4x4 matrix, A = R . K with K upper
    triangular (QR). Scales are 1 for a rigid transform; a reflection shows as a negative scale."""
    a = np.asarray(matrix, dtype=float)[:3, :3]
    q, k = np.linalg.qr(a)
    signs = np.sign(np.diag(k))
    signs[signs == 0] = 1.0
    q, k = q * signs, (k.T * signs).T
    if np.linalg.det(q) < 0:          # keep R a rotation; the reflection goes to the last scale
        q[:, 2] *= -1.0
        k[2, :] *= -1.0
    scales = np.diag(k).copy()
    shears = (k[0, 1] / scales[0], k[0, 2] / scales[0], k[1, 2] / scales[1])
    return q, scales, tuple(float(v) for v in shears)


def linearSummary(matrix, center):
    """{"shift" (mm, of the centre), "angles" (deg), "scales" (%, 100 = none), "shears", "rigid"} of a 4x4."""
    m = np.asarray(matrix, dtype=float)
    rotation, scales, shears = decomposeAffine(m)
    c = np.asarray(center, dtype=float)
    shift = m[:3, :3] @ c + m[:3, 3] - c
    rigid = bool(np.allclose(scales, 1.0, atol=1e-4) and np.allclose(shears, 0.0, atol=1e-4))
    return {"shift": tuple(float(v) for v in shift), "angles": mc.eulerAngles(rotation),
            "scales": tuple(float(100.0 * s) for s in scales), "shears": shears, "rigid": rigid}


def displacementStats(points, movedPoints):
    """{"mean", "max" (mm, length of the displacement), "maxAxis" (largest |d| along R, A, S)} of point pairs."""
    d = np.asarray(movedPoints, dtype=float) - np.asarray(points, dtype=float)
    if d.size == 0:
        return {"mean": 0.0, "max": 0.0, "maxAxis": (0.0, 0.0, 0.0)}
    lengths = np.linalg.norm(d, axis=1)
    return {"mean": float(lengths.mean()), "max": float(lengths.max()),
            "maxAxis": tuple(float(v) for v in np.abs(d).max(axis=0))}


def samplePoints(bounds, steps=7):
    """steps^3 points over the middle half of the bounds (xmin, xmax, ymin, ymax, zmin, zmax)."""
    return mc.limitSamplePoints(bounds, steps)


def summaryLines(steps, linear, center, deformation=None, manual=None, manualCenter=None):
    """Text lines of the applied transform: the automatic linear part, the B-spline deformation (stats of
    displacementStats) and the manual correction (4x4 or None; its shift is that of manualCenter, default where the
    linear part puts the centre)."""
    lines = [f"Steps: {stepsText(steps)}" if steps else "Not registered"]
    if linear is not None:
        s = linearSummary(linear, center)
        lines.append(f"Shift R, A, S: {mc.vectorText(s['shift'])} mm")
        lines.append(f"Angles about R, A, S: {mc.vectorText(s['angles'])} deg")
        if not s["rigid"]:
            lines.append(f"Scaling R, A, S: {mc.vectorText(s['scales'])} %")
            lines.append("Shear xy, xz, yz: " + ", ".join(f"{v:.3f}" for v in s["shears"]))
    if deformation is not None:
        lines.append(f"Deformation: mean {deformation['mean']:.1f} mm, max {deformation['max']:.1f} mm "
                     f"(largest R, A, S: {mc.vectorText(deformation['maxAxis'])} mm)")
    if manual is not None and not np.allclose(manual, np.eye(4), atol=1e-6):
        if manualCenter is None:
            moved = np.asarray(linear if linear is not None else np.eye(4), float) @ np.append(
                np.asarray(center, float), 1.0)
            manualCenter = moved[:3]
        s = linearSummary(manual, manualCenter)
        lines.append(f"Manual: shift {mc.vectorText(s['shift'])} mm, angles {mc.vectorText(s['angles'])} deg")
    return lines


# ---------------------------------------------------------------------------
# Checkerboard and output grid
# ---------------------------------------------------------------------------
def checkerboard(shape, spacing, squareMm=DEFAULT_CHECKER_MM, columnAxis=None):
    """uint8 array of shape (k, j, i): 1 and 0 in squares squareMm wide; spacing (i, j, k) in mm. columnAxis (0 = i,
    1 = j, 2 = k): the pattern does not change along that axis (columns through the volume, so every slice across
    it shows the same 2D checkerboard); None: cubes."""
    if squareMm <= 0:
        raise ValueError("The checkerboard square size must be positive.")
    k, j, i = (np.arange(n) for n in shape)
    cells = [np.floor(index * float(spacing[a]) / squareMm).astype(int) for a, index in enumerate((i, j, k))]
    if columnAxis is not None:
        cells[columnAxis] = np.zeros_like(cells[columnAxis])
    ci, cj, ck = cells
    return ((ck[:, None, None] + cj[None, :, None] + ci[None, None, :]) % 2).astype(np.uint8)


# the patient axis across the slices of a view: S for axial, A for coronal, R for sagittal (0 = R, 1 = A, 2 = S)
SLICE_NORMALS = {"Axial": 2, "Coronal": 1, "Sagittal": 0}


def closestAxis(ijkToRas, rasAxis):
    """Index (0 = i, 1 = j, 2 = k) of the grid axis that runs closest to the patient axis rasAxis (0 = R, 1 = A,
    2 = S)."""
    m = np.asarray(ijkToRas, dtype=float)[:3, :3]
    directions = m / np.linalg.norm(m, axis=0)
    return int(np.argmax(np.abs(directions[rasAxis, :])))


def gridSpacing(ijkToRas):
    m = np.asarray(ijkToRas, dtype=float)
    return np.linalg.norm(m[:3, :3], axis=0)


def outputSpacing(mode, petSpacing, fixedSpacing, customSpacing=None):
    """Voxel size (i, j, k) of the Harden grid for one of OUTPUT_GRIDS."""
    if mode == OUTPUT_PET_SPACING:
        return tuple(float(v) for v in petSpacing)
    if mode == OUTPUT_FIXED_GRID:
        return tuple(float(v) for v in fixedSpacing)
    if mode == OUTPUT_CUSTOM:
        values = tuple(float(v) for v in (customSpacing or ()))
        if len(values) != 3 or min(values) <= 0:
            raise ValueError("Give a positive voxel size along i, j and k.")
        return values
    raise ValueError(f"Unknown output grid {mode!r}")


def workingSpacing(fixedSpacing, petSpacing, voxelMm=None):
    """Voxel size (i, j, k) of the CT/MR copy BRAINSFit registers to: the PET voxel size (or voxelMm along every
    axis), never finer than the CT/MR itself. None when the CT/MR is not finer than that along any axis (it is used
    as it is). A fine MR (0.5 mm) makes the B-spline step use many GB of memory; the PET resolution limits what
    the registration can match anyway."""
    fixedSpacing = np.asarray(fixedSpacing, dtype=float)
    target = np.full(3, float(voxelMm)) if voxelMm else np.asarray(petSpacing, dtype=float)
    spacing = np.maximum(fixedSpacing, target)
    if np.allclose(spacing, fixedSpacing, rtol=1e-3):
        return None
    return tuple(float(v) for v in spacing)


def poolToSpacing(voxels, ijkToWorld, spacingMm):
    """Average blocks of voxels (whole voxel counts per axis) so the voxel size gets as close to spacingMm as it can
    without exceeding it. Returns (voxels, IJK-to-world of the block centres); the input itself when no axis is finer
    than half of spacingMm."""
    from . import sitkreg
    matrix = np.asarray(ijkToWorld, dtype=float)
    spacing = np.linalg.norm(matrix[:3, :3], axis=0)
    factors = [max(1, int(np.floor(float(spacingMm) / s + 1e-6))) if s > 0 else 1 for s in spacing]
    if factors == [1, 1, 1]:
        return voxels, matrix
    return sitkreg.blockMean(voxels, matrix, factors)


def smoothingSigmas(inputSpacing, outputSpacing):
    """Gaussian standard deviations (voxels of the input, i, j, k) against aliasing before resampling to a coarser
    grid: half the ratio of the voxel sizes along each axis that gets coarser, 0 elsewhere."""
    ratio = np.asarray(outputSpacing, dtype=float) / np.asarray(inputSpacing, dtype=float)
    return tuple(float(0.5 * r) if r > 1.0 + 1e-6 else 0.0 for r in ratio)


def outputGrid(fixedDims, fixedIjkToRas, petSpacing):
    """(dims (i, j, k), ijkToRas 4x4) of a grid with the directions and the field of view of the CT/MR grid and the
    given voxel size (petSpacing): the first voxel's corner is the CT/MR first voxel's corner."""
    m = np.asarray(fixedIjkToRas, dtype=float)
    spacing = gridSpacing(m)
    directions = m[:3, :3] / spacing
    petSpacing = np.asarray(petSpacing, dtype=float)
    length = np.asarray(fixedDims, dtype=float) * spacing
    dims = np.maximum(1, np.round(length / petSpacing - 1e-6).astype(int))
    corner = m[:3, 3] - directions @ (spacing / 2.0)
    out = np.eye(4)
    out[:3, :3] = directions * petSpacing
    out[:3, 3] = corner + directions @ (petSpacing / 2.0)
    return tuple(int(v) for v in dims), out


def extendGrid(dims, ijkToRas, points):
    """(dims, ijkToRas) of a grid (from outputGrid) grown by whole voxels along its own axes until it covers the
    world points (e.g. the corners of the registered PET), so a CT/MR smaller than the PET does not cut the PET. The
    voxels stay where they were (the grown grid is aligned with the original); unchanged when it covers them."""
    m = np.asarray(ijkToRas, dtype=float)
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    if not len(points):
        return tuple(int(v) for v in dims), m
    ijk = (np.linalg.inv(m) @ np.c_[points, np.ones(len(points))].T)[:3].T     # continuous voxel indices
    low = np.minimum(0, np.floor(ijk.min(axis=0) + 0.5 + 1e-6)).astype(int)    # voxel i covers [i - 0.5, i + 0.5]
    high = np.maximum(np.asarray(dims, dtype=int) - 1, np.ceil(ijk.max(axis=0) - 0.5 - 1e-6)).astype(int)
    out = np.array(m)
    out[:3, 3] = m[:3, :3] @ low + m[:3, 3]
    return tuple(int(v) for v in high - low + 1), out


# ---------------------------------------------------------------------------
# Layout: 2 rows (axial, coronal) x 3 columns (PET, fusion, CT/MR), with the checkerboard column on the left: 4
# ---------------------------------------------------------------------------
ROWS = (("Axial", "#F34A33"), ("Coronal", "#6EB04B"))
COLUMNS = (("Checker", "Checkerboard"), ("PET", "PET"), ("Fusion", "Fusion"), ("Anat", "CT/MR"))
VIEW_PREFIX = "DynReg"
VIEW_GROUPS = {"Axial": 81, "Coronal": 82}


def viewName(orientation, column):
    return f"{VIEW_PREFIX}{orientation}{column}"


def columns(withCheckerboard):
    return [key for key, _ in COLUMNS if withCheckerboard or key != "Checker"]


def layoutXml(withCheckerboard=False):
    """Slicer layout: the axial row on top, the coronal row below; the columns PET, fusion, CT/MR, and with
    withCheckerboard the checkerboard first. Each row is one view group, so its views link among themselves."""
    titles = dict(COLUMNS)
    rows = []
    for orientation, color in ROWS:
        items = "".join(
            f'<item><view class="vtkMRMLSliceNode" singletontag="{viewName(orientation, key)}">'
            f'<property name="orientation" action="default">{orientation}</property>'
            f'<property name="viewlabel" action="default">{orientation[0]} {titles[key]}</property>'
            f'<property name="viewcolor" action="default">{color}</property>'
            f'<property name="viewgroup" action="default">{VIEW_GROUPS[orientation]}</property>'
            '</view></item>'
            for key in columns(withCheckerboard))
        rows.append(f'<item><layout type="horizontal" split="true">{items}</layout></item>')
    return f'<layout type="vertical" split="true">{"".join(rows)}</layout>'


def percentileWindow(values, low=1.0, high=99.5):
    """(lower, upper) percentiles of the finite values (a sample of at most ~400k), (0, 1) when empty."""
    sample = np.asarray(values).ravel()
    if sample.size > 400000:
        sample = sample[::int(math.ceil(sample.size / 400000.0))]
    sample = sample[np.isfinite(sample)]
    if not sample.size:
        return 0.0, 1.0
    lower, upper = (float(v) for v in np.percentile(sample, [low, high]))
    return (lower, upper) if upper > lower else (lower, lower + 1.0)
