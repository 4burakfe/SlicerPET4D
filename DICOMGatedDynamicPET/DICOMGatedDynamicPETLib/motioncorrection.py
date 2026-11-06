"""Pure helpers of Dynamic PET Motion Correction (numpy only, no Slicer imports): frame activities and the
low-activity rule, the motion of a rigid transform (shift and angles), the motion limit, and the texts of the
motion rows on the timeline.

Conventions
- Transforms are 4x4 matrices in RAS (mm). The *motion* of a frame is the transform that takes a point of the
  reference to where the same anatomy is in the frame (the inverse of the correction that BRAINSFit returns).
- The shift of a rigid motion is the displacement of a centre point (the centre of the reference volume), not
  the matrix translation, which depends on where the RAS origin is.
- Angles are in degrees about the R (left-right), A (anterior-posterior) and S (inferior-superior) axes, from
  R = Rz(S) . Ry(A) . Rx(R).
"""

import math

import numpy as np

METHODS = ("Rigid", "Affine", "BSpline")
SECOND_PASS_METHODS = ("Rigid", "Affine", "BSpline")
DEFAULT_METHOD = "Rigid"
DEFAULT_SECOND_METHOD = "Rigid"
DEFAULT_LOW_ACTIVITY_PERCENT = 1.0
DEFAULT_MAX_MOTION_MM = 200.0
DEFAULT_SAMPLING = 0.05
DEFAULT_BSPLINE_GRID = 5
# BRAINSFit initializeTransformMode of the first pass: (title, mode); the second pass starts from the first
INITIALIZATIONS = (("Geometric centers", "useGeometryAlign"), ("Centers of mass", "useMoments"), ("None", "Off"))
DEFAULT_INITIALIZATION = "useGeometryAlign"

# frame status after registration
REGISTERED = "registered"
REFERENCE = "reference"
LOW_ACTIVITY = "low activity"
OVER_LIMIT = "over limit"
FAILED = "failed"
NOT_RUN = "not run"
UNCHANGED = (REFERENCE, LOW_ACTIVITY, OVER_LIMIT, FAILED, NOT_RUN)   # frames kept as they were

# the six rigid motion components of the motion plots: (key, component, label, colour)
COMPONENTS = (
    ("shift", 0, "x (R)", "#D62728"),
    ("shift", 1, "y (A)", "#2CA02C"),
    ("shift", 2, "z (S)", "#1F77B4"),
    ("angles", 0, "pitch (about R)", "#FF7F0E"),
    ("angles", 1, "roll (about A)", "#9467BD"),
    ("angles", 2, "yaw (about S)", "#17BECF"),
)

STATUS_SHORT = {REFERENCE: "ref", LOW_ACTIVITY: "low", OVER_LIMIT: "limit", FAILED: "failed", NOT_RUN: ""}


def showsMotionRows(method):
    """Shift / angle rows are shown for a rigid registration only (the first pass when there are two).

    A rigid transform is exactly 3 shifts and 3 angles. An affine one adds scaling and shear, so splitting it into
    a rotation and a shift is not unique, and a B-spline has a different displacement at every point; neither has
    one shift and one set of angles per frame."""
    return method == "Rigid"


def transformTypes(method, twoPass=False, secondMethod=DEFAULT_SECOND_METHOD):
    """BRAINSFit transform type of each pass: ["Rigid"] or ["Rigid", "Affine"]. The first pass registers the
    frames to the reference, the second to the summed image of the frames corrected by the first. A B-spline
    first pass has no second pass."""
    if method not in METHODS:
        raise ValueError(f"Unknown registration method {method!r}")
    if not twoPass or method == "BSpline":
        return [method]
    if secondMethod not in SECOND_PASS_METHODS:
        raise ValueError(f"Unknown second pass method {secondMethod!r}")
    return [method, secondMethod]


def methodText(method, twoPass=False, secondMethod=DEFAULT_SECOND_METHOD):
    """'Rigid' or 'Rigid, then Affine to the motion corrected sum'."""
    types = transformTypes(method, twoPass, secondMethod)
    return types[0] if len(types) == 1 else f"{types[0]}, then {types[1]} to the motion corrected sum"


def rowsPass(method, twoPass=False, secondMethod=DEFAULT_SECOND_METHOD):
    """Index of the pass whose rigid motion the shift / angle rows show: the second when both are rigid (it is the
    final motion), else the first."""
    types = transformTypes(method, twoPass, secondMethod)
    return 1 if types == ["Rigid", "Rigid"] else 0


# ---------------------------------------------------------------------------
# Low activity frames
# ---------------------------------------------------------------------------
def frameActivity(array):
    """Activity of a frame: the sum of its voxels (all frames share the geometry, so the voxel volume does not
    matter); non finite voxels are left out."""
    values = np.asarray(array, dtype=np.float64)
    return float(np.sum(values[np.isfinite(values)]))


def lowActivityFrames(activities, percent=DEFAULT_LOW_ACTIVITY_PERCENT):
    """True for every frame whose activity is below percent % of the highest frame activity."""
    activities = np.asarray(activities, dtype=float)
    if activities.size == 0:
        return []
    highest = float(np.max(activities))
    if not highest > 0:
        return [False] * activities.size
    return [bool(a < highest * percent / 100.0) for a in activities]


def maxActivityFrame(activities):
    """Index of the frame with the highest activity (the default reference); 0 when empty."""
    return int(np.argmax(activities)) if len(activities) else 0


# ---------------------------------------------------------------------------
# Rigid motion
# ---------------------------------------------------------------------------
def eulerAngles(rotation):
    """(about R, about A, about S) in degrees of a 3x3 rotation R = Rz . Ry . Rx."""
    r = np.asarray(rotation, dtype=float)
    sy = -r[2, 0]
    ay = math.degrees(math.asin(max(-1.0, min(1.0, sy))))
    if abs(sy) < 1.0 - 1e-9:
        ax = math.degrees(math.atan2(r[2, 1], r[2, 2]))
        az = math.degrees(math.atan2(r[1, 0], r[0, 0]))
    else:     # gimbal lock: put it all in the angle about R
        ax = math.degrees(math.atan2(-r[1, 2], r[1, 1]))
        az = 0.0
    return ax, ay, az


def rotationMatrix(ax, ay, az):
    """3x3 rotation of eulerAngles (degrees)."""
    x, y, z = (math.radians(a) for a in (ax, ay, az))
    rx = np.array([[1, 0, 0], [0, math.cos(x), -math.sin(x)], [0, math.sin(x), math.cos(x)]])
    ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
    rz = np.array([[math.cos(z), -math.sin(z), 0], [math.sin(z), math.cos(z), 0], [0, 0, 1]])
    return rz @ ry @ rx


def rigidMotion(motionMatrix, center):
    """(shift (3, mm) of the centre point, angles (3, degrees)) of a 4x4 rigid motion matrix."""
    m = np.asarray(motionMatrix, dtype=float)
    c = np.asarray(center, dtype=float)
    moved = m[:3, :3] @ c + m[:3, 3]
    return tuple(float(v) for v in moved - c), eulerAngles(m[:3, :3])


# ---------------------------------------------------------------------------
# Motion limit
# ---------------------------------------------------------------------------
def limitSamplePoints(bounds, steps=3):
    """Points where the motion is checked against the limit: a steps^3 grid over the middle half of the volume
    bounds (xmin, xmax, ymin, ymax, zmin, zmax), so a rotation is not judged by the far corners of the field of
    view. Returns an (n, 3) array."""
    axes = []
    for a in range(3):
        low, high = float(bounds[2 * a]), float(bounds[2 * a + 1])
        middle, half = (low + high) / 2.0, (high - low) / 4.0
        axes.append(np.linspace(middle - half, middle + half, steps) if steps > 1 else np.array([middle]))
    grid = np.meshgrid(*axes, indexing="ij")
    return np.stack([g.ravel() for g in grid], axis=1)


def largestDisplacement(points, movedPoints):
    """Largest absolute displacement along R, A and S over the points: (dR, dA, dS) in mm."""
    d = np.abs(np.asarray(movedPoints, dtype=float) - np.asarray(points, dtype=float))
    return tuple(float(v) for v in d.max(axis=0)) if d.size else (0.0, 0.0, 0.0)


def exceedsLimit(displacement, limitMm):
    """True when any axis moves more than limitMm (None or <= 0: no limit)."""
    if limitMm is None or limitMm <= 0:
        return False
    return any(abs(v) > limitMm for v in displacement)


# ---------------------------------------------------------------------------
# Texts
# ---------------------------------------------------------------------------
def _fmt(value):
    text = f"{value:.1f}"
    return "0.0" if text == "-0.0" else text


def vectorText(values):
    """'1.2, -0.4, 3.0'."""
    return ", ".join(_fmt(v) for v in values)


def cellText(values, width, charWidth=6.0):
    """Text of one cell of a motion row in a frame block width pixels wide: the three values when they fit, the
    largest one (by size, signed) when that fits, else nothing."""
    if values is None:
        return ""
    full = vectorText(values)
    if len(full) * charWidth + 6 <= width:
        return full
    largest = _fmt(max(values, key=abs))
    if len(largest) * charWidth + 4 <= width:
        return largest
    return ""


def statusCellText(status, width, charWidth=6.0):
    text = STATUS_SHORT.get(status, "")
    return text if len(text) * charWidth + 4 <= width else ""


def magnitudeLevel(values, low, high):
    """0 (below low), 1 (below high) or 2 of the largest absolute value, for the cell colour."""
    if values is None:
        return 0
    largest = max(abs(v) for v in values)
    return 0 if largest < low else (1 if largest < high else 2)


def frameMotionTooltip(entry):
    """Lines of the timeline tooltip for a frame's motion entry {"status", "shift", "angles", "message"}."""
    if not entry:
        return ""
    lines = [f"Motion correction: {entry.get('status', '')}"]
    if entry.get("shift") is not None:
        lines.append(f"Shift R, A, S: {vectorText(entry['shift'])} mm")
    if entry.get("angles") is not None:
        lines.append(f"Angles about R, A, S: {vectorText(entry['angles'])} deg")
    if entry.get("message"):
        lines.append(entry["message"])
    return "\n".join(lines)


def summaryText(statuses):
    """'12 registered, 2 low activity, 1 over limit'."""
    order = (REGISTERED, REFERENCE, LOW_ACTIVITY, OVER_LIMIT, FAILED)
    counts = [(s, sum(1 for x in statuses if x == s)) for s in order]
    return ", ".join(f"{n} {s}" for s, n in counts if n)


def plotLimit(values, minimum):
    """Half height of a motion plot axis that is symmetric about 0: the largest absolute value plus 10 %, at
    least minimum."""
    largest = max([abs(v) for v in values] or [0.0])
    return max(minimum, largest * 1.1)


def plotTicks(limit, count=3):
    """Ticks of a symmetric axis -limit..limit: 0 and about count round steps either side."""
    if not limit > 0:
        return [0.0]
    rough = limit / float(count)
    magnitude = 10.0 ** math.floor(math.log10(rough))
    step = next(m * magnitude for m in (1, 2, 2.5, 5, 10) if m * magnitude >= rough * (1 - 1e-9))
    n = int(math.floor(limit / step + 1e-9))
    return [float(f"{i * step:.10g}") for i in range(-n, n + 1)]
