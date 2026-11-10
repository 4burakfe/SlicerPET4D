"""Iterative (group-wise) template building for a normal database (numpy only, no Slicer imports).

Pass 0 registers every subject to the MNI target (e.g. the synthetic PET) and averages them: template 0. Pass k
registers every subject to template k-1 (starting from its previous matrix) and averages again: template k. The
subjects' maps are always resampled from their native images (subjects/<id>/<id>_native.*: one resampling, no blur
builds up); only the matrices are kept from pass to pass (subjects/<id>/transform.json).

Drift: registering to one's own average lets the group slowly grow, shrink or move. After each pass the change of
every subject's matrix since pass 0 (D_i = T_i^k (T_i^0)^-1) is averaged and its inverse applied to all, so the
group's mean position and size stay those of pass 0 (the MNI target's space) while the subjects line up better with
each other (as in unbiased template building: Guimond et al. 2000, Fonov et al. 2011 for the ICBM 152 2009c).

Stop: when the subjects' matrices move less than STOP_MM on average between two passes (RMS over the corners of a
brain-sized box), or after MAX_PASSES.
"""
import csv
import json
import os

import numpy as np

from . import normaldb

TEMPLATES_DIR = "templates"
ITERATIONS_LOG = "template_iterations.csv"     # in <database>/qc/
MAX_PASSES = 5          # pass 0 (to the MNI target) and up to 4 passes to the group's own template
STOP_MM = 0.5           # mean RMS corner displacement between two passes below this: converged
# a box around the brain in the MNI space (mm, RAS): its corners measure how far a matrix moves the brain
BRAIN_BOX = ((-70.0, 70.0), (-105.0, 70.0), (-50.0, 80.0))


def boxCorners(box=BRAIN_BOX):
    """The 8 corners of a box ((x0, x1), (y0, y1), (z0, z1)) as homogeneous points [8, 4]."""
    (x0, x1), (y0, y1), (z0, z1) = box
    return np.array([[x, y, z, 1.0] for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)])


def displacementMm(a, b, box=BRAIN_BOX):
    """RMS distance (mm) between the box corners moved by matrix a and by matrix b (4x4, the same direction)."""
    corners = boxCorners(box)
    pa = corners @ np.asarray(a, dtype=float).T
    pb = corners @ np.asarray(b, dtype=float).T
    return float(np.sqrt(np.mean(np.sum((pa[:, :3] - pb[:, :3]) ** 2, axis=1))))


def meanMatrix(matrices):
    """The average of 4x4 affine matrices (element-wise; fine for the small differences between subjects' drifts)."""
    stack = np.asarray([np.asarray(m, dtype=float) for m in matrices])
    if stack.size == 0:
        return np.eye(4)
    mean = stack.mean(axis=0)
    mean[3] = (0.0, 0.0, 0.0, 1.0)
    return mean


def driftCorrected(current, initial):
    """(corrected matrices, drift) with the group's mean change since pass 0 taken out: D_i = current_i initial_i^-1,
    drift = mean(D_i), corrected_i = drift^-1 current_i. current / initial: {id: 4x4} (subject to template, to
    parent); subjects without an initial matrix are left as they are (not in the mean)."""
    shared = [k for k in current if k in initial]
    if not shared:
        return dict(current), np.eye(4)
    drift = meanMatrix([np.asarray(current[k], dtype=float) @ np.linalg.inv(np.asarray(initial[k], dtype=float))
                        for k in shared])
    inverse = np.linalg.inv(drift)
    return {k: inverse @ np.asarray(m, dtype=float) for k, m in current.items()}, drift


def scaleOf(matrix):
    """Volume scale of a 4x4 matrix (cube root of the determinant of its linear part)."""
    return float(np.cbrt(np.linalg.det(np.asarray(matrix, dtype=float)[:3, :3])))


def passStatistics(current, previous, box=BRAIN_BOX):
    """{"n", "meanMm", "maxMm"}: how far the subjects' matrices moved between two passes ({id: 4x4})."""
    moves = [displacementMm(current[k], previous[k], box) for k in current if k in previous]
    if not moves:
        return {"n": 0, "meanMm": float("nan"), "maxMm": float("nan")}
    return {"n": len(moves), "meanMm": float(np.mean(moves)), "maxMm": float(np.max(moves))}


def converged(stats, stopMm=STOP_MM):
    """Whether a pass moved the subjects less than stopMm on average."""
    return bool(stats.get("n")) and float(stats["meanMm"]) < float(stopMm)


def correlation(a, b, mask=None):
    """Pearson correlation of two maps (inside mask, else where either is non-zero); nan when it cannot be told."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        return float("nan")
    inside = (np.asarray(mask) > 0) if mask is not None else ((a != 0) | (b != 0))
    inside &= np.isfinite(a) & np.isfinite(b)
    if inside.sum() < 3:
        return float("nan")
    x, y = a[inside], b[inside]
    if x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


# --- files ------------------------------------------------------------------------------------------------
def transformPath(folder, subjectId):
    return os.path.join(folder, normaldb.transformName(subjectId))


def templateFile(passIndex):
    """Relative path of a pass's template."""
    return f"{TEMPLATES_DIR}/pass_{int(passIndex)}.nii.gz"


def writeTransform(folder, subjectId, passIndex, matrix, source=""):
    """Append (or replace) a pass's matrix (subject's native image to the MNI space, to parent) in the subject's
    transforms/<id>.json."""
    path = transformPath(folder, subjectId)
    document = readTransforms(folder, subjectId) or {"id": subjectId, "passes": []}
    if source:
        document["source"] = source
    passes = [p for p in document.get("passes", []) if int(p.get("pass", -1)) != int(passIndex)]
    passes.append({"pass": int(passIndex), "matrix": [float(v) for v in np.asarray(matrix, dtype=float).ravel()]})
    document["passes"] = sorted(passes, key=lambda p: p["pass"])
    document["current"] = {"matrix": passes[-1]["matrix"], "target": f"template pass {int(passIndex) - 1}"
                           if passIndex else "MNI target", "deformable": False, "made": normaldb.now()}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(document, f, indent=1)
    return path


def writeCurrent(folder, subjectId, matrix, target, deformable=False):
    """The matrix the subject's maps were last made with (native image to the MNI space, to parent; the linear part
    when a deformable part was used too) in its transform.json, with the template it was registered to."""
    path = transformPath(folder, subjectId)
    document = readTransforms(folder, subjectId) or {"id": subjectId, "passes": []}
    document["current"] = {"matrix": [float(v) for v in np.asarray(matrix, dtype=float).ravel()],
                           "target": str(target or ""), "deformable": bool(deformable), "made": normaldb.now()}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(document, f, indent=1)
    return path


def readTransforms(folder, subjectId):
    path = transformPath(folder, subjectId)
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


def passMatrix(document, passIndex):
    """The 4x4 matrix of a pass from a transforms document, None when it has none."""
    for entry in (document or {}).get("passes", []):
        if int(entry.get("pass", -1)) == int(passIndex):
            return np.asarray(entry["matrix"], dtype=float).reshape(4, 4)
    return None


LOG_COLUMNS = ("pass", "target", "subjects", "failed", "meanMoveMm", "maxMoveMm", "driftScale", "driftShiftMm",
               "templateCorrelation", "seconds")


def writeIterationsLog(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
