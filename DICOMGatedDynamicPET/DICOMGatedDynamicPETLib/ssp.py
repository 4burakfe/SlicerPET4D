"""3D stereotactic surface projection (3D-SSP; Minoshima et al., J Nucl Med 1995) of the Static Neuro Review
(numpy + scipy, no Slicer).

    surface     the brain surface (the union of the regions, made in Slicer with VTK): its vertices (RAS mm), outward
                normals and triangles; for the medial views, the surfaces of the two hemispheres (the union cut at
                the midline, x = 0 in MNI space) after it, each vertex's part in "parts" (0 brain, 1 left, 2 right)
    projection  each vertex takes the maximum of a map along its inward normal, from the surface to DEPTH_MM (20 mm),
                sampled every STEP_MM with trilinear interpolation; the ray stops where it leaves the brain (the
                union of the regions, when given): no samples beyond it
    scores      Minoshima: every normal's map (smoothed as the voxel maps) is projected, and the patient's projection
                is scored against the vertex-wise mean / SD of the normals' projections (z, or the Crawford-Howell t);
                voxel: the voxel score map itself projected (the most extreme score along the normal), quicker and
                possible without the normals' projections, but not the same as the Minoshima scores
    statistics  the normals' vertex mean / SD per group in an .npz with the surface (stats/ssp.npz, made with the
                mean / SD maps; or on the fly, cached in the database's cache/)
"""

import hashlib
import json
import os

import numpy as np

from . import neuroreview as nr

DEPTH_MM = 20.0
STEP_MM = 1.0
SOURCE_MINOSHIMA, SOURCE_VOXEL = "minoshima", "voxel"
SOURCES = ((SOURCE_MINOSHIMA, "Minoshima (normals' projections)"), (SOURCE_VOXEL, "Voxel score map projected"))
DEFAULT_SOURCE = SOURCE_MINOSHIMA
MAP_SCORE, MAP_PATIENT = "score", "patient"
MAPS = ((MAP_SCORE, "Score (z / t)"), (MAP_PATIENT, "Patient (projected values)"))
SSP_FILE = "stats/ssp.npz"
# the surface's version in the .npz meta: 4, contoured on a coarser grid (closed, in one piece; 3 was decimated and
# split again, its pieces' normals pointing in or out at random: a speckled surface), the hemispheres (medial
# views) only from regions with sides; an older stored surface is not used
SURFACE_VERSION = 4

PART_BRAIN, PART_LEFT, PART_RIGHT = 0, 1, 2
# camera of each view: (direction from the brain centre to the camera, view up), RAS; the medial views look at one
# hemisphere from the other side
CAMERAS = {
    "Top": ((0.0, 0.0, 1.0), (0.0, 1.0, 0.0)),
    "Bottom": ((0.0, 0.0, -1.0), (0.0, 1.0, 0.0)),
    "Ant": ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    "Post": ((0.0, -1.0, 0.0), (0.0, 0.0, 1.0)),
    "Left": ((-1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    "Right": ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    "LeftMedial": ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    "RightMedial": ((-1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
}
VIEW_PARTS = {"LeftMedial": PART_LEFT, "RightMedial": PART_RIGHT}       # the other views: PART_BRAIN


def offsets(depthMm=DEPTH_MM, stepMm=STEP_MM):
    """Distances (mm) sampled along the inward normal: 0, step, ..., depth."""
    count = int(np.floor(depthMm / stepMm + 1e-6))
    values = [i * stepMm for i in range(count + 1)]
    if values[-1] < depthMm - 1e-6:
        values.append(depthMm)
    return np.array(values, dtype=np.float64)


def sideLabels(key):
    """("R" / "L" on the left of the view, on the right) of a view whose screen left / right is the patient's right /
    left (the superior, inferior, anterior and posterior views, as their cameras are set); None for the others."""
    direction, up = CAMERAS[key]
    right = np.cross(-np.asarray(direction, np.float64), np.asarray(up, np.float64))     # VTK: the screen's right
    if abs(right[0]) < 0.9:
        return None
    return ("L", "R") if right[0] > 0 else ("R", "L")


def viewPart(key):
    return VIEW_PARTS.get(key, PART_BRAIN)


def hemispheres(shape, ijkToRas, midlineX=0.0):
    """(left, right) boolean arrays (k, j, i) of a grid: the voxels with RAS x below / above midlineX."""
    m = np.asarray(ijkToRas, dtype=np.float64)
    k, j, i = (np.arange(n, dtype=np.float64) for n in shape)
    x = (m[0, 0] * i[None, None, :] + m[0, 1] * j[None, :, None] + m[0, 2] * k[:, None, None] + m[0, 3])
    x = np.broadcast_to(x, tuple(shape))
    return x < midlineX, x >= midlineX


def regionSide(name):
    """"L" / "R" of a region named with its side ("Precuneus L", "Pons R (reference)"), None for the others."""
    text = str(name or "").strip()
    if text.lower().endswith(nr.REFERENCE_MARK):
        text = text[:-len(nr.REFERENCE_MARK)].strip()
    side = text[-2:].upper()
    return side[1] if side in (" L", " R") else None


def hemisphereMasks(masks, shape):
    """(left, right) boolean arrays of the regions {name: mask} named with a side ("... L" / "... R"; the others,
    e.g. a bilateral reference, left out; a voxel of both sides left): the hemispheres of the medial views. None for
    bilateral regions (no medial views: the brain is not cut at the midline)."""
    left = np.zeros(shape, dtype=bool)
    right = np.zeros(shape, dtype=bool)
    for name, mask in masks.items():
        side = regionSide(name)
        if side == "L":
            left |= np.asarray(mask, dtype=bool)
        elif side == "R":
            right |= np.asarray(mask, dtype=bool)
    if not left.any() or not right.any():
        return None
    return left, right & ~left


def combine(brain, left=None, right=None):
    """One surface of the brain's and the hemispheres' surfaces (vertices after one another, "parts" per vertex)."""
    pieces = [(PART_BRAIN, brain)] + [(part, s) for part, s in ((PART_LEFT, left), (PART_RIGHT, right))
                                      if s is not None and len(s["points"])]
    points, normals, triangles, parts = [], [], [], []
    offset = 0
    for part, s in pieces:
        count = len(s["points"])
        points.append(np.asarray(s["points"], np.float64))
        normals.append(np.asarray(s["normals"], np.float64))
        triangles.append(np.asarray(s["triangles"], np.int64) + offset)
        parts.append(np.full(count, part, dtype=np.int8))
        offset += count
    return {"points": np.concatenate(points), "normals": np.concatenate(normals),
            "triangles": np.concatenate(triangles), "parts": np.concatenate(parts)}


def hasMedial(surface):
    """Whether a surface has both hemispheres' surfaces (the medial views)."""
    return hasPart(surface, PART_LEFT) and hasPart(surface, PART_RIGHT)


def openEdges(triangles):
    """Number of edges of a triangle mesh in one triangle only (0: closed; cracks or holes otherwise)."""
    triangles = np.asarray(triangles, dtype=np.int64)
    if not len(triangles):
        return 0
    edges = np.sort(np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]]), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return int(np.sum(counts == 1))


def clippingRange(points, center, distance, margin=20.0):
    """(near, far) of a camera distance mm from center (parallel projection) that holds every point: the views
    otherwise drawn black or cut until the first interaction."""
    points = np.asarray(points, dtype=np.float64)
    radius = float(np.max(np.linalg.norm(points - np.asarray(center, np.float64), axis=1))) if len(points) else 100.0
    return max(0.1, distance - radius - margin), distance + radius + margin


def parts(surface):
    """The part of each vertex (all PART_BRAIN for a surface without hemispheres)."""
    found = surface.get("parts")
    return np.asarray(found, np.int8) if found is not None else np.zeros(len(surface["points"]), np.int8)


def hasPart(surface, part):
    return bool(np.any(parts(surface) == part))


def piece(surface, part):
    """(surface of one part, its vertex indices in the whole): the triangles of that part, renumbered."""
    index = np.flatnonzero(parts(surface) == part)
    renumber = np.full(len(surface["points"]), -1, dtype=np.int64)
    renumber[index] = np.arange(len(index))
    triangles = np.asarray(surface["triangles"], np.int64)
    keep = np.all(renumber[triangles] >= 0, axis=1) if len(triangles) else np.zeros(0, bool)
    return ({"points": np.asarray(surface["points"])[index], "normals": np.asarray(surface["normals"])[index],
             "triangles": renumber[triangles[keep]]}, index)


def brainPart(surface):
    """(the brain's surface, its vertex indices): the whole surface, or its brain part when it has parts."""
    if surface.get("parts") is None:
        return surface, np.arange(len(surface["points"]))
    return piece(surface, PART_BRAIN)


def rasToIjk(points, ijkToRas):
    """(n, 3) RAS points as (n, 3) IJK coordinates of a grid (4 x 4 ijkToRas)."""
    inverse = np.linalg.inv(np.asarray(ijkToRas, dtype=np.float64))
    points = np.asarray(points, dtype=np.float64)
    return points @ inverse[:3, :3].T + inverse[:3, 3]


def project(array, ijkToRas, points, outwardNormals, mode="max", depthMm=DEPTH_MM, stepMm=STEP_MM, chunk=20000,
            mask=None):
    """Vertex values of a map (numpy array (k, j, i) on the grid ijkToRas): along each vertex's inward normal (the
    outward normals negated, unit length), the maximum (mode "max") or the value of the largest magnitude ("extreme",
    for signed score maps). Outside the grid the map is 0; NaN is taken as 0. mask (boolean, the brain, on the same
    grid): each ray stops where it first leaves the mask after entering it (the samples before it enters are kept;
    the first sample always counts)."""
    from scipy import ndimage
    array = np.nan_to_num(np.asarray(array, dtype=np.float32))
    if mask is not None:
        mask = np.asarray(mask, dtype=np.float32)
        if mask.shape != array.shape:
            raise ValueError(f"The mask {mask.shape} is not on the grid of the map {array.shape}.")
    points = np.asarray(points, dtype=np.float64)
    normals = np.asarray(outwardNormals, dtype=np.float64)
    lengths = np.linalg.norm(normals, axis=1)
    lengths[lengths == 0] = 1.0
    inward = -normals / lengths[:, None]
    distances = offsets(depthMm, stepMm)
    out = np.zeros(len(points), dtype=np.float32)
    for start in range(0, len(points), chunk):
        p = points[start:start + chunk]
        d = inward[start:start + chunk]
        samples = p[:, None, :] + distances[None, :, None] * d[:, None, :]         # (v, s, 3) RAS
        ijk = rasToIjk(samples.reshape(-1, 3), ijkToRas)
        coordinates = ijk[:, ::-1].T                                                 # (k, j, i) rows
        values = ndimage.map_coordinates(array, coordinates, order=1, mode="constant", cval=0.0)
        values = values.reshape(len(p), len(distances))
        if mask is not None:                                                        # stop where the ray leaves it
            inside = ndimage.map_coordinates(mask, coordinates, order=0, mode="constant", cval=0.0)
            inside = inside.reshape(len(p), len(distances)) > 0
            started = np.maximum.accumulate(inside, axis=1)
            valid = ~np.maximum.accumulate(started & ~inside, axis=1)
            valid[:, 0] = True
        if mode == "extreme":
            if mask is not None:
                values = np.where(valid, values, 0.0)
            index = np.argmax(np.abs(values), axis=1)
            out[start:start + chunk] = values[np.arange(len(p)), index]
        else:
            if mask is not None:
                values = np.where(valid, values, -np.inf)
            out[start:start + chunk] = values.max(axis=1)
    return out


def scores(patient, mean, sd, n=None, kind=nr.SCORE_Z, floorFraction=nr.SD_FLOOR_FRACTION):
    """Vertex z (or t) of the patient's projection against the normals' (negative: below the normals); the SD with
    the floor of the voxel scores (a fraction of its median)."""
    patient = np.asarray(patient, dtype=np.float32)
    mean = np.asarray(mean, dtype=np.float32)
    sd = nr.sdFloor(sd, np.ones(np.shape(sd), dtype=bool), floorFraction)
    factor = nr.tFactor(n) if kind == nr.SCORE_T else 1.0
    return ((patient - mean) / (sd * np.float32(factor))).astype(np.float32)


def signature(points, normals, depthMm, stepMm, extra=()):
    """Short hash of a surface (vertices, normals rounded to 0.01 mm) and the projection, the key of the cached
    projections."""
    digest = hashlib.sha1()
    digest.update(str(len(points)).encode())
    for values in (points, normals):
        digest.update(np.round(np.asarray(values, dtype=np.float64), 2).astype(np.float32).tobytes())
    digest.update(f"{depthMm:g}/{stepMm:g}".encode())
    for value in extra:
        digest.update(str(value).encode("utf-8"))
    return digest.hexdigest()[:16]


def write(path, surface, groups, meta):
    """The surface ({"points", "normals", "triangles"}) and the normals' vertex statistics ({group key: {"mean", "sd",
    "n", "name"}}) in an .npz; meta (depth, step, FWHM, map type, version, ...) as JSON inside."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    arrays = {"points": np.asarray(surface["points"], np.float32), "normals": np.asarray(surface["normals"], np.float32),
              "triangles": np.asarray(surface["triangles"], np.int32)}
    if surface.get("parts") is not None:
        arrays["parts"] = np.asarray(surface["parts"], np.int8)
    names = {}
    for key, group in groups.items():
        arrays[f"mean__{key}"] = np.asarray(group["mean"], np.float32)
        arrays[f"sd__{key}"] = np.asarray(group["sd"], np.float32)
        names[key] = {"name": group.get("name") or key, "n": int(group.get("n") or 0)}
    arrays["meta"] = np.frombuffer(json.dumps(dict(meta, groups=names)).encode("utf-8"), dtype=np.uint8)
    temporary = path + ".tmp.npz"
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def read(path):
    """(surface, groups, meta) of write(); None when the file is missing or not readable."""
    try:
        with np.load(path) as data:
            meta = json.loads(bytes(data["meta"]).decode("utf-8"))
            surface = {"points": data["points"].astype(np.float64), "normals": data["normals"].astype(np.float64),
                       "triangles": data["triangles"].astype(np.int64)}
            if "parts" in data.files:
                surface["parts"] = data["parts"].astype(np.int8)
            groups = {}
            for key, info in (meta.get("groups") or {}).items():
                groups[key] = {"mean": data[f"mean__{key}"], "sd": data[f"sd__{key}"], "n": info.get("n"),
                               "name": info.get("name")}
        return surface, groups, meta
    except (OSError, KeyError, ValueError):
        return None


def sspPath(folder, header):
    """The normals' projections of a database (stats/ssp.npz), None when missing."""
    relative = ((header or {}).get("statistics") or {}).get("ssp")
    path = os.path.join(folder, relative) if relative else ""
    return path if path and os.path.isfile(path) else None


def cachePath(folder, key):
    return os.path.join(folder, nr.CACHE_DIR, f"ssp_{key}.npz")


def sameSurface(a, b, tolerance=0.05):
    """Whether two surfaces have the same vertices (within tolerance mm)."""
    pa, pb = np.asarray(a["points"]), np.asarray(b["points"])
    return pa.shape == pb.shape and bool(np.allclose(pa, pb, atol=tolerance))


def surfaceColors(table, low, high, count=256):
    """count x 4 RGBA of the patient's projected values: the color map from low to high (opaque)."""
    table = np.asarray(table, dtype=float)
    index = np.clip(np.rint(np.linspace(0, len(table) - 1, count)).astype(int), 0, len(table) - 1)
    rgba = np.ones((count, 4))
    rgba[:, :3] = table[index]
    return rgba


def _frame(focal, position, viewUp):
    """(direction of projection, up, right) unit vectors of a camera."""
    direction = np.asarray(focal, np.float64) - np.asarray(position, np.float64)
    direction /= np.linalg.norm(direction) or 1.0
    up = np.asarray(viewUp, np.float64)
    up = up - np.dot(up, direction) * direction
    up /= np.linalg.norm(up) or 1.0
    return direction, up, np.cross(direction, up)


def panOffset(center, focal, position, viewUp):
    """(right, up) mm of a camera's focal point from the centre, on its screen: its pan."""
    _, up, right = _frame(focal, position, viewUp)
    offset = np.asarray(focal, np.float64) - np.asarray(center, np.float64)
    return float(np.dot(offset, right)), float(np.dot(offset, up))


def pannedCamera(center, pan, focal, position, viewUp):
    """(focal point, position) of a camera moved to the screen pan (right, up) of another view, its own direction,
    up and distance kept (the views share zoom and pan, not rotation)."""
    direction, up, right = _frame(focal, position, viewUp)
    focal = np.asarray(focal, np.float64)
    distance = float(np.linalg.norm(focal - np.asarray(position, np.float64)))
    depth = float(np.dot(focal - np.asarray(center, np.float64), direction))
    newFocal = np.asarray(center, np.float64) + pan[0] * right + pan[1] * up + depth * direction
    return newFocal, newFocal - distance * direction
