"""Static Neuro PET/SPECT Review: the patient compared with a normal database (numpy only, no Slicer).

    voxel scores    z = (patient - mean) / SD and the Crawford-Howell t = (patient - mean) / (SD * sqrt(1 + 1/n))
                    (df n - 1), both negative where the patient is below the normals (hypometabolism / hypoperfusion);
                    computed in a mask (the brain mask and the database mean above a fraction of its maximum) with
                    an SD floor (a fraction of the median SD in the mask), so the edge of the brain and the
                    near-constant voxels do not blow up
    regional scores the regions' means of the normals (their mean, SD, n per age group) computed with the mean / SD
                    maps by the Neuro PET Database Manager (stats/regions.json), or on the fly from the subjects'
                    maps for another segmentation (cached in the database's cache/ folder); for a predefined
                    database without them, an approximation from the mean / SD maps (the mean of the voxel SDs
                    over-estimates the SD of the regional mean, so that score is conservative)
    display         the colours of the score maps (a diverging colour map over a window symmetric about 0; the
                    side above the normals grey when only the side below is wanted) and the
                    Review layout (4 x 4: PET, score map, 3D-SSP views and the region table)
"""

import hashlib
import json
import math
import os

import numpy as np

from . import normaldb

SCORE_Z, SCORE_T = "z", "t"
SCORES = ((SCORE_T, "t (Crawford-Howell)"), (SCORE_Z, "z"))
DEFAULT_SCORE = SCORE_T
MASK_FRACTION = 0.3             # voxels scored: the database mean above 30 % of its maximum (and in the brain mask)
SD_FLOOR_FRACTION = 0.2         # SD floor: 20 % of the median SD in the mask
DEFAULT_MAXIMUM = 5.0           # score window: -5 ... +5, 0 in the middle
SCORE_COLORMAPS = ("ColdToHotRainbow", "DivergingBlueRed")     # Slicer's colour tables; the first is the default
GREY = (0.62, 0.62, 0.62)       # the side above the normals with "below the normals only"
OUTSIDE = -1.0e6                # score map voxels outside the scored mask (shown black: below the display threshold)
REGIONS_FILE = normaldb.STATS_DIR + "/regions.json"
CACHE_DIR = "cache"
DROPPED_SEGMENT_WORDS = ("optic chiasm",)       # left out of the review (overlay, surface, table)


# ---------------------------------------------------------------------------
# Voxel scores
# ---------------------------------------------------------------------------
def scoreMask(mean, brainMask=None, fraction=MASK_FRACTION):
    """Voxels scored: finite database mean above fraction x its maximum, inside brainMask when given (same shape)."""
    mean = np.asarray(mean, dtype=np.float32)
    finite = np.isfinite(mean)
    if not finite.any():
        return np.zeros(mean.shape, dtype=bool)
    top = float(np.max(mean[finite]))
    mask = finite & (mean > fraction * top) if top > 0 else np.zeros(mean.shape, dtype=bool)
    if brainMask is not None:
        brainMask = np.asarray(brainMask)
        if brainMask.shape != mean.shape:
            raise ValueError("The brain mask is not on the grid of the database maps.")
        mask &= brainMask > 0
    return mask


def sdFloor(sd, mask, fraction=SD_FLOOR_FRACTION):
    """The SD with a floor of fraction x its median inside the mask (and never 0)."""
    sd = np.nan_to_num(np.asarray(sd, dtype=np.float32))
    inside = sd[np.asarray(mask, dtype=bool)]
    inside = inside[inside > 0]
    floor = float(np.median(inside)) * fraction if inside.size else 0.0
    floor = max(floor, 1e-6)
    return np.maximum(sd, floor).astype(np.float32)


def tFactor(n):
    """sqrt(1 + 1/n) of the Crawford-Howell t (1 without a known n)."""
    try:
        n = int(n)
    except (TypeError, ValueError):
        return 1.0
    return math.sqrt(1.0 + 1.0 / n) if n > 0 else 1.0


def scoreMap(patient, mean, sd, mask, kind=DEFAULT_SCORE, n=None, floorFraction=SD_FLOOR_FRACTION):
    """Voxel z or t map (negative: the patient below the mean), 0 outside the mask."""
    patient = np.asarray(patient, dtype=np.float32)
    mean = np.asarray(mean, dtype=np.float32)
    if patient.shape != mean.shape or np.shape(sd) != mean.shape:
        raise ValueError(f"The patient's map {patient.shape} is not on the grid of the database maps {mean.shape}.")
    mask = np.asarray(mask, dtype=bool) & np.isfinite(patient)
    denominator = sdFloor(sd, mask, floorFraction)
    if kind == SCORE_T:
        denominator = denominator * np.float32(tFactor(n))
    out = np.zeros(mean.shape, dtype=np.float32)
    out[mask] = (patient[mask] - mean[mask]) / denominator[mask]
    return out


def regionScore(value, mean, sd, n=None, kind=DEFAULT_SCORE):
    """z or t of one regional value (negative: below the mean); None when it cannot be computed."""
    try:
        value, mean, sd = float(value), float(mean), float(sd)
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(value) and math.isfinite(mean) and math.isfinite(sd)) or sd <= 0:
        return None
    return (value - mean) / (sd * (tFactor(n) if kind == SCORE_T else 1.0))


# ---------------------------------------------------------------------------
# Age group
# ---------------------------------------------------------------------------
def statisticsGroups(header):
    """[(key, name, n)] of the computed mean / SD maps of a database: the master first."""
    groups = ((header or {}).get("statistics") or {}).get("groups") or {}
    out = []
    for key, entry in groups.items():
        if entry.get("mean") and entry.get("sd"):
            out.append((key, entry.get("name") or key, entry.get("n")))
    out.sort(key=lambda g: (g[0] != normaldb.MASTER, str(g[1]).lower()))
    return out


def patientGroup(header, age):
    """(group key, why) of the maps a patient is compared with: the age group of the patient's age when the
    database is age stratified and that group has maps, else the master."""
    available = {key for key, _, _ in statisticsGroups(header)}
    if not header or not header.get("ageStratified"):
        return normaldb.MASTER, "the database has no age groups"
    if age is None:
        return normaldb.MASTER, "the age is not known"
    name = normaldb.ageGroupOf(age, header.get("ageGroups") or [])
    if name is None:
        return normaldb.MASTER, f"no age group holds {age:g} years"
    key = normaldb.groupKey(name)
    if key not in available:
        return normaldb.MASTER, f"the age group '{name}' has no mean / SD maps"
    return key, f"age group '{name}'"


# ---------------------------------------------------------------------------
# Regions
# ---------------------------------------------------------------------------
def isDroppedSegment(name):
    text = str(name or "").strip().lower()
    return any(word in text for word in DROPPED_SEGMENT_WORDS)


def regionMeans(values, masks):
    """{name: mean of the finite values in the mask} (NaN when the mask holds none)."""
    values = np.asarray(values, dtype=np.float32)
    out = {}
    for name, mask in masks.items():
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != values.shape:
            raise ValueError(f"The region '{name}' is not on the grid of the map.")
        inside = values[mask]
        inside = inside[np.isfinite(inside)]
        out[name] = float(inside.mean()) if inside.size else float("nan")
    return out


def regionStatistics(subjectValues, groups):
    """{group key: {"name", "n", "regions": {region: [mean, SD, n]}}} from subjectValues {subject ID: {region:
    value}} and groups {key: {"name", "ids"}} (normaldb.statisticsGroups); groups with fewer than two subjects are
    left out, and per region the subjects with a value (SD with n - 1)."""
    out = {}
    for key, group in groups.items():
        ids = [i for i in group.get("ids") or [] if i in subjectValues]
        if len(ids) < 2:
            continue
        names = []
        for i in ids:
            for name in subjectValues[i]:
                if name not in names:
                    names.append(name)
        regions = {}
        for name in names:
            column = np.array([subjectValues[i].get(name, np.nan) for i in ids], dtype=np.float64)
            column = column[np.isfinite(column)]
            if column.size >= 2:
                regions[name] = [float(column.mean()), float(column.std(ddof=1)), int(column.size)]
        out[key] = {"name": group.get("name") or key, "n": len(ids), "regions": regions}
    return out


def unionMask(masks, shape):
    """All the regions together (the brain the 3D-SSP surface is made of)."""
    union = np.zeros(shape, dtype=bool)
    for mask in masks.values():
        union |= np.asarray(mask, dtype=bool)
    return union


def approximateRegions(meanMap, sdMap, masks):
    """{region: [mean of the mean map, mean of the SD map, None]}: a predefined database without regional
    statistics. The mean voxel SD is larger than the SD of the regional mean, so scores from it are conservative."""
    means = regionMeans(meanMap, masks)
    sds = regionMeans(sdMap, masks)
    return {name: [means[name], sds[name], None] for name in masks}


def masksSignature(masks, extra=()):
    """Short hash of the regions (names and voxels) and of extra values, the key of the cached regional
    statistics."""
    digest = hashlib.sha1()
    for name in sorted(masks):
        mask = np.asarray(masks[name], dtype=bool)
        digest.update(name.encode("utf-8"))
        digest.update(str(mask.shape).encode())
        digest.update(np.packbits(mask.ravel()).tobytes())
    for value in extra:
        digest.update(str(value).encode("utf-8"))
    return digest.hexdigest()[:16]


def regionsDocument(statistics, mapKey, version, settings=None, signature=""):
    return {"maps": mapKey, "version": version, "settings": settings or {}, "signature": signature,
            "groups": statistics}


def writeJson(path, document):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as stream:
        json.dump(document, stream, indent=1, ensure_ascii=False)
    os.replace(temporary, path)


def readJson(path):
    try:
        with open(path, encoding="utf-8") as stream:
            document = json.load(stream)
        return document if isinstance(document, dict) else None
    except (OSError, ValueError):
        return None


def regionsPath(folder, header):
    """The regional statistics of a database (stats/regions.json), None when missing."""
    relative = ((header or {}).get("statistics") or {}).get("regions")
    path = os.path.join(folder, relative) if relative else ""
    return path if path and os.path.isfile(path) else None


def cachePath(folder, signature, mapKey, version):
    return os.path.join(folder, CACHE_DIR, f"regions_{mapKey}_v{version}_{signature}.json")


def matchRegions(statistics, groupKey, names):
    """{name: [mean, SD, n]} of the group for the names found (case and spaces ignored), and the names missing."""
    regions = ((statistics or {}).get(groupKey) or {}).get("regions") or {}
    byClean = {" ".join(k.lower().split()): v for k, v in regions.items()}
    found, missing = {}, []
    for name in names:
        entry = byClean.get(" ".join(name.lower().split()))
        if entry is None:
            missing.append(name)
        else:
            found[name] = entry
    return found, missing


def tableRows(patientValues, patientSuv, reference, kind=DEFAULT_SCORE):
    """Rows of the region table: (name, SUVmean, value compared, DB mean, DB SD, n, score) per region of
    patientValues {name: value in the database's map type}; patientSuv {name: SUV mean}; reference {name: [mean,
    SD, n]} (None entries: no normal values). The scores use their own n (the region's) when given."""
    rows = []
    for name, value in patientValues.items():
        entry = reference.get(name)
        mean = sd = n = None
        if entry:
            mean, sd, n = entry[0], entry[1], entry[2] if len(entry) > 2 else None
        score = regionScore(value, mean, sd, n, kind) if entry else None
        rows.append((name, patientSuv.get(name), value, mean, sd, n, score))
    return rows


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------
def divergingColors(table, hypoOnly=False, count=256, grey=GREY):
    """count x 4 RGBA (0..1) of a score map shown over a window symmetric about 0 ([-maximum, maximum]): the colour
    table (n x 3 RGB 0..1) across it, 0 in the middle; with hypoOnly the side above 0 (above the normals) grey."""
    table = np.asarray(table, dtype=float)
    index = np.clip(np.rint(np.linspace(0, len(table) - 1, count)).astype(int), 0, len(table) - 1)
    rgba = np.ones((count, 4), dtype=float)
    rgba[:, :3] = table[index]
    if hypoOnly:
        rgba[np.linspace(-1.0, 1.0, count) > 0, :3] = grey
    return rgba


def maskedScores(scores, mask, factor=1.0):
    """The score map divided by factor (t from z) with OUTSIDE where the mask is not (shown black)."""
    out = np.full(np.shape(scores), OUTSIDE, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    out[mask] = np.asarray(scores, dtype=np.float32)[mask] / np.float32(factor)
    return out


# Layout: rows 1-2 slice views (PET with the region outlines; the score map), rows 3-4 the 3D-SSP views, the
# region table on the right
LAYOUT_ID = 7810
ORIENTATIONS = (("Axial", "#F34A33"), ("Coronal", "#6EB04B"), ("Sagittal", "#EDD54C"))
PET_ROW, SCORE_ROW = "PET", "Score"
VIEW_PREFIX = "NeuroRev"
VIEW_GROUP = 93
SSP_VIEWS = (("Top", "Superior"), ("Ant", "Anterior"), ("Left", "Left"),
             ("Bottom", "Inferior"), ("Post", "Posterior"), ("Right", "Right"))
TABLE_VIEW = VIEW_PREFIX + "Table"


def sliceViewName(row, orientation):
    return f"{VIEW_PREFIX}{row}{orientation}"


def sspViewName(key):
    return f"{VIEW_PREFIX}SSP{key}"


def layoutXml():
    def sliceView(row, orientation, color, label):
        return (f'<item><view class="vtkMRMLSliceNode" singletontag="{sliceViewName(row, orientation)}">'
                f'<property name="orientation" action="default">{orientation}</property>'
                f'<property name="viewlabel" action="default">{label}</property>'
                f'<property name="viewcolor" action="default">{color}</property>'
                f'<property name="viewgroup" action="default">{VIEW_GROUP}</property>'
                '</view></item>')

    def threeD(key, title):
        return (f'<item><view class="vtkMRMLViewNode" singletontag="{sspViewName(key)}">'
                f'<property name="viewlabel" action="default">{title}</property>'
                f'<property name="viewgroup" action="default">{VIEW_GROUP + 1}</property>'
                '</view></item>')

    rows = []
    for row, prefix in ((PET_ROW, ""), (SCORE_ROW, "S")):
        rows.append('<item><layout type="horizontal" split="true">' + "".join(
            sliceView(row, o, c, prefix + o[0]) for o, c in ORIENTATIONS) + '</layout></item>')
    for start in (0, 3):
        rows.append('<item><layout type="horizontal" split="true">' + "".join(
            threeD(k, t) for k, t in SSP_VIEWS[start:start + 3]) + '</layout></item>')
    left = f'<layout type="vertical" split="true">{"".join(rows)}</layout>'
    table = (f'<view class="vtkMRMLTableViewNode" singletontag="{TABLE_VIEW}">'
             '<property name="viewlabel" action="default">R</property></view>')
    return (f'<layout type="horizontal" split="true"><item splitSize="750">{left}</item>'
            f'<item splitSize="250">{table}</item></layout>')
