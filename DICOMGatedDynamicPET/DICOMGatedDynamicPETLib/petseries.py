"""PET series from DICOM: static, ECG gated and dynamic. Recognition, frame/slice ordering, geometry, SUV,
tracer, decay correction, frame timing and ECG gating (pydicom and numpy only, no Slicer dependency; the
DICOM plugin is DICOMGatedDynamicPET.py).

Kind of series (Series Type (0054,1000) first value; only when it is missing or unknown the frame counts of
the header, gates first):
    GATED     GATED\\IMAGE, or Number of Time Slots (0054,0071) > 1: one frame per cardiac gate
    DYNAMIC   DYNAMIC\\IMAGE, or Number of Time Slices (0054,0101) > 1: one frame per time frame
    STATIC    anything else (STATIC, WHOLE BODY): one volume

A gated or dynamic series is a stack of single slice images, one per slice and frame. Image Index
(0054,1330) orders them:
    ImageIndex = (frame - 1) x Number of Slices (0054,0081) + slice
(the PET image IOD; more than one R-R interval would add a level, which is folded into the gates).
Without a usable Image Index the images are grouped by Trigger Time (0018,1060) for gated series, by
Frame Reference Time (0054,1300) then acquisition date/time for dynamic series, and last by repeated slice
positions in instance order. Slices are sorted along the normal of the image plane.

Pixel values are the stored value x Rescale Slope + Intercept of each image (slopes differ per image).
SUVbw follows the QIBA vendor-neutral pseudo-code (also what Slicer's PET SUV plugin computes):
    SUVbw = activity concentration (Bq/ml) x body weight (g) / injected dose decayed (Bq)
    SUVlbm (SUL) uses the lean body mass (g, Janmahasatian 2005, as PERCIST), SUVbsa the body surface area
    (cm2, Du Bois) instead of the body weight; both need the patient height, SUVlbm also the sex.
    Units BQML   Decay Correction START -> dose decayed to the series date/time if not later than the first
                 acquisition, else the GE scan start (0009,100d), else the first acquisition date/time
                 (one scale for the whole series);
                 ADMIN -> no decay;
                 NONE  -> images are not decay corrected: the dose is decayed to each frame's own
                 start (one scale per frame), the same frame starts as the frame timing table
                 (frameStarts: acquisition time, else the end of the previous frame).
    Units CNTS   Philips: SUV scale factor (7053,1000), Bq/ml from the activity concentration scale
                 factor (7053,1009).
    Units GML    already SUVbw.
The dose, its time and the half life come from the Radiopharmaceutical Information Sequence (0054,0016);
when the dose is missing GE's own tracer activity record (GEMS_PETD_01 private group) is used.
Each gate of a GE gated PET is a full activity concentration image (normalised to its own frame
duration), so the gates are on the scale of the static image, and their mean (not sum) is the static.
"""

import datetime
import json
import math
import os

import numpy as np

STATIC, GATED, DYNAMIC = "STATIC", "GATED", "DYNAMIC"
# Node attribute prefix of each kind. GatedPET.* is also read by the Tarvos (Myocardial Perfusion) extension.
ATTRIBUTE_PREFIXES = {STATIC: "StaticPET.", GATED: "GatedPET.", DYNAMIC: "DynamicPET."}
UNITS_SUV = "SUVbw"
UNITS_SUL = "SUVlbm"
UNITS_SUVBSA = "SUVbsa"
UNITS_ACTIVITY = "Bq/ml"
# SUV normalisation: body weight, lean body mass (Janmahasatian), body surface area (Du Bois)
SUV_METHODS = {"bw": UNITS_SUV, "lbm": UNITS_SUL, "bsa": UNITS_SUVBSA}
SUV_METHOD_TITLES = {"bw": "body weight (SUVbw)", "lbm": "lean body mass (SUVlbm, SUL, Janmahasatian)",
                     "bsa": "body surface area (SUVbsa, Du Bois)"}

GE_PET_CREATOR = "GEMS_PETD_01"
PHILIPS_CREATOR = "Philips PET Private Group"


# ---------------------------------------------------------------------------
# Small DICOM helpers
# ---------------------------------------------------------------------------
def _value(ds, keyword, default=None):
    value = getattr(ds, keyword, None)
    if value is None or value == "":
        return default
    return value


def _float(value, default=None):
    try:
        if value is None or value == "":
            return default
        if isinstance(value, (list, tuple)) or (hasattr(value, "__len__") and not isinstance(value, (str, bytes))):
            value = value[0]
        number = float(value)
        return number if math.isfinite(number) else default
    except (TypeError, ValueError, IndexError):
        return default


def _text(ds, keyword):
    """Text of an element; multi-valued elements joined with '\\'."""
    value = _value(ds, keyword)
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    try:
        return "\\".join(str(v).strip() for v in value)
    except TypeError:
        return str(value).strip()


def _textList(value):
    if value is None:
        return []
    if isinstance(value, str):
        return [v.strip().upper() for v in value.split("\\")]
    try:
        return [str(v).strip().upper() for v in value]
    except TypeError:
        return [str(value).strip().upper()]


def privateValue(ds, group, creator, element):
    """Value of a private element (group, xx element) of a creator, or None. Finds the creator's block
    itself, so it works whichever block (0x10, 0x11, ...) the scanner put it in."""
    try:
        for block in range(0x10, 0x100):
            tag = (group << 16) | block
            if tag not in ds:
                continue
            if str(ds[tag].value).strip() == creator:
                full = (group << 16) | (block << 8) | element
                return ds[full].value if full in ds else None
    except Exception:
        pass
    return None


def parseDateTime(dateText, timeText=None):
    """datetime from DICOM DA + TM, or a DT when timeText is None; None when unreadable.
    UTC offsets are dropped (all times of one study are in the same zone)."""
    text = str(dateText or "").strip()
    if timeText is not None:
        if not text or not str(timeText).strip():
            return None
        text = text[:8] + str(timeText).strip()
    if not text:
        return None
    for sign in ("+", "-"):
        cut = text.find(sign, 8)
        if cut > 0:
            text = text[:cut]
    date, _, rest = text.partition(".")
    fraction = rest[:6].ljust(6, "0") if rest else "000000"
    date = date.ljust(14, "0")[:14]
    try:
        stamp = datetime.datetime.strptime(date, "%Y%m%d%H%M%S")
        return stamp + datetime.timedelta(microseconds=int(fraction or 0))
    except ValueError:
        return None


def _acquisitionStart(ds):
    """Acquisition Date + Time of an image, else its Acquisition DateTime; None without an acquisition time
    (a date alone would read as midnight)."""
    time = _value(ds, "AcquisitionTime")
    if time is not None and str(time).strip():
        return parseDateTime(_value(ds, "AcquisitionDate") or _value(ds, "SeriesDate"), time)
    stamp = str(_value(ds, "AcquisitionDateTime", "") or "").strip()
    return parseDateTime(stamp) if len(stamp) > 8 else None


def _seconds(later, earlier):
    if later is None or earlier is None:
        return None
    return (later - earlier).total_seconds()


def _round(value, digits=1):
    number = _float(value)
    if number is None:
        return None
    number = round(number, digits)
    return int(number) if float(number).is_integer() else number


# ---------------------------------------------------------------------------
# Recognition
# ---------------------------------------------------------------------------
def seriesKind(modality, seriesType, timeSlots=None, timeSlices=None):
    """STATIC, GATED or DYNAMIC for a PET series from its Modality, Series Type (text 'GATED\\IMAGE' or list),
    Number of Time Slots and Number of Time Slices as stored in the header (strings are fine); None when
    the series is not PET."""
    if str(modality or "").strip().upper() != "PT":
        return None
    types = _textList(seriesType)
    first = types[0] if types else ""
    if first in (GATED, DYNAMIC):   # an explicit Series Type wins over the frame counts
        return first
    if first in ("STATIC", "WHOLE BODY"):
        return STATIC
    if (_float(timeSlots, 0) or 0) > 1:
        return GATED
    if (_float(timeSlices, 0) or 0) > 1:
        return DYNAMIC
    return STATIC


def datasetKind(ds):
    return seriesKind(_value(ds, "Modality"), _value(ds, "SeriesType"), _value(ds, "NumberOfTimeSlots"),
                      _value(ds, "NumberOfTimeSlices"))


def expectedFrames(ds, kind):
    """Number of gates / time frames the header announces (0 when it does not say)."""
    keyword = {GATED: "NumberOfTimeSlots", DYNAMIC: "NumberOfTimeSlices"}.get(kind)
    return int(_float(_value(ds, keyword), 0) or 0) if keyword else 1


# ---------------------------------------------------------------------------
# Ordering
# ---------------------------------------------------------------------------
def sliceNormal(ds):
    orientation = [float(v) for v in ds.ImageOrientationPatient]
    row, column = np.array(orientation[:3]), np.array(orientation[3:])
    normal = np.cross(row, column)
    return normal / (np.linalg.norm(normal) or 1.0)


def _position(ds):
    return np.array([float(v) for v in ds.ImagePositionPatient])


def _depth(ds, normal):
    return float(np.dot(_position(ds), normal))


def sortSlices(datasets):
    normal = sliceNormal(datasets[0])
    return sorted(datasets, key=lambda ds: _depth(ds, normal))


def groupFrames(datasets, kind=None):
    """Split the images of a series into frames (gates or time frames) of slices: (frames, method), where
    frames is a list (one per frame, in frame order) of lists of datasets sorted along the slice normal.
    A static series gives one frame. Raises ValueError when the images cannot be split into frames of the
    same slices (arrangeFrames tolerates missing frames and slices)."""
    if not datasets:
        raise ValueError("No images")
    kind = kind or datasetKind(datasets[0]) or STATIC
    normal = sliceNormal(datasets[0])
    if kind == STATIC:
        frame = sortSlices(datasets)
        depths = [round(_depth(ds, normal), 2) for ds in frame]
        if len(set(depths)) != len(depths):
            raise ValueError("Two images at the same slice position in a static series")
        return [frame], "slice position"
    expected = expectedFrames(datasets[0], kind)

    problems = []
    for attempt in _groupingAttempts(datasets, kind, normal):
        result = attempt()
        if result is None:
            continue
        frames, method, _ = result
        problem = _checkFrames(frames, normal, expected)
        if problem is None:
            frames = [sorted(frame, key=lambda ds: _depth(ds, normal)) for frame in frames]
            return frames, method
        problems.append(f"{method}: {problem}")
    noun = "gates" if kind == GATED else "time frames"
    raise ValueError(f"Cannot split the images into {noun}. " + "; ".join(problems or ["only one frame found"]))


def _groupingAttempts(datasets, kind, normal, tolerant=False):
    """Ways to split a gated / dynamic series into frames, most reliable first. Each attempt returns None
    or (frames, method, frame numbers (1-based, None when only the order is known))."""

    def byKey(keyOf, method, numberOf=None):
        groups = {}
        for ds in datasets:
            groups.setdefault(keyOf(ds), []).append(ds)
        if len(groups) < 2:
            return None
        keys = sorted(groups)
        return [groups[k] for k in keys], method, [numberOf(k) for k in keys] if numberOf else None

    attempts = []
    nSlices = int(_float(_value(datasets[0], "NumberOfSlices"), 0) or 0)
    if nSlices and all(_float(_value(ds, "ImageIndex")) for ds in datasets):
        attempts.append(lambda: byKey(lambda ds: (int(_float(ds.ImageIndex)) - 1) // nSlices,
                                      "Image Index (0054,1330)", lambda k: k + 1))
    if kind == GATED and all(_float(_value(ds, "TriggerTime")) is not None for ds in datasets):
        attempts.append(lambda: byKey(lambda ds: round(_float(ds.TriggerTime), 1), "Trigger Time (0018,1060)"))
    if kind == DYNAMIC:
        if all(_float(_value(ds, "FrameReferenceTime")) is not None for ds in datasets):
            attempts.append(lambda: byKey(lambda ds: round(_float(ds.FrameReferenceTime), 1),
                                          "Frame Reference Time (0054,1300)"))
        if all(_acquisitionStart(ds) is not None for ds in datasets):
            attempts.append(lambda: byKey(_acquisitionStart, "acquisition date/time"))
    attempts.append(lambda: _byRepeatedPositions(datasets, normal, tolerant))
    return attempts


def groupGates(datasets):
    """Gates of a gated series (see groupFrames)."""
    return groupFrames(datasets, GATED)


def _byRepeatedPositions(datasets, normal, tolerant=False):
    """Images at the same slice position, taken in instance (then image index) order, are the frames.
    Tolerant: positions with fewer images than others are missing them in the last frames."""
    order = sorted(datasets, key=lambda ds: (_float(_value(ds, "InstanceNumber"), 0), _float(_value(ds, "ImageIndex"), 0)))
    positions = {}
    for ds in order:
        positions.setdefault(round(_depth(ds, normal), 2), []).append(ds)
    counts = {len(v) for v in positions.values()}
    nFrames = max(counts)
    if nFrames < 2 or (len(counts) != 1 and not tolerant):
        return None
    frames = [[positions[p][f] for p in positions if f < len(positions[p])] for f in range(nFrames)]
    return frames, "repeated slice positions", None


def _checkFrames(frames, normal, expected):
    sizes = {len(f) for f in frames}
    if len(sizes) != 1:
        return f"frames have different numbers of slices ({sorted(sizes)})"
    if expected > 1 and len(frames) != expected:
        return f"{len(frames)} frames found, the header says {expected}"
    reference = sorted(round(_depth(ds, normal), 2) for ds in frames[0])
    if len(set(reference)) != len(reference):
        return "a frame has two images at the same slice position"
    for frame in frames[1:]:
        positions = sorted(round(_depth(ds, normal), 2) for ds in frame)
        if np.max(np.abs(np.array(positions) - np.array(reference))) > 0.05:
            return "frames do not cover the same slice positions"
    return None


def _listText(numbers, limit=12):
    numbers = list(numbers)
    text = ", ".join(str(n) for n in numbers[:limit])
    return text + (f" and {len(numbers) - limit} more" if len(numbers) > limit else "")


def sliceGrid(depths, nominal=None):
    """Regular slice positions (mm along the normal) covering the given depths: (grid, problem). The step
    is the most common distance between neighbouring slices, so missing slices leave empty places in the
    grid. problem is None, or a text when the depths do not sit on a regular grid (then the grid is the
    depths themselves)."""
    depths = sorted(set(round(d, 2) for d in depths))
    if len(depths) < 2:
        return depths, None
    steps = np.diff(depths)
    step = float(np.median(steps[steps <= np.percentile(steps, 50) + 1e-6]))
    if nominal and nominal > 0 and abs(nominal - step) <= max(0.1, 0.01 * nominal):
        step = float(nominal)
    tolerance = max(0.1, 0.01 * step)
    places = [(d - depths[0]) / step for d in depths]
    if step <= 0 or any(abs(p - round(p)) * step > tolerance for p in places):
        return depths, f"uneven slice spacing ({steps.min():.3f} to {steps.max():.3f} mm)"
    count = int(round(places[-1])) + 1
    return [depths[0] + k * step for k in range(count)], None


def arrangeFrames(datasets, kind=None):
    """groupFrames that tolerates missing frames and slices: a dict with
        frames    list (frame order) of lists of the datasets present, sorted along the slice normal
        slots     per frame, a list as long as grid with the dataset at each slice position, or None
        grid      slice positions (mm along the normal) shared by all frames
        numbers   frame (gate) number of each frame, 1-based, as the header numbers them
        method    how the images were split into frames
        warnings  texts for the user (missing frames, missing slices, uneven spacing, duplicates)
    Raises ValueError only when the images cannot be split into frames at all."""
    if not datasets:
        raise ValueError("No images")
    kind = kind or datasetKind(datasets[0]) or STATIC
    normal = sliceNormal(datasets[0])
    noun, word = ("gates", "gate") if kind == GATED else ("time frames", "frame")
    warnings = []
    expected = expectedFrames(datasets[0], kind)

    def withoutDuplicates(frame, label):
        seen, kept = set(), []
        for ds in sorted(frame, key=lambda d: (_depth(d, normal), _float(_value(d, "InstanceNumber"), 0))):
            depth = round(_depth(ds, normal), 2)
            if depth in seen:
                continue
            seen.add(depth)
            kept.append(ds)
        if len(kept) != len(frame):
            warnings.append(f"{label}: {len(frame) - len(kept)} image(s) at an already used slice position "
                            "were skipped")
        return kept

    if kind == STATIC:
        frames, method, numbers = [withoutDuplicates(datasets, "static series")], "slice position", [1]
    else:
        best = None
        for attempt in _groupingAttempts(datasets, kind, normal, tolerant=True):
            result = attempt()
            if result is None:
                continue
            frames, method, numbers = result
            duplicates = sum(len(f) - len({round(_depth(ds, normal), 2) for ds in f}) for f in frames)
            if duplicates == 0:
                best = result
                break
            best = best or result
        if best is None:
            raise ValueError(f"Cannot split the images into {noun}: only one frame found")
        frames, method, numbers = best
        missing = sorted(set(range(1, max(expected, numbers[-1]) + 1)) - set(numbers)) if numbers else []
        numbers = numbers or list(range(1, len(frames) + 1))
        frames = [withoutDuplicates(f, f"{word} {n}") for f, n in zip(frames, numbers)]
        if missing:
            warnings.append(f"{len(missing)} of {max(expected, numbers[-1])} {noun} missing ({word} "
                            f"{_listText(missing)}); loaded the {len(frames)} present")
        elif expected > 1 and len(frames) != expected:
            warnings.append(f"{len(frames)} {noun} found, the header says {expected}")

    nominal = _float(_value(datasets[0], "SpacingBetweenSlices"))
    grid, problem = sliceGrid([_depth(ds, normal) for f in frames for ds in f], nominal)
    if problem:
        warnings.append(problem.capitalize() + "; the volume is built with the median spacing, distances "
                        "along the slices may be off")
    slots, gaps = [], {}
    for frame, number in zip(frames, numbers):
        row = [None] * len(grid)
        for ds in frame:
            k = int(np.argmin([abs(_depth(ds, normal) - g) for g in grid]))
            row[k] = ds
        slots.append(row)
        empty = [k + 1 for k, ds in enumerate(row) if ds is None]
        if empty:
            gaps[number] = empty
    if gaps:
        where = "the series" if kind == STATIC else \
            (f"{len(gaps)} of {len(frames)} {noun}" if len(gaps) < len(frames) else f"every {word}")
        examples = "; ".join(f"{word} {n}: slice {_listText(e, 6)}" if kind != STATIC else f"slice {_listText(e, 6)}"
                             for n, e in list(gaps.items())[:4])
        more = f"; {len(gaps) - 4} more {noun}" if len(gaps) > 4 else ""
        inner = sum(int(interpolatedSlices(row).sum()) for row in slots)
        total = sum(len(e) for e in gaps.values())
        if inner == total:
            fate = "interpolated from the slices before and after"
        elif inner:
            fate = (f"{inner} between present slices interpolated from the slices before and after, "
                    f"{total - inner} at the ends of the stack left empty (0)")
        else:
            fate = "they are at the ends of the stack and left empty (0)"
        warnings.append(f"{total} slice(s) missing in {where} ({examples}{more}); {fate}")
    announced = int(_float(_value(datasets[0], "NumberOfSlices"), 0) or 0)
    if announced > len(grid):
        warnings.append(f"{len(grid)} slice positions found, the header says {announced} (slices missing at "
                        "the ends of the stack cannot be placed)")
    frames = [sorted(f, key=lambda ds: _depth(ds, normal)) for f in frames]
    return {"frames": frames, "slots": slots, "grid": grid, "numbers": numbers, "method": method,
            "warnings": warnings}


def missingImagesWarning(header, kind, nFiles):
    """Quick check on the first header, for the DICOM browser: a text when the series has fewer images than
    the header announces (frames x slices), else None."""
    frames = expectedFrames(header, kind) if kind != STATIC else 1
    slices = int(_float(_value(header, "NumberOfSlices"), 0) or 0)
    if frames and slices and nFiles < frames * slices:
        return (f"{frames * slices - nFiles} of {frames * slices} images missing; the series will be loaded "
                "with empty slices or fewer frames")
    return None


# ---------------------------------------------------------------------------
# Geometry and pixels
# ---------------------------------------------------------------------------
def geometry(sortedSlices):
    """(origin LPS, spacing (column, row, slice) mm, direction 3x3 with column/row/normal unit vectors as
    columns) of a frame. Raises ValueError for uneven slice spacing (more than 1 % or 0.1 mm)."""
    first = sortedSlices[0]
    orientation = [float(v) for v in first.ImageOrientationPatient]
    row, column = np.array(orientation[:3]), np.array(orientation[3:])
    normal = sliceNormal(first)
    rowSpacing, columnSpacing = (float(v) for v in first.PixelSpacing)   # between rows, between columns
    depths = np.array([_depth(ds, normal) for ds in sortedSlices])
    if len(depths) > 1:
        steps = np.diff(depths)
        sliceSpacing = float(np.median(steps))
        if sliceSpacing <= 0 or np.max(np.abs(steps - sliceSpacing)) > max(0.1, 0.01 * sliceSpacing):
            raise ValueError(f"Uneven slice spacing ({steps.min():.3f} to {steps.max():.3f} mm)")
    else:
        sliceSpacing = _float(_value(first, "SliceThickness"), 1.0)
    direction = np.column_stack([row, column, normal])
    return _position(first), (columnSpacing, rowSpacing, sliceSpacing), direction


def gridGeometry(slots, grid):
    """geometry of a frame given by its slots on the shared slice grid (some slots may be None): the origin
    is the first grid position, the slice spacing the grid step."""
    reference = next(ds for ds in slots if ds is not None)
    orientation = [float(v) for v in reference.ImageOrientationPatient]
    row, column = np.array(orientation[:3]), np.array(orientation[3:])
    normal = sliceNormal(reference)
    rowSpacing, columnSpacing = (float(v) for v in reference.PixelSpacing)
    if len(grid) > 1:
        sliceSpacing = float(np.median(np.diff(grid)))
    else:
        sliceSpacing = _float(_value(reference, "SliceThickness"), 1.0)
    position = _position(reference)
    origin = position + (grid[0] - float(np.dot(position, normal))) * normal
    return origin, (columnSpacing, rowSpacing, sliceSpacing), np.column_stack([row, column, normal])


def lpsToRasMatrix(origin, spacing, direction):
    """4x4 IJK to RAS matrix (list of lists) for a volume whose voxel (i, j, k) is column i, row j, slice k."""
    flip = np.diag([-1.0, -1.0, 1.0])
    matrix = np.eye(4)
    matrix[:3, :3] = flip @ direction @ np.diag(spacing)
    matrix[:3, 3] = flip @ np.asarray(origin, dtype=float)
    return matrix.tolist()


def rescaled(ds):
    """Pixel values of one image after its own Rescale Slope / Intercept (float64, rows x columns)."""
    pixels = ds.pixel_array.astype(np.float64)
    slope = _float(_value(ds, "RescaleSlope"), 1.0)
    intercept = _float(_value(ds, "RescaleIntercept"), 0.0)
    return pixels * slope + intercept


def frameArray(sortedSlices, scale=1.0, interpolate=True):
    """Frame volume as a float32 array (slices, rows, columns) times scale. Missing slices (None, see the
    slots of arrangeFrames) between two present slices are linearly interpolated along the slice axis when
    interpolate is set (see interpolatedSlices); the others are left 0."""
    reference = next(ds for ds in sortedSlices if ds is not None)
    shape = (int(reference.Rows), int(reference.Columns))
    planes = [rescaled(ds) if ds is not None else np.zeros(shape) for ds in sortedSlices]
    array = np.stack(planes).astype(np.float64)
    if interpolate:
        array = fillSliceGaps(array, presentSlices(sortedSlices))
    return (array * scale).astype(np.float32)


def presentSlices(slots):
    """Boolean array (slices,) of the slots that hold an image."""
    return np.array([ds is not None for ds in slots])


def interpolatedSlices(slots):
    """Boolean array (slices,) of the missing slots that have a present slice on both sides, i.e. the ones
    frameArray interpolates."""
    present = presentSlices(slots)
    if not present.any():
        return np.zeros_like(present)
    indices = np.flatnonzero(present)
    inside = np.zeros_like(present)
    inside[indices[0]:indices[-1] + 1] = True
    return inside & ~present


def fillSliceGaps(array, present):
    """Copy of array (slices, rows, columns) with the missing slices between present ones linearly
    interpolated from the nearest present slice below and above, weighted by distance on the slice grid.
    This is what hardening Slicer's acquisition geometry regularization transform does for a gap in a
    scalar volume (linear resampling across it). Missing slices at the ends of the stack stay as they are."""
    array = np.array(array, dtype=np.float64, copy=True)
    indices = np.flatnonzero(present)
    for below, above in zip(indices[:-1], indices[1:]):
        for k in range(below + 1, above):
            w = (k - below) / float(above - below)
            array[k] = (1.0 - w) * array[below] + w * array[above]
    return array


gateArray = frameArray


class SliceWiseMean:
    """Weighted mean of frame arrays, slice by slice. A slice comes from the frames that really have it; only
    where no frame has it are the interpolated slices of the frames used (and where none has those either,
    the result is 0)."""

    def __init__(self):
        self.total = self.weights = self.filledTotal = self.filledWeights = None

    def add(self, array, slots, weight=1.0):
        array = np.asarray(array, dtype=np.float64)
        present = presentSlices(slots).astype(np.float64) * weight
        filled = (presentSlices(slots) | interpolatedSlices(slots)).astype(np.float64) * weight
        parts = (array * present[:, None, None], present, array * filled[:, None, None], filled)
        if self.total is None:
            self.total, self.weights, self.filledTotal, self.filledWeights = parts
        else:
            self.total += parts[0]
            self.weights += parts[1]
            self.filledTotal += parts[2]
            self.filledWeights += parts[3]

    def mean(self):
        real = self.total / np.maximum(self.weights, 1e-12)[:, None, None]
        filled = self.filledTotal / np.maximum(self.filledWeights, 1e-12)[:, None, None]
        return np.where((self.weights > 0)[:, None, None], real, filled).astype(np.float32)


def durationWeightedMean(arrays, durations):
    """Activity concentration over the whole acquisition from time frames: sum(C_i x d_i) / sum(d_i)
    (a plain mean when the durations are unknown)."""
    weights = [d if d and d > 0 else None for d in durations]
    if any(w is None for w in weights):
        weights = [1.0] * len(arrays)
    total = None
    for array, weight in zip(arrays, weights):
        part = np.asarray(array, dtype=np.float64) * weight
        total = part if total is None else total + part
    return total / float(sum(weights))


# ---------------------------------------------------------------------------
# Tracer, acquisition and corrections
# ---------------------------------------------------------------------------
def _radiopharmaceutical(ds):
    sequence = _value(ds, "RadiopharmaceuticalInformationSequence")
    return sequence[0] if sequence else None


def _codeMeaning(item, keyword):
    sequence = _value(item, keyword)
    if not sequence:
        return ""
    return str(_value(sequence[0], "CodeMeaning", "") or _value(sequence[0], "CodeValue", "") or "").strip()


def radiopharmaceuticalInfo(ds):
    """Tracer of a PET series from its Radiopharmaceutical Information Sequence (0054,0016):
    {"Radiopharmaceutical": Radiopharmaceutical (0018,0031) text, else its code meaning,
     "Radionuclide": radionuclide code meaning}. Empty fields are left out."""
    radio = _radiopharmaceutical(ds)
    if radio is None:
        return {}
    name = str(_value(radio, "Radiopharmaceutical", "") or "").strip() or \
        _codeMeaning(radio, "RadiopharmaceuticalCodeSequence")
    nuclide = _codeMeaning(radio, "RadionuclideCodeSequence")
    return {key: value for key, value in (("Radiopharmaceutical", name), ("Radionuclide", nuclide)) if value}


def acquisitionInfo(datasets):
    """Series-level fields worth keeping with the volume, as {key: text}; empty ones are left out:
    study / series / scan start date-times, scanner, reconstruction and the corrections applied
    (Corrected Image (0028,0051): DECY decay, ATTN attenuation, SCAT scatter, DTIM dead time, RAN randoms,
    NORM normalisation), decay correction and its reference date-time, patient weight / height / sex."""
    first = datasets[0]
    start, startSource = scanStart(datasets)
    study = parseDateTime(_value(first, "StudyDate"), _value(first, "StudyTime"))
    series = parseDateTime(_value(first, "SeriesDate"), _value(first, "SeriesTime"))
    info = {
        "StudyStart": study.isoformat() if study else "",
        "SeriesStart": series.isoformat() if series else "",
        "ScanStart": start.isoformat() if start else "",
        "ScanStartSource": startSource if start else "",
        "Manufacturer": _text(first, "Manufacturer"),
        "ScannerModel": _text(first, "ManufacturerModelName"),
        "ReconstructionMethod": _text(first, "ReconstructionMethod"),
        "ConvolutionKernel": _text(first, "ConvolutionKernel"),
        "CorrectedImage": _text(first, "CorrectedImage"),
        "AttenuationCorrectionMethod": _text(first, "AttenuationCorrectionMethod"),
        "ScatterCorrectionMethod": _text(first, "ScatterCorrectionMethod"),
        "RandomsCorrectionMethod": _text(first, "RandomsCorrectionMethod"),
        "DecayCorrection": _text(first, "DecayCorrection").upper(),
        "DecayCorrectionDateTime": _isoText(_value(first, "DecayCorrectionDateTime")),
        "DICOMUnits": _text(first, "Units").upper(),
        "PatientWeightKg": _numberText(_value(first, "PatientWeight")),
        "PatientHeightCm": _numberText(_float(_value(first, "PatientSize")) * 100.0
                                       if _float(_value(first, "PatientSize")) else None),
        "PatientSex": _text(first, "PatientSex"),
        "SeriesDescription": _text(first, "SeriesDescription"),
        "SeriesInstanceUID": _text(first, "SeriesInstanceUID"),
    }
    radio = _radiopharmaceutical(first)
    if radio is not None:
        info["RadionuclideHalfLifeS"] = _numberText(_value(radio, "RadionuclideHalfLife"))
        info["RadionuclidePositronFraction"] = _numberText(_value(radio, "RadionuclidePositronFraction"))
        info["InjectedDoseMBq"] = _numberText(_float(_value(radio, "RadionuclideTotalDose")) / 1e6
                                              if _float(_value(radio, "RadionuclideTotalDose")) else None)
        injection = _injectionTime(first, radio)
        info["InjectionTime"] = injection.isoformat() if injection else ""
        info["InjectionToScanStartS"] = _numberText(_seconds(start, injection))
    info.update(radiopharmaceuticalInfo(first))
    return {k: v for k, v in info.items() if v not in (None, "")}


def _isoText(value):
    stamp = parseDateTime(value) if value else None
    return stamp.isoformat() if stamp else ""


def _numberText(value, digits=6):
    number = _float(value)
    if number is None:
        return ""
    return f"{number:.{digits}g}"


def _injectionTime(first, radio):
    if radio is None:
        return None
    injection = parseDateTime(_value(radio, "RadiopharmaceuticalStartDateTime"))
    if injection is None:
        injection = parseDateTime(_value(first, "SeriesDate") or _value(first, "StudyDate"),
                                  _value(radio, "RadiopharmaceuticalStartTime"))
    return injection


def scanStart(datasets):
    """Start of the acquisition, the decay reference of a decay-corrected (START) series:
    (datetime, source text)."""
    first = datasets[0]
    series = parseDateTime(_value(first, "SeriesDate"), _value(first, "SeriesTime"))
    acquisitions = [a for a in (_acquisitionStart(ds) for ds in datasets) if a is not None]
    firstAcquisition = min(acquisitions) if acquisitions else None
    if series is not None and (firstAcquisition is None or series <= firstAcquisition):
        return series, "series date/time"
    geStart = privateValue(first, 0x0009, GE_PET_CREATOR, 0x0D)
    if geStart:
        stamp = parseDateTime(geStart)
        if stamp is not None:
            return stamp, "GE scan start (0009,100d)"
    if firstAcquisition is not None:
        return firstAcquisition, "first acquisition date/time"
    return series, "series date/time"


# ---------------------------------------------------------------------------
# SUV
# ---------------------------------------------------------------------------
def suvInfo(datasets, frameStarts=None, decayCorrectedTo=None):
    """SUVbw conversion of a PET series: a dict with
        activityScale   factor from rescaled pixel values to Bq/ml (None when not possible)
        suvScale        factor from rescaled pixel values to SUVbw (None when not possible)
        frameSuvScales  for Decay Correction NONE with frameStarts given: one SUV factor per frame (the dose
                        decayed to each frame's start), else None (suvScale holds for every frame)
                        decayCorrectedTo: the frames of a NONE series are decay corrected to this time by the
                        loader (decayCorrectionFactors), so one scale with the dose decayed to it
        units           DICOM Units (0054,1001)
        and the inputs (WeightKg, InjectedDoseBq, DecayedDoseBq, InjectionTime, ScanStart, HalfLifeS,
        DecayCorrection, DecayReference, SUVSource) for the node attributes,
        problems        why SUV is not available (list of text)."""
    first = datasets[0]
    units = str(_value(first, "Units", "") or "").strip().upper()
    info = {"units": units, "activityScale": None, "suvScale": None, "frameSuvScales": None, "problems": []}
    height = _float(_value(first, "PatientSize"))
    info.update(PatientWeightKg=_float(_value(first, "PatientWeight")),
                PatientHeightCm=height * 100.0 if height and height > 0 else None,
                PatientSex=str(_value(first, "PatientSex", "") or "").strip().upper())

    if units == "GML":
        info.update(suvScale=1.0, SUVSource="image already in SUVbw (Units GML)")
        info["problems"].append("activity concentration not available (image stored in SUVbw)")
        return info

    if units == "CNTS":
        suvFactor = _float(privateValue(first, 0x7053, PHILIPS_CREATOR, 0x00))
        activityFactor = _float(privateValue(first, 0x7053, PHILIPS_CREATOR, 0x09))
        if activityFactor:
            info["activityScale"] = activityFactor
        if suvFactor:
            info.update(suvScale=suvFactor, SUVSource="Philips SUV scale factor (7053,1000)")
        else:
            info["problems"].append("counts image without the Philips SUV scale factor")
        return info

    if units not in ("BQML", ""):
        info["problems"].append(f"units {units} not supported for SUV")
        return info
    info["activityScale"] = 1.0
    if not units:
        info["problems"].append("Units (0054,1001) missing, assumed Bq/ml")

    weight = _float(_value(first, "PatientWeight"))
    if not weight or weight <= 0:
        info["problems"].append("patient weight missing")
    radio = _radiopharmaceutical(first)
    dose = _float(_value(radio, "RadionuclideTotalDose")) if radio is not None else None
    halfLife = _float(_value(radio, "RadionuclideHalfLife")) if radio is not None else None
    injection = _injectionTime(first, radio)
    source = "Radiopharmaceutical Information Sequence"
    if not dose:
        # GE's own record of the measured tracer activity, MBq, and when it was measured
        geDose = _float(privateValue(first, 0x0009, GE_PET_CREATOR, 0x38))
        geTime = parseDateTime(privateValue(first, 0x0009, GE_PET_CREATOR, 0x39))
        if geDose and geTime:
            dose, injection, source = geDose * 1e6, geTime, "GE tracer activity (0009,1038)"
    if not dose or dose <= 0:
        info["problems"].append("injected dose missing")
    if not halfLife or halfLife <= 0:
        info["problems"].append("half life missing")

    decayCorrection = str(_value(first, "DecayCorrection", "") or "").strip().upper()
    decayed = None
    frameDecayed = None
    if dose and halfLife:
        if decayCorrection == "ADMIN":
            decayed, reference = dose, "ADMIN (none)"
            info["ScanStart"] = ""
        else:
            if decayCorrection == "NONE" and decayCorrectedTo is not None:
                start, reference = decayCorrectedTo, "first frame start (frames decay corrected by the loader)"
                frameStarts = None
            elif decayCorrection == "NONE":
                stamps = [a for a in (_acquisitionStart(ds) for ds in datasets) if a is not None]
                start, reference = (min(stamps) if stamps else None), "first acquisition date/time"
                if frameStarts and len(frameStarts) > 1:
                    reference = "start of each frame (images not decay corrected)"
            else:
                start, reference = scanStart(datasets)
            if injection is None:
                info["problems"].append("injection time missing")
            elif start is None:
                info["problems"].append("scan time missing")
            else:
                elapsed = (start - injection).total_seconds()
                if elapsed < 0:
                    info["problems"].append("injection time after the scan start")
                else:
                    decayed = dose * math.pow(2.0, -elapsed / halfLife)
                    if decayCorrection == "NONE" and frameStarts and len(frameStarts) > 1:
                        unknown = [str(f + 1) for f, s in enumerate(frameStarts) if s is None]
                        if unknown:
                            info["problems"].append(f"start time of frame {_listText(unknown, 6)} unknown "
                                                    "(images not decay corrected)")
                        else:
                            frameDecayed = [dose * math.pow(2.0, -(s - injection).total_seconds() / halfLife)
                                            for s in frameStarts]
                info["ScanStart"] = start.isoformat()
        info["DecayReference"] = reference
    info.update({"WeightKg": weight, "InjectedDoseBq": dose, "HalfLifeS": halfLife,
                 "InjectionTime": injection.isoformat() if injection else "",
                 "DecayCorrection": decayCorrection, "DecayedDoseBq": decayed, "SUVSource": source})
    if weight and decayed and not info["problems"]:
        info["suvScale"] = weight * 1000.0 / decayed
        if frameDecayed:
            info["frameSuvScales"] = [weight * 1000.0 / d for d in frameDecayed]
    return info


def isSuvUnits(units):
    return units in SUV_METHODS.values()


def suvMethodOf(units):
    """"bw" / "lbm" / "bsa" of SUV units (None for Bq/ml)."""
    return next((m for m, u in SUV_METHODS.items() if u == units), None)


def leanBodyMassKg(weightKg, heightCm, sex):
    """Lean body mass, kg (Janmahasatian 2005): 9270 W / (6680 + 216 BMI) for men, 9270 W / (8780 + 244 BMI)
    for women. None without weight, height or sex (M / F)."""
    if not weightKg or not heightCm or sex not in ("M", "F"):
        return None
    bmi = weightKg / (heightCm / 100.0) ** 2
    return 9270.0 * weightKg / ((6680.0 + 216.0 * bmi) if sex == "M" else (8780.0 + 244.0 * bmi))


def bodySurfaceAreaM2(weightKg, heightCm):
    """Body surface area, m2 (Du Bois): 0.007184 W^0.425 H^0.725. None without weight or height."""
    if not weightKg or not heightCm:
        return None
    return 0.007184 * weightKg ** 0.425 * heightCm ** 0.725


def suvScales(suv, method="bw"):
    """SUV factors of suvInfo for a normalisation method: (scale, frameScales or None, extra attributes,
    problems). scale is None when that SUV cannot be computed."""
    scale, frameScales = suv["suvScale"], suv["frameSuvScales"]
    if method == "bw" or scale is None:
        return scale, frameScales, {"SUVMethod": SUV_METHOD_TITLES["bw"]}, list(suv["problems"])
    weight, height, sex = suv.get("PatientWeightKg"), suv.get("PatientHeightCm"), suv.get("PatientSex")
    if not weight:
        return None, None, {}, ["patient weight missing"]
    if method == "lbm":
        lbm = leanBodyMassKg(weight, height, sex)
        if lbm is None:
            return None, None, {}, ["patient height or sex (M / F) missing for the lean body mass"]
        factor, extra = lbm / weight, {"LeanBodyMassKg": f"{lbm:.4g}"}
    elif method == "bsa":
        bsa = bodySurfaceAreaM2(weight, height)
        if bsa is None:
            return None, None, {}, ["patient height missing for the body surface area"]
        factor, extra = bsa * 1e4 / (weight * 1000.0), {"BodySurfaceAreaM2": f"{bsa:.4g}"}
    else:
        raise ValueError(f"Unknown SUV method {method}")
    extra["SUVMethod"] = SUV_METHOD_TITLES[method]
    return scale * factor, [f * factor for f in frameScales] if frameScales else None, extra, []


SUV_ATTRIBUTE_KEYS = ("WeightKg", "InjectedDoseBq", "DecayedDoseBq", "HalfLifeS", "InjectionTime", "ScanStart",
                      "DecayCorrection", "DecayReference", "SUVSource")


def suvAttributes(suv):
    """The SUV inputs of suvInfo as {key: text} for node attributes."""
    out = {}
    for key in SUV_ATTRIBUTE_KEYS:
        value = suv.get(key)
        if value not in (None, ""):
            out[key] = value if isinstance(value, str) else f"{value:.6g}"
    if suv["problems"]:
        out["SUVProblems"] = "; ".join(suv["problems"])
    return out


# ---------------------------------------------------------------------------
# Dynamic frame timing
# ---------------------------------------------------------------------------
FRAME_TIMING_COLUMNS = (
    ("Frame", "frame number (1 = first)"),
    ("StartDateTime", "frame start, clock date/time"),
    ("StartFromInjectionS", "frame start from the injection, s"),
    ("StartFromStudyStartS", "frame start from the study start = start of the first frame, s"),
    ("StartFromScanStartS", "frame start from the scan start (decay reference of START), s"),
    ("DurationS", "frame duration, s"),
    ("DurationMs", "frame duration, ms"),
    ("MidFromInjectionS", "frame middle from the injection, s (kinetic modelling time)"),
    ("MidFromScanStartS", "frame middle from the scan start, s"),
    ("EndFromInjectionS", "frame end from the injection, s"),
    ("FrameReferenceTimeMs", "Frame Reference Time (0054,1300), ms"),
    ("DecayFactor", "Decay Factor (0054,1321) applied to the frame"),
    ("StartSource", "where the frame start comes from"),
    ("LoaderDecayFactor", "decay correction applied by the loader to a not decay corrected frame (NONE)"),
)
START_FROM_ACQUISITION = "acquisition time"
START_FROM_PREVIOUS = "end of the previous frame"
START_FROM_SCAN = "scan start"


def frameStarts(frames, datasets=None):
    """Clock start of each time frame and where it comes from: a list of (datetime or None, source), in
    this order of preference:
        acquisition time   earliest Acquisition Date/Time of the frame's slices
        end of the previous frame   previous start + its Actual Frame Duration (frames back to back, as in
                           dynamic framing); used when the frame has no acquisition time
        scan start         first frame only, without acquisition time (scanStart)
    Frame Reference Time (0054,1300) is not used: the standard does not say whether it is the frame's start
    or middle. Both the frame timing table and the per-frame SUV of non decay corrected images use these
    starts, so they cannot disagree."""
    datasets = datasets or [ds for frame in frames for ds in frame]
    previousEnd = None
    out = []
    for f, frame in enumerate(frames):
        stamps = [a for a in (_acquisitionStart(ds) for ds in frame) if a is not None]
        if stamps:
            start, source = min(stamps), START_FROM_ACQUISITION
        elif previousEnd is not None:
            start, source = previousEnd, START_FROM_PREVIOUS
        elif f == 0:
            start, source = scanStart(datasets)[0], START_FROM_SCAN
        else:
            start, source = None, None
        durationMs = _float(_value(frame[0], "ActualFrameDuration"))
        previousEnd = start + datetime.timedelta(milliseconds=durationMs) \
            if (start is not None and durationMs is not None) else None
        out.append((start, source if start is not None else None))
    return out


def isDecayCorrected(ds):
    """False for images not decay corrected (Decay Correction (0054,1102) NONE), else True."""
    return str(_value(ds, "DecayCorrection", "") or "").strip().upper() != "NONE"


def decayCorrectionFactors(frames, datasets=None):
    """Decay correction of the time frames of a series that is not decay corrected (Decay Correction NONE) to
    the start of the first frame, as scanners do for START: (reference datetime, factors, problem). Frame f
    is multiplied by
        2^((start_f - reference) / T1/2) x lambda d_f / (1 - exp(-lambda d_f))
    i.e. the decay from the reference to the frame start times the decay during the frame (the DICOM Decay
    Factor (0054,1321) of a decay corrected frame). Uses the frame starts of frameStarts and Actual Frame
    Duration; factors is None with the reason in problem when a start, a duration or the half life is
    missing."""
    datasets = datasets or [ds for frame in frames for ds in frame]
    radio = _radiopharmaceutical(datasets[0])
    halfLife = _float(_value(radio, "RadionuclideHalfLife")) if radio is not None else None
    if not halfLife or halfLife <= 0:
        return None, None, "half life missing"
    starts = frameStartTimes(frames, datasets)
    if any(start is None for start in starts):
        return None, None, "a frame start is unknown"
    durations = [_float(_value(frame[0], "ActualFrameDuration")) for frame in frames]
    if any(d is None or d <= 0 for d in durations):
        return None, None, "a frame duration is missing"
    rate = math.log(2.0) / halfLife
    reference = starts[0]
    factors = []
    for start, durationMs in zip(starts, durations):
        x = rate * durationMs / 1000.0
        factors.append(math.pow(2.0, (start - reference).total_seconds() / halfLife) * x / -math.expm1(-x))
    return reference, factors, None


def frameStartWarnings(frames, numbers=None, datasets=None):
    """Texts for the user about the frame starts: frames without an acquisition time (start estimated) and
    frames that start more than 1 s before the previous one ends (inconsistent times or durations)."""
    numbers = list(numbers or range(1, len(frames) + 1))
    starts = frameStarts(frames, datasets)
    warnings = []
    estimated = [n for n, (start, source) in zip(numbers, starts) if source != START_FROM_ACQUISITION]
    if estimated:
        unknown = [n for n, (start, _) in zip(numbers, starts) if start is None]
        text = (f"Acquisition time missing in frame {_listText(estimated, 6)}; the start is taken from the end "
                "of the previous frame (or the scan start) for the timing and for SUV")
        if unknown:
            text += f"; frame {_listText(unknown, 6)} has no start at all"
        warnings.append(text)
    overlaps = []
    for f in range(1, len(frames)):
        (start, source), (previous, _) = starts[f], starts[f - 1]
        durationMs = _float(_value(frames[f - 1][0], "ActualFrameDuration"))
        if source == START_FROM_ACQUISITION and previous is not None and durationMs is not None:
            if (start - previous).total_seconds() < durationMs / 1000.0 - 1.0:
                overlaps.append(numbers[f])
    if overlaps:
        warnings.append(f"Frame {_listText(overlaps, 6)} starts before the previous frame ends (acquisition "
                        "times and frame durations disagree); the timing follows the acquisition times")
    return warnings


def frameTiming(frames, datasets=None, numbers=None):
    """Timing of each time frame of a dynamic (or gated) series: a list of dicts keyed like
    FRAME_TIMING_COLUMNS (None where the header does not tell). The frame starts are those of frameStarts
    (acquisition time, else the end of the previous frame), with their source in StartSource. The study start
    is the start of the first frame of the PET (the DICOM Study Time is often the registration or the CT, not
    the PET), so frame 1 starts at 0 from the study start."""
    datasets = datasets or [ds for frame in frames for ds in frame]
    first = datasets[0]
    radio = _radiopharmaceutical(first)
    injection = _injectionTime(first, radio)
    scan, _ = scanStart(datasets)
    timings = []
    for frame, (start, source) in zip(frames, frameStarts(frames, datasets)):
        durationMs = _float(_value(frame[0], "ActualFrameDuration"))
        duration = durationMs / 1000.0 if durationMs is not None else None
        timings.append((start, source, durationMs, duration))
    study = timings[0][0] if timings else None
    rows = []
    for number, frame, (start, source, durationMs, duration) in zip(numbers or range(1, len(frames) + 1), frames,
                                                                     timings):
        middle = start + datetime.timedelta(seconds=duration / 2.0) if (start and duration is not None) else None
        end = start + datetime.timedelta(seconds=duration) if (start and duration is not None) else None
        rows.append({
            "Frame": number,
            "StartDateTime": start.isoformat() if start else None,
            "StartFromInjectionS": _round(_seconds(start, injection), 3),
            "StartFromStudyStartS": _round(_seconds(start, study), 3),
            "StartFromScanStartS": _round(_seconds(start, scan), 3),
            "DurationS": _round(duration, 3),
            "DurationMs": _round(durationMs, 1),
            "MidFromInjectionS": _round(_seconds(middle, injection), 3),
            "MidFromScanStartS": _round(_seconds(middle, scan), 3),
            "EndFromInjectionS": _round(_seconds(end, injection), 3),
            "FrameReferenceTimeMs": _round(_value(frame[0], "FrameReferenceTime"), 1),
            "DecayFactor": _round(_value(frame[0], "DecayFactor"), 6),
            "StartSource": source,
            "LoaderDecayFactor": None,
        })
    return rows



def frameStartTimes(frames, datasets=None):
    """Clock start of each frame (see frameStarts; None when unknown)."""
    return [start for start, _ in frameStarts(frames, datasets)]


def timingLists(rows):
    """Per-frame timing as JSON lists for the sequence node, {key: json text} (DurationS, ...)."""
    keys = [k for k, _ in FRAME_TIMING_COLUMNS if k not in ("Frame", "StartDateTime")]
    return {f"Frame{key}": json.dumps([row.get(key) for row in rows]) for key in keys}


# ---------------------------------------------------------------------------
# ECG gating
# ---------------------------------------------------------------------------
def gatingInfo(gates):
    """ECG gating fields of a gated PET as {key: value}, keyed like the gating attributes of the DICOM SPECT
    Recon extension (read by the Gated QC step of Tarvos). 0 = not recorded."""
    first = gates[0][0]
    info = {"GatingTimeSlots": len(gates), "GatingInfoSource": "DICOM PET gating attributes"}
    for key, keyword in (("GatingLowRRMs", "LowRRValue"), ("GatingHighRRMs", "HighRRValue"),
                         ("GatingIntervalsAcquired", "IntervalsAcquired"),
                         ("GatingIntervalsRejected", "IntervalsRejected"), ("GatingSkipBeats", "SkipBeats")):
        value = _round(_value(first, keyword))
        if value is not None and (value or key in ("GatingIntervalsRejected", "GatingSkipBeats")):
            info[key] = value
    flag = str(_value(first, "BeatRejectionFlag", "") or "").strip()
    if flag:
        info["GatingBeatRejection"] = flag
    framing = str(_value(first, "FramingType", "") or "").strip()
    if framing:
        info["GatingFramingType"] = framing

    nominal = _round(_value(first, "NominalInterval"))
    rate = _round(_value(first, "HeartRate"))
    if nominal:
        info["GatingNominalIntervalMs"], info["GatingNominalIntervalSource"] = nominal, "DICOM nominal interval"
    elif rate:
        info["GatingNominalIntervalMs"], info["GatingNominalIntervalSource"] = round(60000.0 / rate, 1), "DICOM heart rate"
    elif info.get("GatingLowRRMs") and info.get("GatingHighRRMs"):
        info["GatingNominalIntervalMs"] = round(0.5 * (info["GatingLowRRMs"] + info["GatingHighRRMs"]), 1)
        info["GatingNominalIntervalSource"] = "middle of the accepted R-R window"
    if rate:
        info["GatingHeartRateBpm"], info["GatingHeartRateSource"] = rate, "DICOM heart rate"
    elif info.get("GatingNominalIntervalMs"):
        info["GatingHeartRateBpm"] = round(60000.0 / info["GatingNominalIntervalMs"], 1)
        info["GatingHeartRateSource"] = "from the R-R interval"

    acquired, rejected = info.get("GatingIntervalsAcquired"), info.get("GatingIntervalsRejected")
    if acquired is not None and rejected is not None and acquired + rejected > 0:
        info["GatingRejectedPercent"] = round(100.0 * rejected / (acquired + rejected), 1)
    mean = info.get("GatingNominalIntervalMs")
    if mean:
        info["GatingSlotWidthMs"] = round(mean / len(gates), 1)
        if info.get("GatingLowRRMs") and info.get("GatingHighRRMs"):
            info["GatingRRWindowPercent"] = round(50.0 * (info["GatingHighRRMs"] - info["GatingLowRRMs"]) / mean, 1)
    info["GatingBeatsScope"] = "acquisition"
    return info


def gateTiming(gate):
    """(trigger time ms, frame duration ms) of one gate (None when not recorded)."""
    first = gate[0]
    return _round(_value(first, "TriggerTime")), _round(_value(first, "ActualFrameDuration"))


# ---------------------------------------------------------------------------
# Test data
# ---------------------------------------------------------------------------
def writeSyntheticSeries(folder, kind=GATED, frames=4, slices=6, rows=16, columns=16, shuffle=True,
                         radiopharmaceutical="Fluorodeoxyglucose", description=None, decayCorrection="START",
                         frameDurationsS=None):
    """Write a small PET series (Bq/ml, per image slope) and return (files, expected SUVbw arrays per frame).
    Frame f has the value 1000 x f Bq/ml at every voxel; 70 kg, 170 cm, 370 MBq F-18 injected 60 min before
    the series time (static and gated). Dynamic frames start at the series time, back to back, with the
    given durations (default 30 s each). A static series is written with one frame (frames is ignored)."""
    import random
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid
    os.makedirs(folder, exist_ok=True)
    frames = 1 if kind == STATIC else frames
    description = description or f"Synthetic {kind.lower()} PET"
    durations = list(frameDurationsS or [30.0] * frames)
    study, series, frameOfReference = generate_uid(), generate_uid(), generate_uid()
    seriesStart = datetime.datetime(2026, 1, 1, 10, 0, 0)
    injection = datetime.datetime(2026, 1, 1, 9, 0, 0)
    if kind == DYNAMIC:
        injection = seriesStart   # dynamic studies start at the injection
    starts = [seriesStart + datetime.timedelta(seconds=sum(durations[:f])) for f in range(frames)]
    files, order = [], [(f, s) for f in range(frames) for s in range(slices)]
    if shuffle:
        random.Random(1).shuffle(order)
    for n, (f, s) in enumerate(order):
        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.128"
        meta.MediaStorageSOPInstanceUID = generate_uid()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds = Dataset()
        ds.file_meta = meta
        ds.SOPClassUID, ds.SOPInstanceUID = meta.MediaStorageSOPClassUID, meta.MediaStorageSOPInstanceUID
        ds.Modality, ds.SeriesDescription = "PT", description
        ds.Manufacturer, ds.ManufacturerModelName = "Synthetic", "Phantom PET"
        ds.PatientName, ds.PatientID, ds.PatientWeight, ds.PatientSize, ds.PatientSex = \
            "Phantom^PET", "PHANTOM", "70", "1.70", "O"
        ds.StudyInstanceUID, ds.SeriesInstanceUID, ds.FrameOfReferenceUID = study, series, frameOfReference
        ds.StudyDate = ds.SeriesDate = ds.AcquisitionDate = "20260101"
        ds.StudyTime, ds.SeriesTime = "095500", seriesStart.strftime("%H%M%S")
        ds.AcquisitionTime = starts[f].strftime("%H%M%S.%f")
        ds.SeriesType = [{STATIC: "STATIC", GATED: "GATED", DYNAMIC: "DYNAMIC"}[kind], "IMAGE"]
        ds.Units, ds.DecayCorrection = "BQML", decayCorrection
        ds.CorrectedImage = ["DECY", "ATTN", "SCAT", "DTIM", "RAN", "NORM"] if decayCorrection != "NONE" else \
            ["ATTN", "SCAT", "DTIM", "RAN", "NORM"]
        ds.ReconstructionMethod = "OSEM 3i21s"
        ds.NumberOfSlices = slices
        if kind == GATED:
            ds.NumberOfTimeSlots = frames
            ds.TriggerTime = f * 100
            ds.ImageIndex = f * slices + s + 1
            ds.ActualFrameDuration = 100
        elif kind == DYNAMIC:
            ds.NumberOfTimeSlices = frames
            ds.ImageIndex = f * slices + s + 1
            ds.ActualFrameDuration = int(round(durations[f] * 1000))
            ds.FrameReferenceTime = round((sum(durations[:f]) + durations[f] / 2.0) * 1000.0, 1)
            ds.DecayFactor = "1.0"
        else:
            ds.ImageIndex = s + 1
            ds.ActualFrameDuration = 600000
        ds.InstanceNumber = n + 1
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        ds.ImagePositionPatient = [-40.0, -40.0, -100.0 + 3.27 * s]
        ds.PixelSpacing, ds.SliceThickness = [5.0, 5.0], 3.27
        radio = Dataset()
        radio.RadionuclideTotalDose, radio.RadionuclideHalfLife = 370e6, 6586.2
        radio.RadiopharmaceuticalStartDateTime = injection.strftime("%Y%m%d%H%M%S")
        if radiopharmaceutical:
            radio.Radiopharmaceutical = radiopharmaceutical
        nuclide = Dataset()
        nuclide.CodeValue, nuclide.CodingSchemeDesignator, nuclide.CodeMeaning = "C-111A1", "SRT", "^18^Fluorine"
        radio.RadionuclideCodeSequence = [nuclide]
        ds.RadiopharmaceuticalInformationSequence = [radio]
        ds.Rows, ds.Columns = rows, columns
        ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
        ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 1
        slope = 1000.0 * (f + 1) / 20000.0
        ds.RescaleSlope, ds.RescaleIntercept = f"{slope:.8g}", "0"
        ds.PixelData = np.full((rows, columns), 20000, dtype=np.int16).tobytes()
        path = os.path.join(folder, f"img{n:04d}.dcm")
        try:
            ds.save_as(path, enforce_file_format=True)          # pydicom 3
        except TypeError:
            ds.is_little_endian, ds.is_implicit_VR = True, False  # pydicom 2
            ds.save_as(path, write_like_original=False)
        files.append(path)
    expected = []
    for f in range(frames):
        reference = starts[f] if decayCorrection == "NONE" else seriesStart
        elapsed = (reference - injection).total_seconds()
        decayed = 370e6 * 2.0 ** (-elapsed / 6586.2)
        expected.append(np.full((slices, rows, columns), 1000.0 * (f + 1) * 70000.0 / decayed))
    return files, expected
