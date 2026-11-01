"""Pure helpers of the Review/Edit spherical ROIs (numpy only, no Slicer imports): voxels of a sphere, the
time-activity curve of a sphere over the frames, and the plot helpers of the timeline.

A sphere's region is (slices, mask): slices cut the bounding box out of an array indexed [k, j, i], mask is a
boolean array of that box. A region with no voxels (the sphere is outside the volume) has an empty mask."""

import math

import numpy as np

ROI_COLORS = ("#FFD23F", "#4CC9F0", "#FF6B6B", "#7BD15A", "#C77DFF", "#FF9F1C", "#F15BB5", "#E8E8E8")


def sphereMask(shape_kji, ijkToRas, centerRas, radiusMm):
    """Region of the voxels of an array of the given shape whose centres are within radiusMm of centerRas.
    ijkToRas is the 4x4 matrix of the volume. A sphere that holds no voxel centre gives the nearest voxel when
    the centre is inside the volume, and an empty region when it is outside."""
    matrix = np.asarray(ijkToRas, dtype=float)
    rasToIjk = np.linalg.inv(matrix)
    centerIjk = (rasToIjk @ np.array([centerRas[0], centerRas[1], centerRas[2], 1.0]))[:3]
    sizes = np.array(shape_kji[::-1])
    reach = radiusMm * np.linalg.norm(rasToIjk[:3, :3], axis=1)    # half the sphere's box along i, j, k, voxels
    low = np.maximum(np.ceil(centerIjk - reach).astype(int), 0)
    high = np.minimum(np.floor(centerIjk + reach).astype(int), sizes - 1)
    if np.all(low <= high):
        k, j, i = np.ogrid[low[2]:high[2] + 1, low[1]:high[1] + 1, low[0]:high[0] + 1]
        di, dj, dk = i - centerIjk[0], j - centerIjk[1], k - centerIjk[2]
        distance2 = sum((matrix[r, 0] * di + matrix[r, 1] * dj + matrix[r, 2] * dk) ** 2 for r in range(3))
        mask = distance2 <= radiusMm * radiusMm
        if mask.any():
            return tuple(slice(low[a], high[a] + 1) for a in (2, 1, 0)), mask
    nearest = np.rint(centerIjk).astype(int)
    if np.all(nearest >= 0) and np.all(nearest < sizes):
        return tuple(slice(nearest[a], nearest[a] + 1) for a in (2, 1, 0)), np.ones((1, 1, 1), dtype=bool)
    return (slice(0, 0), slice(0, 0), slice(0, 0)), np.zeros((0, 0, 0), dtype=bool)


def regionMean(array_kji, region):
    """Mean of the voxels of a sphereMask region; None when it has no voxels or the mean is not finite."""
    slices, mask = region
    if not mask.any():
        return None
    value = float(np.mean(array_kji[slices][mask]))
    return value if math.isfinite(value) else None


def sphereMean(array_kji, ijkToRas, centerRas, radiusMm):
    """Mean of the voxels of the sphere (see sphereMask); None when the sphere is outside the volume."""
    return regionMean(array_kji, sphereMask(array_kji.shape, ijkToRas, centerRas, radiusMm))


def sphereCurve(frames, centerRas, radiusMm):
    """Time-activity curve of a sphere: its mean in every frame of frames, an iterable of (array_kji, ijkToRas)
    (or None for a frame without an image, which gives a None value). The voxels are found once for frames that
    share their geometry."""
    values, key, region = [], None, None
    for frame in frames:
        if frame is None:
            values.append(None)
            continue
        array, ijkToRas = frame
        geometry = (array.shape, np.asarray(ijkToRas, dtype=float).tobytes())
        if geometry != key:
            key, region = geometry, sphereMask(array.shape, ijkToRas, centerRas, radiusMm)
        values.append(regionMean(array, region))
    return values


def frameMidpoints(intervals):
    """Middle of every frame of dynamicviews.frameIntervals: (start + end) / 2, seconds."""
    return [(frame["start"] + frame["end"]) / 2.0 for frame in intervals]


def niceAxis(maxValue, tickCount=4):
    """Ticks of a y axis from 0 up to at least maxValue, about tickCount steps of 1, 2, 2.5 or 5 x 10^n."""
    if not maxValue or maxValue <= 0 or not math.isfinite(maxValue):
        return [0.0, 1.0]
    rough = maxValue / float(tickCount)
    magnitude = 10.0 ** math.floor(math.log10(rough))
    step = next(m * magnitude for m in (1, 2, 2.5, 5, 10) if m * magnitude >= rough * (1 - 1e-9))
    count = int(math.ceil(maxValue / step - 1e-9))
    return [float(f"{i * step:.10g}") for i in range(count + 1)]


def formatValue(value):
    """Short text of an activity or SUV: '12,345', '12.35', '0.1234'; '?' when unknown."""
    if value is None:
        return "?"
    return f"{value:,.0f}" if abs(value) >= 1000 else f"{value:.4g}"


def nextColor(used):
    """First color of ROI_COLORS not in used; when all are used, the one used least."""
    used = list(used)
    return min(ROI_COLORS, key=lambda color: used.count(color))


def colorRgb(color):
    """(r, g, b) of 0 - 1 of a '#RRGGBB' color."""
    return tuple(int(color[i:i + 2], 16) / 255.0 for i in (1, 3, 5))
