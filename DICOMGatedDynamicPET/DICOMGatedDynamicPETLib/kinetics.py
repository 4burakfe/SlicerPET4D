"""Kinetic modelling of dynamic PET time-activity curves (numpy only, no Slicer imports), so other modules (and
scripts) can use it as a library:

    from DICOMGatedDynamicPETLib import kinetics
    frames = kinetics.Frames(startsS, endsS)                  # seconds from time zero (the injection)
    blood = kinetics.FrameInput(frames, bloodMeans)          # image-derived input (or kinetics.SampledInput)
    result = kinetics.fit("2TC", frames, tissueMeans, inputFunction=blood)
    result["values"]["K1"], result["values"]["VT"], result["metrics"]["R2"]

Times are minutes inside (rate constants per minute); Frames takes seconds or minutes (unit=). Every model works on
the mean of a region per frame (a TAC) and, for the voxel-wise maps, on arrays of frames x voxels.

Models (MODELS; "input" is the plasma input function, "reference" a region without specific binding):
    1TC         one tissue compartment, plasma input: K1, k2, vB (and a blood delay); VT = K1 / k2
    2TC         two tissue compartments, reversible: K1, k2, k3, k4, vB; VT = K1/k2 (1 + k3/k4), BPnd = k3/k4
    2TCi        two tissue compartments, irreversible (k4 = 0): K1, k2, k3, vB; Ki = K1 k3 / (k2 + k3)
    Patlak      Patlak plot, plasma input: Ki (slope) and V0 (intercept) from t*
    Logan       Logan plot, plasma input: VT (slope) from t*
    LoganRef    Logan plot, reference region (Logan 1996), optional k2': DVR (slope), BPnd = DVR - 1
    SRTM        simplified reference tissue model (Lammertsma & Hume 1996): R1, k2, BPnd; k2' = k2 / R1
    SUVR        ratio of the target to the reference over a time window; BPnd = SUVR - 1

The compartment models and SRTM are fitted by weighted least squares (Levenberg-Marquardt with bounds) to the frame
means of the model (the model is computed on a fine time grid and averaged over every frame). The tissue model of
the plasma input models is (1 - vB) C_T + vB C_B with C_B the whole blood curve (the image-derived input itself
unless a separate blood curve is given). Graphical methods use the points from t* on, fixed or found automatically
(the earliest t* from which the line fits every point within a relative error, as PMOD does).

Fit metrics (on the frames used): R2 (1 - SSres / SStot), RMSE, NRMSE (RMSE / mean of the data) and AIC
(n ln(WSS / n) + 2 p, with the weighted sum of squares and p fitted parameters). The graphical methods give them for
their straight line.

Weighting: WEIGHT_DURATION (frame duration, default), WEIGHT_UNIFORM, WEIGHT_COUNTS (duration / activity: the
variance of a decay corrected frame grows with its activity and shrinks with its duration).
"""

import csv
import io
import math

import numpy as np

WEIGHT_DURATION = "duration"
WEIGHT_UNIFORM = "uniform"
WEIGHT_COUNTS = "counts"
WEIGHTINGS = ((WEIGHT_DURATION, "Frame duration"), (WEIGHT_UNIFORM, "Uniform"),
              (WEIGHT_COUNTS, "Frame duration / activity"))

DEFAULT_TSTAR_MIN = 20.0
DEFAULT_MAX_ERROR = 10.0       # % relative error of the automatic t*
MIN_LINEAR_POINTS = 3
METRICS = ("R2", "RMSE", "NRMSE", "AIC")


# ---------------------------------------------------------------------------
# Parameters and models
# ---------------------------------------------------------------------------
class Parameter:
    """A parameter of a fitted model: initial value, bounds, unit, and whether it is fitted (False: kept at its
    initial value)."""

    def __init__(self, name, initial, lower, upper, unit="", fitted=True, description=""):
        self.name, self.initial, self.lower, self.upper = name, float(initial), float(lower), float(upper)
        self.unit, self.fitted, self.description = unit, fitted, description

    def copy(self, **changes):
        out = Parameter(self.name, self.initial, self.lower, self.upper, self.unit, self.fitted, self.description)
        for key, value in changes.items():
            setattr(out, key, value)
        return out

    def asDict(self):
        return {"name": self.name, "initial": self.initial, "lower": self.lower, "upper": self.upper,
                "unit": self.unit, "fitted": self.fitted}


def _p(name, initial, lower, upper, unit, description, fitted=True):
    return Parameter(name, initial, lower, upper, unit, fitted, description)


_K1 = _p("K1", 0.1, 0.0, 3.0, "ml/ccm/min", "influx from plasma to tissue")
_K2 = _p("k2", 0.1, 0.0, 3.0, "1/min", "efflux from tissue to plasma")
_K3 = _p("k3", 0.05, 0.0, 2.0, "1/min", "transfer to the specific (bound) compartment")
_K4 = _p("k4", 0.05, 0.0, 2.0, "1/min", "return from the specific compartment")
_VB = _p("vB", 0.05, 0.0, 0.6, "", "blood volume fraction")
_DELAY = _p("delay", 0.0, -0.5, 1.0, "min", "delay of the input function (fitted when ticked)", fitted=False)


class Model:
    """A kinetic model: key, title, input kind ("plasma", "reference" or both for a ratio), method ("compartment",
    "srtm", "graphical", "ratio"), its fitted parameters and the values it reports (in order)."""

    def __init__(self, key, title, inputKind, method, parameters, reports, description, voxelwise=False):
        self.key, self.title, self.inputKind, self.method = key, title, inputKind, method
        self.parameters, self.reports, self.description = parameters, reports, description
        self.voxelwise = voxelwise

    def defaultParameters(self):
        return [p.copy() for p in self.parameters]

    @property
    def usesReference(self):
        return self.inputKind == "reference"


MODELS = {
    "1TC": Model("1TC", "1TC (one tissue compartment)", "plasma", "compartment", [_K1, _K2, _VB, _DELAY],
                 ("K1", "k2", "vB", "delay", "VT"),
                 "One tissue compartment with a plasma input; VT = K1 / k2.", voxelwise=True),
    "2TC": Model("2TC", "2TC reversible (K1, k2, k3, k4)", "plasma", "compartment", [_K1, _K2, _K3, _K4, _VB, _DELAY],
                 ("K1", "k2", "k3", "k4", "vB", "delay", "VT", "VND", "BPnd"),
                 "Two tissue compartments with a plasma input; VT = K1/k2 (1 + k3/k4), BPnd = k3/k4."),
    "2TCi": Model("2TCi", "2TC irreversible (k4 = 0)", "plasma", "compartment", [_K1, _K2, _K3, _VB, _DELAY],
                  ("K1", "k2", "k3", "vB", "delay", "Ki"),
                  "Two tissue compartments, irreversible trapping; Ki = K1 k3 / (k2 + k3)."),
    "Patlak": Model("Patlak", "Patlak plot (plasma input)", "plasma", "graphical", [],
                    ("Ki", "V0", "t*"),
                    "Patlak graphical analysis of irreversible uptake: Ki is the slope from t* on.", voxelwise=True),
    "Logan": Model("Logan", "Logan plot (plasma input)", "plasma", "graphical", [],
                   ("VT", "intercept", "t*"),
                   "Logan graphical analysis of reversible binding: VT is the slope from t* on.", voxelwise=True),
    "LoganRef": Model("LoganRef", "Logan plot (reference region)", "reference", "graphical", [],
                      ("DVR", "BPnd", "intercept", "t*"),
                      "Logan reference tissue method: DVR is the slope from t* on, BPnd = DVR - 1.", voxelwise=True),
    "SRTM": Model("SRTM", "SRTM (simplified reference tissue)", "reference", "srtm",
                  [_p("R1", 1.0, 0.0, 5.0, "", "K1 / K1' (relative delivery)"),
                   _p("k2", 0.1, 0.001, 3.0, "1/min", "efflux of the target region"),
                   _p("BPnd", 1.0, -0.9, 20.0, "", "non-displaceable binding potential")],
                  ("R1", "k2", "k2'", "BPnd"),
                  "Simplified reference tissue model; k2' = k2 / R1. The voxel-wise maps use basis functions.",
                  voxelwise=True),
    "SUVR": Model("SUVR", "SUVR (ratio to the reference)", "reference", "ratio", [],
                  ("SUVR", "BPnd"),
                  "Mean of the target over the time window divided by the reference's; BPnd = SUVR - 1.",
                  voxelwise=True),
}
MODEL_ORDER = ("2TC", "2TCi", "1TC", "Patlak", "Logan", "SRTM", "LoganRef", "SUVR")

UNITS = {"K1": "ml/ccm/min", "k2": "1/min", "k3": "1/min", "k4": "1/min", "delay": "min", "Ki": "ml/ccm/min",
         "VT": "ml/ccm", "VND": "ml/ccm", "V0": "ml/ccm", "k2'": "1/min", "t*": "min"}


def model(key):
    try:
        return MODELS[key]
    except KeyError:
        raise ValueError(f"Unknown kinetic model {key!r}; one of {', '.join(MODEL_ORDER)}") from None


def defaultOptions(key=None):
    """Options of fit() and parametricMaps(); every model ignores the ones it does not use."""
    options = {"weighting": WEIGHT_DURATION, "tStar": DEFAULT_TSTAR_MIN, "autoTStar": False,
               "maxError": DEFAULT_MAX_ERROR, "k2ref": None, "window": None, "parameters": None, "starts": 4,
               "clipNegative": True}
    if key is not None:
        options["parameters"] = model(key).defaultParameters()
    return options


# ---------------------------------------------------------------------------
# Frames and curves
# ---------------------------------------------------------------------------
class Frames:
    """Frame timing relative to time zero (the injection): starts and ends in seconds (unit="s", default) or minutes.
    Inside everything is minutes. Frames starting before time zero are cut at it; a frame that ends at or before
    time zero is not used (valid False)."""

    def __init__(self, starts, ends, unit="s"):
        scale = 1.0 / 60.0 if unit == "s" else 1.0
        self.rawStart = np.asarray(starts, dtype=float) * scale
        self.rawEnd = np.asarray(ends, dtype=float) * scale
        if self.rawStart.shape != self.rawEnd.shape or self.rawStart.ndim != 1 or not self.rawStart.size:
            raise ValueError("Frames need one start and one end per frame")
        if np.any(self.rawEnd <= self.rawStart):
            raise ValueError("Every frame must end after it starts")
        self.start = np.maximum(self.rawStart, 0.0)
        self.end = np.maximum(self.rawEnd, 0.0)
        self.valid = self.end > self.start
        if not self.valid.any():
            raise ValueError("No frame after time zero")
        self.duration = np.where(self.valid, self.end - self.start, 0.0)
        self.mid = (self.start + self.end) / 2.0

    @classmethod
    def fromIntervals(cls, intervals, timeZeroS=None):
        """From dynamicviews.frameIntervals (seconds on the study time axis); time zero in seconds on that axis
        (None: the start of the first frame)."""
        zero = intervals[0]["start"] if timeZeroS is None else float(timeZeroS)
        return cls([f["start"] - zero for f in intervals], [f["end"] - zero for f in intervals])

    def __len__(self):
        return self.start.size

    @property
    def lastEnd(self):
        return float(self.end[-1] if self.end.size else 1.0)

    def grid(self, maxPoints=32768):
        """Fine uniform time grid from 0 to the end of the last frame (minutes): a tenth of the shortest frame,
        between 0.1 s and 6 s, and at most maxPoints points."""
        shortest = float(np.min(self.duration[self.valid]))
        step = min(max(shortest / 10.0, 1.0 / 600.0), 0.1)
        step = max(step, self.lastEnd / (maxPoints - 1))
        count = int(math.ceil(self.lastEnd / step)) + 1
        return np.arange(count) * step

    def integralAtMid(self, values):
        """Integral from 0 to every frame's middle of a frame-sampled curve (values: frames, or frames x voxels),
        each frame's mean held over the frame; a gap between frames and the time before the first frame are linear
        (from 0 at time zero)."""
        values = np.asarray(values, dtype=float)
        out = np.zeros_like(values)
        total = np.zeros(values.shape[1:])
        previousEnd, previousValue = 0.0, np.zeros(values.shape[1:])
        for i in range(len(self)):
            if not self.valid[i]:
                continue
            gap = self.start[i] - previousEnd
            if gap > 0:
                total = total + gap * (previousValue + values[i]) / 2.0
            out[i] = total + values[i] * (self.mid[i] - self.start[i])
            total = total + values[i] * self.duration[i]
            previousEnd, previousValue = self.end[i], values[i]
        return out

    def frameMeans(self, grid, curve):
        """Mean of a curve on the grid (grid points, or grid points x voxels) over every frame."""
        curve = np.asarray(curve, dtype=float)
        step = grid[1] - grid[0] if grid.size > 1 else 1.0
        cumulative = np.concatenate([np.zeros((1,) + curve.shape[1:]),
                                     np.cumsum((curve[1:] + curve[:-1]) * step / 2.0, axis=0)])
        startI = _interpRows(self.start, grid, cumulative)
        endI = _interpRows(self.end, grid, cumulative)
        duration = np.where(self.valid, self.end - self.start, 1.0).reshape((-1,) + (1,) * (curve.ndim - 1))
        return (endI - startI) / duration

    def weights(self, mode=WEIGHT_DURATION, values=None):
        """Fit weights of the frames (mean 1 over the valid frames; 0 for a frame that is not valid)."""
        if mode == WEIGHT_UNIFORM:
            w = np.ones(len(self))
        elif mode == WEIGHT_COUNTS and values is not None:
            activity = np.abs(np.asarray(values, dtype=float))
            peak = float(np.nanmax(activity)) if np.any(np.isfinite(activity)) else 1.0
            floor = max(peak * 1e-3, 1e-12)
            w = self.duration / np.maximum(activity, floor)
        else:
            w = self.duration.copy()
        w = np.where(self.valid & np.isfinite(w), w, 0.0)
        mean = w[w > 0].mean() if np.any(w > 0) else 1.0
        return w / mean


def _interpRows(x, xp, fp):
    """np.interp of every column of fp (rows along xp)."""
    if fp.ndim == 1:
        return np.interp(x, xp, fp)
    flat = fp.reshape(fp.shape[0], -1)
    out = np.empty((len(x), flat.shape[1]))
    position = np.clip(np.searchsorted(xp, x, side="right") - 1, 0, len(xp) - 2)
    span = xp[position + 1] - xp[position]
    fraction = np.clip((x - xp[position]) / np.where(span > 0, span, 1.0), 0.0, 1.0)
    out[:] = flat[position] * (1 - fraction)[:, None] + flat[position + 1] * fraction[:, None]
    return out.reshape((len(x),) + fp.shape[1:])


class FrameInput:
    """An input (or reference) curve known as frame means: an image-derived blood pool, or a reference region. On
    the fine grid it is linear through the frame middles, from 0 at time zero, and constant after the last
    middle."""

    def __init__(self, frames, values):
        self.frames = frames
        self.values = np.asarray(values, dtype=float)
        if self.values.shape != (len(frames),):
            raise ValueError("The input curve needs one value per frame")

    def onGrid(self, grid, delay=0.0):
        valid = self.frames.valid
        times = self.frames.mid[valid]
        values = self.values[valid]
        if times[0] > 0:
            times, values = np.concatenate([[0.0], times]), np.concatenate([[0.0], values])
        return np.interp(grid - delay, times, values, left=0.0, right=values[-1])

    def frameMeans(self):
        return self.values

    def integralAtMid(self):
        return self.frames.integralAtMid(self.values)


class SampledInput:
    """An input curve measured at its own times (minutes from time zero, e.g. a plasma curve from a CSV file): linear
    between the samples, from 0 at time zero when the first sample is later, constant after the last."""

    def __init__(self, frames, times, values):
        self.frames = frames
        order = np.argsort(np.asarray(times, dtype=float))
        self.times = np.asarray(times, dtype=float)[order]
        self.values = np.asarray(values, dtype=float)[order]
        if self.times.size < 2:
            raise ValueError("The input curve needs at least two samples")
        self._grid = frames.grid()
        self._curve = self.onGrid(self._grid)

    def onGrid(self, grid, delay=0.0):
        times, values = self.times, self.values
        if times[0] > 0:
            times, values = np.concatenate([[0.0], times]), np.concatenate([[0.0], values])
        return np.interp(grid - delay, times, values, left=0.0, right=values[-1])

    def frameMeans(self):
        return self.frames.frameMeans(self._grid, self._curve)

    def integralAtMid(self):
        step = self._grid[1] - self._grid[0]
        cumulative = np.concatenate([[0.0], np.cumsum((self._curve[1:] + self._curve[:-1]) * step / 2.0)])
        return np.interp(self.frames.mid, self._grid, cumulative)


INPUT_PEAK_MAX_MIN = 5.0        # a blood curve peaks within the first minutes ...
INPUT_LATE_MAX_FRACTION = 0.5   # ... its last frames are well below the peak ...
INPUT_WASHOUT_MIN = 4.0         # ... and a few minutes after the peak it has fallen well below it
INPUT_WASHOUT_MAX_FRACTION = 0.6


def inputCurveWarning(frames, values):
    """Text when a frame-sampled curve used as the plasma input does not look like blood (it peaks late, or stays
    near its peak at the end, as a tissue region does), else None."""
    values = np.asarray(values, dtype=float)
    use = frames.valid & np.isfinite(values)
    if use.sum() < 3:
        return None
    mids, curve = frames.mid[use], values[use]
    peak = int(np.argmax(curve))
    if curve[peak] <= 0:
        return None
    late = float(curve[-1]) / float(curve[peak])
    after = mids[peak] + INPUT_WASHOUT_MIN
    washout = float(np.interp(after, mids, curve)) / float(curve[peak]) if mids[-1] >= after else None
    if (mids[peak] > INPUT_PEAK_MAX_MIN or (late > INPUT_LATE_MAX_FRACTION and mids[-1] > 3 * INPUT_PEAK_MAX_MIN)
            or (washout is not None and washout > INPUT_WASHOUT_MAX_FRACTION)):
        details = f"peak at {mids[peak]:.1f} min"
        if washout is not None:
            details += f", {100 * washout:.0f} % of the peak {INPUT_WASHOUT_MIN:g} min later"
        return (f"the input curve does not look like blood ({details}): is the blood pool segment a tissue "
                f"region? Use the reference methods (SRTM, Logan reference) for a reference region")
    return None


def readCurveCsv(text, timeUnit="s"):
    """(times in minutes, values) of a two-column CSV / TSV text (time, value); a header line and other columns are
    skipped. timeUnit "s" or "min" (a header naming min / s wins)."""
    rows = []
    dialect = csv.Sniffer().sniff(text[:2048], delimiters=",;\t ") if text.strip() else csv.excel
    for row in csv.reader(io.StringIO(text), dialect):
        cells = [c.strip() for c in row if c.strip()]
        if len(cells) < 2:
            continue
        try:
            rows.append((float(cells[0]), float(cells[1])))
        except ValueError:
            header = cells[0].lower()
            if not rows:
                if "min" in header:
                    timeUnit = "min"
                elif "(s)" in header or "[s]" in header or "sec" in header:
                    timeUnit = "s"
    if len(rows) < 2:
        raise ValueError("The curve file needs at least two rows of time, value")
    times, values = np.array(rows).T
    return (times / 60.0 if timeUnit == "s" else times), values


# ---------------------------------------------------------------------------
# Convolution and the model curves
# ---------------------------------------------------------------------------
class _Convolver:
    """Convolution of a curve on a uniform grid with decaying exponentials (FFT, trapezoidal rule)."""

    def __init__(self, grid, signal):
        self.grid = grid
        self.step = grid[1] - grid[0]
        self.n = grid.size
        self.size = 1 << int(math.ceil(math.log2(2 * self.n)))
        self.signal = np.asarray(signal, dtype=float)
        self.spectrum = np.fft.rfft(self.signal, self.size)

    def exp(self, rate):
        kernel = np.exp(-max(rate, 0.0) * self.grid)
        full = np.fft.irfft(self.spectrum * np.fft.rfft(kernel, self.size), self.size)[:self.n]
        return self.step * (full - 0.5 * (self.signal[0] * kernel + self.signal * kernel[0]))


def compartmentCurve(values, grid, plasmaConvolver, blood, kind):
    """Tissue curve on the grid of a compartment model: values {K1, k2, k3, k4, vB}; blood is the whole blood curve
    on the grid. kind "1TC", "2TC" or "2TCi"."""
    K1, k2, vB = values["K1"], values["k2"], values.get("vB", 0.0)
    if kind == "1TC":
        tissue = K1 * plasmaConvolver.exp(k2)
    else:
        k3 = values["k3"]
        k4 = 0.0 if kind == "2TCi" else values["k4"]
        total = k2 + k3 + k4
        root = math.sqrt(max(total * total - 4.0 * k2 * k4, 0.0))
        a1, a2 = (total - root) / 2.0, (total + root) / 2.0
        if a2 - a1 < 1e-7:      # equal rates: nudged apart (the response is continuous there)
            a1, a2 = max(a1 - 5e-8, 0.0), a2 + 5e-8
        tissue = K1 / (a2 - a1) * ((k3 + k4 - a1) * plasmaConvolver.exp(a1)
                                   + (a2 - k3 - k4) * plasmaConvolver.exp(a2))
    return (1.0 - vB) * tissue + vB * blood


def srtmCurve(values, reference, referenceConvolver):
    """Target curve on the grid of SRTM: values {R1, k2, BPnd}; reference on the grid."""
    R1, k2, bp = values["R1"], values["k2"], values["BPnd"]
    k2a = k2 / (1.0 + bp)
    return R1 * reference + (k2 - R1 * k2a) * referenceConvolver.exp(k2a)


def derivedValues(key, v):
    """Values computed from the fitted parameters (VT, Ki, BPnd, k2', ...)."""
    out = {}
    if key == "1TC":
        out["VT"] = _ratio(v["K1"], v["k2"])
    elif key == "2TC":
        out["VND"] = _ratio(v["K1"], v["k2"])
        out["BPnd"] = _ratio(v["k3"], v["k4"])
        out["VT"] = None if out["VND"] is None or out["BPnd"] is None else out["VND"] * (1.0 + out["BPnd"])
    elif key == "2TCi":
        out["Ki"] = _ratio(v["K1"] * v["k3"], v["k2"] + v["k3"])
    elif key == "SRTM":
        out["k2'"] = _ratio(v["k2"], v["R1"])
    return out


def _ratio(a, b):
    return float(a) / float(b) if b and abs(b) > 1e-12 and math.isfinite(a) and math.isfinite(b) else None


# ---------------------------------------------------------------------------
# Least squares
# ---------------------------------------------------------------------------
def levenbergMarquardt(residuals, initial, lower, upper, maxIterations=200, tolerance=1e-10):
    """Minimum of sum(residuals(p) ** 2) with lower <= p <= upper (projected Levenberg-Marquardt, forward difference
    Jacobian). Returns (p, cost)."""
    lower, upper = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    p = np.clip(np.asarray(initial, dtype=float), lower, upper)
    r = residuals(p)
    cost = float(r @ r)
    damping = 1e-3
    for _ in range(maxIterations):
        jacobian = np.empty((r.size, p.size))
        for j in range(p.size):
            h = 1e-6 * max(abs(p[j]), 1e-3)
            step = -h if p[j] + h > upper[j] else h
            q = p.copy()
            q[j] += step
            jacobian[:, j] = (residuals(q) - r) / step
        normal = jacobian.T @ jacobian
        gradient = jacobian.T @ r
        diagonal = np.diag(normal).copy()
        diagonal[diagonal <= 0] = 1e-12
        improved = False
        while damping < 1e12:
            try:
                delta = np.linalg.solve(normal + damping * np.diag(diagonal), -gradient)
            except np.linalg.LinAlgError:
                damping *= 10.0
                continue
            candidate = np.clip(p + delta, lower, upper)
            rc = residuals(candidate)
            newCost = float(rc @ rc)
            if np.isfinite(newCost) and newCost < cost:
                change = cost - newCost
                p, r, cost = candidate, rc, newCost
                damping = max(damping / 10.0, 1e-12)
                improved = True
                break
            damping *= 10.0
        if not improved or change <= tolerance * max(cost, 1e-30):
            break
    return p, cost


def fitMetrics(observed, fitted, weights, parameterCount):
    """R2, RMSE, NRMSE (RMSE / mean of observed), AIC and n of the points with a weight above 0."""
    observed, fitted, weights = (np.asarray(a, dtype=float) for a in (observed, fitted, weights))
    used = (weights > 0) & np.isfinite(observed) & np.isfinite(fitted)
    n = int(used.sum())
    if n == 0:
        return {"R2": None, "RMSE": None, "NRMSE": None, "AIC": None, "n": 0}
    residual = observed[used] - fitted[used]
    ssRes = float(residual @ residual)
    ssTot = float(np.sum((observed[used] - observed[used].mean()) ** 2))
    rmse = math.sqrt(ssRes / n)
    mean = float(observed[used].mean())
    wss = float(np.sum(weights[used] * residual ** 2))
    return {"R2": 1.0 - ssRes / ssTot if ssTot > 0 else None, "RMSE": rmse,
            "NRMSE": rmse / abs(mean) if mean else None,
            "AIC": n * math.log(wss / n) + 2 * parameterCount if wss > 0 else None, "n": n}


def _parameterList(key, parameters):
    """Model parameters with the user's changes (a list of Parameter or of dicts, or {name: dict}) applied."""
    base = {p.name: p.copy() for p in model(key).parameters}
    if parameters:
        items = parameters.values() if isinstance(parameters, dict) else parameters
        for item in items:
            data = item.asDict() if isinstance(item, Parameter) else dict(item)
            name = data.get("name")
            if name in base:
                for field in ("initial", "lower", "upper", "fitted"):
                    if data.get(field) is not None:
                        setattr(base[name], field, data[field] if field == "fitted" else float(data[field]))
    out = [base[p.name] for p in model(key).parameters]
    for p in out:
        p.lower, p.upper = min(p.lower, p.upper), max(p.lower, p.upper)
        p.initial = min(max(p.initial, p.lower), p.upper)
    return out


def _starts(key, initial, names, count):
    """Starting points of a compartment fit: the user's, then the rates scaled."""
    starts = [initial]
    factors = [(0.5, 2.0), (2.0, 0.5), (0.25, 1.0), (1.0, 0.25), (4.0, 4.0)]
    for f in factors[:max(0, count - 1)]:
        start = initial.copy()
        for i, name in enumerate(names):
            if name in ("k3", "R1"):
                start[i] *= f[0]
            elif name in ("k4", "k2", "BPnd"):
                start[i] *= f[1]
        starts.append(start)
    return starts


# ---------------------------------------------------------------------------
# Fits
# ---------------------------------------------------------------------------
def fit(key, frames, tissue, inputFunction=None, reference=None, blood=None, options=None):
    """Fit a model to a TAC (frame means). key: a MODELS key. inputFunction: FrameInput / SampledInput (plasma
    models); reference: FrameInput (reference models); blood: whole blood frame means for vB (default the input).
    Returns {"model", "values" (reported values, None when unknown), "parameters" (fitted ones), "metrics",
    "fitted" (frame means of the model or None), "curve" ((times, values) of the model on a coarse grid, or None),
    "graphical" (x, y, used, slope, intercept, tStar, xLabel, yLabel) or None, "weights", "warnings"}."""
    spec = model(key)
    opts = defaultOptions()
    opts.update(options or {})
    tissue = np.asarray(tissue, dtype=float)
    if tissue.shape != (len(frames),):
        raise ValueError("The tissue curve needs one value per frame")
    if spec.inputKind == "plasma" and inputFunction is None:
        raise ValueError(f"{spec.title} needs an input function (blood pool or plasma curve)")
    if spec.inputKind == "reference" and reference is None:
        raise ValueError(f"{spec.title} needs a reference region")
    weights = frames.weights(opts["weighting"], tissue)
    weights = np.where(np.isfinite(tissue), weights, 0.0)
    if spec.method == "compartment":
        result = _fitCompartment(key, frames, tissue, inputFunction, blood, weights, opts)
    elif spec.method == "srtm":
        result = _fitSrtm(frames, tissue, reference, weights, opts)
    elif spec.method == "graphical":
        result = _fitGraphical(key, frames, tissue, inputFunction, reference, weights, opts)
    else:
        result = _ratio_fit(frames, tissue, reference, opts)
    result["model"] = key
    result["weights"] = weights
    result.setdefault("warnings", [])
    if spec.inputKind == "plasma" and isinstance(inputFunction, FrameInput):
        warning = inputCurveWarning(frames, inputFunction.values)
        if warning:
            result["warnings"].append(warning)
    result["values"] = {name: _finite(result["values"].get(name)) for name in spec.reports}
    return result


def _finite(value):
    return float(value) if value is not None and np.isfinite(value) else None


def _coarseCurve(grid, curve, points=400):
    stride = max(1, grid.size // points)
    return grid[::stride].copy(), np.asarray(curve)[::stride].copy()


def _fitCompartment(key, frames, tissue, inputFunction, blood, weights, opts):
    params = _parameterList(key, opts.get("parameters"))
    grid = frames.grid()
    bloodGrid = (FrameInput(frames, blood).onGrid(grid) if blood is not None else None)
    free = [p for p in params if p.fitted]
    fixed = {p.name: p.initial for p in params}
    names = [p.name for p in free]
    sqrtW = np.sqrt(weights)
    cache = {}

    def curveFor(values):
        delay = values.get("delay", 0.0)
        entry = cache.get(delay)
        if entry is None:
            plasma = inputFunction.onGrid(grid, delay)
            entry = (_Convolver(grid, plasma), bloodGrid if bloodGrid is not None else plasma)
            if len(cache) > 64:
                cache.clear()
            cache[delay] = entry
        convolver, wholeBlood = entry
        return compartmentCurve(values, grid, convolver, wholeBlood, key)

    def valuesOf(vector):
        values = dict(fixed)
        values.update(zip(names, vector))
        return values

    def residuals(vector):
        means = frames.frameMeans(grid, curveFor(valuesOf(vector)))
        out = sqrtW * (means - np.nan_to_num(tissue))
        return np.nan_to_num(out, nan=1e30)

    warnings = []
    if free:
        initial = np.array([p.initial for p in free])
        lower, upper = [p.lower for p in free], [p.upper for p in free]
        best = None
        for start in _starts(key, initial, names, int(opts.get("starts") or 1)):
            p, cost = levenbergMarquardt(residuals, np.clip(start, lower, upper), lower, upper)
            if best is None or cost < best[1]:
                best = (p, cost)
        vector = best[0]
        atBound = [n for n, v, lo, hi in zip(names, vector, lower, upper)
                   if (abs(v - lo) < 1e-9 * max(1, abs(lo)) and lo != 0) or abs(v - hi) < 1e-9 * max(1, abs(hi))]
        if atBound:
            warnings.append("at a bound: " + ", ".join(atBound))
    else:
        vector = np.array([])
    values = valuesOf(vector)
    curve = curveFor(values)
    fitted = frames.frameMeans(grid, curve)
    values.update(derivedValues(key, values))
    if not any(p.name == "delay" and p.fitted for p in params):
        values["delay"] = values.get("delay", 0.0)
    return {"values": values, "parameters": {n: float(v) for n, v in zip(names, vector)},
            "metrics": fitMetrics(tissue, fitted, weights, len(free)), "fitted": fitted,
            "curve": _coarseCurve(grid, curve), "graphical": None, "warnings": warnings}


# --- SRTM ------------------------------------------------------------------------------------------
BASIS_RATES = np.geomspace(0.006, 1.2, 80)     # 1/min, k2a = k2 / (1 + BPnd) of the basis functions


def basisFit(design, data, weights, basis):
    """Basis function fit (Gunn et al. 1997) of data (frames, or frames x voxels): for every basis column B the
    weighted linear fit of data ~ design @ [a] + b B; the basis with the smallest residual is kept. design: frames x
    m fixed columns; basis: frames x nBasis. Returns (coefficients (m + 1) x voxels, index of the basis, wss)."""
    data = np.asarray(data, dtype=float)
    single = data.ndim == 1
    if single:
        data = data[:, None]
    sqrtW = np.sqrt(weights)[:, None]
    use = weights > 0
    yw = (sqrtW * np.nan_to_num(data))[use]
    best = np.full(data.shape[1], np.inf)
    bestIndex = np.zeros(data.shape[1], dtype=int)
    bestCoef = np.zeros((design.shape[1] + 1, data.shape[1]))
    for b in range(basis.shape[1]):
        x = np.column_stack([design, basis[:, b]])
        xw = (sqrtW * x)[use]
        pinv = np.linalg.pinv(xw)
        coef = pinv @ yw
        wss = np.sum((yw - xw @ coef) ** 2, axis=0)
        better = wss < best
        best[better], bestIndex[better] = wss[better], b
        bestCoef[:, better] = coef[:, better]
    if single:
        return bestCoef[:, 0], int(bestIndex[0]), float(best[0])
    return bestCoef, bestIndex, best


def _referenceBasis(frames, reference, rates=BASIS_RATES):
    grid = frames.grid()
    refGrid = reference.onGrid(grid)
    convolver = _Convolver(grid, refGrid)
    basis = np.column_stack([frames.frameMeans(grid, convolver.exp(rate)) for rate in rates])
    return grid, refGrid, convolver, basis


def srtmBasis(frames, data, reference, weights, rates=BASIS_RATES):
    """SRTM by basis functions for data (frames, or frames x voxels): {"R1", "k2", "BPnd"} (arrays for voxels)."""
    _, _, _, basis = _referenceBasis(frames, reference, rates)
    design = reference.frameMeans()[:, None]
    coef, index, _ = basisFit(design, data, weights, basis)
    theta3 = rates[index]
    R1, theta2 = coef[0], coef[1]
    k2 = theta2 + R1 * theta3
    with np.errstate(divide="ignore", invalid="ignore"):
        bp = k2 / theta3 - 1.0
    return {"R1": R1, "k2": k2, "BPnd": bp}


def _fitSrtm(frames, tissue, reference, weights, opts):
    params = _parameterList("SRTM", opts.get("parameters"))
    grid, refGrid, convolver, _ = _referenceBasis(frames, reference, BASIS_RATES[:1])
    sqrtW = np.sqrt(weights)
    free = [p for p in params if p.fitted]
    names = [p.name for p in free]
    fixed = {p.name: p.initial for p in params}
    def curveFor(values):
        return srtmCurve(values, refGrid, convolver)

    def valuesOf(vector):
        values = dict(fixed)
        values.update(zip(names, vector))
        return values

    def residuals(vector):
        means = frames.frameMeans(grid, curveFor(valuesOf(vector)))
        return np.nan_to_num(sqrtW * (means - np.nan_to_num(tissue)), nan=1e30)

    warnings = []
    starts = []
    if free:
        initial = np.array([p.initial for p in free])
        lower, upper = [p.lower for p in free], [p.upper for p in free]
        if all(p.fitted for p in params):
            guess = srtmBasis(frames, tissue, reference, weights)
            starts.append(np.array([guess[n] for n in names], dtype=float))
        starts += _starts("SRTM", initial, names, int(opts.get("starts") or 1))
        best = None
        for start in starts:
            if not np.all(np.isfinite(start)):
                continue
            p, cost = levenbergMarquardt(residuals, np.clip(start, lower, upper), lower, upper)
            if best is None or cost < best[1]:
                best = (p, cost)
        vector = best[0]
        atBound = [n for n, v, lo, hi in zip(names, vector, lower, upper)
                   if abs(v - lo) < 1e-9 * max(1, abs(lo)) or abs(v - hi) < 1e-9 * max(1, abs(hi))]
        if atBound:
            warnings.append("at a bound: " + ", ".join(atBound))
    else:
        vector = np.array([])
    values = valuesOf(vector)
    curve = curveFor(values)
    fitted = frames.frameMeans(grid, curve)
    values.update(derivedValues("SRTM", values))
    return {"values": values, "parameters": {n: float(v) for n, v in zip(names, vector)},
            "metrics": fitMetrics(tissue, fitted, weights, len(free)), "fitted": fitted,
            "curve": _coarseCurve(grid, curve), "graphical": None, "warnings": warnings}


# --- graphical methods ------------------------------------------------------------------------------
def linearFit(x, y, weights):
    """Weighted straight line y = slope x + intercept: (slope, intercept), None when fewer than 2 points."""
    use = (weights > 0) & np.isfinite(x) & np.isfinite(y)
    if use.sum() < 2:
        return None
    w, xs, ys = weights[use], x[use], y[use]
    sw = w.sum()
    mx, my = (w * xs).sum() / sw, (w * ys).sum() / sw
    sxx = (w * (xs - mx) ** 2).sum()
    if sxx <= 0:
        return None
    slope = (w * (xs - mx) * (ys - my)).sum() / sxx
    return float(slope), float(my - slope * mx)


def autoTStar(frames, x, y, weights, maxError=DEFAULT_MAX_ERROR, minPoints=MIN_LINEAR_POINTS):
    """Index of the first frame of the linear part: the earliest frame from which a line through it and all later
    frames has no point off by more than maxError % (relative to the point); when there is none, the start with the
    smallest such error (at least minPoints points)."""
    usable = np.flatnonzero((weights > 0) & np.isfinite(x) & np.isfinite(y))
    best = None
    for position, first in enumerate(usable):
        points = usable[position:]
        if points.size < minPoints:
            break
        mask = np.zeros_like(weights)
        mask[points] = weights[points]
        line = linearFit(x, y, mask)
        if line is None:
            continue
        predicted = line[0] * x[points] + line[1]
        error = float(np.max(np.abs(predicted - y[points]) / np.maximum(np.abs(y[points]), 1e-12))) * 100.0
        if error <= maxError:
            return int(first)
        if best is None or error < best[1]:
            best = (int(first), error)
    if best is not None:
        return best[0]
    return int(usable[max(0, usable.size - minPoints)]) if usable.size else 0


def graphicalAxes(key, frames, tissue, inputFunction=None, reference=None, k2ref=None):
    """x and y of the Patlak / Logan plot for a TAC (frames) or frames x voxels, with their labels."""
    tissue = np.asarray(tissue, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        if key == "Patlak":
            cp = inputFunction.frameMeans()
            integral = inputFunction.integralAtMid()
            cp = cp.reshape((-1,) + (1,) * (tissue.ndim - 1))
            integral = integral.reshape(cp.shape)
            x = np.broadcast_to(integral / cp, tissue.shape).copy()
            y = tissue / cp
            labels = ("∫Cp / Cp (min)", "Ct / Cp")
        elif key == "Logan":
            integralTissue = frames.integralAtMid(tissue)
            integral = inputFunction.integralAtMid().reshape((-1,) + (1,) * (tissue.ndim - 1))
            x = integral / tissue
            y = integralTissue / tissue
            labels = ("∫Cp / Ct (min)", "∫Ct / Ct (min)")
        else:
            integralTissue = frames.integralAtMid(tissue)
            ref = reference.frameMeans()
            integral = reference.integralAtMid()
            if k2ref:
                integral = integral + ref / float(k2ref)
            integral = integral.reshape((-1,) + (1,) * (tissue.ndim - 1))
            x = integral / tissue
            y = integralTissue / tissue
            labels = ("(∫Cref + Cref/k2') / Ct (min)" if k2ref else "∫Cref / Ct (min)", "∫Ct / Ct (min)")
    return x, y, labels


def _tStarMask(frames, tStar, weights):
    return np.where(frames.mid >= tStar - 1e-9, weights, 0.0)


def _fitGraphical(key, frames, tissue, inputFunction, reference, weights, opts):
    x, y, labels = graphicalAxes(key, frames, tissue, inputFunction, reference, opts.get("k2ref"))
    usable = np.where(np.isfinite(x) & np.isfinite(y), weights, 0.0)
    warnings = []
    if opts.get("autoTStar"):
        first = autoTStar(frames, x, y, usable, float(opts.get("maxError") or DEFAULT_MAX_ERROR))
        tStar = float(frames.mid[first])
    else:
        tStar = float(opts.get("tStar") if opts.get("tStar") is not None else DEFAULT_TSTAR_MIN)
    used = _tStarMask(frames, tStar, usable)
    if np.count_nonzero(used) < MIN_LINEAR_POINTS:
        warnings.append(f"fewer than {MIN_LINEAR_POINTS} frames after t*")
    line = linearFit(x, y, used)
    slope, intercept = line if line else (None, None)
    values = {"t*": tStar}
    fitted = None
    curve = None
    if key == "Patlak":
        values.update({"Ki": slope, "V0": intercept})
        if line:
            cp = inputFunction.frameMeans()
            fitted = np.where(used > 0, slope * inputFunction.integralAtMid() + intercept * cp, np.nan)
    elif key == "Logan":
        values.update({"VT": slope, "intercept": intercept})
    else:
        values.update({"DVR": slope, "BPnd": slope - 1.0 if slope is not None else None, "intercept": intercept})
    linePoints = slope * x + intercept if line else np.full_like(x, np.nan)
    metrics = fitMetrics(y, linePoints, used, 2)
    return {"values": values, "parameters": {}, "metrics": metrics, "fitted": fitted, "curve": curve,
            "graphical": {"x": x, "y": y, "used": used > 0, "slope": slope, "intercept": intercept, "tStar": tStar,
                          "xLabel": labels[0], "yLabel": labels[1]},
            "warnings": warnings}


# --- SUVR ---------------------------------------------------------------------------------------------
def defaultWindow(frames, minutes=30.0):
    """(start, end) minutes of the SUVR window: the last 30 min of the series (all of it when shorter)."""
    end = frames.lastEnd
    return max(0.0, end - minutes), end


def windowWeights(frames, window):
    """Duration of every frame inside the window (minutes)."""
    start, end = window
    overlap = np.clip(np.minimum(frames.end, end) - np.maximum(frames.start, start), 0.0, None)
    return np.where(frames.valid, overlap, 0.0)


def _ratio_fit(frames, tissue, reference, opts):
    window = tuple(opts.get("window") or defaultWindow(frames))
    w = windowWeights(frames, window)
    warnings = []
    if w.sum() <= 0:
        warnings.append("no frame in the time window")
        return {"values": {}, "parameters": {}, "metrics": dict.fromkeys(METRICS), "fitted": None, "curve": None,
                "graphical": None, "warnings": warnings, "window": window}
    target = float(np.nansum(w * tissue) / w.sum())
    ref = float(np.nansum(w * reference.frameMeans()) / w.sum())
    suvr = _ratio(target, ref)
    return {"values": {"SUVR": suvr, "BPnd": suvr - 1.0 if suvr is not None else None}, "parameters": {},
            "metrics": dict.fromkeys(METRICS), "fitted": None, "curve": None, "graphical": None,
            "warnings": warnings, "window": window}


# ---------------------------------------------------------------------------
# Voxel-wise maps
# ---------------------------------------------------------------------------
VOXEL_MAPS = {"1TC": ("K1", "k2", "vB", "VT"), "Patlak": ("Ki", "V0"), "Logan": ("VT",),
              "LoganRef": ("DVR", "BPnd"), "SRTM": ("BPnd", "R1", "k2"), "SUVR": ("SUVR",)}


def parametricMaps(key, frames, data, inputFunction=None, reference=None, blood=None, options=None):
    """Voxel-wise values of a model for data (frames x voxels): {name: array of voxels} (VOXEL_MAPS[key]). The
    compartment models with more than one tissue compartment have no voxel-wise version. The graphical methods
    use options["tStar"] (fixed, minutes), the same for every voxel. With options["clipNegative"] (default) negative
    values (noise in voxels with little uptake, not physical for these parameters) are set to 0."""
    spec = model(key)
    if key not in VOXEL_MAPS:
        raise ValueError(f"{spec.title} has no voxel-wise version (use the region fit)")
    opts = defaultOptions()
    opts.update(options or {})
    data = np.asarray(data, dtype=float)
    if data.ndim != 2 or data.shape[0] != len(frames):
        raise ValueError("The voxel data must be frames x voxels")
    mode = opts["weighting"] if opts["weighting"] != WEIGHT_COUNTS else WEIGHT_DURATION
    weights = frames.weights(mode)
    with np.errstate(divide="ignore", invalid="ignore"):
        if key == "SRTM":
            values = srtmBasis(frames, data, reference, weights)
        elif key == "1TC":
            grid = frames.grid()
            plasma = _Convolver(grid, inputFunction.onGrid(grid))
            basis = np.column_stack([frames.frameMeans(grid, plasma.exp(rate)) for rate in BASIS_RATES])
            wholeBlood = blood if blood is not None else inputFunction.frameMeans()
            coef, index, _ = basisFit(np.asarray(wholeBlood, dtype=float)[:, None], data, weights, basis)
            vB, scaled = coef[0], coef[1]
            k2 = BASIS_RATES[index]
            K1 = scaled / (1.0 - vB)
            values = {"K1": K1, "k2": k2, "vB": vB, "VT": K1 / k2}
        elif key == "SUVR":
            window = tuple(opts.get("window") or defaultWindow(frames))
            w = windowWeights(frames, window)
            target = (w[:, None] * np.nan_to_num(data)).sum(axis=0) / w.sum()
            ref = float((w * reference.frameMeans()).sum() / w.sum())
            values = {"SUVR": target / ref}
        else:
            x, y, _ = graphicalAxes(key, frames, data, inputFunction, reference, opts.get("k2ref"))
            used = _tStarMask(frames, float(opts["tStar"]), weights)[:, None] * np.isfinite(x) * np.isfinite(y)
            x, y = np.nan_to_num(x), np.nan_to_num(y)
            sw = used.sum(axis=0)
            mx = (used * x).sum(axis=0) / sw
            my = (used * y).sum(axis=0) / sw
            sxx = (used * (x - mx) ** 2).sum(axis=0)
            slope = (used * (x - mx) * (y - my)).sum(axis=0) / sxx
            intercept = my - slope * mx
            values = {"Patlak": {"Ki": slope, "V0": intercept}, "Logan": {"VT": slope},
                      "LoganRef": {"DVR": slope, "BPnd": slope - 1.0}}[key]
    out = {}
    for name in VOXEL_MAPS[key]:
        array = np.asarray(values[name], dtype=float)
        if array.ndim == 0:
            array = np.full(data.shape[1], float(array))
        array = np.where(np.isfinite(array), array, 0.0)
        if opts.get("clipNegative"):
            array = np.maximum(array, 0.0)
        out[name] = array.astype(np.float32)
    return out


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def resultColumns(key):
    """Columns of a results table of a model: the reported values, then the fit metrics."""
    spec = model(key)
    metrics = () if spec.method == "ratio" else METRICS
    return list(spec.reports) + list(metrics)


def columnTitle(name):
    unit = UNITS.get(name)
    return f"{name} ({unit})" if unit else name


def resultRow(result):
    """Values of a fit() result in the order of resultColumns."""
    out = []
    for name in resultColumns(result["model"]):
        value = result["values"].get(name) if name in result["values"] else result["metrics"].get(name)
        out.append(value)
    return out


def formatNumber(value, digits=4):
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return ""
    if value == 0:
        return "0"
    if abs(value) >= 1e5 or abs(value) < 1e-4:
        return f"{value:.{digits - 1}e}"
    return f"{value:.{digits}g}"


def resultsCsv(rows, key, extraColumns=(), delimiter=","):
    """CSV text of results: rows are (name, result, extra values); columns: Region, extra columns, the model's
    values and metrics, Warnings."""
    out = io.StringIO()
    writer = csv.writer(out, delimiter=delimiter, lineterminator="\n")
    writer.writerow(["Region", *extraColumns, *[columnTitle(c) for c in resultColumns(key)], "Warnings"])
    for name, result, extra in rows:
        values = ["" if v is None else f"{v:.6g}" for v in resultRow(result)]
        writer.writerow([name, *extra, *values, "; ".join(result.get("warnings") or [])])
    return out.getvalue()


def tacCsv(frames, curves, units=""):
    """CSV text of TACs: frame start, end, middle (min from time zero) and every curve ({name: values})."""
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    suffix = f" ({units})" if units else ""
    writer.writerow(["Start (min)", "End (min)", "Mid (min)", *[f"{n}{suffix}" for n in curves]])
    for i in range(len(frames)):
        writer.writerow([f"{frames.rawStart[i]:.6g}", f"{frames.rawEnd[i]:.6g}", f"{frames.mid[i]:.6g}",
                         *[f"{float(v[i]):.6g}" for v in curves.values()]])
    return out.getvalue()
