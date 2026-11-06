"""Review/Edit frame editing (numpy only, no Slicer).

The edited series is a list of frames; each frame is a list of atoms, the frames as they were acquired:
    {"key", "start", "duration", "label", "origin"}
    key       unique text ("o3": frame 3 of the original series, "a1.2": frame 2 of the first added series)
    start     seconds on the editor's time axis (0 = the start of the first original frame)
    duration  seconds
    label     text for the user ("F3", "Late static F1")
    origin    "original" or "added"
A frame of one atom is the atom itself; a merged frame has several, and its image is their duration weighted
mean. Every atom also has a scale (FrameEdit.scales, 1 by default): the decay correction factor that keeps the
values of an added series continuous with the others (Bq/ml with different decay reference times).

Splitting a merged frame gives its atoms back; the original series is never changed (FrameEdit only holds keys).
Every operation (add, merge, split, remove, discard) can be undone and redone.

compareSeries checks a series to add against the edited series: units, tracer, decay correction (and the factor
that aligns the two), patient, scanner, the time between them.
"""

import datetime
import math

import numpy as np

ORIGINAL, ADDED = "original", "added"
GAP_TOLERANCE_S = 1.0               # frames closer than this are back to back; overlaps up to this are allowed
DEFAULT_LONG_GAP_S = 30 * 60.0      # warn when the series are further apart than this
INJECTION_TOLERANCE_S = 60.0
HALF_LIFE_TOLERANCE = 0.005


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def parseIso(text):
    """datetime of an ISO text (None when empty or unreadable)."""
    if isinstance(text, datetime.datetime):
        return text
    if not text:
        return None
    try:
        return datetime.datetime.fromisoformat(str(text).strip())
    except ValueError:
        return None


def number(value):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def unitsKind(units):
    """"SUV" for SUVbw / SUVlbm / SUVbsa, "Bq/ml" for activity concentration, None when unknown."""
    text = str(units or "")
    if text.startswith("SUV"):
        return "SUV"
    if text.lower() in ("bq/ml", "bqml"):
        return "Bq/ml"
    return None


def formatDuration(seconds):
    if seconds is None:
        return "?"
    seconds = int(round(seconds))
    hours, rest = divmod(abs(seconds), 3600)
    minutes, secs = divmod(rest, 60)
    sign = "-" if seconds < 0 else ""
    return f"{sign}{hours}:{minutes:02d}:{secs:02d}" if hours else f"{sign}{minutes}:{secs:02d}"


def formatClock(stamp):
    return stamp.strftime("%Y-%m-%d %H:%M:%S") if stamp else "?"


def _round(value, digits=3):
    if value is None:
        return None
    value = round(float(value), digits)
    return int(value) if value.is_integer() else value


# ---------------------------------------------------------------------------
# Series description and comparison
# ---------------------------------------------------------------------------
SERIES_KEYS = ("name", "kind", "units", "frames", "injection", "halfLife", "radionuclide", "decayCorrection",
               "decayCorrected", "decayReference", "patientName", "patientID", "manufacturer", "model", "serial",
               "station")


def seriesInfo(**values):
    """A series description for compareSeries; frames: [{"start": datetime, "duration": s}] (clock times)."""
    info = {key: None for key in SERIES_KEYS}
    info.update(values)
    info["frames"] = list(info["frames"] or [])
    return info


def seriesStart(info):
    starts = [f["start"] for f in info["frames"] if f.get("start") is not None]
    return min(starts) if starts else None


def seriesEnd(info):
    ends = [f["start"] + datetime.timedelta(seconds=f["duration"]) for f in info["frames"]
            if f.get("start") is not None and f.get("duration") is not None]
    return max(ends) if ends else None


def totalDuration(info):
    durations = [f.get("duration") for f in info["frames"]]
    return sum(durations) if durations and all(d is not None for d in durations) else None


def decayText(info):
    if not info.get("decayCorrected"):
        return "not decay corrected"
    correction = info.get("decayCorrection") or "?"
    return f"{correction} to {formatClock(info.get('decayReference'))}" if info.get("decayReference") else correction


def _differs(a, b):
    a, b = str(a or "").strip(), str(b or "").strip()
    return bool(a and b and a.lower() != b.lower())


def compareSeries(base, added, longGapS=DEFAULT_LONG_GAP_S):
    """Can `added` join `base` (the edited series), and how. Returns
        {"errors": [...]         reasons it cannot,
         "warnings": [...]       things the user should confirm (patient, scanner, time between, injection),
         "rows": [(title, base text, added text)]   for the dialog,
         "reference": datetime   decay reference of the result (Bq/ml, both decay corrected), else None,
         "scaleBase", "scaleAdded": factors applied to the base and the added frames,
         "note": text            what is done with the decay correction}
    SUV images are decay independent (each one's dose is decayed to its own reference): both stay as they are.
    Bq/ml images decay corrected to different times are put on the reference of the series that starts first:
    the other is multiplied by 2^((its reference - that reference) / half life)."""
    errors, warnings = [], []
    out = {"errors": errors, "warnings": warnings, "reference": None, "scaleBase": 1.0, "scaleAdded": 1.0,
           "note": ""}
    baseStart, addedStart = seriesStart(base), seriesStart(added)
    baseKind, addedKind = unitsKind(base["units"]), unitsKind(added["units"])
    if baseKind is None or addedKind is None:
        errors.append("The units of " + ("the added series" if baseKind else "the edited series") + " are unknown.")
    elif baseKind != addedKind:
        errors.append(f"One series is in SUV, the other in Bq/ml ({base['units']} / {added['units']}).")
    elif base["units"] != added["units"]:
        errors.append(f"Different SUV normalisations ({base['units']} / {added['units']}).")
    if baseStart is None or addedStart is None:
        errors.append("The acquisition (clock) time of " + ("the added series" if baseStart else "the edited series")
                      + " is unknown, so it cannot be placed on the time axis.")
    if any(f.get("duration") is None or f["duration"] <= 0 for f in added["frames"]):
        errors.append("A frame duration of the added series is unknown.")
    halfLives = (number(base["halfLife"]), number(added["halfLife"]))
    if _differs(base["radionuclide"], added["radionuclide"]):
        errors.append(f"Different radionuclides ({base['radionuclide']} / {added['radionuclide']}).")
    elif all(halfLives) and abs(halfLives[0] - halfLives[1]) > HALF_LIFE_TOLERANCE * halfLives[0]:
        errors.append(f"Different half lives ({halfLives[0]:g} s / {halfLives[1]:g} s).")

    if not errors and baseKind == "Bq/ml":
        if bool(base["decayCorrected"]) != bool(added["decayCorrected"]):
            errors.append("One series is decay corrected and the other is not; their Bq/ml values cannot be joined.")
        elif base["decayCorrected"]:
            references = (base["decayReference"], added["decayReference"])
            halfLife = halfLives[0] or halfLives[1]
            if not all(references):
                errors.append("The decay correction time of " + ("the added series" if references[0] else
                                                                 "the edited series") + " is unknown.")
            elif not halfLife:
                errors.append("The half life is unknown, so the decay correction cannot be aligned.")
            else:
                first = base if baseStart <= addedStart else added
                reference = first["decayReference"]
                out["reference"] = reference
                for key, info in (("scaleBase", base), ("scaleAdded", added)):
                    out[key] = 2.0 ** ((info["decayReference"] - reference).total_seconds() / halfLife)
                other, scale = (added, out["scaleAdded"]) if first is base else (base, out["scaleBase"])
                if abs(scale - 1.0) < 1e-9:
                    out["note"] = f"Both are decay corrected to {formatClock(reference)}: no change."
                else:
                    which = "added" if other is added else "edited"
                    out["note"] = (f"Bq/ml, decay corrected to different times: the {which} frames are decay corrected "
                                   f"to {formatClock(reference)} (the reference of the series that starts first), "
                                   f"x {scale:.6g}.")
        else:
            out["note"] = "Neither series is decay corrected: the values are joined as stored."
    elif not errors:
        out["note"] = "SUV: each series is normalised to its own decayed dose, so the values join as they are."

    if not errors:
        baseEnd, addedEnd = seriesEnd(base), seriesEnd(added)
        if baseEnd and addedEnd:
            gap = (addedStart - baseEnd).total_seconds() if addedStart >= baseStart else \
                (baseStart - addedEnd).total_seconds()
            if gap > longGapS:
                warnings.append(f"{formatDuration(gap)} between the two series (more than "
                                f"{formatDuration(longGapS)}).")
    if _differs(base["patientID"], added["patientID"]):
        warnings.append(f"Different patient IDs ({base['patientID']} / {added['patientID']}).")
    if _differs(base["patientName"], added["patientName"]):
        warnings.append(f"Different patient names ({base['patientName']} / {added['patientName']}).")
    devices = [(title, base[key], added[key]) for title, key in
               (("manufacturer", "manufacturer"), ("model", "model"), ("serial number", "serial"),
                ("station", "station")) if _differs(base[key], added[key])]
    if devices:
        warnings.append("Different scanner: " + ", ".join(f"{t} {a} / {b}" for t, a, b in devices) + ".")
    injections = (base["injection"], added["injection"])
    if all(injections) and abs((injections[0] - injections[1]).total_seconds()) > INJECTION_TOLERANCE_S:
        warnings.append(f"Different injection times ({formatClock(injections[0])} / {formatClock(injections[1])}): "
                        "two injections?")
    out["rows"] = comparisonRows(base, added)
    return out


def _scanner(info):
    parts = [str(info.get(k) or "").strip() for k in ("manufacturer", "model")]
    text = " ".join(p for p in parts if p) or "?"
    extra = [f"{t} {info[k]}" for t, k in (("SN", "serial"), ("station", "station")) if info.get(k)]
    return text + (f" ({', '.join(extra)})" if extra else "")


def comparisonRows(base, added):
    """(title, edited series, added series) rows for the Add frame dialog."""
    def patient(info):
        return " / ".join(str(info.get(k) or "?") for k in ("patientName", "patientID"))

    def halfLife(info):
        value = number(info.get("halfLife"))
        return f"{value:g} s" + (f" ({info['radionuclide']})" if info.get("radionuclide") else "") if value else "?"

    rows = [("Series", base["name"], added["name"]),
            ("Type", base.get("kind") or "?", added.get("kind") or "?"),
            ("Patient (name / ID)", patient(base), patient(added)),
            ("Units", base["units"] or "?", added["units"] or "?"),
            ("Injection time", formatClock(base["injection"]), formatClock(added["injection"])),
            ("Acquisition start", formatClock(seriesStart(base)), formatClock(seriesStart(added))),
            ("Acquisition end", formatClock(seriesEnd(base)), formatClock(seriesEnd(added))),
            ("Acquisition time (sum of frames)", formatDuration(totalDuration(base)),
             formatDuration(totalDuration(added))),
            ("Frames", str(len(base["frames"])), str(len(added["frames"]))),
            ("Decay correction (Bq/ml, SUV reference)", decayText(base), decayText(added)),
            ("Half life", halfLife(base), halfLife(added)),
            ("Scanner", _scanner(base), _scanner(added))]
    return rows


# ---------------------------------------------------------------------------
# The edited frame list
# ---------------------------------------------------------------------------
class EditError(ValueError):
    pass


class FrameEdit:
    """The frames of the edited series. originals: atoms of the original series, in order."""

    def __init__(self, originals):
        self.originals = [dict(atom, origin=ORIGINAL) for atom in originals]
        self.atoms = {atom["key"]: atom for atom in self.originals}
        self.scales = {}
        self.addedSeries = []        # names of the added series, in order (for the keys and the notes)
        self.reference = None        # decay reference after an alignment (None: that of the original)
        self.frames = []
        self.undoStack, self.redoStack = [], []
        self.addCount = 0            # added series so far, never undone: atom keys are never used twice
        self.reset()

    # --- state -----------------------------------------------------------------------------------
    def reset(self):
        """Back to the original series, forgetting the history too."""
        self._clear()
        self.undoStack, self.redoStack = [], []

    def _clear(self):
        self.atoms = {atom["key"]: atom for atom in self.originals}
        self.frames = [[atom["key"]] for atom in self.originals]
        self.scales = {}
        self.addedSeries = []
        self.reference = None

    # --- undo / redo ------------------------------------------------------------------------------
    UNDO_LIMIT = 100

    def _state(self):
        return ([list(f) for f in self.frames], dict(self.scales), list(self.addedSeries), self.reference,
                dict(self.atoms))

    def _setState(self, state):
        frames, scales, added, reference, atoms = state
        self.frames, self.scales, self.addedSeries, self.reference = [list(f) for f in frames], dict(scales), \
            list(added), reference
        self.atoms = dict(atoms)

    def _remember(self, label):
        """Called by every operation just before it changes the frames."""
        self.undoStack.append((label, self._state()))
        del self.undoStack[:-self.UNDO_LIMIT]
        self.redoStack = []

    def canUndo(self):
        return bool(self.undoStack)

    def canRedo(self):
        return bool(self.redoStack)

    def undoText(self):
        return self.undoStack[-1][0] if self.undoStack else ""

    def redoText(self):
        return self.redoStack[-1][0] if self.redoStack else ""

    def undo(self):
        """Undo the last operation; returns its label (None when there is nothing to undo)."""
        if not self.undoStack:
            return None
        label, state = self.undoStack.pop()
        self.redoStack.append((label, self._state()))
        self._setState(state)
        return label

    def redo(self):
        if not self.redoStack:
            return None
        label, state = self.redoStack.pop()
        self.undoStack.append((label, self._state()))
        self._setState(state)
        return label

    def discard(self):
        """Back to the original series; can be undone."""
        if not self.isEdited():
            return
        self._remember("Discard changes")
        self._clear()

    def isEdited(self):
        return self.frames != [[atom["key"]] for atom in self.originals] or \
            any(abs(s - 1.0) > 1e-12 for s in self.scales.values())

    def scale(self, key):
        return self.scales.get(key, 1.0)

    def frameAtoms(self, index):
        return [self.atoms[key] for key in self.frames[index]]

    def isFrameEdited(self, index):
        keys = self.frames[index]
        return len(keys) > 1 or self.atoms[keys[0]]["origin"] != ORIGINAL or abs(self.scale(keys[0]) - 1.0) > 1e-12

    def isOriginalFrame(self, index):
        """The original frame number (index) when the frame is an untouched original frame, else None."""
        if self.isFrameEdited(index):
            return None
        return self.originals.index(self.atoms[self.frames[index][0]])

    def frameStart(self, index):
        return min(a["start"] for a in self.frameAtoms(index))

    def frameEnd(self, index):
        return max(a["start"] + a["duration"] for a in self.frameAtoms(index))

    def frameDuration(self, index):
        """Acquisition time of the frame: the sum of its atoms' durations (gaps do not count)."""
        return sum(a["duration"] for a in self.frameAtoms(index))

    def frameWeights(self, index):
        """(atom key, weight) of the duration weighted mean of a frame, the weights summing to 1, the scales
        not included."""
        atoms = self.frameAtoms(index)
        total = sum(a["duration"] for a in atoms)
        return [(a["key"], a["duration"] / total if total > 0 else 1.0 / len(atoms)) for a in atoms]

    def frameSources(self, index):
        """Text of what a frame is made of: 'F3', 'F3 + F4 (merged)', 'Late static F1 (added) x 1.03'."""
        atoms = self.frameAtoms(index)
        parts = []
        for atom in atoms:
            text = atom["label"]
            scale = self.scale(atom["key"])
            if abs(scale - 1.0) > 1e-9:
                text += f" x {scale:.4g}"
            parts.append(text)
        text = " + ".join(parts)
        if len(atoms) > 1:
            text += " (merged)"
        elif atoms[0]["origin"] == ADDED:
            text += " (added)"
        elif abs(self.scale(atoms[0]["key"]) - 1.0) > 1e-9:
            text += " (decay aligned)"
        return text

    def intervals(self, axis=None):
        """The frames as dynamicviews.frameIntervals items (index, number, start, end, duration) with "edited"
        and "sources"; the first carries timeBase / clockAtZero / injectionAt from axis (the original's first
        interval)."""
        out = []
        for i in range(len(self.frames)):
            out.append({"index": i, "number": i + 1, "start": self.frameStart(i), "end": self.frameEnd(i),
                        "duration": self.frameDuration(i), "edited": self.isFrameEdited(i),
                        "sources": self.frameSources(i)})
        if out and axis:
            for key in ("timeBase", "clockAtZero", "injectionAt"):
                out[0][key] = axis.get(key)
        return out

    # --- operations ----------------------------------------------------------------------------
    def overlaps(self, atoms):
        """[(new atom label, frame number)] of the new atoms that overlap a frame (by more than the tolerance)."""
        found = []
        for atom in atoms:
            start, end = atom["start"], atom["start"] + atom["duration"]
            for i in range(len(self.frames)):
                for other in self.frameAtoms(i):
                    if min(end, other["start"] + other["duration"]) - max(start, other["start"]) > GAP_TOLERANCE_S:
                        found.append((atom["label"], i + 1))
                        break
        return found

    def nextAddedKeys(self, count):
        """Keys the atoms of the next added series will get."""
        return [f"a{self.addCount + 1}.{i + 1}" for i in range(count)]

    def add(self, name, atoms, scaleExisting=1.0, scaleAdded=1.0, reference=None):
        """Add the frames of a series (atoms without keys: start, duration, label); every frame present is
        multiplied by scaleExisting, the new ones by scaleAdded (decay alignment, compareSeries). Returns the
        indexes of the new frames."""
        if not atoms:
            raise EditError("The series has no frames.")
        overlapping = self.overlaps(atoms)
        if overlapping:
            raise EditError("The added frames overlap the series in time: " + ", ".join(
                f"{label} with frame {n}" for label, n in overlapping[:6]) + ".")
        self._remember(f"Add {name}")
        self.addCount += 1
        number = self.addCount
        self.addedSeries.append(name)
        if abs(scaleExisting - 1.0) > 1e-12:
            for key in self.atoms:
                self.scales[key] = self.scale(key) * scaleExisting
        keys = []
        for i, atom in enumerate(atoms):
            key = f"a{number}.{i + 1}"
            self.atoms[key] = dict(atom, key=key, origin=ADDED, series=name)
            if abs(scaleAdded - 1.0) > 1e-12:
                self.scales[key] = scaleAdded
            self.frames.append([key])
            keys.append(key)
        if reference is not None:
            self.reference = reference
        self._sort()
        return [i for i, frame in enumerate(self.frames) if frame[0] in keys]

    def _sort(self):
        self.frames.sort(key=lambda keys: min(self.atoms[k]["start"] for k in keys))

    @staticmethod
    def consecutive(indexes):
        indexes = sorted(indexes)
        return bool(indexes) and indexes == list(range(indexes[0], indexes[-1] + 1))

    def largestGap(self, indexes):
        """Longest time between neighbouring frames of the selection (s; 0 when back to back)."""
        indexes = sorted(indexes)
        return max([self.frameStart(b) - self.frameEnd(a) for a, b in zip(indexes, indexes[1:])] or [0.0])

    def merge(self, indexes):
        """Merge neighbouring frames into one; returns its index."""
        indexes = sorted(set(indexes))
        if len(indexes) < 2:
            raise EditError("Select two or more frames to merge.")
        if not self.consecutive(indexes):
            raise EditError("Only neighbouring frames can be merged.")
        keys = sorted((k for i in indexes for k in self.frames[i]), key=lambda k: self.atoms[k]["start"])
        self._remember("Merge")
        self.frames[indexes[0]:indexes[-1] + 1] = [keys]
        return indexes[0]

    def canSplit(self, indexes):
        return any(len(self.frames[i]) > 1 for i in indexes if 0 <= i < len(self.frames))

    def split(self, indexes):
        """Split the selected merged frames back into the frames they were made of; returns their new indexes."""
        chosen = {tuple(self.frames[i]) for i in indexes if 0 <= i < len(self.frames) and len(self.frames[i]) > 1}
        if not chosen:
            raise EditError("Select a merged frame to split.")
        self._remember("Split")
        out = []
        for keys in self.frames:
            if tuple(keys) in chosen:
                out.extend([k] for k in keys)
            else:
                out.append(keys)
        self.frames = out
        self._sort()
        keys = {k for frame in chosen for k in frame}
        return [i for i, frame in enumerate(self.frames) if frame[0] in keys]

    def remove(self, indexes):
        indexes = sorted(set(i for i in indexes if 0 <= i < len(self.frames)))
        if not indexes:
            raise EditError("Select the frames to remove.")
        if len(indexes) >= len(self.frames):
            raise EditError("At least one frame has to stay.")
        self._remember("Remove")
        for i in reversed(indexes):
            del self.frames[i]

    # --- images -------------------------------------------------------------------------------
    def frameArray(self, index, arrayOf):
        """Image of an edited frame: the duration weighted mean of its atoms' arrays (arrayOf(key)), each times its
        scale. A frame of one unscaled atom returns that atom's array itself."""
        weights = self.frameWeights(index)
        if len(weights) == 1 and abs(self.scale(weights[0][0]) - 1.0) < 1e-12:
            return arrayOf(weights[0][0])
        out = None
        for key, weight in weights:
            part = np.asarray(arrayOf(key), dtype=np.float64) * (weight * self.scale(key))
            out = part if out is None else out + part
        return out.astype(np.float32)

    # --- timing of the saved series -----------------------------------------------------------
    def timingRows(self, clockAtZero=None, injection=None, scanStart=None):
        """Frame timing of the edited series, keyed like petseries.FRAME_TIMING_COLUMNS. The study start is the
        start of the new first frame; clock times come from clockAtZero (the clock time of 0 on the editor's
        time axis), injection and scanStart (datetimes, any None)."""
        rows = []
        zero = self.frameStart(0) if self.frames else 0.0

        def clock(t):
            return clockAtZero + datetime.timedelta(seconds=t) if clockAtZero is not None else None

        def since(t, stamp):
            return _round((clock(t) - stamp).total_seconds()) if clock(t) is not None and stamp is not None else None

        for i in range(len(self.frames)):
            start, end, duration = self.frameStart(i), self.frameEnd(i), self.frameDuration(i)
            middle = (start + end) / 2.0
            startClock = clock(start)
            rows.append({
                "Frame": i + 1,
                "StartDateTime": startClock.isoformat() if startClock else None,
                "StartFromInjectionS": since(start, injection),
                "StartFromStudyStartS": _round(start - zero),
                "StartFromScanStartS": since(start, scanStart),
                "DurationS": _round(duration),
                "DurationMs": _round(duration * 1000.0, 1),
                "MidFromInjectionS": since(middle, injection),
                "MidFromScanStartS": since(middle, scanStart),
                "EndFromInjectionS": since(end, injection),
                "FrameReferenceTimeMs": None,
                "DecayFactor": None,
                "StartSource": "Review/Edit: " + self.frameSources(i),
                "LoaderDecayFactor": None,
            })
        return rows
