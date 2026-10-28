"""Pure helpers of the Dynamic PET Editor (no Slicer imports): view grid, layout XML, frame intervals and time
labels."""

import datetime
import math

MAIN_VIEW = "DynPETMain"
VIEW_GROUP = 78    # editor views are linked among themselves only
FRAME_VIEW_PREFIX = "DynPETF"
EMPTY_VIEW_PREFIX = "DynPETEmpty"


def gridShape(count):
    """(rows, columns) of a grid of count views that is close to square (columns >= rows)."""
    if count <= 0:
        return 0, 0
    columns = int(math.ceil(math.sqrt(count)))
    rows = int(math.ceil(count / float(columns)))
    return rows, columns


def frameViewName(index):
    """Layout name of the small view of frame index (0-based)."""
    return f"{FRAME_VIEW_PREFIX}{index + 1}"


def emptyViewName(index):
    """Layout name of the index-th (0-based) empty view that fills the last grid row."""
    return f"{EMPTY_VIEW_PREFIX}{index + 1}"


def emptyViewCount(count):
    rows, columns = gridShape(count)
    return rows * columns - count


def _sliceViewXml(name, label, color):
    return (f'<item><view class="vtkMRMLSliceNode" singletontag="{name}">'
            '<property name="orientation" action="default">Axial</property>'
            f'<property name="viewlabel" action="default">{label}</property>'
            f'<property name="viewcolor" action="default">{color}</property>'
            f'<property name="viewgroup" action="default">{VIEW_GROUP}</property>'
            '</view></item>')


def layoutXml(count, mainShare=55):
    """Slicer layout of the editor: the main axial view on the left (mainShare % of the width) and a near-square
    grid of count small axial views, one per frame, on the right. The last row is filled with empty views, so
    every frame view has the same size and shape."""
    rows, columns = gridShape(count)
    grid = []
    for r in range(rows):
        cells = []
        for c in range(columns):
            index = r * columns + c
            if index < count:
                cells.append(_sliceViewXml(frameViewName(index), str(index + 1), "#3A7F7F"))
            else:
                cells.append(_sliceViewXml(emptyViewName(index - count), "", "#202020"))
        grid.append('<item><layout type="horizontal">' + "".join(cells) + "</layout></item>")
    right = '<layout type="vertical">' + "".join(grid) + "</layout>"
    main = _sliceViewXml(MAIN_VIEW, "D", "#F34A33")
    main = main.replace("<item>", f'<item splitSize="{mainShare * 10}">', 1)
    return ('<layout type="horizontal" split="true">' + main
            + f'<item splitSize="{(100 - mainShare) * 10}">' + right + "</item></layout>")


def formatTime(seconds):
    """m:ss (h:mm:ss from an hour on), with tenths when the time is not a whole second; '?' when unknown."""
    if seconds is None:
        return "?"
    sign = "-" if seconds < 0 else ""
    seconds = abs(seconds)
    tenths = int(round(seconds * 10))
    whole, tenth = divmod(tenths, 10)
    hours, rest = divmod(whole, 3600)
    minutes, secs = divmod(rest, 60)
    text = f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"
    return sign + text + (f".{tenth}" if tenth else "")


def _number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def frameIntervals(rows):
    """Frames as [{"index", "number", "start", "end", "duration"}] in seconds from the study start, from per-frame
    rows of {"Frame", "StartFromStudyStartS", "DurationS", "StartFromScanStartS", "IndexValue"} (texts or numbers,
    any missing). Missing starts follow the previous frame's end; missing durations reach the next start (1 s
    for the last frame). The first item also carries "timeBase" (which reference the times use) and, from the
    rows' "StartDateTime" (ISO clock time) and "StartFromInjectionS", "clockAtZero" (clock date-time of time 0)
    and "injectionAt" (the injection on the same time axis, s); each None when unknown."""
    out, base = [], "study start"
    starts = [_number(r.get("StartFromStudyStartS")) for r in rows]
    if all(s is None for s in starts):
        starts = [_number(r.get("StartFromScanStartS")) for r in rows]
        base = "scan start"
    durations = [_number(r.get("DurationS")) for r in rows]
    if all(s is None for s in starts):
        base = "frame order" if all(d is None for d in durations) else "first frame start"
    running = 0.0
    for i, row in enumerate(rows):
        start = starts[i] if starts[i] is not None else running
        duration = durations[i]
        if duration is None or duration <= 0:
            nextStart = next((s for s in starts[i + 1:] if s is not None), None)
            duration = (nextStart - start) if nextStart is not None and nextStart > start else 1.0
        number = row.get("Frame")
        number = int(_number(number)) if _number(number) is not None else i + 1
        out.append({"index": i, "number": number, "start": start, "end": start + duration, "duration": duration})
        running = start + duration
    if out:
        out[0]["timeBase"] = base
        out[0]["clockAtZero"], out[0]["injectionAt"] = None, None
        for row, interval in zip(rows, out):
            if out[0]["clockAtZero"] is None and row.get("StartDateTime"):
                try:
                    stamp = datetime.datetime.fromisoformat(str(row["StartDateTime"]))
                    out[0]["clockAtZero"] = stamp - datetime.timedelta(seconds=interval["start"])
                except ValueError:
                    pass
            fromInjection = _number(row.get("StartFromInjectionS"))
            if out[0]["injectionAt"] is None and fromInjection is not None:
                out[0]["injectionAt"] = interval["start"] - fromInjection
    return out


def formatClock(secondsOfDay):
    """HH:MM:SS of seconds after midnight."""
    seconds = int(round(secondsOfDay)) % 86400
    return f"{seconds // 3600:02d}:{seconds // 60 % 60:02d}:{seconds % 60:02d}"


def rulerRows(intervals):
    """Rows of the timeline ruler as (title, offset, kind): the label of time t is t + offset, formatted as a
    clock time (kind "clock") or as a duration ("duration"). Clock time, time from the study (or scan) start,
    and time from the injection, the ones that are known."""
    if not intervals:
        return []
    first = intervals[0]
    rows = []
    clock = first.get("clockAtZero")
    if clock is not None:
        rows.append(("Clock", clock.hour * 3600 + clock.minute * 60 + clock.second + clock.microsecond / 1e6,
                     "clock"))
    title = {"study start": "Study", "scan start": "Scan"}.get(first.get("timeBase"), "Time")
    rows.append((title, 0.0, "duration"))
    if first.get("injectionAt") is not None:
        rows.append(("Injection", -first["injectionAt"], "duration"))
    return rows


def frameTimes(intervals, index):
    """Text of a frame's start - end on every ruler row: 'study 5:20 - 5:50, injection 0:20 - 0:50, clock
    10:00:20'."""
    frame = intervals[index]
    parts = []
    for title, offset, kind in rulerRows(intervals):
        if kind == "clock":
            parts.append(f"clock {formatClock(frame['start'] + offset)} - {formatClock(frame['end'] + offset)}")
        else:
            parts.append(f"{title.lower()} {formatTime(frame['start'] + offset)} - {formatTime(frame['end'] + offset)}")
    return ", ".join(parts)


def frameLabel(interval):
    """Overlay text of a frame: 'F3  0:20 - 0:50'."""
    return f"F{interval['number']}  {formatTime(interval['start'])} - {formatTime(interval['end'])}"


def frameAt(intervals, t):
    """Index of the frame shown at time t: the last frame that starts at or before t (0 before the first)."""
    index = 0
    for i, frame in enumerate(intervals):
        if frame["start"] <= t + 1e-9:
            index = i
        else:
            break
    return index


def defaultPlaySpeed(intervals, seconds=60.0):
    """Playback speed (acquisition seconds per real second) that plays the whole series in the given seconds."""
    if not intervals:
        return 1.0
    return max((intervals[-1]["end"] - intervals[0]["start"]) / seconds, 1e-3)


def speedText(speed):
    """'x90 (1 s = 1:30)'."""
    factor = f"{speed:.0f}" if speed >= 10 else f"{speed:.2g}"
    return f"x{factor}  (1 s = {formatTime(speed)})"


def niceTickStep(span, width, minSpacing=70.0):
    """Ruler step (s) giving at least minSpacing pixels between ticks over width pixels for span seconds."""
    steps = (0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600, 7200)
    if span <= 0 or width <= 0:
        return steps[-1]
    for step in steps:
        if width * step / span >= minSpacing:
            return step
    return steps[-1]


def zoomRange(start, end, center, factor, lowest, highest, minimumSpan=1.0):
    """Time window [start, end] scaled by factor (< 1 zooms in) around center, kept inside [lowest, highest]
    and at least minimumSpan seconds wide (or the whole range when that is shorter)."""
    full = highest - lowest
    span = min(full, max(min(minimumSpan, full), (end - start) * factor))
    ratio = (center - start) / (end - start) if end > start else 0.5
    newStart = center - ratio * span
    newStart = min(max(newStart, lowest), highest - span)
    return newStart, newStart + span
