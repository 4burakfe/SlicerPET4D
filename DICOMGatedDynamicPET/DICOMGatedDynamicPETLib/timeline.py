"""Timeline widget of Review/Edit: a time ruler over a track of frame blocks, each as wide as its duration. The
ruler has a row per time reference (dynamicviews.rulerRows): clock time, time from the study start and time from
the injection.

With curves (setCurves: the time-activity curves of the spherical ROIs) a plot band sits between the ruler and a
shorter frame track. It uses the same time axis, so a curve's points stay in the middle of their frame blocks at
any zoom; the current frame's points are larger. Without curves there is no band.

With motion rows (setMotion: the result of Dynamic PET Motion Correction) two rows sit right above the frame
track, on the same time axis: the shift (mm along R, A, S) and the angles (degrees about R, A, S) of every frame,
or its status (reference, low activity, over limit, failed). A narrow block shows the largest value only; hovering
gives all of them.

Only a time window [viewStart, viewEnd] is drawn (zoom): set it with setViewRange (the editor's two-handle time
slider) or with the mouse wheel over the timeline (zooms around the cursor); Shift + wheel pans in time.
Ctrl + wheel over the plot band zooms its value axis (from 0 up), a double click on the band resets it, and a
double click elsewhere shows the whole series. Click a frame to make it current;
Ctrl+click toggles it in the selection, Shift+click selects a range; Left / Right step through the frames.
Callbacks (set by the editor):
    onCurrentFrameChanged(index)     the user picked a frame
    onSelectionChanged(indexes)      the selection changed (sorted list)
    onViewRangeChanged(start, end)   the wheel zoomed the time window
"""

import qt

from .dynamicviews import formatClock, formatTime, niceTickStep, panRange, rulerRows, zoomRange
from .motioncorrection import cellText, frameMotionTooltip, magnitudeLevel, statusCellText
from .roicurves import formatValue, frameMidpoints, niceAxis

_BACKGROUND = qt.QColor(30, 30, 33)
_RULER = qt.QColor(44, 44, 48)
_DIM_TEXT = qt.QColor(150, 150, 155)
_FRAME = qt.QColor(32, 178, 160)
_FRAME_CURRENT = qt.QColor(64, 224, 200)
_SELECTED = qt.QColor(255, 255, 255)
_PLAYHEAD = qt.QColor(235, 64, 52)
_TRACK = qt.QColor(38, 38, 42)
_FRAME_TEXT = qt.QColor(10, 30, 28)
_GRID = qt.QColor(62, 62, 68)
_MOTION_LEVELS = (qt.QColor(150, 220, 150), qt.QColor(240, 200, 90), qt.QColor(255, 110, 90))
_STATUS_TEXT = qt.QColor(160, 160, 165)


class FrameTimeline(qt.QWidget):
    RULER_ROW_HEIGHT = 17
    LABEL_WIDTH = 64
    MARGIN = 8
    CURVE_TRACK_HEIGHT = 30       # frame track under the plot band
    CURVE_BAND_MIN_HEIGHT = 110
    CURVE_BAND_GAP = 4
    POINT_RADIUS = 2.5
    CURRENT_POINT_RADIUS = 4.5
    HOVER_DISTANCE = 10           # pixels from a curve point that show its tooltip
    MOTION_ROW_HEIGHT = 16
    MOTION_TITLES = ("Shift mm", "Angle \u00b0")
    SHIFT_LEVELS_MM = (2.0, 5.0)      # cell colour: green below 2 mm, amber below 5 mm, red above
    ANGLE_LEVELS_DEG = (1.0, 3.0)

    def __init__(self, parent=None):
        qt.QWidget.__init__(self, parent)
        self.frames = []
        self.currentIndex = -1
        self.selected = set()
        self.anchor = None
        self.viewStart, self.viewEnd = 0.0, 1.0
        self.playTime = None          # playback position (s); the playhead follows it while playing
        self.rulers = []
        self.rulerHeight = self.RULER_ROW_HEIGHT + 4
        self.curves = []              # {"name", "color", "values"}: one value (or None) per frame
        self.curveUnits = ""
        self.curveTicks = [0.0, 1.0]
        self.valueZoom = 1.0          # Ctrl + wheel over the plot band: its top is the highest value / valueZoom
        self.motion = []              # per frame {"status", "shift", "angles", "message"} or None; empty: no rows
        self.onCurrentFrameChanged = None
        self.onSelectionChanged = None
        self.onViewRangeChanged = None
        self.setMinimumHeight(110)
        self.setFocusPolicy(qt.Qt.StrongFocus)
        self.setMouseTracking(True)

    # --- data ----------------------------------------------------------------------------------
    def setFrames(self, frames):
        """frames: dynamicviews.frameIntervals items. Shows the whole series."""
        self.frames = list(frames)
        self.rulers = rulerRows(self.frames)
        self.rulerHeight = self.RULER_ROW_HEIGHT * max(1, len(self.rulers)) + 4
        self._updateMinimumHeight()
        self.selected = set()
        self.currentIndex = 0 if self.frames else -1
        if self.frames:
            self.viewStart, self.viewEnd = self.fullRange()
        self.update()

    def setCurves(self, curves, units=""):
        """curves: [{"name", "color" ('#RRGGBB'), "values" (per frame, None allowed)}]; none hides the plot band."""
        self.curves = [c for c in curves if any(v is not None for v in c["values"])]
        self.curveUnits = units or ""
        self._updateCurveTicks()
        self._updateMinimumHeight()
        self.update()

    def _updateCurveTicks(self):
        highest = max([v for c in self.curves for v in c["values"] if v is not None] or [0.0])
        self.curveTicks = niceAxis(highest / self.valueZoom)

    def setValueZoom(self, zoom):
        self.valueZoom = min(max(float(zoom), 1.0), 1000.0)
        self._updateCurveTicks()
        self.update()

    def setMotion(self, entries):
        """entries: per frame {"status", "shift" (3 mm or None), "angles" (3 deg or None), "message"} or None;
        an empty list removes the motion rows."""
        self.motion = list(entries or [])
        self._updateMinimumHeight()
        self.update()

    def _motionHeight(self):
        return 2 * self.MOTION_ROW_HEIGHT + 2 if self.motion else 0

    def _updateMinimumHeight(self):
        extra = self.CURVE_TRACK_HEIGHT + self.CURVE_BAND_MIN_HEIGHT + self.CURVE_BAND_GAP + 16 if self.curves else 90
        self.setMinimumHeight(self.rulerHeight + extra + self._motionHeight())

    def fullRange(self):
        return (self.frames[0]["start"], self.frames[-1]["end"]) if self.frames else (0.0, 1.0)

    def setViewRange(self, start, end):
        if end - start > 1e-6 and (abs(start - self.viewStart) > 1e-9 or abs(end - self.viewEnd) > 1e-9):
            self.viewStart, self.viewEnd = float(start), float(end)
            self.update()

    def setPlayTime(self, t):
        self.playTime = t
        self.update()

    def setCurrentIndex(self, index):
        if index != self.currentIndex:
            self.currentIndex = index
            self.update()

    def selectedIndexes(self):
        return sorted(self.selected)

    # --- geometry ------------------------------------------------------------------------------
    def _trackRect(self):
        left = self.LABEL_WIDTH
        top = self.rulerHeight + 6 + self._motionHeight()
        height = max(10, self.height - top - 10)
        if self.curves:
            top, height = top + height - min(self.CURVE_TRACK_HEIGHT, height), min(self.CURVE_TRACK_HEIGHT, height)
        return left, top, max(1, self.width - left - self.MARGIN), height

    def _bandRect(self):
        """The plot band: between the ruler and the frame track."""
        left, trackTop, width, _ = self._trackRect()
        top = self.rulerHeight + 6
        return left, top, width, max(10, trackTop - self.CURVE_BAND_GAP - self._motionHeight() - top)

    def _y(self, value):
        _, top, _, height = self._bandRect()
        return top + height - value * (height - 6) / max(self.curveTicks[-1], 1e-12)

    def _x(self, t):
        left, _, width, _ = self._trackRect()
        return left + (t - self.viewStart) * width / max(self.viewEnd - self.viewStart, 1e-9)

    def _time(self, x):
        left, _, width, _ = self._trackRect()
        return self.viewStart + (x - left) * (self.viewEnd - self.viewStart) / float(width)

    def _indexAt(self, x):
        left, _, width, _ = self._trackRect()
        if not left <= x <= left + width:
            return None
        for i, frame in enumerate(self.frames):
            x0, x1 = self._x(frame["start"]), self._x(frame["end"])
            if x0 <= x < max(x1, x0 + 2):
                return i
        return None

    # --- painting ------------------------------------------------------------------------------
    def paintEvent(self, event):
        painter = qt.QPainter()
        painter.begin(self)
        try:
            painter.setRenderHint(qt.QPainter.Antialiasing, True)
            painter.fillRect(0, 0, self.width, self.height, _BACKGROUND)
            painter.fillRect(0, 0, self.width, self.rulerHeight, _RULER)
            left, top, width, height = self._trackRect()
            painter.fillRect(left, top, width, height, _TRACK)
            painter.setPen(qt.QPen(_DIM_TEXT))
            painter.drawText(qt.QRect(4, top, self.LABEL_WIDTH - 8, height), qt.Qt.AlignVCenter | qt.Qt.AlignLeft,
                             "Frames")
            if not self.frames:
                painter.drawText(qt.QRect(left, top, width, height), qt.Qt.AlignCenter, "No dynamic series")
                return
            self._paintRulerTitles(painter)
            if self.curves:
                self._paintCurveAxis(painter)
            if self.motion:
                self._paintMotionTitles(painter)
            painter.setClipRect(qt.QRect(left, 0, width, self.height))
            self._paintRuler(painter)
            if self.curves:
                self._paintCurves(painter)
            self._paintFrames(painter, top, height)
            if self.motion:
                self._paintMotion(painter)
            self._paintPlayhead(painter)
        finally:
            painter.end()

    def _smallFont(self, painter):
        font = painter.font()
        font.setPointSize(max(7, font.pointSize() - 1))
        painter.setFont(font)

    def _paintRulerTitles(self, painter):
        painter.setPen(qt.QPen(_DIM_TEXT))
        self._smallFont(painter)
        for r, (title, _, _) in enumerate(self.rulers):
            painter.drawText(qt.QRect(4, 2 + r * self.RULER_ROW_HEIGHT, self.LABEL_WIDTH - 8, self.RULER_ROW_HEIGHT),
                             qt.Qt.AlignLeft | qt.Qt.AlignVCenter, title)

    def _paintRuler(self, painter):
        """One row per time reference; ticks at round values of that reference."""
        painter.setPen(qt.QPen(_DIM_TEXT))
        self._smallFont(painter)
        _, _, width, _ = self._trackRect()
        tick = niceTickStep(self.viewEnd - self.viewStart, width, 80.0)
        for r, (_, offset, kind) in enumerate(self.rulers):
            top = 2 + r * self.RULER_ROW_HEIGHT
            bottom = top + self.RULER_ROW_HEIGHT
            value = (int((self.viewStart + offset) // tick)) * tick
            while value - offset <= self.viewEnd + 1e-9:
                t = value - offset
                if t >= self.viewStart - 1e-9:
                    x = int(self._x(t))
                    painter.drawLine(x, bottom - 5, x, bottom)
                    text = formatClock(value) if kind == "clock" else formatTime(value)
                    painter.drawText(qt.QRect(x + 3, top, 90, self.RULER_ROW_HEIGHT),
                                     qt.Qt.AlignLeft | qt.Qt.AlignVCenter, text)
                value += tick

    def _paintFrames(self, painter, top, height):
        for i, frame in enumerate(self.frames):
            if frame["end"] < self.viewStart or frame["start"] > self.viewEnd:
                continue
            x0, x1 = self._x(frame["start"]), self._x(frame["end"])
            rect = qt.QRectF(x0 + 1, top + 4, max(1.0, x1 - x0 - 2), height - 8)
            painter.setBrush(qt.QBrush(_FRAME_CURRENT if i == self.currentIndex else _FRAME))
            painter.setPen(qt.QPen(qt.QBrush(_SELECTED), 2) if i in self.selected else qt.QPen(qt.Qt.NoPen))
            painter.drawRoundedRect(rect, 3, 3)
            if rect.width() > 24:
                painter.setPen(qt.QPen(_FRAME_TEXT))
                text = f"F{frame['number']}"
                if rect.width() > 70:
                    text += (" " if self.curves else "\n") + formatTime(frame["duration"])
                painter.drawText(rect, qt.Qt.AlignCenter, text)

    def _motionRowRect(self, row):
        left, trackTop, width, _ = self._trackRect()
        top = trackTop - self._motionHeight() + row * self.MOTION_ROW_HEIGHT
        return left, top, width, self.MOTION_ROW_HEIGHT

    def _paintMotionTitles(self, painter):
        painter.save()
        self._smallFont(painter)
        painter.setPen(qt.QPen(_DIM_TEXT))
        for row, title in enumerate(self.MOTION_TITLES):
            _, top, _, height = self._motionRowRect(row)
            painter.drawText(qt.QRect(4, top, self.LABEL_WIDTH - 8, height), qt.Qt.AlignLeft | qt.Qt.AlignVCenter,
                             title)
        painter.restore()

    def _paintMotion(self, painter):
        """The shift and angle rows: one cell per frame block, the text coloured by size."""
        painter.save()
        self._smallFont(painter)
        charWidth = max(4.0, painter.fontMetrics().averageCharWidth())
        for row, key in enumerate(("shift", "angles")):
            left, top, width, height = self._motionRowRect(row)
            painter.fillRect(left, top, width, height - 1, _TRACK)
            levels = self.SHIFT_LEVELS_MM if key == "shift" else self.ANGLE_LEVELS_DEG
            for i, frame in enumerate(self.frames):
                entry = self.motion[i] if i < len(self.motion) else None
                if not entry or frame["end"] < self.viewStart or frame["start"] > self.viewEnd:
                    continue
                x0, x1 = self._x(frame["start"]), self._x(frame["end"])
                rect = qt.QRectF(x0 + 1, top, max(1.0, x1 - x0 - 2), height - 1)
                values = entry.get(key)
                if values is not None:
                    text = cellText(values, rect.width(), charWidth)
                    painter.setPen(qt.QPen(_MOTION_LEVELS[magnitudeLevel(values, *levels)]))
                else:
                    text = statusCellText(entry.get("status"), rect.width(), charWidth) if row == 0 else ""
                    painter.setPen(qt.QPen(_STATUS_TEXT))
                if text:
                    painter.drawText(rect, qt.Qt.AlignCenter, text)
            painter.setPen(qt.QPen(_GRID))
            lastX = None
            for frame in self.frames:
                x = int(self._x(frame["start"]))
                if lastX is None or x - lastX >= 3:
                    painter.drawLine(x, top, x, top + height - 2)
                    lastX = x
        painter.restore()

    def _paintCurveAxis(self, painter):
        """Value labels of the plot band and the units, in the label column."""
        left, top, _, height = self._bandRect()
        painter.save()
        self._smallFont(painter)
        self._smallFont(painter)
        painter.setPen(qt.QPen(_DIM_TEXT))
        for tick in self.curveTicks:
            painter.drawText(qt.QRect(14, int(self._y(tick)) - 8, left - 18, 16),
                             qt.Qt.AlignRight | qt.Qt.AlignVCenter, formatValue(tick))
        painter.translate(2, top + height)
        painter.rotate(-90)
        painter.drawText(qt.QRect(0, 0, height, 14), qt.Qt.AlignCenter, self.curveUnits)
        painter.restore()

    def _paintCurves(self, painter):
        """The band's background and grid (the frame boundaries, the value ticks), and a line through the middle
        of the frames for each curve; a frame without a value breaks the line."""
        left, top, width, height = self._bandRect()
        painter.save()
        painter.setClipRect(qt.QRect(left, top, width, height))
        painter.fillRect(left, top, width, height, _TRACK)
        painter.setPen(qt.QPen(_GRID))
        for tick in self.curveTicks:
            painter.drawLine(left, int(self._y(tick)), left + width, int(self._y(tick)))
        lastX = None
        for frame in self.frames:
            for t in (frame["start"], frame["end"]):
                x = int(self._x(t))
                if lastX is None or x - lastX >= 3:
                    painter.drawLine(x, top, x, top + height)
                    lastX = x
        midpoints = frameMidpoints(self.frames)
        for curve in self.curves:
            color = qt.QColor(curve["color"])
            path, drawing = qt.QPainterPath(), False
            for t, value in zip(midpoints, curve["values"]):
                if value is None:
                    drawing = False
                elif drawing:
                    path.lineTo(self._x(t), self._y(value))
                else:
                    path.moveTo(self._x(t), self._y(value))
                    drawing = True
            painter.setBrush(qt.QBrush(qt.Qt.NoBrush))
            painter.setPen(qt.QPen(qt.QBrush(color), 1.5))
            painter.drawPath(path)
            painter.setBrush(qt.QBrush(color))
            painter.setPen(qt.QPen(qt.Qt.NoPen))
            for t, value in zip(midpoints, curve["values"]):
                if value is not None:
                    painter.drawEllipse(qt.QPointF(self._x(t), self._y(value)), self.POINT_RADIUS, self.POINT_RADIUS)
            if 0 <= self.currentIndex < len(curve["values"]) and curve["values"][self.currentIndex] is not None:
                painter.setPen(qt.QPen(qt.QBrush(_SELECTED), 1.5))
                painter.drawEllipse(qt.QPointF(self._x(midpoints[self.currentIndex]),
                                               self._y(curve["values"][self.currentIndex])),
                                    self.CURRENT_POINT_RADIUS, self.CURRENT_POINT_RADIUS)
        painter.restore()

    def _paintPlayhead(self, painter):
        if not (0 <= self.currentIndex < len(self.frames)):
            return
        x0 = self._x(self.playTime if self.playTime is not None else self.frames[self.currentIndex]["start"])
        painter.setPen(qt.QPen(qt.QBrush(_PLAYHEAD), 2))
        painter.drawLine(int(x0), 0, int(x0), self.height)
        painter.setBrush(qt.QBrush(_PLAYHEAD))
        painter.drawEllipse(qt.QPointF(x0, 5.0), 4, 4)

    # --- interaction ---------------------------------------------------------------------------
    def mousePressEvent(self, event):
        index = self._indexAt(event.pos().x())
        if index is None:
            return
        modifiers = event.modifiers()
        if modifiers & qt.Qt.ControlModifier:
            self.selected ^= {index}
            self.anchor = index
        elif modifiers & qt.Qt.ShiftModifier and self.anchor is not None:
            low, high = sorted((self.anchor, index))
            self.selected = set(range(low, high + 1))
        else:
            self.selected = {index}
            self.anchor = index
        self._setCurrent(index)
        if self.onSelectionChanged:
            self.onSelectionChanged(self.selectedIndexes())
        self.update()

    def _curvePointAt(self, x, y):
        """(curve, frame index) of the curve point nearest to the position, within HOVER_DISTANCE pixels; or None."""
        left, top, width, height = self._bandRect()
        if not self.curves or not (left <= x <= left + width and top <= y <= top + height):
            return None
        best, nearest = None, self.HOVER_DISTANCE ** 2
        midpoints = frameMidpoints(self.frames)
        for curve in self.curves:
            for index, value in enumerate(curve["values"]):
                if value is not None and index < len(midpoints):
                    distance = (self._x(midpoints[index]) - x) ** 2 + (self._y(value) - y) ** 2
                    if distance <= nearest:
                        best, nearest = (curve, index), distance
        return best

    def mouseMoveEvent(self, event):
        point = self._curvePointAt(event.pos().x(), event.pos().y())
        if point is not None:
            curve, index = point
            units = f" {self.curveUnits}" if self.curveUnits else ""
            self.toolTip = (f"{curve['name']}, frame {self.frames[index]['number']}: "
                            f"{formatValue(curve['values'][index])}{units}")
            return
        index = self._indexAt(event.pos().x())
        if index is None:
            self.toolTip = ""
            return
        frame = self.frames[index]
        tip = (f"Frame {frame['number']}: {formatTime(frame['start'])} - {formatTime(frame['end'])} "
               f"({formatTime(frame['duration'])})")
        if index < len(self.motion) and self.motion[index]:
            tip += "\n" + frameMotionTooltip(self.motion[index])
        self.toolTip = tip

    def _inBand(self, x, y):
        left, top, width, height = self._bandRect()
        return bool(self.curves) and left <= x <= left + width and top <= y <= top + height

    def mouseDoubleClickEvent(self, event):
        if self._inBand(event.pos().x(), event.pos().y()):
            self.setValueZoom(1.0)
            return
        if self.frames:
            self.setViewRange(*self.fullRange())
            if self.onViewRangeChanged:
                self.onViewRangeChanged(self.viewStart, self.viewEnd)

    def wheelEvent(self, event):
        if not self.frames:
            return
        steps = event.angleDelta().y() / 120.0
        if not steps:
            return
        modifiers = event.modifiers()
        if modifiers & qt.Qt.ControlModifier:
            if self.curves:
                self.setValueZoom(self.valueZoom * 1.25 ** steps)
            event.accept()
            return
        if modifiers & qt.Qt.ShiftModifier:
            start, end = panRange(self.viewStart, self.viewEnd, steps, *self.fullRange())
        else:
            start, end = zoomRange(self.viewStart, self.viewEnd, self._time(event.pos().x()), 0.8 ** steps,
                                   *self.fullRange())
        self.setViewRange(start, end)
        if self.onViewRangeChanged:
            self.onViewRangeChanged(self.viewStart, self.viewEnd)
        event.accept()

    def keyPressEvent(self, event):
        if not self.frames:
            return
        if event.key() == qt.Qt.Key_Left:
            self._setCurrent(max(0, self.currentIndex - 1))
        elif event.key() == qt.Qt.Key_Right:
            self._setCurrent(min(len(self.frames) - 1, self.currentIndex + 1))
        elif event.key() == qt.Qt.Key_Escape:
            self.selected = set()
            if self.onSelectionChanged:
                self.onSelectionChanged([])
        else:
            qt.QWidget.keyPressEvent(self, event)
            return
        self.update()

    def _setCurrent(self, index):
        self.currentIndex = index
        if self.onCurrentFrameChanged:
            self.onCurrentFrameChanged(index)
