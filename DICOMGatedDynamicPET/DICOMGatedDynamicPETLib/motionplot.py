"""Motion plots of Dynamic PET Motion Correction: the rigid motion of every frame over time, on a white background.
One plot per kind: MotionPlot("shift") draws x, y, z (mm along R, A, S), MotionPlot("angles") pitch, roll, yaw
(degrees about R, A, S), each component in its own colour. One point per frame, in the middle of the frame; a
frame that was not registered has no point and breaks the line.

The time axis is the timeline's (the same label column and margins), so the two line up: setViewRange follows the
timeline's time window, and the mouse wheel zooms time here too (onViewRangeChanged). Ctrl + wheel zooms the
value axis, Shift + wheel pans in time, a double click resets the value zoom. Click picks the frame.
"""

import qt

from .dynamicviews import formatTime, panRange, zoomRange
from .motioncorrection import COMPONENTS, plotLimit, plotTicks, vectorText
from .roicurves import frameMidpoints

UNITS = {"shift": "mm", "angles": "deg"}
MINIMUM_LIMIT = {"shift": 2.0, "angles": 1.0}   # smallest half height of the value axis
_BACKGROUND = qt.QColor(255, 255, 255)
_AXIS_TEXT = qt.QColor(70, 70, 75)
_GRID = qt.QColor(225, 225, 228)
_ZERO = qt.QColor(150, 150, 155)


class MotionPlot(qt.QWidget):
    LABEL_WIDTH = 64      # as FrameTimeline, so the time axes line up
    MARGIN = 8
    LEGEND_HEIGHT = 18
    POINT_RADIUS = 2.5

    def __init__(self, kind="shift", parent=None):
        qt.QWidget.__init__(self, parent)
        self.kind = kind
        self.series = [(component, label, colour) for key, component, label, colour in COMPONENTS if key == kind]
        self.frames = []
        self.motion = []
        self.currentIndex = -1
        self.viewStart, self.viewEnd = 0.0, 1.0
        self.valueZoom = 1.0
        self.visibleSeries = [True] * len(self.series)
        self.legendRects = []         # set while painting, used by clicks on the legend
        self.onCurrentFrameChanged = None
        self.onViewRangeChanged = None
        self.setMinimumHeight(110)
        self.setMouseTracking(True)

    # --- data ------------------------------------------------------------------------------------
    def setFrames(self, frames):
        self.frames = list(frames)
        if self.frames:
            self.viewStart, self.viewEnd = self.frames[0]["start"], self.frames[-1]["end"]
        self.update()

    def setMotion(self, entries):
        """Per frame {"shift", "angles"} (None when not registered); empty: nothing to plot."""
        self.motion = list(entries or [])
        self.update()

    def setViewRange(self, start, end):
        if end - start > 1e-6:
            self.viewStart, self.viewEnd = float(start), float(end)
            self.update()

    def setCurrentIndex(self, index):
        self.currentIndex = index
        self.update()

    def fullRange(self):
        return (self.frames[0]["start"], self.frames[-1]["end"]) if self.frames else (0.0, 1.0)

    def _values(self, component):
        return [(e.get(self.kind)[component] if e and e.get(self.kind) is not None else None) for e in self.motion]

    def _limit(self):
        values = [v for component, _, _ in self.series for v in self._values(component) if v is not None]
        return plotLimit(values, MINIMUM_LIMIT.get(self.kind, 1.0)) / self.valueZoom

    # --- geometry --------------------------------------------------------------------------------
    def _plotRect(self):
        top = self.LEGEND_HEIGHT + 6
        return self.LABEL_WIDTH, top, max(1, self.width - self.LABEL_WIDTH - self.MARGIN), \
            max(10, self.height - top - 6)

    def _x(self, t):
        left, _, width, _ = self._plotRect()
        return left + (t - self.viewStart) * width / max(self.viewEnd - self.viewStart, 1e-9)

    def _time(self, x):
        left, _, width, _ = self._plotRect()
        return self.viewStart + (x - left) * (self.viewEnd - self.viewStart) / float(width)

    def _y(self, value, limit):
        _, top, _, height = self._plotRect()
        return top + height / 2.0 - value / max(limit, 1e-12) * (height / 2.0 - 4)

    def _indexAt(self, x):
        t = self._time(x)
        for i, frame in enumerate(self.frames):
            if frame["start"] <= t < frame["end"]:
                return i
        return None

    # --- painting --------------------------------------------------------------------------------
    def paintEvent(self, event):
        painter = qt.QPainter()
        painter.begin(self)
        try:
            painter.setRenderHint(qt.QPainter.Antialiasing, True)
            painter.fillRect(0, 0, self.width, self.height, _BACKGROUND)
            font = painter.font()
            font.setPointSize(max(7, font.pointSize() - 1))
            painter.setFont(font)
            self._paintLegend(painter)
            left, top, width, height = self._plotRect()
            if not self.motion or not self.frames:
                painter.setPen(qt.QPen(_AXIS_TEXT))
                painter.drawText(qt.QRect(left, top, width, height), qt.Qt.AlignCenter,
                                 "Register with the rigid method to plot the motion of the frames")
                return
            limit = self._limit()
            self._paintAxes(painter, limit)
            painter.setClipRect(qt.QRect(left, top, width, height))
            self._paintCurrent(painter)
            midpoints = frameMidpoints(self.frames)
            for visible, (component, _, colour) in zip(self.visibleSeries, self.series):
                if visible:
                    self._paintSeries(painter, midpoints, self._values(component), limit, qt.QColor(colour))
        finally:
            painter.end()

    def _legendRects(self, metrics):
        rects, x = [], self.LABEL_WIDTH
        for _, label, _ in self.series:
            w = 22 + metrics.boundingRect(label).width() + 4
            rects.append(qt.QRect(x, 2, w, self.LEGEND_HEIGHT))
            x += w + 10
        return rects

    def _paintLegend(self, painter):
        self.legendRects = self._legendRects(painter.fontMetrics())
        for visible, rect, (_, label, colour) in zip(self.visibleSeries, self.legendRects, self.series):
            pen = qt.QPen(qt.QColor(colour))
            pen.setWidthF(2.0)
            painter.setPen(pen)
            y = rect.top() + rect.height() // 2
            painter.drawLine(rect.left(), y, rect.left() + 16, y)
            painter.setPen(qt.QPen(_AXIS_TEXT if visible else _GRID))
            painter.drawText(qt.QRect(rect.left() + 20, rect.top(), rect.width() - 20, rect.height()),
                             qt.Qt.AlignLeft | qt.Qt.AlignVCenter, label)

    def _paintAxes(self, painter, limit):
        left, top, width, height = self._plotRect()
        painter.setPen(qt.QPen(_GRID))
        for frame in self.frames:
            x = int(self._x(frame["start"]))
            if left <= x <= left + width:
                painter.drawLine(x, top, x, top + height)
        for tick in plotTicks(limit):
            y = int(self._y(tick, limit))
            painter.setPen(qt.QPen(_ZERO if tick == 0 else _GRID))
            painter.drawLine(left, y, left + width, y)
            painter.setPen(qt.QPen(_AXIS_TEXT))
            painter.drawText(qt.QRect(2, y - 8, left - 6, 16), qt.Qt.AlignRight | qt.Qt.AlignVCenter, f"{tick:g}")
        painter.drawText(qt.QRect(2, 2, left - 6, 16), qt.Qt.AlignLeft | qt.Qt.AlignVCenter, UNITS.get(self.kind, ""))

    def _paintCurrent(self, painter):
        if 0 <= self.currentIndex < len(self.frames):
            frame = self.frames[self.currentIndex]
            _, top, _, height = self._plotRect()
            x0, x1 = self._x(frame["start"]), self._x(frame["end"])
            painter.fillRect(qt.QRectF(x0, top, max(1.0, x1 - x0), height), qt.QColor(255, 235, 200))

    def _paintSeries(self, painter, midpoints, values, limit, colour):
        path, drawing = qt.QPainterPath(), False
        for t, value in zip(midpoints, values):
            if value is None:
                drawing = False
            elif drawing:
                path.lineTo(self._x(t), self._y(value, limit))
            else:
                path.moveTo(self._x(t), self._y(value, limit))
                drawing = True
        pen = qt.QPen(colour)
        pen.setWidthF(1.8)
        painter.setBrush(qt.QBrush(qt.Qt.NoBrush))
        painter.setPen(pen)
        painter.drawPath(path)
        painter.setBrush(qt.QBrush(colour))
        painter.setPen(qt.QPen(qt.Qt.NoPen))
        for t, value in zip(midpoints, values):
            if value is not None:
                painter.drawEllipse(qt.QPointF(self._x(t), self._y(value, limit)), self.POINT_RADIUS,
                                    self.POINT_RADIUS)

    # --- interaction -----------------------------------------------------------------------------
    def mousePressEvent(self, event):
        for i, rect in enumerate(self.legendRects):     # click a legend entry to hide / show its line
            if rect.contains(event.pos()):
                self.visibleSeries[i] = not self.visibleSeries[i]
                self.update()
                return
        index = self._indexAt(event.pos().x())
        if index is not None and self.onCurrentFrameChanged:
            self.onCurrentFrameChanged(index)

    def mouseDoubleClickEvent(self, event):
        self.valueZoom = 1.0
        self.update()

    def mouseMoveEvent(self, event):
        index = self._indexAt(event.pos().x())
        if index is None or index >= len(self.motion) or not self.motion[index]:
            self.toolTip = ""
            return
        entry, frame = self.motion[index], self.frames[index]
        lines = [f"Frame {frame['number']}: {formatTime(frame['start'])} - {formatTime(frame['end'])}"]
        if entry.get(self.kind) is not None:
            lines.append(("x, y, z: " if self.kind == "shift" else "pitch, roll, yaw: ")
                         + f"{vectorText(entry[self.kind])} {UNITS.get(self.kind, '')}")
        else:
            lines.append(entry.get("status") or "")
        self.toolTip = "\n".join(lines)

    def wheelEvent(self, event):
        steps = event.angleDelta().y() / 120.0
        if not steps or not self.frames:
            return
        modifiers = event.modifiers()
        if modifiers & qt.Qt.ControlModifier:
            self.valueZoom = min(max(self.valueZoom * 1.25 ** steps, 1.0), 1000.0)
            self.update()
        else:
            if modifiers & qt.Qt.ShiftModifier:
                start, end = panRange(self.viewStart, self.viewEnd, steps, *self.fullRange())
            else:
                start, end = zoomRange(self.viewStart, self.viewEnd, self._time(event.pos().x()), 0.8 ** steps,
                                       *self.fullRange())
            self.setViewRange(start, end)
            if self.onViewRangeChanged:
                self.onViewRangeChanged(self.viewStart, self.viewEnd)
        event.accept()
