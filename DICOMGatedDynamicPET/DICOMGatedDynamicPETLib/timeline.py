"""Timeline widget of the Dynamic PET Editor: a time ruler over a track of frame blocks, each as wide as its
duration. The ruler has a row per time reference (dynamicviews.rulerRows): clock time, time from the study
start and time from the injection.

Only a time window [viewStart, viewEnd] is drawn (zoom): set it with setViewRange (the editor's two-handle time
slider) or with the mouse wheel over the timeline (zooms around the cursor). Click a frame to make it current;
Ctrl+click toggles it in the selection, Shift+click selects a range; Left / Right step through the frames.
Callbacks (set by the editor):
    onCurrentFrameChanged(index)     the user picked a frame
    onSelectionChanged(indexes)      the selection changed (sorted list)
    onViewRangeChanged(start, end)   the wheel zoomed the time window
"""

import qt

from .dynamicviews import formatClock, formatTime, niceTickStep, rulerRows, zoomRange

_BACKGROUND = qt.QColor(30, 30, 33)
_RULER = qt.QColor(44, 44, 48)
_DIM_TEXT = qt.QColor(150, 150, 155)
_FRAME = qt.QColor(32, 178, 160)
_FRAME_CURRENT = qt.QColor(64, 224, 200)
_SELECTED = qt.QColor(255, 255, 255)
_PLAYHEAD = qt.QColor(235, 64, 52)
_TRACK = qt.QColor(38, 38, 42)
_FRAME_TEXT = qt.QColor(10, 30, 28)


class FrameTimeline(qt.QWidget):
    RULER_ROW_HEIGHT = 17
    LABEL_WIDTH = 64
    MARGIN = 8

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
        self.setMinimumHeight(self.rulerHeight + 90)
        self.selected = set()
        self.currentIndex = 0 if self.frames else -1
        if self.frames:
            self.viewStart, self.viewEnd = self.fullRange()
        self.update()

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
        return (left, self.rulerHeight + 6, max(1, self.width - left - self.MARGIN),
                max(10, self.height - self.rulerHeight - 16))

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
            painter.setClipRect(qt.QRect(left, 0, width, self.height))
            self._paintRuler(painter)
            self._paintFrames(painter, top, height)
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
                    text += f"\n{formatTime(frame['duration'])}"
                painter.drawText(rect, qt.Qt.AlignCenter, text)

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

    def mouseMoveEvent(self, event):
        index = self._indexAt(event.pos().x())
        if index is None:
            self.toolTip = ""
            return
        frame = self.frames[index]
        self.toolTip = (f"Frame {frame['number']}: {formatTime(frame['start'])} - {formatTime(frame['end'])} "
                        f"({formatTime(frame['duration'])})")

    def wheelEvent(self, event):
        if not self.frames:
            return
        steps = event.angleDelta().y() / 120.0
        if not steps:
            return
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
