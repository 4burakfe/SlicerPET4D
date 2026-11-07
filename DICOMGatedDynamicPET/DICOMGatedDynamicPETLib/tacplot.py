"""Plot of Dynamic PET Modeling (Qt): the time-activity curves of the segments and the fitted model, or the Patlak /
Logan plot of the target.

TAC mode (setCurves): time (min from time zero) against activity. The target and the reference are drawn thick and
opaque, every other segment thin at OTHER_OPACITY; the data are points joined by a line, a fitted model a dashed
line in the same color. setMarkers adds the t* line (graphical methods, the frames used are shaded) or the SUVR time
window. Graphical mode (setGraphical): the points of the Patlak / Logan plot, the ones used (from t*) filled, and
the fitted line. Hovering shows the nearest point; Ctrl + wheel zooms the value axis, wheel zooms time, a double
click resets."""

import math

import qt

from .dynamicviews import zoomRange
from .roicurves import formatValue, niceAxis

OTHER_OPACITY = 0.2
_BACKGROUND = qt.QColor(255, 255, 255)
_AXIS_TEXT = qt.QColor(70, 70, 75)
_GRID = qt.QColor(228, 228, 232)
_MARK = qt.QColor(255, 236, 200)
_TSTAR = qt.QColor(220, 120, 30)


def _niceTicks(low, high, count=6):
    """Ticks between low and high (any sign)."""
    span = high - low
    if span <= 0 or not math.isfinite(span):
        return [low]
    rough = span / count
    magnitude = 10.0 ** math.floor(math.log10(rough))
    step = next(m * magnitude for m in (1, 2, 2.5, 5, 10) if m * magnitude >= rough * (1 - 1e-9))
    first = math.ceil(low / step - 1e-9) * step
    ticks, value = [], first
    while value <= high + step * 1e-6:
        ticks.append(float(f"{value:.10g}"))
        value += step
    return ticks


def _numbers(values):
    """List of floats (None for missing / not finite) of any sequence or array; None stays None."""
    if values is None:
        return None
    out = []
    for value in values:
        try:
            value = float(value)
        except (TypeError, ValueError):
            value = None
        out.append(value if value is not None and math.isfinite(value) else None)
    return out


def _plain(curve):
    curve = dict(curve)
    for key in ("times", "values", "fitTimes", "fitValues"):
        if key in curve:
            curve[key] = _numbers(curve[key])
    return curve


class TacPlot(qt.QWidget):
    MARGIN_LEFT = 64
    MARGIN_RIGHT = 12
    MARGIN_TOP = 24
    MARGIN_BOTTOM = 34

    def __init__(self, parent=None):
        qt.QWidget.__init__(self, parent)
        self.mode = "tac"
        self.curves = []           # dicts: name, color, times, values, emphasis, fitTimes, fitValues
        self.units = ""
        self.markers = {}          # tStar, window (start, end)
        self.graphical = None      # dict of kinetics graphical results + name, color
        self.message = "Pick a dynamic series and a segmentation"
        self.valueZoom = 1.0
        self.timeRange = None      # (start, end) min, None: all
        self.points = []           # (x, y, text) of hover targets, set while painting
        self.setMinimumHeight(160)
        self.setMouseTracking(True)

    # --- data --------------------------------------------------------------------------------------
    def setCurves(self, curves, units="", message=""):
        self.mode = "tac"
        self.curves = [_plain(curve) for curve in (curves or [])]
        self.units = units
        self.message = message
        self.update()

    def setMarkers(self, tStar=None, window=None):
        self.markers = {"tStar": tStar, "window": window}
        self.update()

    def setGraphical(self, graphical):
        """kinetics graphical dict (x, y, used, slope, intercept, tStar, xLabel, yLabel) plus name and color; None
        goes back to the curves."""
        self.graphical = graphical
        self.mode = "graphical" if graphical else "tac"
        self.update()

    def showGraphical(self, show):
        self.mode = "graphical" if show and self.graphical else "tac"
        self.update()

    def resetZoom(self):
        self.valueZoom = 1.0
        self.timeRange = None
        self.update()

    # --- geometry ------------------------------------------------------------------------------------
    def _rect(self):
        return (self.MARGIN_LEFT, self.MARGIN_TOP, max(10, self.width - self.MARGIN_LEFT - self.MARGIN_RIGHT),
                max(10, self.height - self.MARGIN_TOP - self.MARGIN_BOTTOM))

    def _mapper(self, xRange, yRange):
        left, top, width, height = self._rect()
        (x0, x1), (y0, y1) = xRange, yRange
        sx = width / (x1 - x0) if x1 > x0 else 1.0
        sy = height / (y1 - y0) if y1 > y0 else 1.0
        return (lambda x: left + (x - x0) * sx), (lambda y: top + height - (y - y0) * sy)

    # --- painting --------------------------------------------------------------------------------------
    def paintEvent(self, event):
        painter = qt.QPainter()
        painter.begin(self)
        try:
            painter.setRenderHint(qt.QPainter.Antialiasing, True)
            painter.fillRect(0, 0, self.width, self.height, _BACKGROUND)
            font = painter.font()
            font.setPointSize(max(7, font.pointSize() - 1))
            painter.setFont(font)
            self.points = []
            if self.mode == "graphical" and self.graphical:
                self._paintGraphical(painter)
            elif self.curves:
                self._paintCurves(painter)
            else:
                left, top, width, height = self._rect()
                painter.setPen(qt.QPen(_AXIS_TEXT))
                painter.drawText(qt.QRect(0, 0, self.width, self.height), qt.Qt.AlignCenter, self.message or "")
        finally:
            painter.end()

    def _axes(self, painter, xTicks, yTicks, mapX, mapY, xLabel, yLabel):
        left, top, width, height = self._rect()
        for tick in yTicks:
            y = mapY(tick)
            painter.setPen(qt.QPen(_GRID))
            painter.drawLine(qt.QPointF(left, y), qt.QPointF(left + width, y))
            painter.setPen(qt.QPen(_AXIS_TEXT))
            painter.drawText(qt.QRectF(0, y - 8, left - 6, 16), qt.Qt.AlignRight | qt.Qt.AlignVCenter,
                             formatValue(tick))
        for tick in xTicks:
            x = mapX(tick)
            painter.setPen(qt.QPen(_GRID))
            painter.drawLine(qt.QPointF(x, top), qt.QPointF(x, top + height))
            painter.setPen(qt.QPen(_AXIS_TEXT))
            painter.drawText(qt.QRectF(x - 30, top + height + 2, 60, 14), qt.Qt.AlignHCenter | qt.Qt.AlignTop,
                             f"{tick:g}")
        painter.setPen(qt.QPen(_AXIS_TEXT))
        painter.drawRect(qt.QRectF(left, top, width, height))
        painter.drawText(qt.QRectF(left, top + height + 16, width, 16), qt.Qt.AlignHCenter | qt.Qt.AlignTop, xLabel)
        painter.drawText(qt.QRectF(4, 2, left + 200, 18), qt.Qt.AlignLeft | qt.Qt.AlignVCenter, yLabel)

    def _timeRange(self):
        ends = [t for c in self.curves for t in (c.get("times") or []) + (c.get("fitTimes") or []) if t is not None]
        full = (0.0, max(ends) if ends else 1.0)
        return self.timeRange or full

    def _paintCurves(self, painter):
        xRange = self._timeRange()
        values = [v for c in self.curves
                  for v in (c.get("values") or []) + (c.get("fitValues") or [])
                  if v is not None and math.isfinite(v)]
        emphasised = [v for c in self.curves if c.get("emphasis")
                      for v in (c.get("values") or []) + (c.get("fitValues") or [])
                      if v is not None and math.isfinite(v)]
        top = max(emphasised or values or [1.0]) * 1.05 / self.valueZoom
        low = min(0.0, min(values or [0.0]))
        yTicks = [t for t in niceAxis(top)] if low >= 0 else _niceTicks(low, top)
        yRange = (min(low, yTicks[0]), max(yTicks[-1], top))
        mapX, mapY = self._mapper(xRange, yRange)
        left, plotTop, width, height = self._rect()
        window = self.markers.get("window")
        if window:
            x0, x1 = mapX(max(window[0], xRange[0])), mapX(min(window[1], xRange[1]))
            if x1 > x0:
                painter.fillRect(qt.QRectF(x0, plotTop, x1 - x0, height), _MARK)
        self._axes(painter, _niceTicks(*xRange), yTicks, mapX, mapY, "Time from time zero (min)",
                   self.units or "Activity")
        painter.setClipRect(qt.QRect(left, plotTop, width + 1, height + 1))
        tStar = self.markers.get("tStar")
        if tStar is not None:
            pen = qt.QPen(_TSTAR)
            pen.setStyle(qt.Qt.DashLine)
            painter.setPen(pen)
            x = mapX(tStar)
            painter.drawLine(qt.QPointF(x, plotTop), qt.QPointF(x, plotTop + height))
        ordered = [c for c in self.curves if not c.get("emphasis")] + [c for c in self.curves if c.get("emphasis")]
        for curve in ordered:
            self._paintCurve(painter, curve, mapX, mapY)
        painter.setClipping(False)
        self._legend(painter, [c for c in self.curves if c.get("emphasis")], tStar is not None)

    def _color(self, curve):
        color = qt.QColor(curve.get("color") or "#3070d0")
        if not curve.get("emphasis"):
            color.setAlphaF(OTHER_OPACITY)
        return color

    def _paintCurve(self, painter, curve, mapX, mapY):
        color = self._color(curve)
        emphasis = bool(curve.get("emphasis"))
        times, values = curve.get("times") or [], curve.get("values") or []
        path, drawing = qt.QPainterPath(), False
        for t, v in zip(times, values):
            if v is None or not math.isfinite(v):
                drawing = False
                continue
            point = qt.QPointF(mapX(t), mapY(v))
            if drawing:
                path.lineTo(point)
            else:
                path.moveTo(point)
                drawing = True
            if emphasis:
                self.points.append((point.x(), point.y(), f"{curve['name']}: {t:.2f} min, {formatValue(v)} "
                                                          f"{self.units}"))
        pen = qt.QPen(color)
        pen.setWidthF(2.2 if emphasis else 1.0)
        painter.setPen(pen)
        painter.setBrush(qt.QBrush(qt.Qt.NoBrush))
        painter.drawPath(path)
        painter.setPen(qt.QPen(qt.Qt.NoPen))
        painter.setBrush(qt.QBrush(color))
        radius = 3.0 if emphasis else 1.8
        for t, v in zip(times, values):
            if v is not None and math.isfinite(v):
                painter.drawEllipse(qt.QPointF(mapX(t), mapY(v)), radius, radius)
        fitTimes, fitValues = curve.get("fitTimes"), curve.get("fitValues")
        if fitTimes is not None and fitValues is not None and len(fitTimes):
            fitPath, drawing = qt.QPainterPath(), False
            for t, v in zip(fitTimes, fitValues):
                if v is None or not math.isfinite(v):
                    drawing = False
                    continue
                point = qt.QPointF(mapX(t), mapY(v))
                if drawing:
                    fitPath.lineTo(point)
                else:
                    fitPath.moveTo(point)
                    drawing = True
            fitColor = qt.QColor(color).darker(150) if emphasis else color
            pen = qt.QPen(fitColor)
            pen.setWidthF(2.0 if emphasis else 1.0)
            pen.setStyle(qt.Qt.DashLine)
            painter.setPen(pen)
            painter.setBrush(qt.QBrush(qt.Qt.NoBrush))
            painter.drawPath(fitPath)

    def _legend(self, painter, curves, tStar):
        left, top, width, _ = self._rect()
        x = left + width
        metrics = painter.fontMetrics()
        items = []
        for curve in curves:
            items.append((curve["name"] + (f" ({curve['role']})" if curve.get("role") else ""), self._color(curve),
                          qt.Qt.SolidLine))
            if curve.get("fitValues") is not None:
                items.append(("fit", qt.QColor(self._color(curve)).darker(150), qt.Qt.DashLine))
        if tStar:
            items.append(("t*", _TSTAR, qt.Qt.DashLine))
        for text, color, style in reversed(items):
            w = metrics.boundingRect(text).width() + 26
            x -= w + 8
            pen = qt.QPen(color)
            pen.setWidthF(2.0)
            pen.setStyle(style)
            painter.setPen(pen)
            y = top - 11
            painter.drawLine(qt.QPointF(x, y), qt.QPointF(x + 18, y))
            painter.setPen(qt.QPen(_AXIS_TEXT))
            painter.drawText(qt.QRectF(x + 22, y - 8, w, 16), qt.Qt.AlignLeft | qt.Qt.AlignVCenter, text)

    def _paintGraphical(self, painter):
        g = self.graphical
        xs = [float(x) for x in g["x"]]
        ys = [float(y) for y in g["y"]]
        good = [(x, y, bool(u)) for x, y, u in zip(xs, ys, g["used"]) if math.isfinite(x) and math.isfinite(y)]
        if not good:
            painter.setPen(qt.QPen(_AXIS_TEXT))
            painter.drawText(qt.QRect(0, 0, self.width, self.height), qt.Qt.AlignCenter, "No points to plot")
            return
        xMax = max(x for x, _, _ in good) * 1.05
        xMin = min(0.0, min(x for x, _, _ in good))
        yMax = max(y for _, y, _ in good) * 1.05 / self.valueZoom
        yMin = min(0.0, min(y for _, y, _ in good))
        xTicks, yTicks = _niceTicks(xMin, xMax), _niceTicks(yMin, yMax)
        mapX, mapY = self._mapper((xMin, xMax), (yMin, yMax))
        self._axes(painter, xTicks, yTicks, mapX, mapY, g.get("xLabel", "x"), g.get("yLabel", "y"))
        left, top, width, height = self._rect()
        painter.setClipRect(qt.QRect(left, top, width + 1, height + 1))
        color = qt.QColor(g.get("color") or "#3070d0")
        if g.get("slope") is not None:
            usedX = [x for x, _, u in good if u] or [xMin, xMax]
            x0, x1 = min(usedX), xMax
            pen = qt.QPen(color.darker(150))
            pen.setWidthF(2.0)
            pen.setStyle(qt.Qt.DashLine)
            painter.setPen(pen)
            painter.drawLine(qt.QPointF(mapX(x0), mapY(g["slope"] * x0 + g["intercept"])),
                             qt.QPointF(mapX(x1), mapY(g["slope"] * x1 + g["intercept"])))
        for x, y, used in good:
            point = qt.QPointF(mapX(x), mapY(y))
            pen = qt.QPen(color)
            pen.setWidthF(1.5)
            painter.setPen(pen)
            painter.setBrush(qt.QBrush(color) if used else qt.QBrush(qt.Qt.NoBrush))
            painter.drawEllipse(point, 3.5, 3.5)
            self.points.append((point.x(), point.y(), f"x {formatValue(x)}, y {formatValue(y)}"
                                                      + ("" if used else " (before t*)")))
        painter.setClipping(False)
        painter.setPen(qt.QPen(_AXIS_TEXT))
        text = f"{g.get('name', '')}: slope {formatValue(g.get('slope'))}, t* {g.get('tStar', 0):.1f} min"
        painter.drawText(qt.QRectF(left, 2, width, 18), qt.Qt.AlignRight | qt.Qt.AlignVCenter, text)

    # --- interaction ---------------------------------------------------------------------------------
    def mouseMoveEvent(self, event):
        x, y = event.pos().x(), event.pos().y()
        best = None
        for px, py, text in self.points:
            distance = (px - x) ** 2 + (py - y) ** 2
            if distance < 100 and (best is None or distance < best[0]):
                best = (distance, text)
        self.toolTip = best[1] if best else ""

    def mouseDoubleClickEvent(self, event):
        self.resetZoom()

    def wheelEvent(self, event):
        steps = event.angleDelta().y() / 120.0
        if not steps:
            return
        if event.modifiers() & qt.Qt.ControlModifier or self.mode == "graphical":
            self.valueZoom = min(max(self.valueZoom * 1.25 ** steps, 0.2), 1000.0)
        else:
            full = (0.0, max([t for c in self.curves for t in (c.get("times") or []) if t is not None] or [1.0]))
            start, end = self.timeRange or full
            left, _, width, _ = self._rect()
            center = start + (event.pos().x() - left) / float(width) * (end - start)
            self.timeRange = zoomRange(start, end, center, 0.8 ** steps, full[0], full[1], minimumSpan=0.5)
        self.update()
        event.accept()
