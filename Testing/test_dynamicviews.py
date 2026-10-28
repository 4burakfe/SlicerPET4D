"""Tests of the Dynamic PET Editor helpers without Slicer: python -m pytest Testing."""
import os
import sys
import xml.etree.ElementTree as ElementTree

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "DICOMGatedDynamicPET"))
from DICOMGatedDynamicPETLib import dynamicviews as dv  # noqa: E402


def test_grid_shape_is_near_square():
    assert dv.gridShape(0) == (0, 0)
    assert dv.gridShape(1) == (1, 1)
    assert dv.gridShape(5) == (2, 3)
    assert dv.gridShape(9) == (3, 3)
    assert dv.gridShape(20) == (4, 5)
    assert dv.gridShape(26) == (5, 6)
    for n in range(1, 101):
        rows, columns = dv.gridShape(n)
        assert rows * columns >= n and columns - rows <= 1 and (rows - 1) * columns < n


def test_layout_xml():
    root = ElementTree.fromstring(dv.layoutXml(7))
    tags = [v.get("singletontag") for v in root.iter("view")]
    assert tags == ["DynPETMain"] + [f"DynPETF{i}" for i in range(1, 8)] + ["DynPETEmpty1", "DynPETEmpty2"]
    rows = root.find("item[2]/layout")
    assert [len(r.findall("layout/item")) for r in rows.findall("item")] == [3, 3, 3]   # last row filled
    assert dv.emptyViewCount(17) == 3 and dv.emptyViewCount(20) == 0 and dv.emptyViewCount(16) == 0


def test_format_time():
    assert dv.formatTime(0) == "0:00"
    assert dv.formatTime(65) == "1:05"
    assert dv.formatTime(7.5) == "0:07.5"
    assert dv.formatTime(3725) == "1:02:05"
    assert dv.formatTime(None) == "?"


def test_frame_intervals():
    rows = [{"Frame": "1", "StartFromStudyStartS": "300", "DurationS": "10"},
            {"Frame": "2", "StartFromStudyStartS": "310", "DurationS": "30"},
            {"Frame": "4", "StartFromStudyStartS": "400", "DurationS": None}]
    intervals = dv.frameIntervals(rows)
    assert [(i["number"], i["start"], i["end"]) for i in intervals] == [(1, 300, 310), (2, 310, 340), (4, 400, 401)]
    assert intervals[0]["timeBase"] == "study start"
    assert dv.frameLabel(intervals[1]) == "F2  5:10 - 5:40"
    fallback = dv.frameIntervals([{"DurationS": "5"}, {"DurationS": "15"}])
    assert [(i["start"], i["end"]) for i in fallback] == [(0, 5), (5, 20)]
    assert dv.frameIntervals([{}, {}])[0]["timeBase"] == "frame order"


def test_ruler_rows_and_frame_times():
    rows = [{"Frame": "1", "StartFromStudyStartS": "300", "DurationS": "10", "StartDateTime": "2026-01-01T10:00:00",
             "StartFromInjectionS": "0"},
            {"Frame": "2", "StartFromStudyStartS": "310", "DurationS": "30", "StartDateTime": "2026-01-01T10:00:10",
             "StartFromInjectionS": "10"}]
    intervals = dv.frameIntervals(rows)
    assert intervals[0]["injectionAt"] == 300
    assert intervals[0]["clockAtZero"].isoformat() == "2026-01-01T09:55:00"
    titles = [r[0] for r in dv.rulerRows(intervals)]
    assert titles == ["Clock", "Study", "Injection"]
    assert dv.frameTimes(intervals, 1) == "clock 10:00:10 - 10:00:40, study 5:10 - 5:40, injection 0:10 - 0:40"
    assert [r[0] for r in dv.rulerRows(dv.frameIntervals([{"DurationS": "5"}, {"DurationS": "5"}]))] == ["Time"]
    assert dv.formatClock(36000 + 61) == "10:01:01"


def test_tick_step():
    assert dv.niceTickStep(3600, 1000) == 300
    assert dv.niceTickStep(60, 1000) == 5


def test_zoom_range():
    assert dv.zoomRange(0, 100, 50, 0.5, 0, 100) == (25, 75)
    assert dv.zoomRange(0, 100, 0, 0.5, 0, 100) == (0, 50)
    assert dv.zoomRange(25, 75, 50, 4.0, 0, 100) == (0, 100)              # never wider than the series
    assert dv.zoomRange(80, 100, 95, 1.5, 0, 100) == (70, 100)            # kept inside the series
    assert dv.zoomRange(49.8, 50.2, 50, 0.5, 0, 100) == (49.5, 50.5)      # at least 1 s


def test_color_tables():
    from DICOMGatedDynamicPETLib import colortables
    for label in colortables.LABELS:
        table = colortables.table(label)
        assert table.shape == (256, 3) and table.min() >= 0 and table.max() <= 1
        assert tuple(table[0]) == (0, 0, 0)


def test_playback_helpers():
    intervals = dv.frameIntervals([{"StartFromStudyStartS": "0", "DurationS": "30"},
                                   {"StartFromStudyStartS": "30", "DurationS": "300"},
                                   {"StartFromStudyStartS": "330", "DurationS": "5070"}])
    assert [dv.frameAt(intervals, t) for t in (-5, 0, 29.9, 30, 1000, 9999)] == [0, 0, 0, 1, 2, 2]
    assert dv.defaultPlaySpeed(intervals) == 90                      # 1:30:00 in 60 s
    assert dv.speedText(90) == "x90  (1 s = 1:30)"
    assert dv.speedText(0.5) == "x0.5  (1 s = 0:00.5)"
