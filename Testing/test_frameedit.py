"""Tests of the Review/Edit frame editing without Slicer: python -m pytest Testing."""
import datetime
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "DICOMGatedDynamicPET"))
from DICOMGatedDynamicPETLib import frameedit as fe  # noqa: E402

T0 = datetime.datetime(2026, 1, 1, 10, 0, 0)
F18 = 6586.2


def originals(durations=(10, 10, 30, 60)):
    out, start = [], 0.0
    for i, d in enumerate(durations):
        out.append({"key": f"o{i + 1}", "start": start, "duration": float(d), "label": f"F{i + 1}"})
        start += d
    return out


def info(start=T0, durations=(10, 10, 30, 60), units="Bq/ml", reference=None, **values):
    frames, t = [], start
    for d in durations:
        frames.append({"start": t, "duration": float(d)})
        t += datetime.timedelta(seconds=d)
    values.setdefault("decayCorrected", True)
    values.setdefault("decayCorrection", "START")
    values.setdefault("halfLife", F18)
    return fe.seriesInfo(name="series", units=units, frames=frames,
                         decayReference=reference if reference is not None else start, **values)


def test_merge_split_and_edited_flags():
    edit = fe.FrameEdit(originals())
    assert not edit.isEdited()
    assert edit.merge([1, 2]) == 1
    assert edit.isEdited()
    intervals = edit.intervals()
    assert [(i["start"], i["end"], i["duration"], i["edited"]) for i in intervals] == [
        (0, 10, 10, False), (10, 50, 40, True), (50, 110, 60, False)]
    assert intervals[1]["sources"] == "F2 + F3 (merged)"
    assert edit.frameWeights(1) == [("o2", 0.25), ("o3", 0.75)]
    with pytest.raises(fe.EditError):
        edit.merge([0, 2])                          # not neighbours
    with pytest.raises(fe.EditError):
        edit.split([0])                             # not merged
    assert edit.split([1]) == [1, 2]
    assert not edit.isEdited()


def test_merge_of_merged_frames_and_split_gives_the_atoms_back():
    edit = fe.FrameEdit(originals())
    edit.merge([0, 1])
    edit.merge([0, 1])                              # (F1 + F2) + F3
    assert edit.frames[0] == ["o1", "o2", "o3"]
    edit.split([0])
    assert edit.frames == [["o1"], ["o2"], ["o3"], ["o4"]]


def test_duration_weighted_mean():
    edit = fe.FrameEdit(originals())
    arrays = {"o1": np.full(4, 1.0), "o2": np.full(4, 2.0), "o3": np.full(4, 4.0), "o4": np.full(4, 8.0)}
    assert edit.frameArray(0, arrays.get) is arrays["o1"]       # untouched frame: its own array
    edit.merge([1, 2])
    np.testing.assert_allclose(edit.frameArray(1, arrays.get), (2.0 * 10 + 4.0 * 30) / 40)


def test_remove():
    edit = fe.FrameEdit(originals())
    edit.remove([0, 3])
    assert edit.frames == [["o2"], ["o3"]]
    with pytest.raises(fe.EditError):
        edit.remove([0, 1])


def test_add_places_by_time_and_rejects_overlaps():
    edit = fe.FrameEdit(originals())
    late = [{"start": 3600.0, "duration": 600.0, "label": "Late F1"}]
    assert edit.add("Late", late, scaleAdded=1.2) == [4]
    assert edit.isFrameEdited(4) and edit.frameSources(4) == "Late F1 x 1.2 (added)"
    early = [{"start": -100.0, "duration": 50.0, "label": "Early F1"}]
    assert edit.add("Early", early) == [0]
    assert [round(i["start"]) for i in edit.intervals()] == [-100, 0, 10, 20, 50, 3600]
    with pytest.raises(fe.EditError):
        edit.add("Overlap", [{"start": 40.0, "duration": 30.0, "label": "X"}])
    edit.add("Touching", [{"start": 110.5, "duration": 10.0, "label": "T"}])     # within the tolerance
    edit.reset()
    assert not edit.isEdited() and len(edit.frames) == 4


def test_add_scales_the_existing_frames():
    edit = fe.FrameEdit(originals())
    edit.add("Early", [{"start": -600.0, "duration": 300.0, "label": "E"}], scaleExisting=1.1)
    assert all(edit.isFrameEdited(i) for i in range(5))
    assert edit.scale("o1") == pytest.approx(1.1)
    arrays = {k: np.ones(2) for k in ("o1", "o2", "o3", "o4", "a1.1")}
    np.testing.assert_allclose(edit.frameArray(1, arrays.get), 1.1)


def test_compare_bq_ml_aligns_to_the_series_that_starts_first():
    base = info()
    later = T0 + datetime.timedelta(hours=1)
    added = info(start=later, durations=(600,))
    result = fe.compareSeries(base, added)
    assert not result["errors"]
    assert result["scaleBase"] == 1.0
    assert result["scaleAdded"] == pytest.approx(2.0 ** (3600 / F18))
    assert result["reference"] == T0
    assert any("between the two series" in w for w in result["warnings"])      # 1 h > 30 min
    # the added series first: the edited frames are corrected to its reference
    earlier = info(start=T0 - datetime.timedelta(minutes=20), durations=(300,))
    result = fe.compareSeries(base, earlier)
    assert result["scaleAdded"] == 1.0
    assert result["scaleBase"] == pytest.approx(2.0 ** (1200 / F18))
    assert not result["warnings"]                                              # 15 min apart


def test_compare_suv_and_refusals():
    base = info(units="SUVbw")
    added = info(start=T0 + datetime.timedelta(hours=2), durations=(600,), units="SUVbw")
    result = fe.compareSeries(base, added, longGapS=3 * 3600)
    assert not result["errors"] and not result["warnings"]
    assert result["scaleAdded"] == result["scaleBase"] == 1.0
    assert fe.compareSeries(base, info(units="Bq/ml"))["errors"]                # SUV with Bq/ml
    assert fe.compareSeries(base, info(units="SUVlbm"))["errors"]               # other normalisation
    assert fe.compareSeries(info(), info(halfLife=1220.0))["errors"]            # C-11 with F-18
    assert fe.compareSeries(info(), info(decayCorrected=False))["errors"]       # corrected with not corrected
    noTime = info()
    noTime["frames"] = [{"start": None, "duration": 10.0}]
    assert fe.compareSeries(info(), noTime)["errors"]


def test_compare_warnings():
    base = info(patientID="1", patientName="A^B", manufacturer="GE", model="Discovery MI", injection=T0)
    added = info(start=T0 + datetime.timedelta(minutes=5), durations=(60,), patientID="2", patientName="C^D",
                 manufacturer="Siemens", model="Biograph", injection=T0 + datetime.timedelta(minutes=3))
    added["frames"][0]["start"] = T0 + datetime.timedelta(seconds=200)
    warnings = " ".join(fe.compareSeries(base, added)["warnings"])
    for text in ("patient IDs", "patient names", "scanner", "injection"):
        assert text in warnings
    rows = dict((r[0], r[1:]) for r in fe.compareSeries(base, added)["rows"])
    assert rows["Frames"] == ("4", "1")
    assert rows["Scanner"][0] == "GE Discovery MI"


def test_timing_rows():
    edit = fe.FrameEdit(originals())
    edit.merge([0, 1])
    edit.add("Late", [{"start": 3600.0, "duration": 600.0, "label": "L"}])
    rows = edit.timingRows(clockAtZero=T0, injection=T0 - datetime.timedelta(minutes=1), scanStart=T0)
    assert [r["Frame"] for r in rows] == [1, 2, 3, 4]
    assert rows[0]["DurationS"] == 20 and rows[0]["StartFromInjectionS"] == 60
    assert rows[3]["StartDateTime"] == "2026-01-01T11:00:00"
    assert rows[3]["StartFromStudyStartS"] == 3600 and rows[3]["EndFromInjectionS"] == 4260
    assert "merged" in rows[0]["StartSource"]


def test_undo_redo():
    edit = fe.FrameEdit(originals())
    assert not edit.canUndo()
    edit.merge([0, 1])
    edit.add("Late", [{"start": 3600.0, "duration": 600.0, "label": "L"}], scaleExisting=1.5)
    edit.remove([0])
    assert edit.undoText() == "Remove"
    assert edit.undo() == "Remove" and len(edit.frames) == 4
    assert edit.undo() == "Add Late" and len(edit.frames) == 3 and edit.scale("o3") == 1.0
    assert edit.redoText() == "Add Late"
    assert edit.redo() == "Add Late" and edit.frames[-1] == ["a1.1"] and edit.scale("o3") == 1.5
    edit.discard()
    assert not edit.isEdited() and edit.undo() == "Discard changes" and edit.isEdited()
    assert edit.undo() == "Add Late" and edit.canRedo()
    # a new operation clears the redo history; keys of added series are never used twice
    assert edit.nextAddedKeys(2) == ["a2.1", "a2.2"]
    edit.add("Other", [{"start": 5000.0, "duration": 60.0, "label": "O"}])
    assert not edit.canRedo() and edit.frames[-1] == ["a2.1"]
    assert edit.undo() == "Add Other"
    assert edit.undo() == "Merge" and not edit.isEdited() and not edit.canUndo()
    edit.merge([0, 1])
    edit.reset()
    assert not edit.canUndo()
