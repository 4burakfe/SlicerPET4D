"""Tests of DICOMGatedDynamicPETLib.petseries without Slicer: python -m pytest Testing (from the repository root)."""
import os
import sys
import tempfile

import numpy as np
import pytest
import pydicom

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "DICOMGatedDynamicPET"))
from DICOMGatedDynamicPETLib import petseries as ps  # noqa: E402


def _read(files):
    return [pydicom.dcmread(f) for f in files]


def test_kinds():
    assert ps.seriesKind("PT", "GATED\\IMAGE") == ps.GATED
    assert ps.seriesKind("PT", ["DYNAMIC", "IMAGE"]) == ps.DYNAMIC
    assert ps.seriesKind("PT", "STATIC\\IMAGE") == ps.STATIC
    assert ps.seriesKind("PT", "WHOLE BODY\\IMAGE") == ps.STATIC
    assert ps.seriesKind("PT", "", timeSlots="8") == ps.GATED
    assert ps.seriesKind("PT", "", timeSlices="20") == ps.DYNAMIC
    assert ps.seriesKind("CT", "STATIC") is None
    # an explicit Series Type wins over conflicting frame counts
    assert ps.seriesKind("PT", "DYNAMIC\\IMAGE", timeSlots="8", timeSlices="20") == ps.DYNAMIC
    assert ps.seriesKind("PT", "DYNAMIC\\IMAGE", timeSlots="2") == ps.DYNAMIC
    assert ps.seriesKind("PT", "GATED\\IMAGE", timeSlices="20") == ps.GATED
    assert ps.seriesKind("PT", "STATIC\\IMAGE", timeSlots="8") == ps.STATIC
    assert ps.seriesKind("PT", "", timeSlots="8", timeSlices="20") == ps.GATED   # counts only: gates first


def test_static():
    files, expected = ps.writeSyntheticSeries(tempfile.mkdtemp(), kind=ps.STATIC)
    datasets = _read(files)
    assert ps.datasetKind(datasets[0]) == ps.STATIC
    frames, _ = ps.groupFrames(datasets)
    assert len(frames) == 1 and len(frames[0]) == 6
    suv = ps.suvInfo(datasets)
    np.testing.assert_allclose(ps.frameArray(frames[0], suv["suvScale"]), expected[0], rtol=1e-4)
    info = ps.acquisitionInfo(datasets)
    assert info["Radiopharmaceutical"] == "Fluorodeoxyglucose"
    assert info["Radionuclide"] == "^18^Fluorine"
    assert info["DecayCorrection"] == "START"
    assert info["InjectionToScanStartS"] == "3600"
    assert info["PatientHeightCm"] == "170"
    assert info["CorrectedImage"].startswith("DECY")
    assert (info["Manufacturer"], info["ScannerModel"]) == ("Synthetic", "Phantom PET")
    assert (info["PatientName"], info["PatientID"]) == ("Phantom^PET", "PHANTOM")
    timing = ps.frameTiming(frames, datasets)[0]
    assert (timing["StartDateTime"], timing["DurationS"]) == ("2026-01-01T10:00:00", 600)


def test_synthetic_series_options():
    import datetime
    start = datetime.datetime(2026, 1, 2, 11, 30, 0)
    files, _ = ps.writeSyntheticSeries(tempfile.mkdtemp(), kind=ps.STATIC, seriesStart=start,
                                       injection=start - datetime.timedelta(hours=2), patient=("Other^One", "42"),
                                       scanner=("Vendor", "Model X"))
    info = ps.acquisitionInfo(_read(files))
    assert info["ScanStart"] == "2026-01-02T11:30:00" and info["InjectionToScanStartS"] == "7200"
    assert (info["PatientID"], info["ScannerModel"]) == ("42", "Model X")


def test_gated():
    files, expected = ps.writeSyntheticSeries(tempfile.mkdtemp(), kind=ps.GATED)
    datasets = _read(files)
    gates, method = ps.groupGates(datasets)
    assert (len(gates), len(gates[0])) == (4, 6) and method.startswith("Image Index")
    suv = ps.suvInfo(datasets)
    for g, gate in enumerate(gates):
        np.testing.assert_allclose(ps.frameArray(gate, suv["suvScale"]), expected[g], rtol=1e-4)
    assert ps.gatingInfo(gates)["GatingTimeSlots"] == 4


def test_dynamic_timing():
    durations = [10.0, 10.0, 30.0, 60.0, 120.0]
    files, expected = ps.writeSyntheticSeries(tempfile.mkdtemp(), kind=ps.DYNAMIC, frames=5,
                                              frameDurationsS=durations)
    datasets = _read(files)
    assert ps.datasetKind(datasets[0]) == ps.DYNAMIC
    frames, method = ps.groupFrames(datasets)
    assert len(frames) == 5 and method.startswith("Image Index")
    rows = ps.frameTiming(frames)
    assert [r["DurationS"] for r in rows] == durations
    assert [r["DurationMs"] for r in rows] == [d * 1000 for d in durations]
    assert [r["StartFromInjectionS"] for r in rows] == [0, 10, 20, 50, 110]
    assert [r["MidFromInjectionS"] for r in rows] == [5, 15, 35, 80, 170]
    assert [r["StartFromStudyStartS"] for r in rows] == [0, 10, 20, 50, 110]   # from the first frame, not 09:55
    assert rows[2]["StartDateTime"] == "2026-01-01T10:00:20"
    assert rows[1]["FrameReferenceTimeMs"] == 15000
    suv = ps.suvInfo(datasets, ps.frameStartTimes(frames))
    assert suv["frameSuvScales"] is None                       # decay corrected: one scale
    for f, frame in enumerate(frames):
        np.testing.assert_allclose(ps.frameArray(frame, suv["suvScale"]), expected[f], rtol=1e-4)
    mean = ps.durationWeightedMean([ps.frameArray(fr) for fr in frames], [r["DurationS"] for r in rows])
    np.testing.assert_allclose(mean, sum(1000.0 * (i + 1) * d for i, d in enumerate(durations)) / sum(durations))


def test_dynamic_not_decay_corrected():
    durations = [600.0, 600.0, 600.0]
    files, expected = ps.writeSyntheticSeries(tempfile.mkdtemp(), kind=ps.DYNAMIC, frames=3,
                                              frameDurationsS=durations, decayCorrection="NONE", shuffle=False)
    datasets = _read(files)
    frames, _ = ps.groupFrames(datasets)
    suv = ps.suvInfo(datasets, ps.frameStartTimes(frames))
    assert suv["frameSuvScales"] is not None and len(suv["frameSuvScales"]) == 3
    for f, frame in enumerate(frames):
        np.testing.assert_allclose(ps.frameArray(frame, suv["frameSuvScales"][f]), expected[f], rtol=1e-4)
    assert suv["frameSuvScales"][2] > suv["frameSuvScales"][0]   # later frames: less activity left


def _noAcquisitionTime(datasets, frameIndex):
    frames, _ = ps.groupFrames(datasets)
    for ds in frames[frameIndex]:
        del ds.AcquisitionTime
    return frames


def test_not_decay_corrected_frame_without_acquisition_time():
    """Decay Correction NONE: a frame without acquisition time gets the start of the timing table (end of the
    previous frame), not the first frame's time; SUV and timing agree."""
    durations = [600.0, 600.0, 600.0]
    files, expected = ps.writeSyntheticSeries(tempfile.mkdtemp(), kind=ps.DYNAMIC, frames=3,
                                              frameDurationsS=durations, decayCorrection="NONE", shuffle=False)
    reference = ps.frameStartTimes(ps.groupFrames(_read(files))[0])
    datasets = _read(files)
    frames = _noAcquisitionTime(datasets, 2)
    starts = ps.frameStarts(frames)
    assert [source for _, source in starts] == [ps.START_FROM_ACQUISITION] * 2 + [ps.START_FROM_PREVIOUS]
    assert [start for start, _ in starts] == reference              # back to back frames: same start
    suv = ps.suvInfo(datasets, ps.frameStartTimes(frames))
    for f, frame in enumerate(frames):
        np.testing.assert_allclose(ps.frameArray(frame, suv["frameSuvScales"][f]), expected[f], rtol=1e-4)
    rows = ps.frameTiming(frames)
    assert [r["StartFromStudyStartS"] for r in rows] == [0, 600, 1200]
    assert rows[2]["StartSource"] == ps.START_FROM_PREVIOUS
    warnings = ps.frameStartWarnings(frames)
    assert len(warnings) == 1 and "frame 3" in warnings[0]


def test_not_decay_corrected_frame_without_any_start():
    """Without a start for a frame (no acquisition time and no duration before it), SUV is not offered
    instead of borrowing the first frame's time."""
    files, _ = ps.writeSyntheticSeries(tempfile.mkdtemp(), kind=ps.DYNAMIC, frames=3,
                                       frameDurationsS=[600.0] * 3, decayCorrection="NONE", shuffle=False)
    datasets = _read(files)
    frames = _noAcquisitionTime(datasets, 2)
    for ds in frames[1]:
        del ds.ActualFrameDuration
    assert ps.frameStartTimes(frames)[2] is None
    suv = ps.suvInfo(datasets, ps.frameStartTimes(frames))
    assert suv["frameSuvScales"] is None and suv["suvScale"] is None
    assert any("frame 3" in p for p in suv["problems"])
    assert "no start at all" in ps.frameStartWarnings(frames)[0]


def test_overlapping_frames_are_reported():
    files, _ = ps.writeSyntheticSeries(tempfile.mkdtemp(), kind=ps.DYNAMIC, frames=3, frameDurationsS=[600.0] * 3)
    datasets = _read(files)
    frames, _ = ps.groupFrames(datasets)
    for ds in frames[0]:
        ds.ActualFrameDuration = 900000          # 15 min, but frame 2 starts after 10 min
    warnings = ps.frameStartWarnings(frames)
    assert len(warnings) == 1 and "Frame 2 starts before" in warnings[0]
    assert ps.frameStartWarnings(ps.groupFrames(_read(files))[0]) == []


def test_decay_correction_of_not_decay_corrected_frames():
    """NONE: frame f is corrected to the first frame start by 2^(dt/T) x lambda d / (1 - exp(-lambda d)); with
    the frames corrected, one SUV scale (the dose decayed to the first frame start) holds for all frames."""
    durations = [60.0, 600.0, 1200.0]
    files, _ = ps.writeSyntheticSeries(tempfile.mkdtemp(), kind=ps.DYNAMIC, frames=3, frameDurationsS=durations,
                                       decayCorrection="NONE", shuffle=False)
    datasets = _read(files)
    frames, _ = ps.groupFrames(datasets)
    assert not ps.isDecayCorrected(datasets[0])
    reference, factors, problem = ps.decayCorrectionFactors(frames)
    assert problem is None and reference == ps.frameStartTimes(frames)[0]
    halfLife, rate = 6586.2, np.log(2.0) / 6586.2
    starts = [0.0, 60.0, 660.0]
    for factor, start, d in zip(factors, starts, durations):
        assert abs(factor - 2.0 ** (start / halfLife) * rate * d / (1.0 - np.exp(-rate * d))) < 1e-9
    # the first frame only has the decay during the frame: the same as the Decay Factor (0054,1321) a GE
    # SIGNA PET/MR wrote for a 30 s C-11 frame at the scan start (1.00852)
    x = np.log(2.0) / 1223.1 * 30.0
    assert abs(x / -np.expm1(-x) - 1.00852) < 1e-5
    suv = ps.suvInfo(datasets, ps.frameStartTimes(frames), decayCorrectedTo=reference)
    assert suv["frameSuvScales"] is None and "decay corrected by the loader" in suv["DecayReference"]
    assert abs(suv["suvScale"] - 70.0 * 1000.0 / 370e6) < 1e-12   # injection at the first frame start
    # without a duration the frames cannot be corrected
    del frames[1][0].ActualFrameDuration
    assert ps.decayCorrectionFactors(frames)[1] is None
    # START series are decay corrected already
    started = _read(ps.writeSyntheticSeries(tempfile.mkdtemp(), kind=ps.DYNAMIC, frames=2)[0])
    assert ps.isDecayCorrected(started[0])


def test_dynamic_without_image_index():
    files, _ = ps.writeSyntheticSeries(tempfile.mkdtemp(), kind=ps.DYNAMIC, frames=3)
    datasets = _read(files)
    for ds in datasets:
        del ds.ImageIndex
    frames, method = ps.groupFrames(datasets)
    assert len(frames) == 3 and method.startswith("Frame Reference Time")


def test_radiopharmaceutical_codes():
    from pydicom.dataset import Dataset
    radio, code = Dataset(), Dataset()
    code.CodeMeaning = "^82^Rubidium"
    radio.RadionuclideCodeSequence = [code]
    ds = Dataset()
    ds.RadiopharmaceuticalInformationSequence = [radio]
    assert ps.radiopharmaceuticalInfo(ds) == {"Radionuclide": "^82^Rubidium"}
    assert ps.radiopharmaceuticalInfo(Dataset()) == {}


def _drop(datasets, frame=None, slices=()):
    """Datasets without the images of the given frame (0-based, all of it when slices is empty) / slices."""
    nSlices = int(datasets[0].NumberOfSlices)
    out = []
    for ds in datasets:
        f, s = divmod(int(ds.ImageIndex) - 1, nSlices)
        if f == frame and (not slices or s in slices):
            continue
        out.append(ds)
    return out


def test_complete_series_has_no_warnings():
    files, _ = ps.writeSyntheticSeries(tempfile.mkdtemp(), kind=ps.GATED)
    layout = ps.arrangeFrames(_read(files))
    assert layout["warnings"] == [] and layout["numbers"] == [1, 2, 3, 4] and len(layout["grid"]) == 6


def test_missing_gate():
    files, expected = ps.writeSyntheticSeries(tempfile.mkdtemp(), kind=ps.GATED)
    datasets = _drop(_read(files), frame=2)
    import pytest
    with pytest.raises(ValueError):
        ps.groupGates(datasets)
    layout = ps.arrangeFrames(datasets)
    assert layout["numbers"] == [1, 2, 4]
    assert any("gate 3" in w for w in layout["warnings"])
    suv = ps.suvInfo(datasets)
    np.testing.assert_allclose(ps.frameArray(layout["slots"][2], suv["suvScale"]), expected[3], rtol=1e-4)


def test_missing_slices_inside_and_at_end():
    files, expected = ps.writeSyntheticSeries(tempfile.mkdtemp(), kind=ps.DYNAMIC, frames=3)
    datasets = _drop(_read(files), frame=1, slices=(2,))          # a hole in frame 2
    datasets = _drop(datasets, frame=0, slices=(5,))              # top slice of frame 1
    layout = ps.arrangeFrames(datasets)
    assert len(layout["grid"]) == 6 and len(layout["frames"]) == 3
    assert [ds is None for ds in layout["slots"][1]] == [False, False, True, False, False, False]
    assert layout["slots"][0][5] is None
    assert any("slice(s) missing" in w for w in layout["warnings"])
    assert any("interpolated" in w and "left empty" in w for w in layout["warnings"])
    array = ps.frameArray(layout["slots"][1])                    # the hole is interpolated
    assert array.shape[0] == 6 and np.allclose(array[2], 2000.0) and np.allclose(array[3], 2000.0)
    assert np.all(ps.frameArray(layout["slots"][1], interpolate=False)[2] == 0)
    assert np.all(ps.frameArray(layout["slots"][0])[5] == 0)     # nothing above the top gap
    assert list(ps.interpolatedSlices(layout["slots"][1])) == [False, False, True, False, False, False]
    assert not ps.interpolatedSlices(layout["slots"][0]).any()
    origin, spacing, _ = ps.gridGeometry(layout["slots"][1], layout["grid"])
    assert abs(spacing[2] - 3.27) < 1e-6 and abs(origin[2] + 100.0) < 1e-6
    rows = ps.frameTiming(layout["frames"], numbers=layout["numbers"])
    assert [r["Frame"] for r in rows] == [1, 2, 3]


def test_missing_slice_in_every_frame_keeps_spacing():
    files, _ = ps.writeSyntheticSeries(tempfile.mkdtemp(), kind=ps.GATED, slices=8)
    datasets = _read(files)
    for f in range(4):
        datasets = _drop(datasets, frame=f, slices=(3,))
    layout = ps.arrangeFrames(datasets)
    assert len(layout["grid"]) == 8 and all(slots[3] is None for slots in layout["slots"])
    assert any("every gate" in w for w in layout["warnings"])


def test_fill_slice_gaps_is_linear():
    array = np.zeros((7, 2, 2))
    array[0], array[4] = 10.0, 50.0
    present = np.array([True, False, False, False, True, False, False])
    filled = ps.fillSliceGaps(array, present)
    assert [float(v) for v in filled[:, 0, 0]] == [10.0, 20.0, 30.0, 40.0, 50.0, 0.0, 0.0]
    assert np.all(array[1] == 0)                                  # input left alone


def test_slice_wise_mean_prefers_real_slices():
    a, b = np.full((3, 1, 1), 10.0), np.full((3, 1, 1), 30.0)
    a[1] = 999.0                                                  # interpolated value in gate a
    b[2] = 999.0                                                  # interpolated value in gate b
    slotsA, slotsB = [1, None, 1], [1, 1, None]
    mean = ps.SliceWiseMean()
    mean.add(a, slotsA)
    mean.add(b, slotsB)
    assert [float(v) for v in mean.mean()[:, 0, 0]] == [20.0, 30.0, 10.0]
    # a slice no gate has: from the interpolated ones, duration weighted
    mean = ps.SliceWiseMean()
    mean.add(np.full((3, 1, 1), 10.0), [1, None, 1], 1.0)
    mean.add(np.full((3, 1, 1), 40.0), [1, None, 1], 2.0)
    assert np.allclose(mean.mean()[:, 0, 0], 30.0)


def test_examine_warning():
    files, _ = ps.writeSyntheticSeries(tempfile.mkdtemp(), kind=ps.GATED)
    header = pydicom.dcmread(files[0], stop_before_pixels=True)
    assert ps.missingImagesWarning(header, ps.GATED, len(files)) is None
    assert "2 of 24 images missing" in ps.missingImagesWarning(header, ps.GATED, len(files) - 2)


def test_lean_body_mass_and_body_surface_area():
    # Janmahasatian: 80 kg, 180 cm man, BMI 24.69 -> 9270 x 80 / (6680 + 216 x 24.69)
    assert abs(ps.leanBodyMassKg(80, 180, "M") - 9270 * 80 / (6680 + 216 * 80 / 1.8 ** 2)) < 1e-9
    assert ps.leanBodyMassKg(80, 180, "F") < ps.leanBodyMassKg(80, 180, "M")
    assert ps.leanBodyMassKg(80, None, "M") is None and ps.leanBodyMassKg(80, 180, "O") is None
    assert abs(ps.bodySurfaceAreaM2(70, 170) - 1.8097) < 1e-3          # Du Bois


def test_suv_methods():
    files, expected = ps.writeSyntheticSeries(tempfile.mkdtemp(), kind=ps.STATIC)
    datasets = _read(files)
    suv = ps.suvInfo(datasets)
    bw, _, extra, problems = ps.suvScales(suv, "bw")
    assert bw == suv["suvScale"] and not problems and "body weight" in extra["SUVMethod"]
    lbm, _, _, problems = ps.suvScales(suv, "lbm")
    assert lbm is None and "sex" in problems[0]                         # sex O in the synthetic series
    for ds in datasets:
        ds.PatientSex = "F"
    suv = ps.suvInfo(datasets)
    lbm, _, extra, _ = ps.suvScales(suv, "lbm")
    assert abs(lbm / bw - ps.leanBodyMassKg(70, 170, "F") / 70) < 1e-9 and "LeanBodyMassKg" in extra
    bsa, _, extra, _ = ps.suvScales(suv, "bsa")
    assert abs(bsa / bw - ps.bodySurfaceAreaM2(70, 170) * 1e4 / 70000) < 1e-9 and "BodySurfaceAreaM2" in extra
    assert ps.suvMethodOf("SUVlbm") == "lbm" and ps.suvMethodOf("Bq/ml") is None


def test_suv_methods_per_frame():
    files, _ = ps.writeSyntheticSeries(tempfile.mkdtemp(), kind=ps.DYNAMIC, frames=3, frameDurationsS=[600.0] * 3,
                                       decayCorrection="NONE", shuffle=False)
    datasets = _read(files)
    frames, _ = ps.groupFrames(datasets)
    suv = ps.suvInfo(datasets, ps.frameStartTimes(frames))
    _, frameScales, _, _ = ps.suvScales(suv, "bsa")
    factor = ps.bodySurfaceAreaM2(70, 170) * 1e4 / 70000
    np.testing.assert_allclose(frameScales, np.array(suv["frameSuvScales"]) * factor)


def test_summation_weights():
    durations = [60.0, 600.0, 1200.0]
    weights, text = ps.summationWeights(durations)
    assert weights == durations and text == "duration weighted"
    assert ps.summationWeights([60.0, None])[0] is None
    # counts: one static acquisition decay corrected to its start equals the count weighted mean of the decay
    # corrected frames (C-11, 0 - 31 min, a changing distribution)
    halfLife, starts = 1223.4, [0.0, 60.0, 660.0]
    rate = np.log(2.0) / halfLife
    activity = [5.0, 2.0, 1.0]                     # decay corrected concentration in each frame (constant inside)
    counts = [a * np.exp(-rate * s) * -np.expm1(-rate * d) / rate for a, s, d in zip(activity, starts, durations)]
    total = 1860.0
    static = sum(counts) * rate / -np.expm1(-rate * total)        # static frame, decay corrected to its start
    weights, text = ps.summationWeights(durations, starts, halfLife, ps.SUM_COUNTS)
    assert "count weighted" in text
    assert float(ps.weightedMean([np.array(a) for a in activity], weights)) == pytest.approx(static, rel=1e-12)
    assert float(ps.weightedMean([np.array(a) for a in activity], durations)) != pytest.approx(static, rel=1e-3)
    # starts relative to any reference give the same mean
    shifted, _ = ps.summationWeights(durations, [s + 300.0 for s in starts], halfLife, ps.SUM_COUNTS)
    np.testing.assert_allclose(np.array(shifted) / sum(shifted), np.array(weights) / sum(weights))
    # frames not decay corrected already hold counts per second: durations; missing inputs: durations
    assert ps.summationWeights(durations, starts, halfLife, ps.SUM_COUNTS, decayCorrected=False)[0] == durations
    weights, text = ps.summationWeights(durations, None, halfLife, ps.SUM_COUNTS)
    assert weights == durations and "needs" in text
