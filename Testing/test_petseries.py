"""Tests of DICOMGatedDynamicPETLib.petseries without Slicer: python -m pytest Testing (from the repository root)."""
import os
import sys
import tempfile

import numpy as np
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
