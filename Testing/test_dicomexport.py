"""Tests of the DICOM export of dynamic PET without Slicer: python -m pytest Testing."""
import datetime
import os
import sys

import numpy as np
import pydicom

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "DICOMGatedDynamicPET"))
from DICOMGatedDynamicPETLib import dicomexport as de  # noqa: E402
from DICOMGatedDynamicPETLib import petseries as ps  # noqa: E402

F18 = 6586.2


def _testFolder(prefix="tmp"):
    """A temporary folder of a test, removed when Python (Slicer) exits."""
    import atexit
    import shutil
    import tempfile
    folder = tempfile.mkdtemp(prefix=prefix)
    atexit.register(shutil.rmtree, folder, True)
    return folder


def load(files):
    """What the loader makes of a series: (Bq/ml frames, IJK to RAS, timing rows, datasets)."""
    datasets = [pydicom.dcmread(f) for f in files]
    layout = ps.arrangeFrames(datasets, ps.DYNAMIC)
    suv = ps.suvInfo(datasets)
    frames = [ps.frameArray(slots, suv["activityScale"]) for slots in layout["slots"]]
    origin, spacing, direction = ps.gridGeometry(layout["slots"][0], layout["grid"])
    ijkToRas = np.asarray(ps.lpsToRasMatrix(origin, spacing, direction))
    return frames, ijkToRas, ps.frameTiming(layout["frames"], datasets), datasets, layout


def test_export_reads_back_the_same():
    files, _ = ps.writeSyntheticSeries(_testFolder(), kind=ps.DYNAMIC, frames=3, frameDurationsS=[10, 30, 60])
    frames, ijkToRas, rows, datasets, _ = load(files)
    rng = np.random.default_rng(1)
    frames = [f * rng.uniform(0.5, 1.5, f.shape) for f in frames]
    merged = [frames[0], (frames[1] * 30 + frames[2] * 60) / 90]                 # an edit: F2 + F3 merged
    starts = [rows[0]["StartDateTime"], rows[1]["StartDateTime"]]
    out = de.buildDatasets(merged, ijkToRas, [{"StartDateTime": starts[0], "DurationS": 10},
                                              {"StartDateTime": starts[1], "DurationS": 90}],
                           datasets[0], "Bq/ml", "Edited", 901, "START",
                           datetime.datetime(2026, 1, 1, 10, 0, 0), F18, frameNotes=[None, "F2 + F3 (merged)"])
    folder = _testFolder()
    paths = de.writeDatasets(folder, out)
    assert len(paths) == 2 * 6
    back, backMatrix, backRows, backDatasets, layout = load(paths)
    assert layout["method"].startswith("Image Index")
    np.testing.assert_allclose(backMatrix, ijkToRas, atol=1e-4)
    for a, b in zip(back, merged):
        np.testing.assert_allclose(a, b, rtol=1e-4, atol=float(np.abs(b).max()) * 1e-4)
    assert [r["StartFromStudyStartS"] for r in backRows] == [0, 10]
    assert [r["DurationS"] for r in backRows] == [10, 90]
    first = backDatasets[0]
    assert first.StudyInstanceUID == datasets[0].StudyInstanceUID
    assert first.SeriesInstanceUID != datasets[0].SeriesInstanceUID
    assert first.PatientID == datasets[0].PatientID
    assert ps.datasetKind(first) == ps.DYNAMIC
    assert ps.scanStart(backDatasets)[0] == datetime.datetime(2026, 1, 1, 10, 0, 0)
    assert ps.suvInfo(backDatasets)["suvScale"] == ps.suvInfo(datasets)["suvScale"]
    assert "merged" in str(backDatasets[-1].DerivationDescription)
    assert float(backDatasets[-1].DecayFactor) > 1.0


def test_refusals():
    files, _ = ps.writeSyntheticSeries(_testFolder(), kind=ps.DYNAMIC, frames=2)
    frames, ijkToRas, rows, datasets, _ = load(files)
    for kwargs in ({"units": "SUVlbm"}, {"frames": [frames[0], frames[1][:2]]},
                   {"rows": [{"StartDateTime": None, "DurationS": 30}] * 2},
                   {"decayReference": None}):
        args = dict(frames=frames, ijkToRas=ijkToRas, rows=rows, template=datasets[0], units="Bq/ml",
                    description="x", seriesNumber=1, decayCorrection="START",
                    decayReference=datetime.datetime(2026, 1, 1, 10))
        args.update(kwargs)
        try:
            de.buildDatasets(**args)
        except de.ExportError:
            continue
        raise AssertionError(f"not refused: {kwargs}")


def test_decay_factor_and_pixels():
    assert de.decayFactor(0.0, 1e-9, F18) == pytest_approx(1.0)
    assert de.decayFactor(F18, 1e-9, F18) == pytest_approx(2.0)
    pixels, slope = de.scaledPixels(np.array([[0.0, -5.0], [10.0, 2.5]]))
    np.testing.assert_allclose(pixels * slope, [[0.0, -5.0], [10.0, 2.5]], atol=slope)


def pytest_approx(value):
    import pytest
    return pytest.approx(value, rel=1e-6)


def test_streamed_datasets_check_inputs_at_once_and_match_the_list():
    files, _ = ps.writeSyntheticSeries(_testFolder(), kind=ps.DYNAMIC, frames=2)
    frames, ijkToRas, rows, datasets, _ = load(files)
    args = dict(frames=frames, ijkToRas=ijkToRas, rows=rows, template=datasets[0], units="Bq/ml",
                description="x", seriesNumber=1, decayCorrection="START",
                decayReference=datetime.datetime(2026, 1, 1, 10), seriesInstanceUID="1.2.3.4")
    try:
        de.iterDatasets(**dict(args, units="SUVlbm"))     # refused at the call, before any image is built
    except de.ExportError:
        pass
    else:
        raise AssertionError("not refused at the call")
    streamed = de.iterDatasets(**args)
    assert not isinstance(streamed, list)
    paths = de.writeDatasets(_testFolder(), streamed)
    listed = de.buildDatasets(**args)
    assert len(paths) == len(listed) == 2 * 6
    for path, ds in zip(paths, listed):
        written = pydicom.dcmread(path)
        assert written.SeriesInstanceUID == "1.2.3.4" == ds.SeriesInstanceUID
        assert (written.ImageIndex, written.PixelData) == (ds.ImageIndex, ds.PixelData)
        assert written.ImagePositionPatient == ds.ImagePositionPatient


def test_scaled_pixels_without_values():
    image = np.array([[np.nan, -5.0], [10.0, np.inf], [-np.inf, 2.5]])
    pixels, slope = de.scaledPixels(image)
    np.testing.assert_allclose(slope, 10.0 / 32767.0)
    np.testing.assert_allclose(pixels * slope, [[0.0, -5.0], [10.0, 0.0], [0.0, 2.5]], atol=slope)
    pixels, slope = de.scaledPixels(np.full((2, 2), np.nan))
    assert slope == 1.0 and not pixels.any()
