"""Tests of the Static Neuro Review helpers without Slicer: python -m pytest Testing."""
import math
import os
import sys
import xml.dom.minidom

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "DICOMGatedDynamicPET"))
from DICOMGatedDynamicPETLib import neuroreview as nr, normaldb  # noqa: E402


def test_score_mask_and_sd_floor():
    mean = np.array([0.0, 0.4, 5.0, 10.0, np.nan], dtype=np.float32)
    assert nr.scoreMask(mean).tolist() == [False, False, True, True, False]     # above 5 % of the maximum
    brain = np.array([1, 1, 0, 1, 1])
    assert nr.scoreMask(mean, brain).tolist() == [False, False, False, True, False]
    with pytest.raises(ValueError):
        nr.scoreMask(mean, np.ones(3))
    sd = np.array([0.0, 1.0, 1.0, 0.01, 2.0], dtype=np.float32)
    floored = nr.sdFloor(sd, np.array([0, 1, 1, 1, 1], bool), 0.5)
    assert floored.tolist() == pytest.approx([0.5, 1.0, 1.0, 0.5, 2.0])


def test_uptake_mask():
    mean = np.array([[0.0, 2.0, 4.0, 6.0, 8.0, 10.0]], dtype=np.float32)
    scored = nr.scoreMask(mean, fraction=0.3)       # above 30 % of 10
    assert scored.tolist() == [[False, False, True, True, True, True]]
    shown = nr.uptakeMask(mean, scored)             # above 50 %: the white matter (low in the normals) hidden
    assert shown.tolist() == [[False, False, False, True, True, True]]
    assert nr.uptakeMask(mean, scored, 0.75).tolist() == [[False, False, False, False, True, True]]
    assert not nr.uptakeMask(mean, np.zeros_like(scored), 0.3).any()


def test_grey_matter_mask_and_hidden_scores():
    gm = np.array([0.0, 0.3, 0.5, 0.9, np.nan], dtype=np.float32)
    scored = np.array([1, 1, 1, 0, 1], bool)
    assert nr.greyMatterMask(gm, scored).tolist() == [False, False, True, False, False]
    assert nr.greyMatterMask(gm, scored, 0.25).tolist() == [False, True, True, False, False]
    scores = np.array([-3.0, -1.0, 0.5, 2.5, 4.0], dtype=np.float32)
    mask = np.array([1, 1, 1, 1, 0], bool)
    shown = nr.maskedScores(scores, mask, hideBelow=2.0)
    assert shown.tolist() == [-3.0, nr.OUTSIDE, nr.OUTSIDE, 2.5, nr.OUTSIDE]
    assert nr.maskedScores(scores, mask).tolist()[:4] == [-3.0, -1.0, 0.5, 2.5]
    halved = nr.maskedScores(scores, mask, factor=2.0, hideBelow=1.0)     # hidden after the factor (t)
    assert halved.tolist()[:4] == [-1.5, nr.OUTSIDE, nr.OUTSIDE, 1.25]


def test_brightness_colors():
    values = np.array([0.0, 1.0, 2.0, 4.0, -1.0], dtype=np.float32)
    brightness = nr.petBrightness(values, percentile=100.0, gamma=1.0)
    assert brightness.tolist() == pytest.approx([0.0, 0.25, 0.5, 1.0, 0.0])
    rgba = np.array([[0, 0, 1, 1], [0, 1, 0, 1], [1, 0, 0, 1]], dtype=float)       # blue, green, red
    scores = np.array([-5.0, 0.0, 9.0, 5.0, 0.0], dtype=np.float32)
    rgb = nr.brightnessColors(scores, brightness, rgba, 5.0)
    assert rgb.dtype == np.uint8 and rgb.shape == (5, 3)
    assert rgb[0].tolist() == [0, 0, 0]                     # no uptake: black whatever the score
    assert rgb[1].tolist() == [0, 64, 0]                    # 0: green, a quarter bright
    assert rgb[2].tolist() == [128, 0, 0]                   # above the window: red
    assert rgb[3].tolist() == [255, 0, 0]


def test_voxel_scores_sign_and_t():
    mean = np.full(4, 10.0, np.float32)
    sd = np.full(4, 2.0, np.float32)
    patient = np.array([6.0, 10.0, 14.0, 6.0], np.float32)
    mask = np.array([1, 1, 1, 0], bool)
    z = nr.scoreMap(patient, mean, sd, mask, nr.SCORE_Z)
    assert z.tolist() == pytest.approx([-2.0, 0.0, 2.0, 0.0])        # below the mean: negative
    t = nr.scoreMap(patient, mean, sd, mask, nr.SCORE_T, n=4)
    assert t[0] == pytest.approx(-2.0 / math.sqrt(1.25))
    with pytest.raises(ValueError):
        nr.scoreMap(patient[:3], mean, sd, mask)
    assert nr.regionScore(6, 10, 2, kind=nr.SCORE_Z) == pytest.approx(-2.0)
    assert nr.regionScore(6, 10, 2, 4, nr.SCORE_T) == pytest.approx(-2.0 / math.sqrt(1.25))
    assert nr.regionScore(6, 10, 0) is None and nr.regionScore(None, 10, 1) is None


def test_patient_group():
    header = {"ageStratified": True, "ageGroups": [{"name": "Young", "min": 20, "max": 49},
                                                   {"name": "Old", "min": 50, "max": 90}],
              "statistics": {"groups": {"master": {"name": "Master", "mean": "a", "sd": "b", "n": 10},
                                        "age_Young": {"name": "Young", "mean": "c", "sd": "d", "n": 4}}}}
    assert nr.statisticsGroups(header)[0][0] == normaldb.MASTER
    key = normaldb.groupKey("Young")
    assert nr.patientGroup(header, 30)[0] == key
    assert nr.patientGroup(header, None)[0] == normaldb.MASTER
    assert nr.patientGroup(header, 60)[0] == normaldb.MASTER       # Old has no maps
    assert nr.patientGroup({"statistics": header["statistics"]}, 30)[0] == normaldb.MASTER


def test_regions():
    values = np.arange(8, dtype=np.float32)
    masks = {"A": values < 4, "B": values >= 4, "Empty": values > 100}
    means = nr.regionMeans(values, masks)
    assert means["A"] == pytest.approx(1.5) and means["B"] == pytest.approx(5.5) and math.isnan(means["Empty"])
    subjects = {"s1": {"A": 1.0, "B": 2.0}, "s2": {"A": 3.0, "B": 4.0}, "s3": {"A": 5.0}}
    groups = {"master": {"name": "Master", "ids": ["s1", "s2", "s3"]}, "g": {"name": "G", "ids": ["s1"]}}
    stats = nr.regionStatistics(subjects, groups)
    assert set(stats) == {"master"}
    assert stats["master"]["regions"]["A"] == pytest.approx([3.0, 2.0, 3])
    assert stats["master"]["regions"]["B"][2] == 2
    found, missing = nr.matchRegions(stats, "master", ["a ", "C"])
    assert list(found) == ["a "] and missing == ["C"]
    approx = nr.approximateRegions(values, np.ones(8), {"A": masks["A"]})
    assert approx["A"] == pytest.approx([1.5, 1.0, None])
    assert nr.masksSignature(masks) == nr.masksSignature(dict(reversed(list(masks.items()))))
    assert nr.masksSignature(masks) != nr.masksSignature(masks, extra=(2,))
    rows = nr.tableRows({"A": 6.0, "X": 1.0}, {"A": 7.0}, {"A": [10.0, 2.0, 4]}, nr.SCORE_Z)
    assert rows[0] == ("A", 7.0, 6.0, 10.0, 2.0, 4, pytest.approx(-2.0))
    assert rows[1][-1] is None
    assert nr.isDroppedSegment("Optic Chiasm") and not nr.isDroppedSegment("Putamen R")


def test_label_map_and_lookup():
    a = np.zeros((2, 3, 4), dtype=bool)
    b = np.zeros((2, 3, 4), dtype=bool)
    a[0, 0, :2] = True
    b[0, 0, 1:3] = True                             # overlaps A at i = 1: A kept
    labels, names = nr.labelMap({"A": a, "B": b}, a.shape)
    assert names == ["A", "B"] and labels[0, 0].tolist() == [1, 1, 2, 0]
    ijkToRas = np.diag([2.0, 1.0, 1.0, 1.0])      # x = 2 i
    assert nr.labelAt(labels, ijkToRas, (4.0, 0.0, 0.0)) == 2
    assert nr.labelAt(labels, ijkToRas, (0.9, 0.2, 0.0)) == 1
    assert nr.labelAt(labels, ijkToRas, (100.0, 0.0, 0.0)) == 0


def test_inside_point():
    labels = np.zeros((1, 9, 9), dtype=np.int16)
    labels[0, 0, :] = 1                             # an L: its centroid is outside it
    labels[0, :, 0] = 1
    point = nr.insidePoint(labels, 1)
    assert labels[point] == 1
    assert nr.insidePoint(labels, 2) is None


def test_json_roundtrip(tmp_path):
    path = os.path.join(tmp_path, "stats", "regions.json")
    nr.writeJson(path, nr.regionsDocument({"master": {}}, "suvrFile", 3))
    assert nr.readJson(path)["version"] == 3
    assert nr.readJson(os.path.join(tmp_path, "none.json")) is None
    header = {"statistics": {"regions": nr.REGIONS_FILE}}
    assert nr.regionsPath(str(tmp_path), header) == path
    assert nr.regionsPath(str(tmp_path), {}) is None


def test_score_colors():
    ramp = np.linspace(0, 1, 256)[:, None] * np.ones((1, 3))
    rgba = nr.divergingColors(ramp, count=5)               # window -m, -m/2, 0, m/2, m
    assert rgba[:, 3].tolist() == [1, 1, 1, 1, 1]
    assert rgba[0, :3].tolist() == [0, 0, 0] and rgba[4, :3].tolist() == [1, 1, 1]
    hypo = nr.divergingColors(ramp, hypoOnly=True, count=5)
    assert hypo[3, :3].tolist() == list(nr.GREY) and hypo[4, :3].tolist() == list(nr.GREY)
    assert hypo[2, :3].tolist() == pytest.approx([0.5, 0.5, 0.5], abs=0.01)    # 0 keeps its colour
    masked = nr.maskedScores(np.array([2.0, -4.0]), np.array([True, False]), 2.0)
    assert masked.tolist() == [1.0, nr.OUTSIDE]


def test_score_color_tables():
    for label in ("Plasma", "Viridis"):
        table = nr.colorTable(label)
        assert table.shape == (256, 3) and table.min() >= 0 and table.max() <= 1
    assert np.allclose(nr.colorTable("Viridis")[-1], (0xfd / 255, 0xe7 / 255, 0x25 / 255))
    assert nr.colorTable("ColdToHotRainbow") is None
    assert {"Plasma", "Viridis"} <= set(nr.SCORE_COLORMAPS)
    assert nr.SCORE_COLORMAPS[:2] == ("SSP_Turbo", "SSP_IceFire") and nr.DEFAULT_MAXIMUM == 4.0
    for label in [label for label in nr.SCORE_COLORMAPS if label.startswith("SSP")]:
        table = nr.colorTable(label)
        assert table.shape == (256, 3) and table.min() >= 0 and table.max() <= 1, label
    rainbow = nr.colorTable("SSP_Rainbow")
    assert rainbow[0].tolist() == [0, 0, 0] and rainbow[-1].tolist() == [1, 1, 1]
    assert rainbow[126].tolist() == pytest.approx([0, 1, 0])
    steps = nr.colorTable("SSP_Rainbow10")
    assert len({tuple(row) for row in steps}) == 10 and steps[25].tolist() == [0, 0, 0]
    assert steps[26].tolist() == [0.3, 0, 0.5] and steps[-1].tolist() == [1, 1, 1]
    assert nr.colorTable("SSP-PET-DICOM")[63].tolist() == pytest.approx([0, 0, 0.8])


def test_reference_regions_left_out():
    assert nr.isReferenceRegion("Cerebellum Gray Matter (reference)")
    assert nr.isReferenceRegion("My mask (Reference)", "custom")
    assert nr.isReferenceRegion("Cerebellum Gray Matter", "cerebellumGM")
    assert not nr.isReferenceRegion("Cerebellum White Matter", "cerebellumGM")
    assert nr.isReferenceRegion("Cerebellum White Matter", "cerebellum")
    assert nr.isReferenceRegion("Brainstem", "pons") and nr.isReferenceRegion("Pons", "brainstem")
    assert not nr.isReferenceRegion("Brainstem", "cerebellumGM")
    assert nr.isReferenceRegion("Pons R", "pons") and nr.isReferenceRegion("Brainstem L", "pons")
    assert nr.isReferenceRegion("Cerebellum Gray Matter L", "cerebellumGM")
    assert not nr.isReferenceRegion("Putamen R", "pons")
    assert not nr.isReferenceRegion("Precuneus", "global") and not nr.isReferenceRegion("Precuneus")


def test_layout_xml():
    document = xml.dom.minidom.parseString(nr.layoutXml())
    tags = [v.getAttribute("singletontag") for v in document.getElementsByTagName("view")]
    assert len(tags) == 13 and nr.TABLE_VIEW in tags
    assert nr.sliceViewName(nr.SCORE_ROW, "Sagittal") in tags and nr.sspViewName("Right") in tags
    dual = xml.dom.minidom.parseString(nr.layoutXml(nr.LAYOUT_DUAL, ' name="main"'))
    layouts = dual.documentElement.getElementsByTagName("layout")
    assert dual.documentElement.tagName == "viewports"
    assert layouts[0].getAttribute("name") == "main"
    floating = [l for l in layouts if l.getAttribute("dockable") == "true"]
    assert len(floating) == 1 and len(floating[0].getElementsByTagName("view")) == 6
    assert len(xml.dom.minidom.parseString(nr.layoutXml(nr.LAYOUT_SLICES)).getElementsByTagName("view")) == 7
    assert len(xml.dom.minidom.parseString(nr.layoutXml(nr.LAYOUT_SSP)).getElementsByTagName("view")) == 6
    for layoutId in nr.LAYOUT_IDS:          # only the table's splitter: the views keep equal sizes
        assert nr.layoutXml(layoutId).count('split="true"') == (0 if layoutId == nr.LAYOUT_SSP else 1)
    assert nr.defaultLayout(2) == nr.LAYOUT_DUAL and nr.defaultLayout(1) == nr.LAYOUT_SINGLE
