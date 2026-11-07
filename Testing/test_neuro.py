"""Tests of the Dynamic Neuro Receptor PET helpers without Slicer: python -m pytest Testing."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "DICOMGatedDynamicPET"))
from DICOMGatedDynamicPETLib import neuro  # noqa: E402

CSV = """Mindboggle ID,Label Name,RH Label,LH Labels,Notes,Dice Kappa
2002,Caudal Anterior Cingulate,30,81,,0.79
16,Brainstem,11,62,"Completed filling, removed labelled voxels",0.65
45,Cerebellum Gray Matter,46,97,"Completed filling using threshold for CGM, removed cerebellum labels",0.83
51,Putamen,21,72,Corrected uniformity using threshold,0.87
"""


def test_find_atlas_files(tmp_path):
    icbm = tmp_path / "mni_icbm152_nlin_sym_09c_nifti" / "mni_icbm152_nlin_sym_09c"
    icbm.mkdir(parents=True)
    for name in ("mni_icbm152_t1_tal_nlin_sym_09c.nii", "mni_icbm152_t2_tal_nlin_sym_09c.nii",
                 "mni_icbm152_t1_tal_nlin_sym_09c_mask.nii", "mni_icbm152_t1_tal_nlin_sym_09c_eye_mask.nii"):
        (icbm / name).write_bytes(b"")
    cerebra = tmp_path / "CerebrA"
    cerebra.mkdir()
    (cerebra / "mni_icbm152_CerebrA_tal_nlin_sym_09c.nii.gz").write_bytes(b"")
    (cerebra / "CerebrA_LabelDetails.csv").write_text(CSV)
    found = neuro.findAtlasFiles(str(tmp_path))
    assert found["T1"].endswith("mni_icbm152_t1_tal_nlin_sym_09c.nii")
    assert found["T2"].endswith("t2_tal_nlin_sym_09c.nii")
    assert found[neuro.BRAIN_MASK].endswith("_mask.nii") and "eye" not in found[neuro.BRAIN_MASK]
    assert found[neuro.CEREBRA_LABELS].endswith(".nii.gz")
    assert found[neuro.CEREBRA_CSV].endswith("CerebrA_LabelDetails.csv")
    missing = neuro.missingFiles(found)
    assert "mni_icbm152_pd_tal_nlin_sym_09c.nii" in missing and "mni_icbm152_t1_tal_nlin_sym_09c.nii" not in missing
    assert neuro.findAtlasFiles(str(tmp_path / "nope")) == {}


@pytest.mark.parametrize("args, expected", [
    (dict(modality="CT"), "CT"),
    (dict(modality="MR", texts=["Sag T1 MPRAGE"]), "T1"),
    (dict(modality="MR", texts=["T1W_3D_TFE"]), "T1"),
    (dict(modality="MR", texts=["AX T2 FLAIR"]), "T2"),
    (dict(modality="MR", texts=["Ax T1 FLAIR post"]), "T1"),
    (dict(modality="MR", texts=["t2_tse_tra"]), "T2"),
    (dict(modality="MR", texts=["PD axial"]), "PD"),
    (dict(modality="MR", texts=["3D BRAVO"]), "T1"),
    (dict(modality="MR", texts=["series 5"], repetitionTime="4000", echoTime="100"), "T2"),
    (dict(modality="MR", texts=["series 5"], repetitionTime="500", echoTime="10"), "T1"),
    (dict(modality="MR", texts=["series 5"], repetitionTime="2300", echoTime="2.9", inversionTime="900"), "T1"),
    (dict(modality="MR", texts=["series 5"], repetitionTime="3000", echoTime="15"), "PD"),
    (dict(modality="MR", texts=["series 5"]), None),
    (dict(texts=["patient CT 2.5mm"]), "CT"),
    (dict(texts=["ACT scan"]), None),
])
def test_classify(args, expected):
    contrast, reason = neuro.classifyAnatomy(**args)
    assert contrast == expected, reason
    assert reason


def test_template_for_contrast():
    assert neuro.templateForContrast("T2") == "T2"
    assert neuro.templateForContrast("PD") == "PD"
    for c in ("T1", "CT", None):
        assert neuro.templateForContrast(c) == "T1"


def test_read_labels_text_and_file(tmp_path):
    labels = neuro.readCerebraLabels(CSV)
    names = ["Caudal Anterior Cingulate", "Brainstem", "Cerebellum Gray Matter", "Putamen"]
    assert [s["name"] for s in labels] == names
    assert labels[3]["right"] == 21 and labels[3]["left"] == 72
    path = tmp_path / "labels.csv"
    path.write_text("﻿" + CSV, encoding="utf-8")
    assert neuro.readCerebraLabels(str(path)) == labels
    assert neuro.structureValues(labels, "cerebellum gray matter") == (46, 97)
    assert neuro.structureValues(labels, "nope") == ()
    with pytest.raises(ValueError):
        neuro.readCerebraLabels("a,b\n1,2\n")


def test_real_label_file():
    """The CerebrA file shipped with the atlas: 51 structures, labels 1 - 102 each used once."""
    path = os.environ.get("CEREBRA_CSV")
    if not path or not os.path.isfile(path):
        pytest.skip("set CEREBRA_CSV to CerebrA_LabelDetails.csv")
    labels = neuro.readCerebraLabels(path)
    assert len(labels) == 51
    values = sorted(v for s in labels for v in (s["right"], s["left"]))
    assert values == list(range(1, 103))


def test_label_table_and_remap():
    labels = neuro.readCerebraLabels(CSV)
    entries, mapping = neuro.labelTable(labels)
    names = {v: n for v, n, _ in entries}
    assert names[21] == "Putamen R" and names[72] == "Putamen L" and len(entries) == 8
    assert all(mapping[v] == v for v in names)
    assert [v for v, _, _ in entries] == sorted(names)
    merged, mergedMap = neuro.labelTable(labels, mergeSides=True)
    assert [n for _, n, _ in merged] == [s["name"] for s in labels]
    assert mergedMap[21] == mergedMap[72] == 4
    array = np.array([[0, 21, 72], [46, 97, 5]], dtype=np.float32)
    np.testing.assert_array_equal(neuro.remapLabels(array, mergedMap), [[0, 4, 4], [3, 3, 0]])
    np.testing.assert_array_equal(neuro.remapLabels(array.astype(np.int16), mapping), [[0, 21, 72], [46, 97, 0]])
    for _, _, color in entries:
        assert all(0.0 <= c <= 1.0 for c in color)


def test_bilateral_files(tmp_path):
    labels = neuro.readCerebraLabels(CSV)
    text = neuro.bilateralCsvText(labels)
    assert text.splitlines()[0] == "Label,Label Name,RH Label,LH Label"
    assert text.splitlines()[4] == "4,Putamen,21,72"
    entries = neuro.readBilateralLabels(text)
    assert [(v, n) for v, n, _ in entries] == [(1, "Caudal Anterior Cingulate"), (2, "Brainstem"),
                                               (3, "Cerebellum Gray Matter"), (4, "Putamen")]
    # the same values as the merged label table
    merged, _ = neuro.labelTable(labels, mergeSides=True)
    assert [(v, n) for v, n, _ in merged] == [(v, n) for v, n, _ in entries]
    with pytest.raises(ValueError):
        neuro.readBilateralLabels(CSV)
    # where they are written, and found once there
    cerebra = tmp_path / "CerebrA"
    cerebra.mkdir()
    (cerebra / "mni_icbm152_CerebrA_tal_nlin_sym_09c.nii").write_bytes(b"")
    (cerebra / "CerebrA_LabelDetails.csv").write_text(CSV)
    found = neuro.findAtlasFiles(str(tmp_path))
    labelsPath, csvPath = neuro.bilateralPaths(found)
    assert labelsPath == str(cerebra / "mni_icbm152_CerebrA_bilateral_tal_nlin_sym_09c.nii")
    assert csvPath == str(cerebra / "CerebrA_bilateral_LabelDetails.csv")
    assert all("bilateral" not in m for m in neuro.missingFiles(found))
    (cerebra / "mni_icbm152_CerebrA_bilateral_tal_nlin_sym_09c.nii").write_bytes(b"")
    (cerebra / "CerebrA_bilateral_LabelDetails.csv").write_text(text)
    found = neuro.findAtlasFiles(str(tmp_path))
    assert found[neuro.BILATERAL_LABELS].endswith("bilateral_tal_nlin_sym_09c.nii")
    assert found[neuro.CEREBRA_LABELS].endswith("mni_icbm152_CerebrA_tal_nlin_sym_09c.nii")
    assert found[neuro.BILATERAL_CSV].endswith("CerebrA_bilateral_LabelDetails.csv")
    assert found[neuro.CEREBRA_CSV].endswith("CerebrA_LabelDetails.csv")
    assert neuro.bilateralPaths({}) == (None, None)


def test_cerebra_semicolon_csv_and_pons(tmp_path):
    text = ("Mindboggle ID;Label Name;RH Label;LH Labels;Notes;Dice Kappa\n"
            "16;Brainstem;11;62;Completed filling, removed labelled voxels;0,65\n"
            "444;Pons;103;104;;0\n")
    structures = neuro.readCerebraLabels(text)
    assert [s["name"] for s in structures] == ["Brainstem", "Pons"]
    assert structures[0]["notes"] == "Completed filling, removed labelled voxels"
    assert neuro.structureValues(structures, "pons") == (103, 104)
    folder = tmp_path / "mni_icbm152_nlin_sym_09c"
    folder.mkdir()
    (folder / "mni_icbm152_CerebrA_tal_nlin_sym_09c.nii").write_bytes(b"")
    (folder / "mni_icbm152_CerebrA_tal_nlin_sym_09c_wpons.seg.nrrd").write_bytes(b"")
    (folder / "mni_icbm152_CerebrA_tal_nlin_sym_09c_wpons.csv").write_text(text)
    found = neuro.findAtlasFiles(str(tmp_path))
    assert found[neuro.PONS_LABELS].endswith("_wpons.seg.nrrd")
    assert found[neuro.PONS_CSV].endswith("_wpons.csv")
    assert found[neuro.CEREBRA_LABELS].endswith("CerebrA_tal_nlin_sym_09c.nii")
    assert not any("wpons" in name for name in neuro.missingFiles({}))
    assert neuro.segmentNamesOfLabels((103, 104), "Pons") == {"segment_103", "segment_104", "pons", "pons r",
                                                              "pons l"}


def test_pons_overlay_helpers(tmp_path):
    text = ("Mindboggle ID;Label Name;RH Label;LH Labels;Notes;Dice Kappa\n"
            "51;Putamen;21;72;;0,87\n16;Brainstem;11;62;;0,65\n444;Pons;103;104;;0\n")
    structures = neuro.readCerebraLabels(text)
    assert neuro.overlayValues(structures) == {103: "Pons R", 104: "Pons L"}
    assert neuro.overlayValues(structures, mergeSides=True) == {3: "Pons"}
    assert neuro.labelsCsv({}) is None
    assert neuro.labelsCsv({neuro.CEREBRA_CSV: "a.csv"}) == "a.csv"
    found = {neuro.CEREBRA_CSV: "a.csv", neuro.PONS_CSV: str(tmp_path / "p.csv"),
             neuro.PONS_LABELS: str(tmp_path / "p.seg.nrrd")}
    assert neuro.hasPons(found) and neuro.labelsCsv(found).endswith("p.csv")
    lateral, bilateral = neuro.ponsOverlayPaths(found)
    assert lateral == str(tmp_path / (neuro.PONS_OVERLAY_STEM + ".nii"))
    assert bilateral == str(tmp_path / (neuro.PONS_BILATERAL_OVERLAY_STEM + ".nii"))
    assert neuro.ponsOverlayPaths({}) == (None, None)
    for path in (lateral, bilateral):
        open(path, "wb").close()
    (tmp_path / "mni_icbm152_CerebrA_tal_nlin_sym_09c_wpons.seg.nrrd").write_bytes(b"")
    again = neuro.findAtlasFiles(str(tmp_path))
    assert again[neuro.PONS_OVERLAY] == lateral and again[neuro.PONS_BILATERAL_OVERLAY] == bilateral
    assert again[neuro.PONS_LABELS].endswith(".seg.nrrd")
