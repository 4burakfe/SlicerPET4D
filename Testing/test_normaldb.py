"""Tests of the Neuro PET normal database helpers without Slicer: python -m pytest Testing."""
import json
import os
import sys
import zipfile

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "DICOMGatedDynamicPET"))
from DICOMGatedDynamicPETLib import normaldb  # noqa: E402

GROUPS = [{"name": "20-39", "min": 20, "max": 39}, {"name": "40-59", "min": 40, "max": 59},
          {"name": "60+", "min": 60, "max": None}]


def header(name="FDG normals", **fields):
    fields.setdefault("nuclide", "F-18")
    fields.setdefault("tracer", "FDG")
    return normaldb.newHeader(name, normaldb.STATIC, normaldb.USER_DEFINED, **fields)


def test_age_groups():
    assert normaldb.validateAgeGroups(GROUPS) == []
    assert normaldb.validateAgeGroups([]) == ["Define at least one age group"]
    errors = normaldb.validateAgeGroups([{"name": "a", "min": 20, "max": 50}, {"name": "b", "min": 50, "max": 60}])
    assert any("overlap" in e for e in errors)
    assert any("twice" in e for e in normaldb.validateAgeGroups([{"name": "a", "min": 1, "max": 2},
                                                                  {"name": "a", "min": 3, "max": 4}]))
    assert any("not valid" in e for e in normaldb.validateAgeGroups([{"name": "a", "min": 50, "max": 20}]))
    assert normaldb.ageGroupOf(39, GROUPS) == "20-39"
    assert normaldb.ageGroupOf(85, GROUPS) == "60+"
    assert normaldb.ageGroupOf(10, GROUPS) is None
    assert normaldb.ageGroupText(GROUPS[2]) == "60+ (60+)"
    assert normaldb.ageSummary([30, 50, None, ""]) == {"n": 2, "mean": 40.0, "sd": 14.1, "min": 30.0, "max": 50.0}


def test_header_validation():
    assert normaldb.validateHeader(header()) == []
    errors = normaldb.validateHeader(header(name=" ", tracer="", nuclide=""))
    assert len(errors) == 3
    assert normaldb.validateHeader(header(ageStratified=True, ageGroups=[])) == ["Define at least one age group"]
    with pytest.raises(KeyError):
        normaldb.newHeader("x", nope=1)


def test_create_list_delete(tmp_path):
    root = str(tmp_path)
    folder = normaldb.createDatabase(root, header(name="FDG / normals"))
    assert os.path.basename(folder) == "FDG _ normals"
    assert os.path.isdir(os.path.join(folder, normaldb.SUBJECTS_DIR))
    assert normaldb.readSubjects(folder) == []
    with pytest.raises(ValueError):
        normaldb.createDatabase(root, header(name="FDG / normals"))
    normaldb.writeSubjects(folder, [{"id": "S1", "age": 44}, {"id": "S2", "age": 61}])
    (tmp_path / "not a db").mkdir()
    (tmp_path / "broken").mkdir()
    (tmp_path / "broken" / normaldb.HEADER_NAME).write_text("{")
    found, errors = normaldb.listDatabases(root)
    assert [h["name"] for _, h in found] == ["FDG / normals"]
    assert len(errors) == 1 and errors[0][0].endswith("broken")
    assert normaldb.listDatabases(root, normaldb.DYNAMIC)[0] == []
    row = normaldb.summary(*found[0])
    assert row["subjects"] == 2 and row["statistics"] == "Not computed" and row["ageGroups"] == "No"
    assert row["reference"] == "Cerebellar grey matter" and row["version"] == 1
    with pytest.raises(ValueError):
        normaldb.deleteDatabase(str(tmp_path / "not a db"), root)
    with pytest.raises(ValueError):
        normaldb.deleteDatabase(folder, str(tmp_path / "broken"))
    normaldb.deleteDatabase(folder, root)
    assert not os.path.exists(folder)


def _maps(tmp_path, names):
    maps = {}
    for name in names:
        entry = {"n": 10}
        for which in ("mean", "sd"):
            path = tmp_path / "in" / f"{normaldb.safeName(name)}_{which}.nii.gz"
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(which.encode())
            entry[which] = str(path)
        maps[name] = entry
    return maps


def test_predefined_and_package(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    h = header(name="cnpet", ageStratified=True, ageGroups=GROUPS[:2], mapType=normaldb.MAP_SUV)
    maps = _maps(tmp_path, [normaldb.MASTER, "20-39"])
    with pytest.raises(ValueError, match="40-59"):
        normaldb.createPredefined(str(root), h, maps)
    assert not (root / "cnpet").exists()
    maps = _maps(tmp_path, [normaldb.MASTER, "20-39", "40-59"])
    maps["40-59"]["n"] = 3
    folder = normaldb.createPredefined(str(root), h, maps)
    saved = normaldb.readHeader(folder)
    assert saved["kind"] == normaldb.PREDEFINED and saved["mapType"] == "SUV"
    assert saved["statistics"]["groups"]["age_40-59"] == {"name": "40-59", "n": 3,
                                                          "mean": "stats/age_40-59_mean.nii.gz",
                                                          "sd": "stats/age_40-59_sd.nii.gz"}
    assert (root / "cnpet" / "stats" / "master_sd.nii.gz").read_bytes() == b"sd"
    assert normaldb.subjectCount(folder, saved) == 10
    assert not os.path.exists(os.path.join(folder, normaldb.SUBJECTS_DIR))
    with pytest.raises(ValueError, match="user defined"):
        normaldb.exportPredefined(folder, str(tmp_path / "x.zip"))


def test_export_import(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    folder = normaldb.createDatabase(str(root), header(name="mine"))
    with pytest.raises(ValueError, match="Compute"):
        normaldb.exportPredefined(folder, str(tmp_path / "x.zip"))
    normaldb.writeSubjects(folder, [{"id": "S1", "age": 40, "suvFile": "subjects/S1_SUV.nii.gz"},
                                    {"id": "S2", "age": 60}])
    (root / "mine" / "subjects" / "S1_SUV.nii.gz").write_bytes(b"patient")
    for which in ("mean", "sd"):
        (root / "mine" / "stats" / f"master_{which}.nii.gz").write_bytes(which.encode())
    (root / "mine" / normaldb.REFERENCE_MASK_NAME).write_bytes(b"mask")
    h = normaldb.readHeader(folder)
    h["referenceMask"] = normaldb.REFERENCE_MASK_NAME
    h["statistics"] = {"computed": "2026-11-07T10:00:00", "stale": True, "groups": {
        normaldb.MASTER: {"name": normaldb.MASTER, "n": 2, "mean": "stats/master_mean.nii.gz",
                          "sd": "stats/master_sd.nii.gz"}}}
    normaldb.writeHeader(folder, h)
    package = str(tmp_path / "mine.zip")
    normaldb.exportPredefined(folder, package)
    with zipfile.ZipFile(package) as archive:
        names = sorted(archive.namelist())
    assert names == ["mine/database.json", "mine/reference_mask.nii.gz", "mine/stats/master_mean.nii.gz",
                     "mine/stats/master_sd.nii.gz"]
    packed = normaldb.readPackage(package)
    assert packed["kind"] == normaldb.PREDEFINED and packed["subjectAges"]["n"] == 2
    with pytest.raises(ValueError, match="already exists"):
        normaldb.importPackage(str(root), package)
    imported = normaldb.importPackage(str(root), package, name="mine copy")
    ih = normaldb.readHeader(imported)
    assert ih["name"] == "mine copy" and ih["kind"] == normaldb.PREDEFINED
    assert normaldb.subjectCount(imported, ih) == 2
    assert open(os.path.join(imported, "stats", "master_mean.nii.gz"), "rb").read() == b"mean"


def test_import_refuses_unsafe_paths(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    package = tmp_path / "evil.zip"
    with zipfile.ZipFile(package, "w") as archive:
        archive.writestr("db/database.json", json.dumps(header(name="evil")))
        archive.writestr("db/../../outside.txt", "x")
    with pytest.raises(ValueError, match="unsafe"):
        normaldb.importPackage(str(root), str(package))
    assert not (root / "evil").exists() and not (tmp_path / "outside.txt").exists()
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad, "w") as archive:
        archive.writestr("readme.txt", "x")
    with pytest.raises(ValueError, match="not a packaged"):
        normaldb.readPackage(str(bad))


def test_mean_and_sd():
    arrays = [np.full((2, 2), v, dtype=np.float32) for v in (1.0, 2.0, 3.0)]
    mean, sd = normaldb.meanAndSd(arrays)
    assert np.allclose(mean, 2.0) and np.allclose(sd, 1.0) and mean.dtype == np.float32
    mean, sd = normaldb.meanAndSd(arrays[:1])
    assert np.allclose(sd, 0.0)
    with pytest.raises(ValueError):
        normaldb.meanAndSd([np.zeros(2), np.zeros(3)])
    with pytest.raises(ValueError):
        normaldb.meanAndSd([])


def test_modality_nuclides_tracers():
    assert normaldb.nuclides(normaldb.SPECT) == ["Tc-99m", "I-123", "Tl-201"]
    assert "F-18" in normaldb.nuclides() and "Tc-99m" not in normaldb.nuclides()
    assert normaldb.tracers("F-18", normaldb.PET)[:2] == ["F-18", "FDG"]
    assert "PiB" not in normaldb.tracers("F-18") and "PiB" in normaldb.tracers("c-11")
    assert normaldb.tracers("Tc-99m", normaldb.SPECT) == ["Tc-99m", "HMPAO", "ECD", "TRODAT-1"]
    assert normaldb.tracers("Tc-99m", normaldb.PET) == ["Tc-99m"]
    assert normaldb.tracers("Lu-177") == ["Lu-177"]
    assert normaldb.tracers("") == []
    spect = normaldb.newHeader("perfusion", modality=normaldb.SPECT, nuclide="Tc-99m", tracer="HMPAO")
    assert spect["mapType"] == normaldb.MAP_RATIO and normaldb.validateHeader(spect) == []
    spect["mapType"] = normaldb.MAP_SUVR
    assert any("Ratio or Counts" in e for e in normaldb.validateHeader(spect))
    assert any("modality" in e for e in normaldb.validateHeader(header(modality="CT")))
    assert normaldb.newHeader("x")["mapType"] == normaldb.MAP_SUVR


def test_old_header_defaults_to_pet(tmp_path):
    folder = normaldb.createDatabase(str(tmp_path), header(name="old"))
    data = json.load(open(normaldb.headerPath(folder)))
    del data["modality"]
    json.dump(data, open(normaldb.headerPath(folder), "w"))
    loaded = normaldb.readHeader(folder)
    assert loaded["modality"] == normaldb.PET
    assert normaldb.summary(folder, loaded)["modality"] == "PET"


def test_region_settings_in_header():
    h = header(regionAtlas="groupedBilateral", regionGrowMm=2.0, growInto=normaldb.GROW_ONLY_WM)
    assert normaldb.validateHeader(h) == []
    assert header()["growInto"] == normaldb.GROW_NO_WM
    assert any("growInto" in e for e in normaldb.validateHeader(header(growInto="nope")))
    assert normaldb.growInto({"growIntoWhiteMatter": True}) == normaldb.GROW_WITH_WM
    assert normaldb.growInto({}) == normaldb.GROW_NO_WM
    assert normaldb.regionAtlasTitle("groupedBilateral") == "CerebrA broad regions, sides merged"
    assert any("growth" in e for e in normaldb.validateHeader(header(regionGrowMm=-1)))
    assert any("region atlas" in e for e in normaldb.validateHeader(header(regionAtlas="nope")))
    assert header()["regionAtlas"] == normaldb.DEFAULT_REGION_ATLAS and header()["regionGrowMm"] == 0.0


def test_normalization_settings(tmp_path):
    assert normaldb.normalizationText(None) == "Not specified"
    db = {"method": "pet", "template": "GM", "engine": "iterative", "linear": "Affine", "deformable": False,
          "brainMask": True}
    text = normaldb.normalizationText(db)
    assert "Grey matter" in text and "Iterative" in text and "no deformable" in text and "skull-stripped" in text
    assert normaldb.normalizationDifferences(db, db) == []
    used = dict(db, template="T1", engine="brainsfit", deformable=True)
    diffs = normaldb.normalizationDifferences(db, used)
    assert len(diffs) == 3 and diffs[0].startswith("template T1 (database: Grey matter")
    assert any("deformable stage on" in d for d in diffs)
    assert normaldb.normalizationDifferences({"method": "anatomy", "template": "T1"},
                                             {"method": "anatomy", "template": "T2"}) == []
    assert normaldb.normalizationDifferences({}, used) == []
    assert any("normalization" in e for e in normaldb.validateHeader(header(normalization={"method": "x"})))
    folder = normaldb.createDatabase(str(tmp_path), header(name="old"))
    data = json.load(open(normaldb.headerPath(folder)))
    del data["normalization"]
    del data["growInto"]
    data["growIntoWhiteMatter"] = True          # before the three growth targets
    json.dump(data, open(normaldb.headerPath(folder), "w"))
    old = normaldb.readHeader(folder)
    assert old["normalization"] == normaldb.emptyNormalization()
    assert old["growInto"] == normaldb.GROW_WITH_WM and "growIntoWhiteMatter" not in old


def test_add_subject(tmp_path):
    folder = normaldb.createDatabase(str(tmp_path), header(name="db"))
    assert normaldb.subjectFileNames("sub/1") == ("subjects/sub_1_SUV.nii.gz", "subjects/sub_1_SUVR.nii.gz")
    norm = {"method": "pet", "template": "GM"}
    h = normaldb.addSubject(folder, {"id": "S1", "age": 52, "sex": "M", "referenceMean": "7.1",
                                     "reference": "Cerebellum Gray Matter (reference)"}, norm)
    assert h["normalization"]["template"] == "GM" and h["statistics"]["stale"] is False
    assert normaldb.readSubjects(folder)[0]["reference"] == "Cerebellum Gray Matter (reference)"
    with pytest.raises(ValueError, match="already"):
        normaldb.addSubject(folder, {"id": "s1"})
    assert normaldb.hasSubject(folder, "s1")
    h["statistics"]["computed"] = "2026-11-08T10:00:00"
    normaldb.writeHeader(folder, h)
    h = normaldb.addSubject(folder, {"id": "S1", "age": 53}, {"method": "anatomy"}, replace=True)
    assert h["statistics"]["stale"] is True and h["normalization"]["method"] == "pet"
    normaldb.addSubject(folder, {"id": "S2", "age": 60})
    rows = normaldb.readSubjects(folder)
    assert [r["id"] for r in rows] == ["S1", "S2"] and rows[0]["age"] == "53" and rows[0]["added"]
    with pytest.raises(ValueError, match="ID"):
        normaldb.addSubject(folder, {"id": " "})
    predefined = normaldb.createPredefined(str(tmp_path), header(name="pre"), _maps(tmp_path, [normaldb.MASTER]))
    with pytest.raises(ValueError, match="user defined"):
        normaldb.addSubject(predefined, {"id": "S1"})
