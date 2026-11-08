"""Tests of the Static Neuro PET/SPECT helpers without Slicer: python -m pytest Testing."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "DICOMGatedDynamicPET"))
from DICOMGatedDynamicPETLib import normaldb, staticneuro as sn  # noqa: E402


def test_identify_nuclide_and_tracer():
    assert sn.identifyNuclide("^18^Fluorine") == "F-18"
    assert sn.identifyNuclide("F18") == "F-18"
    assert sn.identifyNuclide("99mTc") == "Tc-99m"
    assert sn.identifyNuclide("Tc-99m") == "Tc-99m"
    assert sn.identifyNuclide("Iodine-123") == "I-123"
    assert sn.identifyNuclide("^11^Carbon") == "C-11"
    assert sn.identifyNuclide("Fluorodeoxyglucose") is None and sn.identifyNuclide("") is None
    assert sn.identifyTracer("Fluorodeoxyglucose") == "FDG"
    assert sn.identifyTracer("18F-FDG") == "FDG"
    assert sn.identifyTracer("Tc99m HMPAO") == "HMPAO"
    assert sn.identifyTracer("DaTSCAN") == "Ioflupane (DaTscan)"
    assert sn.identifyTracer("AV-1451") == "Flortaucipir"
    assert sn.identifyTracer("C-111A1") is None


def test_image_info_and_units():
    pet = sn.imageInfo({"StaticPET.Kind": "static", "StaticPET.Units": "SUVbw",
                        "StaticPET.Radiopharmaceutical": "Fluorodeoxyglucose",
                        "StaticPET.Radionuclide": "^18^Fluorine"})
    assert (pet["modality"], pet["unitsKind"], pet["nuclide"], pet["tracer"]) == ("PET", "SUV", "F-18", "FDG")
    assert sn.unitWarnings(pet) == []
    bq = dict(pet, units="Bq/ml", unitsKind=sn.UNITS_BQML)
    assert "not affected" in sn.unitWarnings(bq, {"mapType": normaldb.MAP_SUVR})[0]
    assert "load the PET as SUV" in sn.unitWarnings(bq, {"mapType": normaldb.MAP_SUV})[0]
    spect = sn.imageInfo({"SPECTRecon.Units": "Bq/ml", "SPECTRecon.Isotope": "Tc-99m",
                          "SPECTRecon.IsotopeDicomText": "Tc99m Exametazime"})
    assert (spect["modality"], spect["unitsKind"], spect["nuclide"], spect["tracer"]) == \
        ("SPECT", "Bq/ml", "Tc-99m", "HMPAO")
    assert sn.unitWarnings(spect) == []
    assert sn.imageInfo({}, "NM")["modality"] == "SPECT"
    assert "Not recognised" in sn.unitWarnings(sn.imageInfo({}))[0]
    assert sn.imageInfo({"StaticPET.Radiopharmaceutical": "FDG"})["nuclide"] == "F-18"


def test_database_match():
    info = {"modality": "PET", "nuclide": "F-18", "tracer": "FDG"}
    fdg = normaldb.newHeader("fdg", nuclide="F-18", tracer="FDG")
    unknownTracer = normaldb.newHeader("f18", nuclide="F-18", tracer="F-18")
    pib = normaldb.newHeader("pib", nuclide="C-11", tracer="PiB")
    spect = normaldb.newHeader("hmpao", modality="SPECT", nuclide="Tc-99m", tracer="HMPAO")
    assert sn.databaseMatch(fdg, info) == (0, [])
    assert sn.databaseMatch(unknownTracer, info) == (0, [])
    rank, problems = sn.databaseMatch(pib, info)
    assert rank == 3 and len(problems) == 2
    rows = sn.sortDatabases([("d", spect), ("c", pib), ("b", unknownTracer), ("a", fdg)], info)
    assert [r[1]["name"] for r in rows] == ["f18", "fdg", "pib", "hmpao"]


def test_region_settings(tmp_path):
    header = normaldb.newHeader("db", nuclide="F-18", tracer="FDG", regionAtlas="grouped", regionGrowMm=2.0,
                                growIntoWhiteMatter=True, referenceErosionMm=1.0, smoothingFwhmMm=8.0,
                                referenceRegion=normaldb.REF_PONS, referenceMask="reference_mask.nii.gz")
    settings = sn.regionSettings(str(tmp_path), header)
    assert settings["referenceMaskPath"] is None and settings["referenceAtlasRegion"] == "Pons"
    (tmp_path / "reference_mask.nii.gz").write_bytes(b"x")
    settings = sn.regionSettings(str(tmp_path), header)
    assert settings["referenceMaskPath"].endswith("reference_mask.nii.gz") and settings["growMm"] == 2.0
    text = "\n".join(sn.settingsText(settings))
    assert "broad regions" in text and "white matter" in text and "8 mm" in text and "database's mask" in text
    assert sn.masterMeanPath(str(tmp_path), header) is None
    (tmp_path / "stats").mkdir()
    (tmp_path / "stats" / "master_mean.nii.gz").write_bytes(b"m")
    header["statistics"]["groups"][normaldb.MASTER] = {"mean": "stats/master_mean.nii.gz"}
    assert sn.masterMeanPath(str(tmp_path), header).endswith("master_mean.nii.gz")


def test_import_bids(tmp_path):
    root = tmp_path / "cnpet"
    folder = root / "sub-00000016" / "ses-SUV" / "pet"
    folder.mkdir(parents=True)
    image = folder / "sub-00000016_ses-SUV_pet.nii.gz"
    image.write_bytes(b"")
    sidecar = {"Manufacturer": "Siemens", "Units": "Bq/mL", "TracerName": "C-111A1", "TracerRadionuclide": "F18",
               "FrameTimesStart": [5580], "FrameDuration": [180], "ReconMethodName": "PSF 3i21s"}
    (folder / "sub-00000016_ses-SUV_pet.json").write_text(json.dumps(sidecar))
    (root / "participants.tsv").write_text("participant_id\tage\tsex\nsub-00000015\t40\tF\nsub-00000016\t52\tM\n")
    assert sn.parseBidsName(str(image)) == {"sub": "00000016", "ses": "SUV", "suffix": "pet"}
    assert sn.parseBidsName("/x/brain.nii") == {}
    assert sn.importName(str(image)) == "sub-00000016_ses-SUV_pet"
    assert sn.sidecarPath(str(image)).endswith(".json")
    participants = sn.findParticipants(str(image))
    assert participants == str(root / "participants.tsv")
    participant = sn.readParticipant(participants, "00000016")
    attributes = sn.importAttributes(str(image), sidecar, participant)
    assert attributes["StaticPET.Units"] == "SUVbw" and attributes["StaticPET.UnitsSource"] == "file name"
    assert attributes["StaticPET.PatientAge"] == "52" and attributes["StaticPET.PatientSex"] == "M"
    assert attributes["StaticPET.DurationS"] == "180" and attributes["StaticPET.StartFromInjectionS"] == "5580"
    info = sn.imageInfo(attributes)
    assert (info["modality"], info["unitsKind"], info["nuclide"], info["tracer"]) == ("PET", "SUV", "F-18", "")
    other = sn.importAttributes(str(tmp_path / "sub-1_ses-baseline_pet.nii"), sidecar)
    assert other["StaticPET.Units"] == "Bq/ml" and "StaticPET.PatientAge" not in other
    spect = sn.importAttributes(str(tmp_path / "sub-2_spect.nii"), {"TracerRadionuclide": "Tc99m"})
    assert sn.imageInfo(spect)["modality"] == "SPECT" and spect["SPECTRecon.Isotope"] == "Tc-99m"


def test_age_and_sex():
    assert sn.parseAge("052Y") == 52 and sn.parseAge("006M") == 0.5 and sn.parseAge("61") == 61
    assert sn.parseAge("") is None and sn.parseAge("abc") is None
    assert sn.parseSex("F") == "F" and sn.parseSex("male") == "M" and sn.parseSex("Kadın") == "F"
    assert sn.parseSex("") == ""
    assert sn.taggedAgeSex({"StaticPET.PatientAge": "070Y", "StaticPET.PatientSex": "M"}) == (70, "M")
    assert sn.taggedAgeSex({"SPECTRecon.PatientSex": "F"}) == (None, "F")
