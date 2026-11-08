"""
Static Neuro PET/SPECT (3D Slicer module) - workflow app

A step-by-step workflow for a static brain PET or SPECT (FDG, amyloid, perfusion SPECT, ...), built on the Dynamic Neuro
Receptor PET module (same normalization and regions) and the Neuro PET normal databases:

    1. Data                 the static PET (PET4D DICOM plugin) or SPECT (DICOM SPECT reconstruction), or an image
                            file (BIDS PET: sub-<id>_ses-<session>_pet.nii.gz with its .json sidecar and the
                            participants.tsv), the patient's CT / MR, the age and sex (read from the tags, editable)
    2. CT/MR registration   optional: the image registered to the CT / MR (rigid by default) and hardened
    3. Normalization        the normal database (those of the image's modality, nuclide and tracer first; its
                            normalization settings are the defaults, others warn) and the normalization to the MNI
                            ICBM 152 2009c space: the PET/SPECT to an MR template, the CT/MR to the MR template with the
                            same transform applied to the image, or the PET/SPECT to the database's master mean map
    4. Regions              the CerebrA regions: with a database its region settings (atlas, growth, white matter,
                            reference region and its mask) are used, locked unless unlocked
    5. Review               the comparison with the database comes here; adding the patient to a user defined
                            database (SUV and SUVR maps on the database grid, a row in subjects.csv)
"""

import json
import logging
import os
import sys
import time

import ctk
import numpy as np
import qt
import slicer
import vtk
from slicer.ScriptedLoadableModule import ScriptedLoadableModule, ScriptedLoadableModuleTest

_MODULE_DIR = os.path.dirname(os.path.abspath(globals().get("__file__", "") or "."))
if globals().get("__file__") and _MODULE_DIR not in sys.path:
    sys.path.insert(0, _MODULE_DIR)

from DICOMGatedDynamicPETLib import neuro, normaldb  # noqa: E402
from DICOMGatedDynamicPETLib import registration as reg  # noqa: E402
from DICOMGatedDynamicPETLib import resampling  # noqa: E402
from DICOMGatedDynamicPETLib import staticneuro as sn  # noqa: E402
from DICOMGatedDynamicPETLib.hardendialog import hardenDialog  # noqa: E402
from DynamicNeuroPET import (DynamicNeuroPETLogic, DynamicNeuroPETWidget, _button, _ijkToRas, _percentiles,  # noqa
                             _vtkMatrix, ATLAS_BILATERAL, ATLAS_FILE_ATTRIBUTE, METHOD_ANATOMY, METHOD_PET,
                             METHOD_PET_TEMPLATE, NEURO_PREFIX, NORMALIZED_SUFFIX, REFERENCE_SUFFIX, REG_PREFIX,
                             STATE_DONE, STATE_READY, STATE_WAITING, STEP_DATA, STEP_NORM, STEP_REG, STEP_REGIONS,
                             NORMALIZATION_ENGINE, NORMALIZATION_ITERATIVE_DEFORMABLE, NORMALIZATION_ITERATIVE_LINEAR,
                             NORMALIZATION_STEPS, NORMALIZATION_WORKING_MM)

STEP_REVIEW = "review"
STEPS = (
    (STEP_DATA, None, "Data", "Static PET / SPECT of the brain and the patient's CT / MR"),
    (STEP_REG, None, "CT/MR registration", "Optional: the image registered to the CT / MR"),
    (STEP_NORM, None, "Normalization", "Normal database and the MNI ICBM 152 2009c space"),
    (STEP_REGIONS, None, "Regions", "CerebrA regions (as in the normal database)"),
    (STEP_REVIEW, None, "Review", "Comparison with the normal database"),
)
METHODS = ((METHOD_PET, "PET/SPECT to MR template (T1)"),
           (METHOD_ANATOMY, "CT/MR to MR template, same transform to the PET/SPECT"),
           (METHOD_PET_TEMPLATE, "PET/SPECT to PET template (normal database mean)"))
SETTINGS_DB_ROOT = "DICOMGatedDynamicPET/NeuroDatabase/RootFolder"     # shared with the Neuro PET Database Manager
SETTINGS_IMPORT_FOLDER = "DICOMGatedDynamicPET/StaticNeuro/ImportFolder"
AGE_ATTRIBUTE, SEX_ATTRIBUTE = NEURO_PREFIX + "PatientAge", NEURO_PREFIX + "PatientSex"
DATABASE_PARAMETER = NEURO_PREFIX + "Database"
NORMALIZATION_ATTRIBUTE = NEURO_PREFIX + "Normalization"    # JSON of the settings the image was normalized with
REGISTERED_SUFFIX = " registered"
ICON_PATH = os.path.join(_MODULE_DIR, "Resources", "Icons", "StaticNeuroPET.png")
WARNING_STYLE = "<span style='color:#c0392b'>{}</span>"


class StaticNeuroPET(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        parent.title = "Static Neuro PET/SPECT"
        parent.categories = ["Nuclear Medicine.PET4D"]
        parent.dependencies = []
        parent.contributors = ["Burak Demir"]
        parent.helpText = (
            "Workflow for a static brain PET or SPECT: <b>1. Data</b> (static PET / SPECT from DICOM or a file, the "
            "CT/MR, age and sex), <b>2. CT/MR registration</b> (optional), <b>3. Normalization</b> to the MNI ICBM "
            "152 2009c space (PET/SPECT to the T1 template, CT/MR to the MR template, or PET/SPECT to the mean map of "
            "a normal database; the normal database is picked here and gives its normalization settings), <b>4. "
            "Regions</b> (the CerebrA regions as in the database) and <b>5. Review</b> (and adding the patient to a "
            "user defined database). The atlases are not shipped: pick their folder in step 3.")
        parent.acknowledgementText = (
            "Part of the DICOM Gated / Dynamic PET extension. MNI ICBM 152 nonlinear symmetric 2009c template: Fonov "
            "VS et al., NeuroImage 2011;54:313-327. CerebrA: Manera AL et al., Scientific Data 2020;7:237 (CC0).")


def _sameRegion(segmentName, title):
    """A segment name ("Cerebellum Gray Matter (reference)") naming the region title ("Cerebellum Gray Matter")."""
    def clean(text):
        text = (text or "").strip().lower()
        suffix = REFERENCE_SUFFIX.strip().lower()
        return text[:-len(suffix)].strip() if text.endswith(suffix) else text
    return clean(segmentName) == clean(title)


# ---------------------------------------------------------------------------
# Logic
# ---------------------------------------------------------------------------
class StaticNeuroPETLogic(DynamicNeuroPETLogic):
    """The Neuro logic on one static volume (the "series" of the Dynamic Neuro steps is the volume itself):

        logic = StaticNeuroPETLogic()
        volume = logic.loadFromFile("sub-00000016_ses-SUV_pet.nii.gz")
        result = logic.registerToAnatomy(volume, ct)           # optional
        undo, registered = logic.hardenRegistration(volume, ct, linearNode, reg.SAVE_SEPARATELY, reg.OUTPUT_PET_SPACING)
        result = logic.normalize(volume, template, METHOD_PET)
        undo, normalized, _, _ = logic.harden(volume, template, linearNode, reg.SAVE_SEPARATELY, ...)
    """

    # --- the image -----------------------------------------------------------------------------
    @staticmethod
    def firstImage(volume):
        return volume

    @staticmethod
    def displayImage(volume):
        return volume

    @staticmethod
    def referenceImage(volume):
        return volume

    @staticmethod
    def attributes(node):
        return {name: node.GetAttribute(name) for name in (node.GetAttributeNames() or ())} if node else {}

    @classmethod
    def imageInfo(cls, node):
        """staticneuro.imageInfo() of a volume (its tags, else the DICOM modality)."""
        if node is None:
            return sn.imageInfo({})
        attributes = cls.attributes(node)
        modality = ""
        if not any(k.startswith((sn.STATIC_PREFIX, sn.SPECT_PREFIX)) for k in attributes):
            modality = attributes.get("DICOM.Modality") or cls.dicomValues(node, ("0008,0060",))["0008,0060"]
        return sn.imageInfo(attributes, modality)

    @classmethod
    def ageAndSex(cls, node):
        """(age in years or None, sex) of the patient: the values entered on the Data page, else the loaders' tags,
        else the DICOM tags."""
        if node is None:
            return None, ""
        attributes = cls.attributes(node)
        age = sn.parseAge(attributes.get(AGE_ATTRIBUTE))
        sex = sn.parseSex(attributes.get(SEX_ATTRIBUTE))
        if age is None or not sex:
            taggedAge, taggedSex = sn.taggedAgeSex(attributes)
            if age is None and taggedAge is None or not sex and not taggedSex:
                values = cls.dicomValues(node, ("0010,1010", "0010,0040"))
                taggedAge = taggedAge if taggedAge is not None else sn.parseAge(values["0010,1010"])
                taggedSex = taggedSex or sn.parseSex(values["0010,0040"])
            age = age if age is not None else taggedAge
            sex = sex or taggedSex
        return age, sex

    @staticmethod
    def setAgeAndSex(node, age, sex):
        if node is None:
            return
        node.SetAttribute(AGE_ATTRIBUTE, f"{age:g}" if age is not None else "")
        node.SetAttribute(SEX_ATTRIBUTE, sex or "")

    @classmethod
    def isStaticImage(cls, node):
        """A volume the workflow takes as its image: a static PET / SPECT (not an atlas, a template or a helper)."""
        if node is None or node.IsA("vtkMRMLLabelMapVolumeNode") or node.GetHideFromEditors():
            return False
        if node.GetAttribute(ATLAS_FILE_ATTRIBUTE) or node.GetAttribute("DynamicPETRegistration.Helper"):
            return False
        return bool(cls.imageInfo(node)["modality"])

    @staticmethod
    def loadFromFile(path):
        """An image file as a static PET / SPECT volume, with the tags of its BIDS name, .json sidecar and the
        participants.tsv (age, sex)."""
        sidecar = {}
        sidecarFile = sn.sidecarPath(path)
        if sidecarFile:
            try:
                with open(sidecarFile, encoding="utf-8") as f:
                    sidecar = json.load(f)
            except (OSError, ValueError):
                logging.warning(f"Static Neuro: could not read {sidecarFile}", exc_info=True)
        participant = {}
        entities = sn.parseBidsName(path)
        try:
            participant = sn.readParticipant(sn.findParticipants(path), entities.get("sub"))
        except (OSError, ValueError):
            logging.warning("Static Neuro: could not read participants.tsv", exc_info=True)
        node = slicer.util.loadVolume(path, {"name": sn.importName(path), "show": False})
        if node is None:
            raise RuntimeError(f"Could not load {path}")
        attributes = sn.importAttributes(path, sidecar, participant)
        for key, value in attributes.items():
            node.SetAttribute(key, str(value))
        if attributes.get(sn.STATIC_PREFIX + "Frames") or attributes.get(sn.SPECT_PREFIX + "Frames"):
            logging.warning(f"Static Neuro: {os.path.basename(path)} has several frames (dynamic); only the first "
                            "volume is used")
        node.CreateDefaultDisplayNodes()
        display = node.GetDisplayNode()
        colorNode = slicer.mrmlScene.GetFirstNodeByName("PET-DICOM")
        if display is not None and colorNode is not None:
            display.SetAndObserveColorNodeID(colorNode.GetID())
            display.AutoWindowLevelOff()
            low, high = _percentiles(node)
            display.SetWindowLevelMinMax(0.0, high)
        return node

    # --- harden one volume -----------------------------------------------------------------------
    @staticmethod
    def hardenVolume(volume, fixed, linearNode, action, grid, spacing=None, interpolation=reg.DEFAULT_INTERPOLATION,
                     attributes=None, suffix=REGISTERED_SUFFIX, keepFov=False):
        """Resample volume through the registration chain whose moving end is linearNode onto the Harden grid of
        fixed (DynamicPETRegistrationLogic.outputGeometry). reg.OVERWRITE replaces its voxels and geometry,
        reg.SAVE_SEPARATELY makes a new volume '<name><suffix>' next to it. Returns (undo record, volume)."""
        from DynamicPETRegistration import DynamicPETRegistrationLogic
        regLogic = DynamicPETRegistrationLogic
        corners = regLogic.registeredCorners(volume, linearNode) if keepFov else None
        dims, matrix = regLogic.outputGeometry(fixed, volume, grid, spacing, corners)
        worldToMoving, linear = regLogic.worldToPet(linearNode)
        image = resampling.Resampler(_ijkToRas(volume), dims, matrix, worldToMoving, linear,
                                     interpolation).resample(volume.GetImageData())
        attributes = dict(attributes or {})
        attributes.setdefault(REG_PREFIX + "Interpolation", interpolation)
        attributes.setdefault(REG_PREFIX + "VoxelSizeMm", ", ".join(f"{v:.3g}" for v in reg.gridSpacing(matrix)))
        if action == reg.OVERWRITE:
            undo = {"action": action, "node": volume, "image": volume.GetImageData(), "matrix": _ijkToRas(volume),
                    "attributes": {k: volume.GetAttribute(k) for k in attributes}}
            volume.SetAndObserveImageData(image)
            volume.SetIJKToRASMatrix(_vtkMatrix(matrix))
            for key, value in attributes.items():
                volume.SetAttribute(key, value)
            return undo, volume
        scene = slicer.mrmlScene
        node = scene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", scene.GenerateUniqueName(volume.GetName() + suffix))
        node.SetAndObserveImageData(image)
        node.SetIJKToRASMatrix(_vtkMatrix(matrix))
        for key in volume.GetAttributeNames() or ():
            node.SetAttribute(key, volume.GetAttribute(key))
        for key, value in attributes.items():
            node.SetAttribute(key, value)
        try:
            node.SetVoxelValueUnits(volume.GetVoxelValueUnits())
        except Exception:
            pass
        node.CreateDefaultDisplayNodes()
        if volume.GetDisplayNode() is not None:
            node.GetDisplayNode().CopyContent(volume.GetDisplayNode())
        from DynamicPETMotionCorrection import DynamicPETMotionCorrectionLogic
        DynamicPETMotionCorrectionLogic._placeNextTo(volume, [node])
        return {"action": action, "created": [node]}, node

    @staticmethod
    def revertHarden(undo):
        if undo["action"] == reg.OVERWRITE:
            node = undo["node"]
            if not slicer.mrmlScene.IsNodePresent(node):
                return
            node.SetAndObserveImageData(undo["image"])
            node.SetIJKToRASMatrix(_vtkMatrix(undo["matrix"]))
            for key, value in undo["attributes"].items():
                node.SetAttribute(key, value) if value is not None else node.RemoveAttribute(key)
            return
        for node in undo["created"]:
            if slicer.mrmlScene.IsNodePresent(node):
                slicer.mrmlScene.RemoveNode(node)

    # --- CT/MR registration ----------------------------------------------------------------------
    @staticmethod
    def registerToAnatomy(volume, anatomy, settings=None, progress=None):
        """DynamicPETRegistrationLogic.register() of the image to the CT/MR (rigid by default)."""
        from DynamicPETRegistration import DynamicPETRegistrationLogic
        if volume.GetParentTransformNode() is not None:
            raise ValueError(f"'{volume.GetName()}' is under a transform. Harden or remove it first.")
        logic = DynamicPETRegistrationLogic()
        return logic.register(anatomy, volume, settings or logic.defaultSettings(), progress)

    def hardenRegistration(self, volume, anatomy, linearNode, action, grid, spacing=None,
                           interpolation=reg.DEFAULT_INTERPOLATION, description=""):
        """The image resampled onto the CT/MR field of view (grown to the image's where it is larger), tagged as
        registered to it."""
        attributes = {REG_PREFIX + "Fixed": anatomy.GetName(), REG_PREFIX + "Transform": description}
        return self.hardenVolume(volume, anatomy, linearNode, action, grid, spacing, interpolation, attributes,
                                 REGISTERED_SUFFIX, keepFov=True)

    # --- normalization ---------------------------------------------------------------------------
    def harden(self, volume, template, linearNode, action, grid, spacing=None, interpolation="Linear",
               description="", method=METHOD_PET, anatomy=None):
        """The image resampled through the normalization onto the template's field of view; with METHOD_ANATOMY the
        CT/MR is normalized too. Returns (undo, volume, None, normalized CT/MR or None)."""
        attributes = {NEURO_PREFIX + "Template": template.GetName(), NEURO_PREFIX + "Method": method,
                      NEURO_PREFIX + "Transform": description}
        undo, node = self.hardenVolume(volume, template, linearNode, action, grid, spacing, interpolation,
                                       attributes, NORMALIZED_SUFFIX)
        normalized = None
        if method == METHOD_ANATOMY and anatomy is not None:
            try:
                normalized = self.normalizedAnatomy(anatomy, template, linearNode)
            except Exception:
                logging.exception("Static Neuro: could not normalize the CT/MR")
        return undo, node, None, normalized

    # --- adding a patient to a normal database --------------------------------------------------
    def addSubject(self, folder, header, volume, subjectId, gridPath, age, sex, normalization, replace=False,
                   segmentation=None, segmentId=None, referenceName=""):
        """Add a normalized volume to a user defined database: resampled onto the database grid (gridPath: its
        reference mask, or the MNI brain mask; MNI 2009c 1 mm) and saved as subjects/<id>_SUV.nii.gz and its SUVR
        map. The reference region (intensity normalization) is the segment segmentId of segmentation (the regions
        of step 4) or, without it, the mask gridPath. Returns the row written into subjects.csv."""
        scene = slicer.mrmlScene
        gridArray, gridNode = self._loadLabels(gridPath)
        nodes = [gridNode]
        try:
            dims = tuple(int(x) for x in gridNode.GetImageData().GetDimensions())
            image = resampling.Resampler(_ijkToRas(volume), dims, _ijkToRas(gridNode), None, True,
                                         "Linear").resample(volume.GetImageData())
            suvNode = scene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "Neuro subject SUV (temporary)")
            nodes.append(suvNode)
            suvNode.SetIJKToRASMatrix(_vtkMatrix(_ijkToRas(gridNode)))
            suvNode.SetAndObserveImageData(image)
            values = slicer.util.arrayFromVolume(suvNode).astype(np.float32)
            if segmentation is not None and segmentId:
                mask = slicer.util.arrayFromSegmentBinaryLabelmap(segmentation, segmentId, gridNode)
                if mask is None:
                    raise ValueError("The reference segment is empty.")
                mask = np.asarray(mask) > 0
            else:
                mask = gridArray > 0
            ratio, mean = sn.ratioMap(values, mask)
            slicer.util.updateVolumeFromArray(suvNode, values)
            suvrNode = scene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "Neuro subject SUVR (temporary)")
            nodes.append(suvrNode)
            suvrNode.SetIJKToRASMatrix(_vtkMatrix(_ijkToRas(gridNode)))
            slicer.util.updateVolumeFromArray(suvrNode, ratio)
            suvFile, suvrFile = normaldb.subjectFileNames(subjectId)
            os.makedirs(os.path.join(folder, normaldb.SUBJECTS_DIR), exist_ok=True)
            for node, relative in ((suvNode, suvFile), (suvrNode, suvrFile)):
                if not slicer.util.saveNode(node, os.path.join(folder, relative)):
                    raise RuntimeError(f"Could not write {relative}")
        finally:
            for node in nodes:
                if node is not None and scene.IsNodePresent(node):
                    scene.RemoveNode(node)
        attributes = self.attributes(volume)
        scanDate = self.dicomValues(volume, ("0008,0020",))["0008,0020"] or ""
        if len(scanDate) == 8 and scanDate.isdigit():
            scanDate = f"{scanDate[:4]}-{scanDate[4:6]}-{scanDate[6:]}"
        row = {"id": subjectId, "age": "" if age is None else f"{age:g}", "sex": sex or "", "scanDate": scanDate,
               "suvFile": suvFile, "suvrFile": suvrFile, "referenceMean": f"{mean:.6g}",
               "reference": referenceName or "", "normalization": normaldb.normalizationText(normalization),
               "source": attributes.get(sn.IMPORTED_ATTRIBUTE) or volume.GetName()}
        normaldb.addSubject(folder, row, normalization, replace)
        logging.info(f"Static Neuro: '{subjectId}' added to {folder} (reference mean {mean:.4g})")
        return row

    # --- batch: many normal subjects into a database ----------------------------------------------
    BATCH_COLUMNS = ("file", "id", "status", "message", "referenceMean", "seconds")

    def batchPlan(self, header):
        """(plan, registration settings) of a batch into the database: its normalization settings
        (staticneuro.batchNormalization) and the registration settings they give."""
        plan = sn.batchNormalization(header.get("normalization"), neuro.DEFAULT_TEMPLATE, NORMALIZATION_ENGINE,
                                     {reg.ENGINE_ITERATIVE: (NORMALIZATION_ITERATIVE_LINEAR,
                                                             NORMALIZATION_ITERATIVE_DEFORMABLE),
                                      reg.ENGINE_BRAINSFIT: (NORMALIZATION_STEPS[0], "BSpline" in NORMALIZATION_STEPS)})
        settings = self.defaultSettings(METHOD_PET)
        if plan["engine"] == reg.ENGINE_ITERATIVE:
            settings.update(engine=reg.ENGINE_ITERATIVE, iterativeLinear=plan["linear"],
                            iterativeDeformable=plan["deformable"],
                            steps=reg.iterativeSteps(plan["linear"], plan["deformable"]))
        else:
            steps = reg.validateSteps([plan["linear"]] + (["BSpline"] if plan["deformable"] else []))
            settings.update(engine=reg.ENGINE_BRAINSFIT, steps=steps,
                            resample=[NORMALIZATION_WORKING_MM > 0] * len(steps))
        return plan, settings

    def batchAdd(self, folder, files, atlas, qc=True, progress=None):
        """Add BIDS PET files (staticneuro.findPetFiles) of normal subjects to a user defined database, one after the
        other: each loaded, normalized to the MNI template with the database's normalization settings (no manual
        check), resampled onto the template grid and added (logic.addSubject, the database's reference mask). IDs
        already in the database are skipped; a subject that fails is logged and the batch goes on. qc: a PNG of the
        normalized PET over the MNI T1 (3 slices) in <database>/qc/. progress(index, total, text) returns False to
        stop. The log is <database>/qc/batch_<time>.csv. Returns (rows, log path, stopped)."""
        header = normaldb.readHeader(folder)
        if header.get("kind") != normaldb.USER_DEFINED:
            raise ValueError("Subjects can be added to user defined databases only.")
        gridPath = sn.referenceMaskPath(folder, header)
        if not gridPath:
            raise ValueError(f"'{header['name']}' has no reference mask (reference_mask in its folder).")
        plan, settings = self.batchPlan(header)
        key = plan["template"]
        templatePath = atlas.get(key)
        if not templatePath and key == neuro.PSEUDO_PET and neuro.canMakeTemplate(atlas, key):
            templatePath = self.ensurePseudoPet(atlas)
        if not templatePath:
            raise ValueError(f"The template '{key}' of the database's normalization is not in the atlas folder.")
        template = self.loadAtlasVolume(templatePath, name=f"MNI ICBM152 2009c {key}")
        mask = None
        if plan["brainMask"] and neuro.BRAIN_MASK in atlas:
            mask = self.loadAtlasVolume(atlas[neuro.BRAIN_MASK], name="MNI ICBM152 2009c brain mask")
        anatomy = None
        if qc and "T1" in atlas:
            t1 = self.loadAtlasVolume(atlas["T1"], name="MNI ICBM152 2009c T1")
            anatomy = (sn.rasArray(slicer.util.arrayFromVolume(t1), _ijkToRas(t1)), tuple(_ijkToRas(t1).ravel()))
        qcFolder = os.path.join(folder, "qc")
        os.makedirs(qcFolder, exist_ok=True)
        logPath = os.path.join(qcFolder, "batch_" + time.strftime("%Y%m%d_%H%M%S") + ".csv")
        rows, stopped, total = [], False, len(files)
        for index, path in enumerate(files):
            subjectId = sn.bidsSubjectId(path) or normaldb.safeName(sn.splitImageName(path)[0]) or f"subject{index}"
            if progress is not None and not progress(index, total, f"{subjectId} ({index + 1} / {total})"):
                stopped = True
                break
            row = {"file": path, "id": subjectId, "status": "", "message": "", "referenceMean": "", "seconds": ""}
            start = time.time()
            if normaldb.hasSubject(folder, subjectId):
                row.update(status="skipped", message="already in the database")
            else:
                def stepProgress(step, steps, text, index=index, subjectId=subjectId):
                    return progress is None or progress(index, total, f"{subjectId} ({index + 1} / {total}): {text}")
                try:
                    added = self._batchSubject(folder, header, path, subjectId, template, mask, settings, plan,
                                               gridPath, anatomy, qcFolder, stepProgress)
                    if added is None:
                        row.update(status="stopped", message="stopped during the registration")
                        stopped = True
                    else:
                        row.update(status="added", referenceMean=added["referenceMean"],
                                   message=added.get("notes", ""))
                except Exception as error:
                    logging.exception(f"Static Neuro batch: {path} failed")
                    row.update(status="failed", message=str(error).replace("\n", " "))
                header = normaldb.readHeader(folder)
            row["seconds"] = f"{time.time() - start:.0f}"
            rows.append(row)
            self._writeBatchLog(logPath, rows)
            if stopped:
                break
        if not rows:
            self._writeBatchLog(logPath, rows)
        return rows, logPath, stopped

    def _batchSubject(self, folder, header, path, subjectId, template, mask, settings, plan, gridPath, anatomy,
                      qcFolder, progress):
        """One subject of batchAdd(): the row added (with "notes"), None when the registration was stopped."""
        from DynamicPETRegistration import DynamicPETRegistrationLogic
        scene = slicer.mrmlScene
        nodes = []
        try:
            volume = self.loadFromFile(path)
            nodes.append(volume)
            notes = sn.unitWarnings(self.imageInfo(volume), header)
            result = self.normalize(volume, template, METHOD_PET, None, settings, mask, progress)
            nodes.append(result.get("bspline"))
            if result.get("stopped") or not result.get("steps"):
                return None
            chain = DynamicPETRegistrationLogic.transformNode("vtkMRMLLinearTransformNode", "Batch normalization")
            nodes.append(chain)
            chain.SetMatrixTransformToParent(_vtkMatrix(result["linear"]))
            if result.get("bspline") is not None:
                chain.SetAndObserveTransformNodeID(result["bspline"].GetID())
            description = f"Normalized to {template.GetName()} ({reg.stepsText(result['steps'])}, batch)"
            _, normalized, _, _ = self.harden(volume, template, chain, reg.SAVE_SEPARATELY, reg.OUTPUT_FIXED_GRID,
                                              None, "Linear", description, METHOD_PET)
            nodes.append(normalized)
            normalized.SetAttribute(NORMALIZATION_ATTRIBUTE, json.dumps(plan))
            age, sex = self.ageAndSex(normalized)
            if age is None:
                notes.append("age not known")
            row = self.addSubject(folder, header, normalized, subjectId, gridPath, age, sex, plan, False)
            if anatomy is not None:
                try:
                    self.writeQcPicture(os.path.join(qcFolder, normaldb.safeName(subjectId) + "_qc.png"), normalized,
                                        anatomy)
                except Exception as error:
                    logging.exception(f"Static Neuro batch: no QC picture of {subjectId}")
                    notes.append(f"no QC picture ({error})")
            row["notes"] = "; ".join(notes)
            return row
        finally:
            for node in nodes:
                if node is not None and scene.IsNodePresent(node):
                    scene.RemoveNode(node)

    @staticmethod
    def writeQcPicture(path, volume, anatomy):
        """PNG of the volume over the anatomy ((rasArray, ijkToRas values) on the same grid): 3 slices."""
        from vtk.util import numpy_support
        anatomyArray, anatomyMatrix = anatomy
        if not np.allclose(np.asarray(anatomyMatrix).reshape(4, 4), _ijkToRas(volume), atol=1e-3):
            raise ValueError("the image is not on the T1 template grid")
        picture = sn.qcMontage(anatomyArray, sn.rasArray(slicer.util.arrayFromVolume(volume), _ijkToRas(volume)))
        height, width, _ = picture.shape
        image = vtk.vtkImageData()
        image.SetDimensions(width, height, 1)
        scalars = numpy_support.numpy_to_vtk(np.ascontiguousarray(picture[::-1]).reshape(-1, 3), deep=True,
                                             array_type=vtk.VTK_UNSIGNED_CHAR)
        scalars.SetNumberOfComponents(3)
        image.GetPointData().SetScalars(scalars)
        writer = vtk.vtkPNGWriter()
        writer.SetFileName(path)
        writer.SetInputData(image)
        writer.Write()

    def _writeBatchLog(self, path, rows):
        import csv
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.BATCH_COLUMNS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

    # --- normal databases ------------------------------------------------------------------------
    @staticmethod
    def databaseRoot():
        return str(qt.QSettings().value(SETTINGS_DB_ROOT) or "")

    @staticmethod
    def setDatabaseRoot(folder):
        qt.QSettings().setValue(SETTINGS_DB_ROOT, folder or "")

    @staticmethod
    def databases(root, info):
        """[(folder, header, rank, problems)] of the static databases of root, those fitting the image first."""
        found, errors = normaldb.listDatabases(root, normaldb.STATIC)
        for folder, error in errors:
            logging.warning(f"Static Neuro: database {folder} not readable: {error}")
        return sn.sortDatabases(found, info)


# ---------------------------------------------------------------------------
# Widget
# ---------------------------------------------------------------------------
class StaticNeuroPETWidget(DynamicNeuroPETWidget):
    """The Dynamic Neuro workflow on a static PET / SPECT: own Data, CT/MR registration, Regions (normal
    database settings) and Review pages; the Normalization page of the Dynamic Neuro module with the normal database
    and the PET template method."""

    MODULE_NAME = "StaticNeuroPET"
    STEPS = STEPS
    LOGIC_CLASS = StaticNeuroPETLogic
    TOOLBAR_NAME = "StaticNeuroPETWorkflowToolbar"
    TOOLBAR_TITLE = "Static Neuro PET/SPECT"
    TOOLBAR_TOOLTIP = "Static Neuro PET/SPECT: back to the first step"
    ICON = ICON_PATH
    STEP_AFTER_REGIONS = STEP_REVIEW
    NORM_METHODS = METHODS
    LOAD_REGIONS_TEXT = "Load regions and go to review"

    def setup(self):
        self.regSession = None
        self.regResult = None
        self.regUndo = None
        self.databaseRows = []          # [(folder, header, rank, problems)]
        self._databaseKey = None
        self._databaseChosen = False
        DynamicNeuroPETWidget.setup(self)
        self.pages[STEP_REG] = self._buildRegPage()
        self.pages[STEP_REVIEW] = self._buildReviewPage()
        self.refresh()

    def volume(self):
        return self.sequence()

    # --- page 1: data --------------------------------------------------------------------------------
    def _buildDataPage(self):
        page = qt.QWidget()
        layout = qt.QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        box = qt.QGroupBox("Patient data")
        form = qt.QFormLayout(box)
        self.sequenceSelector = self._nodeSelector(["vtkMRMLScalarVolumeNode"],
                                                   "Static PET (PET4D DICOM plugin, e.g. '<series> static SUVbw') or "
                                                   "SPECT (DICOM SPECT reconstruction) of the brain")
        self.sequenceSelector.hideChildNodeTypes = ["vtkMRMLLabelMapVolumeNode"]
        form.addRow("PET / SPECT:", self.sequenceSelector)
        importButton = qt.QPushButton("Import from file...")
        importButton.toolTip = ("Load a PET / SPECT image file (NIfTI, NRRD, ...). BIDS names (sub-<id>_ses-<session>_"
                                "pet.nii.gz) are read with their .json sidecar (tracer, nuclide, units, timing) and "
                                "the participants.tsv of the dataset (age, sex); a session 'SUV' means SUV units.")
        importButton.connect("clicked()", self.onImportFile)
        form.addRow("", importButton)
        self.sequenceInfo = qt.QLabel()
        self.sequenceInfo.wordWrap = True
        form.addRow(self.sequenceInfo)
        ageRow = qt.QHBoxLayout()
        self.ageSpin = qt.QDoubleSpinBox()
        self.ageSpin.setRange(-1.0, 130.0)
        self.ageSpin.setDecimals(0)
        self.ageSpin.specialValueText = "Unknown"
        self.ageSpin.suffix = " years"
        self.ageSpin.toolTip = "Age of the patient at the scan (from the tags; edit it if it is missing or wrong)"
        self.sexCombo = qt.QComboBox()
        for key, title in sn.SEXES:
            self.sexCombo.addItem(title, key)
        self.sexCombo.toolTip = "Sex of the patient (from the tags; edit it if it is missing or wrong)"
        ageRow.addWidget(self.ageSpin)
        ageRow.addWidget(qt.QLabel("Sex:"))
        ageRow.addWidget(self.sexCombo)
        form.addRow("Age:", ageRow)
        self.anatomySelector = self._nodeSelector(["vtkMRMLScalarVolumeNode"],
                                                  "CT or MR of the same patient (optional: registration in step 2, "
                                                  "and the normalization through the CT/MR in step 3)")
        self.anatomySelector.hideChildNodeTypes = ["vtkMRMLLabelMapVolumeNode"]
        form.addRow("CT / MR:", self.anatomySelector)
        self.anatomyInfo = qt.QLabel()
        self.anatomyInfo.wordWrap = True
        form.addRow(self.anatomyInfo)
        self.contrastCombo = qt.QComboBox()
        for key, title in (("", "Automatic"), ("T1", "T1"), ("T2", "T2"), ("PD", "PD"), ("CT", "CT")):
            self.contrastCombo.addItem(title, key)
        self.contrastCombo.toolTip = ("Contrast of the CT/MR, read from the DICOM tags. It picks the template of "
                                      "step 3 when the CT/MR is normalized.")
        form.addRow("Contrast:", self.contrastCombo)
        layout.addWidget(box)
        dicomButton = qt.QPushButton("Open the DICOM browser")
        dicomButton.toolTip = "Load the static PET / SPECT and the CT/MR from DICOM"
        dicomButton.connect("clicked()", lambda: slicer.util.selectModule("DICOM"))
        layout.addWidget(dicomButton)
        note = qt.QLabel("Static brain PET (SUV) or SPECT (counts, Bq/ml or SUV): optionally registered to the CT/MR, "
                         "normalized to the MNI ICBM 152 2009c space, the CerebrA regions loaded and compared with a "
                         "normal database. Steps can be skipped; without a database the review gives the regional "
                         "values only.")
        note.wordWrap = True
        note.setStyleSheet("color: #7d868d")
        layout.addWidget(note)
        layout.addStretch(1)
        self.sequenceSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onDataChanged)
        self.anatomySelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onDataChanged)
        self.contrastCombo.connect("currentIndexChanged(int)", lambda i: self.onDataChanged())
        self.ageSpin.connect("valueChanged(double)", lambda v: self.onAgeSexChanged())
        self.sexCombo.connect("currentIndexChanged(int)", lambda i: self.onAgeSexChanged())
        return page

    def selectInputs(self):
        if self.volume() is None:
            nodes = [n for n in slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode") if self.logic.isStaticImage(n)]
            original = [n for n in nodes if not n.GetAttribute(REG_PREFIX + "Fixed")
                        and not n.GetAttribute(NEURO_PREFIX + "Template")]
            if nodes:
                self.setNode("Sequence", (original or nodes)[-1])
        if self.anatomy() is None:
            from DynamicPETRegistration import fixedCandidates
            candidates = [n for n in fixedCandidates() if not n.GetAttribute(ATLAS_FILE_ATTRIBUTE)
                          and not n.GetAttribute(NEURO_PREFIX + "Template") and n is not self.volume()
                          and not self.logic.imageInfo(n)["modality"]]
            if candidates:
                self.setNode("Anatomy", candidates[0])
        self._syncDataPage()

    def onImportFile(self):
        settings = qt.QSettings()
        folder = str(settings.value(SETTINGS_IMPORT_FOLDER) or "")
        paths = qt.QFileDialog.getOpenFileNames(slicer.util.mainWindow(), "Import PET / SPECT image", folder,
                                                "Images (*.nii *.nii.gz *.nrrd *.nhdr *.mha *.mhd);;All files (*)")
        paths = [str(p) for p in (paths or [])]
        if not paths:
            return
        settings.setValue(SETTINGS_IMPORT_FOLDER, os.path.dirname(paths[0]))
        node = None
        qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)
        try:
            for path in paths:
                node = self.logic.loadFromFile(path)
        except Exception as error:
            logging.exception("Static Neuro: import failed")
            slicer.util.errorDisplay(f"Could not import the image:\n{error}")
        finally:
            qt.QApplication.restoreOverrideCursor()
        if node is not None:
            self.sequenceSelector.setCurrentNode(node)
            slicer.util.setSliceViewerLayers(background=node, fit=True)

    def onDataChanged(self, *args):
        if self._updating:
            return
        if self.sequenceSelector.currentNode() is not self.volume() or \
                self.anatomySelector.currentNode() is not self.anatomy():
            self._clearRegistration()
            self.regUndo = None
        DynamicNeuroPETWidget.onDataChanged(self, *args)
        self._databaseKey = None
        self._syncAgeSex()

    def _syncDataPage(self):
        DynamicNeuroPETWidget._syncDataPage(self)
        self._syncAgeSex()

    def _syncAgeSex(self):
        age, sex = self.logic.ageAndSex(self.volume())
        self._updating = True
        try:
            self.ageSpin.value = age if age is not None else -1.0
            self.sexCombo.setCurrentIndex(max(0, self.sexCombo.findData(sex or "")))
            self.ageSpin.enabled = self.sexCombo.enabled = self.volume() is not None
        finally:
            self._updating = False

    def onAgeSexChanged(self):
        if self._updating or self.volume() is None:
            return
        age = float(self.ageSpin.value)
        self.logic.setAgeAndSex(self.volume(), age if age >= 0 else None, self.sexCombo.currentData or "")

    def _updateDataInfo(self):
        volume = self.volume()
        if volume is None:
            self.sequenceInfo.text = "Pick the static PET / SPECT, or import it from a file."
        else:
            info = self.logic.imageInfo(volume)
            what = " ".join(t for t in (info["modality"] or "Image", info["nuclide"], info["tracer"]) if t)
            parts = [what, f"units {info['units'] or 'unknown'}"]
            if info["source"]:
                parts.append(f"from {info['source']}")
            done = []
            if volume.GetAttribute(REG_PREFIX + "Fixed"):
                done.append(f"registered to '{volume.GetAttribute(REG_PREFIX + 'Fixed')}'")
            if volume.GetAttribute(NEURO_PREFIX + "Template"):
                done.append("normalized")
            text = ", ".join(parts) + (f"; {', '.join(done)}" if done else "") + "."
            warnings = sn.unitWarnings(info)
            if warnings:
                text += "<br>" + WARNING_STYLE.format(" ".join(warnings))
            self.sequenceInfo.text = text
        anatomy = self.anatomy()
        if anatomy is None:
            self.anatomyInfo.text = "No CT/MR: step 2 and the CT/MR normalization are skipped."
        else:
            contrast, reason = self.logic.anatomyContrast(anatomy)
            self.anatomyInfo.text = (f"Recognised as <b>{contrast}</b> ({reason})." if contrast else
                                     f"Contrast not recognised ({reason}); the T1 template is used unless you pick "
                                     "one.")

    # --- page 2: CT/MR registration --------------------------------------------------------------
    def _buildRegPage(self):
        from DynamicPETRegistration import iterativeOptions
        page = qt.QWidget()
        layout = qt.QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        box = qt.QGroupBox("Registration to the CT / MR")
        form = qt.QFormLayout(box)
        self.regInfo = qt.QLabel()
        self.regInfo.wordWrap = True
        form.addRow(self.regInfo)
        self.regEngineCombo = qt.QComboBox()
        for title, key in reg.ENGINES:
            self.regEngineCombo.addItem(title, key)
        self.regEngineCombo.setCurrentIndex(self.regEngineCombo.findData(reg.DEFAULT_ENGINE))
        self.regEngineCombo.toolTip = ("BRAINSFit: rigid (same patient). Iterative multi-resolution: a rigid (or "
                                       "affine) stage, optionally deformable passes.")
        form.addRow("Registration:", self.regEngineCombo)
        self.regIterativeBox = qt.QWidget()
        iterativeForm = qt.QFormLayout(self.regIterativeBox)
        iterativeForm.setContentsMargins(0, 0, 0, 0)
        self.regIterative = iterativeOptions(iterativeForm)
        form.addRow(self.regIterativeBox)
        self.regInitCombo = qt.QComboBox()
        for title, mode in reg.INITIALIZATIONS:
            self.regInitCombo.addItem(title, mode)
        self.regInitCombo.setCurrentIndex(self.regInitCombo.findData(reg.DEFAULT_INITIALIZATION))
        form.addRow("Initialization:", self.regInitCombo)
        self.registerButton = _button("Register")
        self.registerButton.toolTip = "Register the image to the CT/MR and show it in the registration layout"
        form.addRow(self.registerButton)
        row = qt.QHBoxLayout()
        self.regManualButton = qt.QPushButton("Correct manually")
        self.regManualButton.checkable = True
        self.regRevertButton = _button("Revert", "revert")
        self.regRevertButton.toolTip = ("Before Harden: drop the registration. After Harden: remove the registered "
                                        "image (or put an overwritten one back).")
        self.regHardenButton = _button("Harden", "harden")
        self.regHardenButton.toolTip = "Resample the image onto the CT/MR field of view"
        row.addWidget(self.regManualButton)
        row.addWidget(self.regRevertButton)
        row.addWidget(self.regHardenButton)
        form.addRow(row)
        self.regLabel = qt.QLabel("Not registered. This step is optional (e.g. skip it for a PET/MR, or when "
                                  "normalizing the PET/SPECT itself).")
        self.regLabel.wordWrap = True
        self.regLabel.setTextInteractionFlags(qt.Qt.TextSelectableByMouse)
        form.addRow(self.regLabel)
        layout.addWidget(box)
        viewBox = ctk.ctkCollapsibleButton()
        viewBox.text = "View"
        viewBox.collapsed = True
        view = qt.QFormLayout(viewBox)
        self.regOpacity = ctk.ctkSliderWidget()
        self.regOpacity.minimum, self.regOpacity.maximum = 0.0, 1.0
        self.regOpacity.singleStep, self.regOpacity.decimals = 0.05, 2
        self.regOpacity.value = 0.5
        view.addRow("Fusion opacity:", self.regOpacity)
        self.regCheckerCheck = qt.QCheckBox("Checkerboard column")
        view.addRow(self.regCheckerCheck)
        layout.addWidget(viewBox)
        layout.addStretch(1)
        self.regEngineCombo.connect("currentIndexChanged(int)", lambda i: self._updateRegEngine())
        self.registerButton.connect("clicked()", self.onRegister)
        self.regManualButton.connect("toggled(bool)", lambda on: self.regSession and
                                     self.regSession.setManualHandles(on))
        self.regRevertButton.connect("clicked()", self.onRegRevert)
        self.regHardenButton.connect("clicked()", self.onRegHarden)
        self.regOpacity.connect("valueChanged(double)", lambda v: self.regSession and
                                self.regSession.setFusionOpacity(v))
        self.regCheckerCheck.connect("toggled(bool)", lambda on: self.regSession and
                                     self.regSession.setShowCheckerboard(on))
        self._updateRegEngine()
        return page

    def _updateRegEngine(self):
        self.regIterativeBox.visible = self.regEngineCombo.currentData == reg.ENGINE_ITERATIVE

    def regSettings(self):
        from DynamicPETRegistration import DynamicPETRegistrationLogic, iterativeSettings
        settings = DynamicPETRegistrationLogic.defaultSettings()
        settings["initialization"] = self.regInitCombo.currentData
        if self.regEngineCombo.currentData == reg.ENGINE_ITERATIVE:
            settings.update(iterativeSettings(self.regIterative), engine=reg.ENGINE_ITERATIVE)
            settings["steps"] = reg.iterativeSteps(settings["iterativeLinear"], settings["iterativeDeformable"])
        else:
            settings["engine"] = reg.ENGINE_BRAINSFIT
        return settings

    def onRegister(self):
        volume, anatomy = self.volume(), self.anatomy()
        if volume is None or anatomy is None:
            slicer.util.warningDisplay("Pick the PET / SPECT and the CT/MR in step 1.", windowTitle="Registration")
            return
        settings = self.regSettings()
        self._clearRegistration()
        dialog = slicer.util.createProgressDialog(windowTitle="Registration", labelText="Registering...",
                                                  maximum=len(settings["steps"]))

        def progress(index, total, text):
            dialog.setValue(index)
            dialog.setLabelText(f"Registering: {text}")
            slicer.app.processEvents()
            return not dialog.wasCanceled

        qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)
        try:
            result = self.logic.registerToAnatomy(volume, anatomy, settings, progress)
        except Exception as error:
            logging.exception("Static Neuro: registration failed")
            slicer.util.errorDisplay(f"Registration failed:\n{error}")
            return
        finally:
            qt.QApplication.restoreOverrideCursor()
            dialog.close()
        if not result["steps"]:
            self._dropResult(result)
            self.regLabel.text = "Stopped before the first step."
            self._updateRegButtons()
            return
        self.regResult = result
        self._openRegSession()
        lines = reg.summaryLines(result["steps"], result["linear"], result["center"], result["deformation"])
        notes = result.get("notes") or []
        self.regLabel.text = (f"'{volume.GetName()}' registered to '{anatomy.GetName()}'"
                              + (" (stopped)" if result["stopped"] else "") + ":\n" + "\n".join(lines)
                              + ("\n" + " ".join(notes) if notes else "")
                              + "\nCheck it in the views, correct it by hand if needed, then Harden.")
        self._updateRegButtons()

    @staticmethod
    def _dropResult(result):
        bspline = (result or {}).get("bspline")
        if bspline is not None and slicer.mrmlScene.IsNodePresent(bspline):
            slicer.mrmlScene.RemoveNode(bspline)

    def _openRegSession(self):
        from DynamicPETRegistration import RegistrationSession, DynamicPETRegistrationWidget
        self._closeRegSession()
        volume, anatomy = self.volume(), self.anatomy()
        if self.regResult is None or volume is None or anatomy is None:
            return
        DynamicPETRegistrationWidget._closeOtherModules()
        try:
            session = RegistrationSession(anatomy)
            session.fusionOpacity = self.regOpacity.value
            session.showCheckerboard = self.regCheckerCheck.checked
            session.open(volume)
            session.setRegistration(self.regResult["linear"], self.regResult["bspline"])
            session.setFixedWindow(*_percentiles(anatomy))
            self.regSession = session
        except Exception as error:
            logging.exception("Static Neuro: could not open the registration layout")
            slicer.util.errorDisplay(f"Could not open the registration layout:\n{error}")

    def _closeRegSession(self):
        if self.regSession is not None:
            if self.regManualButton.checked:
                self.regManualButton.blockSignals(True)
                self.regManualButton.checked = False
                self.regManualButton.blockSignals(False)
            self.regSession.close()
            self.regSession = None

    def _clearRegistration(self):
        """Drop a pending (not hardened) registration."""
        if not hasattr(self, "regLabel"):
            return
        self._closeRegSession()
        if self.regResult is not None:
            self._dropResult(self.regResult)
            self.regResult = None
            self.regLabel.text = "Not registered."
        self._updateRegButtons()

    def _updateRegButtons(self):
        if not hasattr(self, "regHardenButton"):
            return
        pending = self.regResult is not None and self.regSession is not None
        self.regHardenButton.enabled = pending
        self.regManualButton.enabled = pending
        self.regRevertButton.enabled = pending or self.regUndo is not None
        self.registerButton.enabled = self.volume() is not None and self.anatomy() is not None
        volume, anatomy = self.volume(), self.anatomy()
        if volume is None:
            self.regInfo.text = "Pick the PET / SPECT in step 1."
        elif anatomy is None:
            self.regInfo.text = "No CT/MR picked in step 1: skip this step."
        elif self.logic.petInAnatomySpace(volume, anatomy):
            self.regInfo.text = f"'{volume.GetName()}' is registered to '{anatomy.GetName()}'."
        else:
            self.regInfo.text = f"Register '{volume.GetName()}' to '{anatomy.GetName()}'."

    def onRegHarden(self):
        volume, anatomy = self.volume(), self.anatomy()
        if self.regResult is None or self.regSession is None or volume is None or anatomy is None:
            return
        text = (f"Resample '{volume.GetName()}' through the registration onto the field of view of "
                f"'{anatomy.GetName()}', grown to the image's where it is larger (not with the CT/MR grid option)."
                "\n\nSave separately (default) keeps the original next to the registered image.")
        choice = hardenDialog("Harden registration", text, reg.OUTPUT_GRIDS, reg.gridSpacing(_ijkToRas(volume)),
                              [("Save separately", reg.SAVE_SEPARATELY), ("Overwrite", reg.OVERWRITE)],
                              slicer.util.mainWindow())
        if choice is None:
            return
        description = (f"Registered to {anatomy.GetName()} ({reg.stepsText(self.regResult['steps'])}"
                       + ("" if np.allclose(self.regSession.manualMatrix(), np.eye(4)) else ", corrected by hand")
                       + ")")
        self.regManualButton.checked = False
        qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)
        try:
            undo, node = self.logic.hardenRegistration(volume, anatomy, self.regSession.linear, choice["action"],
                                                       choice["grid"], choice["spacing"], choice["interpolation"],
                                                       description)
        except Exception as error:
            logging.exception("Static Neuro: could not harden the registration")
            slicer.util.errorDisplay(f"Could not harden the registration:\n{error}")
            return
        finally:
            qt.QApplication.restoreOverrideCursor()
        self._clearRegistration()
        self.regUndo = {"undo": undo, "previous": volume}
        self.setNode("Sequence", node)
        self._clearNormalization()
        self.regLabel.text = (description + ".\n" + (f"Overwrote '{volume.GetName()}'" if choice["action"] ==
                                                     reg.OVERWRITE else f"Saved as '{node.GetName()}'") + ".")
        try:
            slicer.util.setSliceViewerLayers(background=anatomy, foreground=node, foregroundOpacity=0.5, fit=True)
        except Exception:
            pass
        self._updateRegButtons()
        self.refresh()

    def onRegRevert(self):
        if self.regResult is not None:
            self._clearRegistration()
            self.regLabel.text = "Not registered."
            return
        if self.regUndo is None:
            return
        record = self.regUndo
        try:
            self.logic.revertHarden(record["undo"])
        except Exception as error:
            logging.exception("Static Neuro: could not revert the registration")
            slicer.util.errorDisplay(f"Could not revert the registration:\n{error}")
            return
        if slicer.mrmlScene.IsNodePresent(record["previous"]):
            self.setNode("Sequence", record["previous"])
        self.regUndo = None
        self._clearNormalization()
        self.regLabel.text = "Reverted: the image is as it was."
        self._updateRegButtons()
        self.refresh()

    # --- page 3: normalization (the Dynamic Neuro page with the normal database and the PET template) -----
    def _buildNormPage(self):
        page = DynamicNeuroPETWidget._buildNormPage(self)
        box = qt.QGroupBox("Normal database")
        form = qt.QFormLayout(box)
        self.dbRootPath = ctk.ctkPathLineEdit()
        self.dbRootPath.filters = ctk.ctkPathLineEdit.Dirs
        self.dbRootPath.settingKey = ""
        self.dbRootPath.currentPath = self.logic.databaseRoot()
        self.dbRootPath.toolTip = "Root folder of the normal databases (the same as in the Neuro PET Database Manager)"
        form.addRow("Database folder:", self.dbRootPath)
        dbRow = qt.QHBoxLayout()
        self.dbCombo = qt.QComboBox()
        self.dbCombo.toolTip = ("The normal database the patient is compared with (and may be added to). Those of the "
                                "image's modality, nuclide and tracer are listed first (and picked); the others are "
                                "marked with '!'. Its normalization settings are put on this page as the default.")
        refreshButton = qt.QPushButton("Refresh")
        refreshButton.toolTip = "Read the database folder again"
        dbRow.addWidget(self.dbCombo, 1)
        dbRow.addWidget(refreshButton)
        form.addRow("Database:", dbRow)
        self.dbInfo = qt.QLabel()
        self.dbInfo.wordWrap = True
        form.addRow(self.dbInfo)
        page.layout().insertWidget(0, box)
        self.dbRootPath.connect("currentPathChanged(QString)", self.onDatabaseRootChanged)
        refreshButton.connect("clicked()", lambda: self._fillDatabases(force=True))
        self.dbCombo.connect("activated(int)", lambda i: self._setDatabase(self.dbCombo.currentData or "",
                                                                           chosen=True))
        for combo in [self.methodCombo, self.templateCombo, self.normEngineCombo] + list(self.normStepCombos):
            combo.connect("currentIndexChanged(int)", lambda i: self._updateDatabaseInfo())
        self.maskCheck.connect("toggled(bool)", lambda on: self._updateDatabaseInfo())
        for widget in self.normIterative.values():
            signal = "toggled(bool)" if isinstance(widget, qt.QCheckBox) else "currentIndexChanged(int)"
            try:
                widget.connect(signal, lambda *args: self._updateDatabaseInfo())
            except Exception:
                pass
        return page

    def _setRowVisible(self, widget, visible):
        widget.visible = visible
        try:
            label = widget.parentWidget().layout().labelForField(widget)
            if label is not None:
                label.visible = visible
        except Exception:
            pass

    def _updateTemplateChoice(self):
        if not hasattr(self, "templateCombo"):
            return
        petTemplate = self.method() == METHOD_PET_TEMPLATE
        self._setRowVisible(self.templateCombo, not petTemplate)
        if not petTemplate:
            DynamicNeuroPETWidget._updateTemplateChoice(self)
            return
        folder = self._database()
        header = self._header(folder)
        if header is None:
            self.templateNote.text = "Pick a normal database above: its master mean map is the template."
        elif not sn.masterMeanPath(folder, header):
            self.templateNote.text = (f"'{header['name']}' has no master mean map yet (statistics not computed): pick "
                                      "another method or database.")
        else:
            self.templateNote.text = (f"Registered to the master mean map of '{header['name']}' (a "
                                      f"{header.get('tracer') or header.get('nuclide') or ''} template in the MNI "
                                      "2009c space).")

    def _template(self):
        if self.method() != METHOD_PET_TEMPLATE:
            return DynamicNeuroPETWidget._template(self)
        folder = self._database()
        header = self._header(folder)
        path = sn.masterMeanPath(folder, header) if header else None
        if not path:
            raise ValueError("Pick a normal database with a master mean map as the template.")
        template = self.logic.loadAtlasVolume(path, name=f"{header['name']} master mean")
        self.setNode("Template", template)
        return template

    def _updateNormButtons(self):
        DynamicNeuroPETWidget._updateNormButtons(self)
        if hasattr(self, "dbCombo") and self.method() == METHOD_PET_TEMPLATE:
            folder = self._database()
            header = self._header(folder)
            self.normalizeButton.enabled = (self.volume() is not None and header is not None
                                            and bool(sn.masterMeanPath(folder, header)))

    @staticmethod
    def _normHardenText(volume, template, method):
        return (f"Resample '{volume.GetName()}' through the normalization onto the field of view of the template "
                f"'{template.GetName()}'."
                + ("\nThe CT/MR is normalized too (on the template grid)." if method == METHOD_ANATOMY else "")
                + "\n\nSave separately (default) keeps the image in the patient space next to the normalized one.")

    # --- normalization settings (stored with a database and on the normalized image) ----------------------
    def normalizationSettings(self):
        """The normalization settings on the page, in the categories of normaldb.NORM_KEYS."""
        from DynamicPETRegistration import iterativeSettings
        method = self.method()
        engine = self.normEngine()
        if engine == reg.ENGINE_ITERATIVE:
            iterative = iterativeSettings(self.normIterative)
            linear, deformable = iterative["iterativeLinear"], bool(iterative["iterativeDeformable"])
        else:
            steps = [c.currentData for c in self.normStepCombos if c.currentData]
            linear = next((s for s in steps if s != "BSpline"), "")
            deformable = "BSpline" in steps
        return {"method": method, "template": "" if method == METHOD_PET_TEMPLATE else
                (self.templateCombo.currentData or ""), "engine": engine, "linear": linear,
                "deformable": deformable, "brainMask": bool(self.maskCheck.checked and self.maskCheck.enabled)}

    @staticmethod
    def usedNormalization(volume):
        """The normalization settings a volume was normalized with (stored at Harden), None when not known."""
        text = volume.GetAttribute(NORMALIZATION_ATTRIBUTE) if volume is not None else None
        try:
            return json.loads(text) if text else None
        except ValueError:
            return None

    def onNormalize(self):
        settings = self.normalizationSettings()
        DynamicNeuroPETWidget.onNormalize(self)
        if self.normResult is not None:
            self.normResult["settings"] = settings
        self._updateDatabaseInfo()

    def onNormHarden(self):
        settings = (self.normResult or {}).get("settings")
        undo = self.normUndo
        DynamicNeuroPETWidget.onNormHarden(self)
        volume = self.volume()
        if self.normUndo is not undo and volume is not None and settings:
            volume.SetAttribute(NORMALIZATION_ATTRIBUTE, json.dumps(settings))
        self._updateDatabaseInfo()

    def _applyNormalizationSettings(self, header):
        """A database's normalization settings put on the page as the default (not when the image is normalized or
        a normalization is pending); the user may change them."""
        norm = header.get("normalization") or {}
        if not normaldb.isSpecified(norm) or self.normResult is not None:
            return
        volume = self.volume()
        if volume is not None and volume.GetAttribute(NEURO_PREFIX + "Template"):
            return
        if norm.get("method"):
            index = self.methodCombo.findData(norm["method"])
            if index >= 0:
                self.methodCombo.setCurrentIndex(index)
                self._methodChosen = True
        if norm.get("template") and self.method() == METHOD_PET:
            index = self.templateCombo.findData(norm["template"])
            if index >= 0 and (not self.atlas or neuro.canMakeTemplate(self.atlas, norm["template"])):
                self.templateCombo.setCurrentIndex(index)
                self._templateChosen = True
        if norm.get("engine"):
            index = self.normEngineCombo.findData(norm["engine"])
            if index >= 0:
                self.normEngineCombo.setCurrentIndex(index)
        if self.normEngine() == reg.ENGINE_ITERATIVE:
            if norm.get("linear"):
                combo = self.normIterative["linear"]
                index = combo.findData(norm["linear"])
                if index >= 0:
                    combo.setCurrentIndex(index)
            if norm.get("deformable") is not None:
                self.normIterative["deformable"].checked = bool(norm["deformable"])
        else:
            if norm.get("linear"):
                index = self.normStepCombos[0].findData(norm["linear"])
                if index >= 0:
                    self.normStepCombos[0].setCurrentIndex(index)
            if norm.get("deformable") is not None and len(self.normStepCombos) > 1:
                combo = self.normStepCombos[1]
                combo.setCurrentIndex(max(0, combo.findData("BSpline" if norm["deformable"] else "")))
        if norm.get("brainMask") is not None and self.maskCheck.enabled:
            self.maskCheck.checked = bool(norm["brainMask"])
        self._updateTemplateChoice()

    # --- normal database ---------------------------------------------------------------------------
    def onDatabaseRootChanged(self, folder):
        self.logic.setDatabaseRoot(folder)
        self._fillDatabases(force=True)

    def _database(self):
        return self._parameter().GetParameter(DATABASE_PARAMETER) or ""

    def _header(self, folder):
        for rowFolder, header, _, _ in self.databaseRows:
            if rowFolder == folder:
                return header
        return None

    def _fillDatabases(self, force=False):
        if not hasattr(self, "dbCombo"):
            return
        info = self.logic.imageInfo(self.volume())
        root = self.logic.databaseRoot()
        key = (root, info["modality"], info["nuclide"], info["tracer"])
        if key == self._databaseKey and not force:
            return
        self._databaseKey = key
        try:
            self.databaseRows = self.logic.databases(root, info)
        except Exception:
            logging.warning("Static Neuro: could not list the databases", exc_info=True)
            self.databaseRows = []
        current = self._database()
        self.dbCombo.blockSignals(True)
        self.dbCombo.clear()
        self.dbCombo.addItem("No database (regional values only)", "")
        for folder, header, rank, problems in self.databaseRows:
            n = normaldb.subjectCount(folder, header)
            text = (("! " if rank else "") + f"{header.get('name')} - {header.get('modality', 'PET')} "
                    f"{header.get('tracer') or header.get('nuclide')}, n = {n}"
                    + (" (predefined)" if header.get("kind") == normaldb.PREDEFINED else ""))
            self.dbCombo.addItem(text, folder)
            self.dbCombo.setItemData(self.dbCombo.count - 1, "Differs from the image: " + "; ".join(problems)
                                     if problems else "Fits the image", qt.Qt.ToolTipRole)
        self.dbCombo.setCurrentIndex(max(0, self.dbCombo.findData(current)))
        self.dbCombo.blockSignals(False)
        if not self._databaseChosen and not current:
            best = next((folder for folder, _, rank, _ in self.databaseRows if rank == 0), "")
            if best:
                self._setDatabase(best, chosen=False)
        elif current and self._header(current) is None:
            self._setDatabase("", chosen=False)
        self._fillAddDatabases()
        self._updateDatabaseInfo()
        self._updateTemplateChoice()

    def _setDatabase(self, folder, chosen=True):
        """Pick the normal database (the folder, '' for none): its normalization settings become the default of the
        Normalization page, its region settings are applied (locked)."""
        self._databaseChosen = self._databaseChosen or chosen
        self._parameter().SetParameter(DATABASE_PARAMETER, folder or "")
        if hasattr(self, "dbCombo"):
            self.dbCombo.blockSignals(True)
            self.dbCombo.setCurrentIndex(max(0, self.dbCombo.findData(folder or "")))
            self.dbCombo.blockSignals(False)
        header = self._header(folder)
        if header is not None:
            rank, problems = sn.databaseMatch(header, self.logic.imageInfo(self.volume()))
            if chosen and rank:
                slicer.util.warningDisplay(f"'{header['name']}' differs from the image: " + "; ".join(problems) + ".",
                                           windowTitle="Normal database")
            self._applyNormalizationSettings(header)
            self._applyDatabaseSettings(folder, header)
            self.unlockCheck.blockSignals(True)
            self.unlockCheck.checked = False
            self.unlockCheck.blockSignals(False)
        self._applyLock()
        self._fillAddDatabases()
        self._updateTemplateChoice()
        self._updateNormButtons()
        self._updateDatabaseInfo()

    def _normalizationWarnings(self, header):
        """Lines about the normalization against the database's: its settings, and how the image (normalized) or
        the page (not yet) differs. Never blocking."""
        norm = header.get("normalization") or {}
        if not normaldb.isSpecified(norm):
            return ["Normalization of the database: not specified (the first subject added sets it)."]
        lines = [f"Normalization of the database: {normaldb.normalizationText(norm)}."]
        volume = self.volume()
        normalized = volume is not None and bool(volume.GetAttribute(NEURO_PREFIX + "Template"))
        if normalized:
            used = self.usedNormalization(volume)
            if used is None:
                lines.append(WARNING_STYLE.format("The image's normalization settings are not known."))
                return lines
            differences = normaldb.normalizationDifferences(norm, used)
            if differences:
                lines.append(WARNING_STYLE.format("The image was normalized otherwise: " + "; ".join(differences)
                                                  + "."))
        else:
            differences = normaldb.normalizationDifferences(norm, self.normalizationSettings())
            if differences:
                lines.append(WARNING_STYLE.format("The settings on this page differ: " + "; ".join(differences)
                                                  + " (allowed, but the patient is not normalized like the "
                                                  "database's subjects)."))
        return lines

    def _updateDatabaseInfo(self):
        if not hasattr(self, "dbInfo") or not hasattr(self, "regionsDbInfo"):
            return
        root = self.logic.databaseRoot()
        folder = self._database()
        header = self._header(folder)
        info = self.logic.imageInfo(self.volume())
        if not root:
            text = "Pick the database folder, or go on without a database (the review gives the regional values only)."
            regionsText = "No normal database: the region settings below are free."
        elif header is None:
            text = (f"{len(self.databaseRows)} static database(s) in the folder. Without a database the review gives "
                    "the regional values only.")
            regionsText = "No normal database: the region settings below are free."
        else:
            lines = []
            rank, problems = sn.databaseMatch(header, info)
            if problems:
                lines.append(WARNING_STYLE.format("Differs from the image: " + "; ".join(problems)))
            warnings = sn.unitWarnings(info, header)
            if warnings:
                lines.append(WARNING_STYLE.format(" ".join(warnings)))
            if not sn.masterMeanPath(folder, header):
                lines.append("Its mean / SD maps are not computed yet.")
            lines.extend(self._normalizationWarnings(header))
            text = "<br>".join(lines)
            settings = sn.regionSettings(folder, header)
            regionLines = [f"Database '{header['name']}':"] + sn.settingsText(settings)
            if self._locked() and not settings["referenceMaskPath"] and not settings["referenceAtlasRegion"]:
                regionLines.append(WARNING_STYLE.format("It has no reference mask copy: unlock to pick the "
                                                        "reference region."))
            regionLines.append("Settings of the database (locked)." if self._locked() else
                               WARNING_STYLE.format("Unlocked: the regions may differ from the database's."))
            regionsText = "<br>".join(regionLines)
        self.dbInfo.text = text
        self.regionsDbInfo.text = regionsText

    # --- page 4: regions (with the database's region settings) ------------------------------------------
    def _buildRegionsPage(self):
        page = DynamicNeuroPETWidget._buildRegionsPage(self)
        box = qt.QGroupBox("Region settings of the normal database")
        form = qt.QFormLayout(box)
        self.regionsDbInfo = qt.QLabel()
        self.regionsDbInfo.wordWrap = True
        form.addRow(self.regionsDbInfo)
        self.unlockCheck = qt.QCheckBox("Unlock the region settings")
        self.unlockCheck.toolTip = ("The regions are drawn as in the database (atlas, growth, reference region) so the "
                                    "patient's values compare with the database's. Unlock to change them.")
        form.addRow(self.unlockCheck)
        page.layout().insertWidget(0, box)
        self.unlockCheck.connect("toggled(bool)", self.onUnlockToggled)
        self.growSpin.connect("valueChanged(double)", lambda v: self._applyLock())
        return page

    def _applyDatabaseSettings(self, folder, header):
        settings = sn.regionSettings(folder, header)
        index = self.targetAtlasCombo.findData(settings["regionAtlas"])
        if index >= 0:
            self.targetAtlasCombo.setCurrentIndex(index)
        self.growSpin.value = settings["growMm"]
        self.growIntoCombo.setCurrentIndex(max(0, self.growIntoCombo.findData(settings["growInto"])))
        self.erodeSpin.value = settings["erodeReferenceMm"]
        if settings["referenceAtlasRegion"]:
            self.referenceAtlasCombo.setCurrentIndex(self.referenceAtlasCombo.findData(ATLAS_BILATERAL))
            self._fillReferenceRegions()
            index = self.referenceRegionCombo.findText(settings["referenceAtlasRegion"])
            if index >= 0:
                self.referenceRegionCombo.setCurrentIndex(index)

    def _locked(self):
        return bool(self._header(self._database())) and not self.unlockCheck.checked

    def _applyLock(self):
        if not hasattr(self, "unlockCheck"):
            return
        locked = self._locked()
        for widget in (self.targetAtlasCombo, self.referenceAtlasCombo, self.referenceRegionCombo, self.growSpin,
                       self.erodeSpin):
            widget.enabled = not locked
        self.growIntoCombo.enabled = not locked and self.growSpin.value > 0
        self.unlockCheck.enabled = bool(self._header(self._database()))

    def onUnlockToggled(self, on):
        if on and not slicer.util.confirmOkCancelDisplay(
                "With other region settings than the database's, the patient's regional values are not drawn the "
                "way the normal values were, so the comparison may be biased. Unlock?", windowTitle="Regions"):
            self.unlockCheck.blockSignals(True)
            self.unlockCheck.checked = False
            self.unlockCheck.blockSignals(False)
        elif not on:
            folder = self._database()
            header = self._header(folder)
            if header is not None:
                self._applyDatabaseSettings(folder, header)
        self._applyLock()
        self._updateDatabaseInfo()

    def _regionsArguments(self):
        arguments = DynamicNeuroPETWidget._regionsArguments(self)
        folder = self._database()
        header = self._header(folder)
        if header is None or not self._locked():
            return arguments
        settings = sn.regionSettings(folder, header)
        if settings["referenceMaskPath"]:
            arguments["referenceMaskPath"] = settings["referenceMaskPath"]
            arguments["referenceName"] = settings["referenceTitle"] or "Reference"
        elif settings["referenceAtlasRegion"]:
            arguments["referenceKind"] = ATLAS_BILATERAL
            arguments["referenceName"] = settings["referenceAtlasRegion"]
        else:
            raise ValueError(f"The database '{header['name']}' has no copy of its reference mask "
                             f"({settings['referenceTitle']}). Unlock the region settings to pick a reference region.")
        return arguments

    def _updateRegionsPage(self):
        DynamicNeuroPETWidget._updateRegionsPage(self)
        if hasattr(self, "dbCombo"):
            self._fillDatabases()
            self._applyLock()

    # --- page 5: review (placeholder) and adding the patient to a database ------------------------------
    def _buildReviewPage(self):
        page = qt.QWidget()
        layout = qt.QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        box = qt.QGroupBox("Review")
        boxLayout = qt.QVBoxLayout(box)
        self.reviewLabel = qt.QLabel()
        self.reviewLabel.wordWrap = True
        boxLayout.addWidget(self.reviewLabel)
        layout.addWidget(box)

        addBox = qt.QGroupBox("Add this patient to a normal database")
        addBox.toolTip = ("For building a user defined database from normal subjects: the normalized image is "
                          "resampled onto the database grid (MNI 2009c, 1 mm) and saved as its SUV map and its SUVR "
                          "map (divided by the mean in the database's reference mask) in subjects/; a row is added "
                          "to subjects.csv. The mean / SD maps are then out of date.")
        form = qt.QFormLayout(addBox)
        self.addDbCombo = qt.QComboBox()
        self.addDbCombo.toolTip = "User defined databases of the database folder (step 3)"
        form.addRow("Database:", self.addDbCombo)
        self.addReferenceCombo = qt.QComboBox()
        self.addReferenceCombo.toolTip = ("Reference region of the SUVR map (intensity normalization): a segment of "
                                          "the regions (step 4; by default their reference region) or the database's "
                                          "reference mask. Warns when it is not the database's reference.")
        form.addRow("Reference region:", self.addReferenceCombo)
        self.subjectIdEdit = qt.QLineEdit()
        self.subjectIdEdit.toolTip = "ID of the subject in the database (file names and subjects.csv)"
        form.addRow("Subject ID:", self.subjectIdEdit)
        self.addSubjectButton = _button("Add to database")
        form.addRow(self.addSubjectButton)
        self.addLabel = qt.QLabel()
        self.addLabel.wordWrap = True
        self.addLabel.setTextInteractionFlags(qt.Qt.TextSelectableByMouse)
        form.addRow(self.addLabel)
        layout.addWidget(addBox)
        layout.addStretch(1)
        self.addSubjectButton.connect("clicked()", self.onAddToDatabase)
        self.addDbCombo.connect("activated(int)", lambda i: self._onAddDatabasePicked())
        self.addReferenceCombo.connect("activated(int)", lambda i: setattr(self, "_addReferenceChosen", True))
        return page

    def _fillAddDatabases(self):
        if not hasattr(self, "addDbCombo"):
            return
        current = self.addDbCombo.currentData or ""
        chosen = getattr(self, "_addDbChosen", False)
        self.addDbCombo.blockSignals(True)
        self.addDbCombo.clear()
        for folder, header, rank, _ in self.databaseRows:
            if header.get("kind") != normaldb.USER_DEFINED:
                continue
            self.addDbCombo.addItem(("! " if rank else "") + f"{header.get('name')} (n = "
                                    f"{normaldb.subjectCount(folder, header)})", folder)
        wanted = current if chosen and current else self._database()
        self.addDbCombo.setCurrentIndex(max(0, self.addDbCombo.findData(wanted)))
        self.addDbCombo.blockSignals(False)

    def _onAddDatabasePicked(self):
        self._addDbChosen = True
        self._fillAddReferences()

    def _fillAddReferences(self):
        """The reference regions to pick from: the segments of the regions (their reference region first) and the
        reference mask of the database picked for adding."""
        if not hasattr(self, "addReferenceCombo"):
            return
        segmentation = self.node("Segmentation")
        folder = self.addDbCombo.currentData or ""
        header = self._header(folder)
        referenceId = self._parameter().GetParameter(NEURO_PREFIX + "ReferenceSegment") or ""
        ids = self.logic.segmentIds(segmentation) if segmentation is not None else []
        key = (segmentation.GetID() if segmentation else "", tuple(ids), referenceId, folder)
        if getattr(self, "_addReferencesFor", None) == key:
            return
        self._addReferencesFor = key
        current = self.addReferenceCombo.currentData
        chosen = getattr(self, "_addReferenceChosen", False)
        self.addReferenceCombo.blockSignals(True)
        self.addReferenceCombo.clear()
        if referenceId in ids:
            ids.remove(referenceId)
            ids.insert(0, referenceId)
        for segmentId in ids:
            self.addReferenceCombo.addItem(segmentation.GetSegmentation().GetSegment(segmentId).GetName(), segmentId)
        if header is not None and sn.referenceMaskPath(folder, header):
            title = sn.regionSettings(folder, header)["referenceTitle"] or "reference"
            self.addReferenceCombo.addItem(f"Database reference mask ({title})", "")
        index = self.addReferenceCombo.findData(current) if chosen and current is not None else -1
        self.addReferenceCombo.setCurrentIndex(index if index >= 0 else 0)
        self.addReferenceCombo.blockSignals(False)

    def _addReference(self, folder, header):
        """(segmentation, segment ID, name) of the reference region picked for adding; (None, None, name) for the
        database's reference mask."""
        segmentId = self.addReferenceCombo.currentData if self.addReferenceCombo.count else None
        segmentation = self.node("Segmentation")
        if segmentId and segmentation is not None:
            name = segmentation.GetSegmentation().GetSegment(segmentId).GetName()
            return segmentation, segmentId, name
        return None, None, sn.regionSettings(folder, header)["referenceTitle"] or "Database reference mask"

    def _updateReviewPage(self):
        if not hasattr(self, "reviewLabel"):
            return
        volume = self.volume()
        segmentation = self.node("Segmentation")
        header = self._header(self._database())
        age, sex = self.logic.ageAndSex(volume)
        lines = ["<b>Coming next</b>: the comparison of the patient with the normal database (regional values and "
                 "voxel z-scores against the mean and SD maps of the patient's age group).", ""]
        lines.append(f"Image: {volume.GetName() if volume else 'none'}"
                     + (" (normalized)" if volume is not None and volume.GetAttribute(NEURO_PREFIX + "Template")
                        else ""))
        lines.append("Age: " + (f"{age:g}" if age is not None else "unknown") + ", sex: "
                     + (dict(sn.SEXES).get(sex) if sex else "unknown"))
        lines.append(f"Regions: {segmentation.GetName() if segmentation else 'not loaded'}")
        if header is not None:
            group = normaldb.ageGroupOf(age, header.get("ageGroups") or []) if header.get("ageStratified") else None
            lines.append(f"Database: {header['name']}" + (f", age group {group}" if group else ""))
        else:
            lines.append("No database: the review will give the regional values (SUV) only.")
        self.reviewLabel.text = "<br>".join(lines)
        if hasattr(self, "subjectIdEdit"):
            key = volume.GetID() if volume is not None else ""
            if getattr(self, "_subjectIdFor", None) != key:
                self._subjectIdFor = key
                self.subjectIdEdit.text = (sn.suggestedSubjectId(self.logic.attributes(volume), volume.GetName())
                                           if volume is not None else "")
            self._fillAddReferences()
            normalized = volume is not None and bool(volume.GetAttribute(NEURO_PREFIX + "Template"))
            self.addSubjectButton.enabled = (normalized and self.addDbCombo.count > 0
                                             and self.addReferenceCombo.count > 0)
            if volume is not None and not normalized:
                self.addLabel.text = "Normalize the image (step 3) to add it to a database."
            elif self.addDbCombo.count == 0 and not self.addLabel.text:
                self.addLabel.text = ("No user defined database in the database folder (create one in the Neuro PET "
                                      "Database Manager).")

    def onAddToDatabase(self):
        volume = self.volume()
        folder = self.addDbCombo.currentData or ""
        header = self._header(folder)
        if volume is None or header is None:
            return
        subjectId = self.subjectIdEdit.text.strip()
        if not subjectId:
            slicer.util.warningDisplay("Type the subject ID.", windowTitle="Add to database")
            return
        segmentation, segmentId, referenceName = self._addReference(folder, header)
        gridPath = sn.referenceMaskPath(folder, header) or (self.atlas or {}).get(neuro.BRAIN_MASK)
        if not gridPath:
            slicer.util.errorDisplay(f"'{header['name']}' has no reference mask and the MNI brain mask is not in the "
                                     "atlas folder: no database grid.", windowTitle="Add to database")
            return
        if segmentId is None and not sn.referenceMaskPath(folder, header):
            slicer.util.errorDisplay("Pick the reference region (load the regions in step 4).",
                                     windowTitle="Add to database")
            return
        age, sex = self.logic.ageAndSex(volume)
        info = self.logic.imageInfo(volume)
        notes = []
        rank, problems = sn.databaseMatch(header, info)
        if problems:
            notes.append("The database differs from the image: " + "; ".join(problems) + ".")
        notes.extend(sn.unitWarnings(info, header))
        used = self.usedNormalization(volume)
        if used is None:
            notes.append("The image's normalization settings are not known.")
        else:
            differences = normaldb.normalizationDifferences(header.get("normalization"), used)
            if differences:
                notes.append("Normalized otherwise than the database: " + "; ".join(differences) + ".")
        expected = sn.regionSettings(folder, header)["referenceTitle"]
        if segmentId and expected and not _sameRegion(referenceName, expected):
            notes.append(f"The reference region '{referenceName}' is not the database's ({expected}).")
        if age is None:
            notes.append("The age is not known" + (" (the database has age groups)." if header.get("ageStratified")
                                                   else "."))
        replace = normaldb.hasSubject(folder, subjectId)
        if replace:
            notes.append(f"Subject '{subjectId}' is already in the database: it will be replaced.")
        text = f"Add '{volume.GetName()}' to '{header['name']}' as '{subjectId}'?"
        if notes:
            text += "\n\n" + "\n".join("- " + n for n in notes)
        if not slicer.util.confirmOkCancelDisplay(text, windowTitle="Add to database"):
            return
        qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)
        try:
            row = self.logic.addSubject(folder, header, volume, subjectId, gridPath, age, sex, used, replace,
                                        segmentation, segmentId, referenceName)
        except Exception as error:
            logging.exception("Static Neuro: could not add the subject")
            slicer.util.errorDisplay(f"Could not add the patient to the database:\n{error}")
            return
        finally:
            qt.QApplication.restoreOverrideCursor()
        self._fillDatabases(force=True)
        count = normaldb.subjectCount(folder, normaldb.readHeader(folder))
        self.addLabel.text = (f"'{subjectId}' added to '{header['name']}' (now {count} subjects; reference mean "
                              f"{float(row['referenceMean']):.4g}). Its mean / SD maps are out of date: compute them "
                              "again in the Neuro PET Database Manager.")

    # --- workflow --------------------------------------------------------------------------------
    def _callStep(self, key, method):
        if key == STEP_REG:
            if method == "exit":
                self._closeRegSession()
            elif method == "enter" and self.regResult is not None:
                self._openRegSession()
            return
        DynamicNeuroPETWidget._callStep(self, key, method)

    def pendingHarden(self, key):
        if key == STEP_REG:
            return self.regResult is not None
        return DynamicNeuroPETWidget.pendingHarden(self, key)

    def _hardenStep(self, key):
        if key == STEP_REG:
            self.onRegHarden()
        else:
            DynamicNeuroPETWidget._hardenStep(self, key)

    def stepStates(self):
        volume, anatomy = self.volume(), self.anatomy()
        states = {STEP_DATA: STATE_DONE if volume is not None else STATE_READY}
        ready = STATE_READY if volume is not None else STATE_WAITING
        states[STEP_REG] = (STATE_DONE if volume is not None and volume.GetAttribute(REG_PREFIX + "Fixed") else
                            ready if anatomy is not None else STATE_WAITING)
        normalized = volume is not None and bool(volume.GetAttribute(NEURO_PREFIX + "Template"))
        states[STEP_NORM] = STATE_DONE if normalized else ready
        segmentation = self.node("Segmentation")
        states[STEP_REGIONS] = (STATE_DONE if segmentation is not None else
                                STATE_READY if normalized else STATE_WAITING)
        states[STEP_REVIEW] = STATE_READY if segmentation is not None else STATE_WAITING
        return states

    def refresh(self):
        DynamicNeuroPETWidget.refresh(self)
        self._updateRegButtons()
        self._updateReviewPage()

    def cleanup(self):
        self._closeRegSession()
        DynamicNeuroPETWidget.cleanup(self)

    def onSceneClose(self, caller, event):
        DynamicNeuroPETWidget.onSceneClose(self, caller, event)
        self._closeRegSession()
        self.regResult = None
        self.regUndo = None
        self._databaseKey = None
        if hasattr(self, "regLabel"):
            self.regLabel.text = "Not registered."


# ---------------------------------------------------------------------------
# Test: import from a BIDS file, harden a volume, the workflow pages
# ---------------------------------------------------------------------------
class StaticNeuroPETTest(ScriptedLoadableModuleTest):
    def setUp(self):
        slicer.mrmlScene.Clear()

    def runTest(self):
        self.setUp()
        self.test_ImportAndHarden()
        self.setUp()
        self.test_Widget()

    def test_ImportAndHarden(self):
        import tempfile
        import vtk
        self.delayDisplay("Static Neuro: import and harden")
        folder = tempfile.mkdtemp()
        source = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "source")
        array = np.zeros((20, 24, 22), dtype=np.float32)
        array[6:14, 8:16, 7:15] = 5.0
        slicer.util.updateVolumeFromArray(source, array)
        path = os.path.join(folder, "sub-7_ses-SUV_pet.nii.gz")
        self.assertTrue(slicer.util.saveNode(source, path))
        with open(os.path.join(folder, "sub-7_ses-SUV_pet.json"), "w") as f:
            json.dump({"TracerName": "FDG", "TracerRadionuclide": "F18", "Units": "Bq/mL"}, f)
        with open(os.path.join(folder, "participants.tsv"), "w") as f:
            f.write("participant_id\tage\tsex\nsub-7\t64\tF\n")
        logic = StaticNeuroPETLogic()
        node = logic.loadFromFile(path)
        info = logic.imageInfo(node)
        self.assertEqual((info["modality"], info["unitsKind"], info["tracer"]), ("PET", sn.UNITS_SUV, "FDG"))
        self.assertEqual(logic.ageAndSex(node), (64, "F"))
        logic.setAgeAndSex(node, 65, "M")
        self.assertEqual(logic.ageAndSex(node), (65, "M"))
        self.assertTrue(logic.isStaticImage(node))

        fixed = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "fixed")
        slicer.util.updateVolumeFromArray(fixed, np.zeros((20, 24, 22), dtype=np.float32))
        linear = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLinearTransformNode", "shift")
        matrix = vtk.vtkMatrix4x4()
        matrix.SetElement(0, 3, 2.0)
        linear.SetMatrixTransformToParent(matrix)
        undo, moved = logic.harden(node, fixed, linear, reg.SAVE_SEPARATELY, reg.OUTPUT_FIXED_GRID)[:2]
        self.assertIsNot(moved, node)
        self.assertEqual(moved.GetAttribute(NEURO_PREFIX + "Template"), "fixed")
        self.assertEqual(moved.GetAttribute(sn.STATIC_PREFIX + "Kind"), "static")
        self.assertAlmostEqual(float(slicer.util.arrayFromVolume(moved).sum()), float(array.sum()), delta=1.0)
        logic.revertHarden(undo)
        self.assertFalse(slicer.mrmlScene.IsNodePresent(moved))
        undo, same = logic.hardenRegistration(node, fixed, linear, reg.OVERWRITE, reg.OUTPUT_FIXED_GRID)
        self.assertIs(same, node)
        self.assertEqual(node.GetAttribute(REG_PREFIX + "Fixed"), "fixed")
        logic.revertHarden(undo)
        self.assertIsNone(node.GetAttribute(REG_PREFIX + "Fixed"))
        self.assertTrue(np.array_equal(slicer.util.arrayFromVolume(node), array))
        self.delayDisplay("Import and harden passed")

    def test_Widget(self):
        self.delayDisplay("Static Neuro: widget")
        widget = slicer.modules.staticneuropet.widgetRepresentation().self()
        node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "static")
        slicer.util.updateVolumeFromArray(node, np.ones((8, 8, 8), dtype=np.float32))
        node.SetAttribute("StaticPET.Kind", "static")
        node.SetAttribute("StaticPET.Units", "SUVbw")
        widget.setNode("Sequence", None)
        widget.selectInputs()
        self.assertIs(widget.volume(), node)
        for key, _, _, _ in STEPS:
            widget.showStep(key)
            self.assertEqual(widget.current, key)
        self.assertEqual(widget.stepStates()[STEP_DATA], STATE_DONE)
        self.assertEqual(widget.stepStates()[STEP_REVIEW], STATE_WAITING)
        self.assertTrue(hasattr(widget, "dbCombo") and hasattr(widget, "addReferenceCombo"))
        self.assertFalse(widget.addSubjectButton.enabled)        # not normalized
        settings = widget.normalizationSettings()
        self.assertIn(settings["method"], [m for m, _ in METHODS])
        self.assertTrue(_sameRegion("Cerebellum Gray Matter" + REFERENCE_SUFFIX, "cerebellum gray matter"))
        # batch: the normalization plan of a database without settings, the QC picture
        plan, settings = StaticNeuroPETLogic().batchPlan(normaldb.newHeader("batch", nuclide="F-18", tracer="FDG"))
        self.assertEqual(plan["method"], METHOD_PET)
        self.assertTrue(settings["steps"])
        import tempfile
        picture = os.path.join(tempfile.mkdtemp(), "qc.png")
        anatomy = sn.rasArray(slicer.util.arrayFromVolume(node), _ijkToRas(node))
        StaticNeuroPETLogic.writeQcPicture(picture, node, (anatomy, tuple(_ijkToRas(node).ravel())))
        self.assertTrue(os.path.getsize(picture) > 0)
        widget.showStep(STEP_DATA)
        self.delayDisplay("Widget passed")
