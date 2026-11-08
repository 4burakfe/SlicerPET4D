"""
Static Neuro PET/SPECT (3D Slicer module) - workflow app

A step-by-step workflow for a static brain PET or SPECT (FDG, amyloid, perfusion SPECT, ...), built on the Dynamic Neuro
Receptor PET module (same normalization and regions) and the Neuro PET normal databases:

    1. Data                 the static PET (PET4D DICOM plugin) or SPECT (DICOM SPECT reconstruction), or an image
                            file (BIDS PET: sub-<id>_ses-<session>_pet.nii.gz with its .json sidecar and the
                            participants.tsv), the patient's CT / MR, the age and sex (read from the tags, editable)
    2. CT/MR registration   optional: the image registered to the CT / MR (rigid by default) and hardened
    3. Normalization        to the MNI ICBM 152 2009c space: the PET/SPECT to the T1 template, the CT/MR to the MR
                            template with the same transform applied to the image, or the PET/SPECT to a PET template
                            (the master mean map of a normal database)
    4. Regions/Template     the normal database the patient is compared with (those of the image's modality, nuclide
                            and tracer first) and the CerebrA regions: with a database its region settings (atlas,
                            growth, white matter, reference region and its mask) are used, locked unless unlocked
    5. Review               placeholder: the comparison with the database comes here (without a database: the
                            regional SUVs)
"""

import json
import logging
import os
import sys

import ctk
import numpy as np
import qt
import slicer
from slicer.ScriptedLoadableModule import ScriptedLoadableModule, ScriptedLoadableModuleTest

_MODULE_DIR = os.path.dirname(os.path.abspath(globals().get("__file__", "") or "."))
if globals().get("__file__") and _MODULE_DIR not in sys.path:
    sys.path.insert(0, _MODULE_DIR)

from DICOMGatedDynamicPETLib import normaldb  # noqa: E402
from DICOMGatedDynamicPETLib import registration as reg  # noqa: E402
from DICOMGatedDynamicPETLib import resampling  # noqa: E402
from DICOMGatedDynamicPETLib import staticneuro as sn  # noqa: E402
from DICOMGatedDynamicPETLib.hardendialog import hardenDialog  # noqa: E402
from DynamicNeuroPET import (DynamicNeuroPETLogic, DynamicNeuroPETWidget, _button, _ijkToRas, _percentiles,  # noqa
                             _vtkMatrix, ATLAS_BILATERAL, ATLAS_FILE_ATTRIBUTE, METHOD_ANATOMY, METHOD_PET,
                             METHOD_PET_TEMPLATE, NEURO_PREFIX, NORMALIZED_SUFFIX, REG_PREFIX, STATE_DONE,
                             STATE_READY, STATE_WAITING, STEP_DATA, STEP_NORM, STEP_REG, STEP_REGIONS)

STEP_REVIEW = "review"
STEPS = (
    (STEP_DATA, None, "Data", "Static PET / SPECT of the brain and the patient's CT / MR"),
    (STEP_REG, None, "CT/MR registration", "Optional: the image registered to the CT / MR"),
    (STEP_NORM, None, "Normalization", "To the MNI ICBM 152 2009c space"),
    (STEP_REGIONS, None, "Regions/Template", "Normal database and CerebrA regions"),
    (STEP_REVIEW, None, "Review", "Comparison with the normal database"),
)
METHODS = ((METHOD_PET, "PET/SPECT to MR template (T1)"),
           (METHOD_ANATOMY, "CT/MR to MR template, same transform to the PET/SPECT"),
           (METHOD_PET_TEMPLATE, "PET/SPECT to PET template (normal database mean)"))
SETTINGS_DB_ROOT = "DICOMGatedDynamicPET/NeuroDatabase/RootFolder"     # shared with the Neuro PET Database Manager
SETTINGS_IMPORT_FOLDER = "DICOMGatedDynamicPET/StaticNeuro/ImportFolder"
AGE_ATTRIBUTE, SEX_ATTRIBUTE = NEURO_PREFIX + "PatientAge", NEURO_PREFIX + "PatientSex"
DATABASE_PARAMETER = NEURO_PREFIX + "Database"
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
            "a normal database), <b>4. Regions/Template</b> (the normal database and the CerebrA regions) and "
            "<b>5. Review</b>. The atlases are not shipped: pick their folder in step 3.")
        parent.acknowledgementText = (
            "Part of the DICOM Gated / Dynamic PET extension. MNI ICBM 152 nonlinear symmetric 2009c template: Fonov "
            "VS et al., NeuroImage 2011;54:313-327. CerebrA: Manera AL et al., Scientific Data 2020;7:237 (CC0).")


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
    """The Dynamic Neuro workflow on a static PET / SPECT: own Data, CT/MR registration, Regions/Template (normal
    database) and Review pages; the Normalization page of the Dynamic Neuro module with the PET template method."""

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

    # --- page 3: normalization (the Dynamic Neuro page, with the PET template) ---------------------------
    def _addTemplateRows(self, form):
        self.templateDbCombo = qt.QComboBox()
        self.templateDbCombo.toolTip = ("Normal database whose master mean map (MNI 2009c, all subjects) is the PET "
                                        "template. Only databases with computed statistics are listed.")
        form.addRow("Template database:", self.templateDbCombo)
        self.templateDbCombo.connect("activated(int)", lambda i: self.onTemplateDatabaseChanged())

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
        if hasattr(self, "templateDbCombo"):
            self._setRowVisible(self.templateDbCombo, petTemplate)
            self._setRowVisible(self.templateCombo, not petTemplate)
        if not petTemplate:
            DynamicNeuroPETWidget._updateTemplateChoice(self)
            return
        folder = self.templateDbCombo.currentData if hasattr(self, "templateDbCombo") else ""
        header = self._header(folder)
        self.templateNote.text = (f"Registered to the master mean map of '{header['name']}' (a "
                                  f"{header.get('tracer') or header.get('nuclide') or ''} template in the MNI 2009c "
                                  "space)." if header else
                                  "Pick a normal database with computed statistics (the root folder is set in step 4).")

    def onTemplateDatabaseChanged(self):
        folder = self.templateDbCombo.currentData or ""
        if folder and not self._database():
            self._setDatabase(folder, chosen=True)
        self._clearNormalization()
        self._updateTemplateChoice()
        self._updateNormButtons()

    def _template(self):
        if self.method() != METHOD_PET_TEMPLATE:
            return DynamicNeuroPETWidget._template(self)
        folder = self.templateDbCombo.currentData or ""
        header = self._header(folder)
        path = sn.masterMeanPath(folder, header) if header else None
        if not path:
            raise ValueError("Pick a normal database with a master mean map as the template.")
        template = self.logic.loadAtlasVolume(path, name=f"{header['name']} master mean")
        self.setNode("Template", template)
        return template

    def _updateNormButtons(self):
        DynamicNeuroPETWidget._updateNormButtons(self)
        if hasattr(self, "templateDbCombo") and self.method() == METHOD_PET_TEMPLATE:
            self.normalizeButton.enabled = self.volume() is not None and bool(self.templateDbCombo.currentData)

    @staticmethod
    def _normHardenText(volume, template, method):
        return (f"Resample '{volume.GetName()}' through the normalization onto the field of view of the template "
                f"'{template.GetName()}'."
                + ("\nThe CT/MR is normalized too (on the template grid)." if method == METHOD_ANATOMY else "")
                + "\n\nSave separately (default) keeps the image in the patient space next to the normalized one.")

    # --- page 4: regions / template --------------------------------------------------------------
    def _buildRegionsPage(self):
        page = DynamicNeuroPETWidget._buildRegionsPage(self)
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
        self.dbCombo.toolTip = ("The normal database the patient is compared with. Those of the image's modality, "
                                "nuclide and tracer are listed first (and picked); the others are marked with '!'.")
        refreshButton = qt.QPushButton("Refresh")
        refreshButton.toolTip = "Read the database folder again"
        dbRow.addWidget(self.dbCombo, 1)
        dbRow.addWidget(refreshButton)
        form.addRow("Database:", dbRow)
        self.dbInfo = qt.QLabel()
        self.dbInfo.wordWrap = True
        form.addRow(self.dbInfo)
        self.unlockCheck = qt.QCheckBox("Unlock the region settings")
        self.unlockCheck.toolTip = ("The regions are drawn as in the database (atlas, growth, reference region) so the "
                                    "patient's values compare with the database's. Unlock to change them.")
        form.addRow(self.unlockCheck)
        page.layout().insertWidget(0, box)
        self.dbRootPath.connect("currentPathChanged(QString)", self.onDatabaseRootChanged)
        refreshButton.connect("clicked()", lambda: self._fillDatabases(force=True))
        self.dbCombo.connect("activated(int)", lambda i: self._setDatabase(self.dbCombo.currentData or "",
                                                                           chosen=True))
        self.unlockCheck.connect("toggled(bool)", self.onUnlockToggled)
        self.growSpin.connect("valueChanged(double)", lambda v: self._applyLock())
        return page

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
        for combo, withMaster in ((self.dbCombo, False), (self.templateDbCombo, True)):
            selected = combo.currentData or ""
            combo.blockSignals(True)
            combo.clear()
            combo.addItem("No database (regional values only)" if not withMaster else "(none)", "")
            for folder, header, rank, problems in self.databaseRows:
                if withMaster and not sn.masterMeanPath(folder, header):
                    continue
                n = normaldb.subjectCount(folder, header)
                text = (("! " if rank else "") + f"{header.get('name')} - {header.get('modality', 'PET')} "
                        f"{header.get('tracer') or header.get('nuclide')}, n = {n}")
                combo.addItem(text, folder)
                combo.setItemData(combo.count - 1, "Differs from the image: " + "; ".join(problems) if problems else
                                  "Fits the image", qt.Qt.ToolTipRole)
            wanted = current if combo is self.dbCombo else (selected or current)
            combo.setCurrentIndex(max(0, combo.findData(wanted)))
            combo.blockSignals(False)
        if not self._databaseChosen and not current:
            best = next((folder for folder, _, rank, _ in self.databaseRows if rank == 0), "")
            if best:
                self._setDatabase(best, chosen=False)
        elif current and self._header(current) is None:
            self._setDatabase("", chosen=False)
        self._updateDatabaseInfo()
        self._updateTemplateChoice()

    def _setDatabase(self, folder, chosen=True):
        """Pick the normal database (the folder, '' for none) and apply its region settings, locked."""
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
            self._applyDatabaseSettings(folder, header)
            self.unlockCheck.blockSignals(True)
            self.unlockCheck.checked = False
            self.unlockCheck.blockSignals(False)
            if hasattr(self, "templateDbCombo") and not self.templateDbCombo.currentData:
                index = self.templateDbCombo.findData(folder)
                if index >= 0:
                    self.templateDbCombo.setCurrentIndex(index)
        self._applyLock()
        self._updateDatabaseInfo()

    def _applyDatabaseSettings(self, folder, header):
        settings = sn.regionSettings(folder, header)
        index = self.targetAtlasCombo.findData(settings["regionAtlas"])
        if index >= 0:
            self.targetAtlasCombo.setCurrentIndex(index)
        self.growSpin.value = settings["growMm"]
        self.growWhiteMatterCheck.checked = settings["growIntoWhiteMatter"]
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
        self.growWhiteMatterCheck.enabled = not locked and self.growSpin.value > 0
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

    def _updateDatabaseInfo(self):
        if not hasattr(self, "dbInfo"):
            return
        root = self.logic.databaseRoot()
        folder = self._database()
        header = self._header(folder)
        info = self.logic.imageInfo(self.volume())
        if not root:
            text = "Pick the database folder, or go on without a database (the review gives the regional values only)."
        elif header is None:
            text = (f"{len(self.databaseRows)} static database(s) in the folder. Without a database the review gives "
                    "the regional values only.")
        else:
            settings = sn.regionSettings(folder, header)
            lines = sn.settingsText(settings)
            rank, problems = sn.databaseMatch(header, info)
            if problems:
                lines.insert(0, WARNING_STYLE.format("Differs from the image: " + "; ".join(problems)))
            warnings = sn.unitWarnings(info, header)
            if warnings:
                lines.append(WARNING_STYLE.format(" ".join(warnings)))
            if header.get("kind") == normaldb.USER_DEFINED and not sn.masterMeanPath(folder, header):
                lines.append("Its statistics are not computed yet.")
            if self._locked() and not settings["referenceMaskPath"] and not settings["referenceAtlasRegion"]:
                lines.append(WARNING_STYLE.format("It has no reference mask copy: unlock to pick the reference "
                                                  "region."))
            lines.append("Settings of the database (locked)." if self._locked() else
                         WARNING_STYLE.format("Unlocked: the regions may differ from the database's."))
            text = "<br>".join(lines)
        self.dbInfo.text = text

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

    # --- page 5: review (placeholder) ---------------------------------------------------------------
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
        layout.addStretch(1)
        return page

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
        widget.showStep(STEP_DATA)
        self.delayDisplay("Widget passed")
