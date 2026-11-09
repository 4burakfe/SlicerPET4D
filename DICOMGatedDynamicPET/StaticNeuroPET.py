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
    5. Review               the comparison with the database (4 x 4 layout: PET with region outlines, voxel z / t
                            map, 3D-SSP views, region table with SUVmean and regional scores; fusion with the CT / MR / template); adding
                            the patient to a user defined database (SUV and SUVR maps on the database grid, a row
                            in subjects.csv)
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

from DICOMGatedDynamicPETLib import colormaps, colortables, neuro, normaldb  # noqa: E402
from DICOMGatedDynamicPETLib import neuroreview as nr  # noqa: E402
from DICOMGatedDynamicPETLib import registration as reg  # noqa: E402
from DICOMGatedDynamicPETLib import resampling  # noqa: E402
from DICOMGatedDynamicPETLib import ssp  # noqa: E402
from DICOMGatedDynamicPETLib import staticneuro as sn  # noqa: E402
from DICOMGatedDynamicPETLib.hardendialog import hardenDialog  # noqa: E402
from DynamicNeuroPET import (DynamicNeuroPETLogic, DynamicNeuroPETWidget, _button, _ijkToRas, _percentiles,  # noqa
                             _badgePixmap, _vtkMatrix, ATLAS_BILATERAL, ATLAS_FILE_ATTRIBUTE, METHOD_ANATOMY,
                             METHOD_PET, METHOD_PET_TEMPLATE, NEURO_PREFIX, NORMALIZATION_GRIDS, NORMALIZED_SUFFIX,
                             REFERENCE_SUFFIX, REG_PREFIX,
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
           (METHOD_ANATOMY, "MR to MR template, same transform to the PET/SPECT (not with a CT)"),
           (METHOD_PET_TEMPLATE, "PET/SPECT to PET template (normal database mean)"))
SETTINGS_DB_ROOT = "DICOMGatedDynamicPET/NeuroDatabase/RootFolder"     # shared with the Neuro PET Database Manager
SETTINGS_IMPORT_FOLDER = "DICOMGatedDynamicPET/StaticNeuro/ImportFolder"
SETTINGS_DISPLAY = "DICOMGatedDynamicPET/StaticNeuro/DisplaySettings"     # JSON of the saved review display settings
AGE_ATTRIBUTE, SEX_ATTRIBUTE = NEURO_PREFIX + "PatientAge", NEURO_PREFIX + "PatientSex"
NAME_ATTRIBUTE, ID_ATTRIBUTE = NEURO_PREFIX + "PatientName", NEURO_PREFIX + "PatientID"
SOURCE_ATTRIBUTE = NEURO_PREFIX + "Source"      # a CT / MR normalized with the image (the volume it was made from)
# Harden of the normalization: the template grid and Overwrite by default
NORM_HARDEN_GRIDS = tuple(sorted(NORMALIZATION_GRIDS, key=lambda option: option[1] != reg.OUTPUT_FIXED_GRID))
NORM_HARDEN_ACTIONS = (("Overwrite", reg.OVERWRITE), ("Save separately", reg.SAVE_SEPARATELY))
STEP_ICON_SIZE = 22                     # px, the height of a step's icon (its pictogram and its state badge)
STEP_GLYPH_COLOR = "#5b6770"
REVIEW_SCORE_WINDOWS = (2.0, 3.0, 4.0, 5.0, 7.0, 10.0)     # the score windows of the review toolbar (+/- SD)
FUSION_TEMPLATES = ("T1", "T2", "PD")   # the MNI templates offered for the fusion of the review
DATABASE_PARAMETER = NEURO_PREFIX + "Database"
NORMALIZATION_ATTRIBUTE = NEURO_PREFIX + "Normalization"    # JSON of the settings the image was normalized with
REGISTERED_SUFFIX = " registered"
ICON_PATH = os.path.join(_MODULE_DIR, "Resources", "Icons", "StaticNeuroPET.png")
WARNING_STYLE = "<span style='color:#c0392b'>{}</span>"
REVIEW_PET_COLORMAP = "Inferno"                  # the top row (the normalized PET / SPECT)
REVIEW_SSP_PATIENT_COLORMAP = "PET-DICOM"       # the 3D-SSP of the patient's projected values
REVIEW_SCORE_COLORMAP = nr.SCORE_COLORMAPS[0]     # diverging: below the normals cold, above hot
SCORE_COLOR_NODE = "Neuro review score colors"
REVIEW_FUSION_OPACITY = 0.6
REVIEW_SCORE_OPACITY = 0.5              # the score map over the MR (fusion): semi-transparent
REVIEW_OUTLINE_THICKNESS = 1            # px, the region outlines on the slices
REVIEW_FLASH_MS = 5000                  # a region clicked in the table: shown alone for 5 s
MONITOR2_PLACE_DELAY_MS = 300           # the Monitor 2 window placed once the layout manager made its views
SSP_TARGET_POINTS = 60000               # vertices of the 3D-SSP surface
SSP_MAX_EDGE_MM = 4.0                   # its triangles' longest edge: the surface is contoured on a coarser grid (not
#                                         decimated: long triangles across the hemispheres' flat midline faces; nor
#                                         split again after it: vtkAdaptiveSubdivisionFilter leaves T-junctions, the
#                                         surface in thousands of pieces whose normals point in or out at random)
SSP_MODEL_NAME = "Neuro 3D-SSP surface"
SSP_MODEL_ATTRIBUTE = NEURO_PREFIX + "SSPModel"     # (view set)/(surface part) of the review's 3D-SSP models
SSP_COLOR_NODE = "Neuro 3D-SSP colors"
SSP_LOW_COLOR_NODE = "Neuro 3D-SSP low score colors"
SSP_CAMERA_DISTANCE = 600.0             # mm from the brain centre (parallel projection)
SSP_SIDE_MARKER_X = (0.17, 0.92)        # the R / L markers of the 3D-SSP views (normalized x; the left one right of
#                                         the color legend), at mid height
SSP_SETUP_DELAY_MS = 500                # the 3D-SSP cameras set again once the new layout's views are drawn


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
# Screens and the Monitor 2 window (as in SlicerPETReviewCompare's dual monitor layouts)
# ---------------------------------------------------------------------------
def _screenCount():
    try:
        return len(list(qt.QGuiApplication.screens()))
    except Exception:
        return 1


def _mainViewportAttributes():
    """XML attributes of the main window viewport, copied from Slicer's own dual monitor layouts, so that the first
    <layout> of <viewports> is shown in the main window."""
    import xml.etree.ElementTree as ElementTree
    try:
        layoutNode = slicer.app.layoutManager().layoutLogic().GetLayoutNode()
        for attributeName in dir(slicer.vtkMRMLLayoutNode):
            if "DualMonitor" not in attributeName:
                continue
            layoutId = getattr(slicer.vtkMRMLLayoutNode, attributeName)
            if not isinstance(layoutId, int) or not layoutNode.IsLayoutDescription(layoutId):
                continue
            root = ElementTree.fromstring(layoutNode.GetLayoutDescription(layoutId))
            if root.tag != "viewports":
                continue
            for element in root.findall("layout"):
                if (element.get("dockable") or "").lower() != "true":
                    name = element.get("name")
                    return f' name="{name}"' if name is not None else ""
    except Exception as error:
        logging.debug(f"Static Neuro: could not read Slicer's dual monitor layouts: {error}")
    return ""


def _secondaryScreen():
    """A screen other than the one showing the main window; None with a single screen."""
    mainWindow = slicer.util.mainWindow()
    screens = list(qt.QGuiApplication.screens())
    if mainWindow is None or len(screens) < 2:
        return None
    center = mainWindow.frameGeometry.center()
    return next((screen for screen in screens if not screen.geometry.contains(center)), None)


def _showOnSecondScreen(window):
    """The floating Monitor 2 window as a normal window (minimize / maximize / close), maximized on the second
    screen; changes only what is not so already (re-applying the flags hides and shows the window again)."""
    wanted = (qt.Qt.Window | qt.Qt.CustomizeWindowHint | qt.Qt.WindowTitleHint | qt.Qt.WindowSystemMenuHint
              | qt.Qt.WindowMinMaxButtonsHint | qt.Qt.WindowCloseButtonHint)
    try:
        hasFlags = (int(window.windowFlags()) & int(wanted)) == int(wanted)
    except (TypeError, ValueError):
        hasFlags = False
    if not hasFlags:
        window.setWindowFlags(wanted)
    screen = _secondaryScreen()
    if screen is None:
        if not window.visible:
            window.show()
        window.raise_()
        return
    onSecond = screen.geometry.contains(window.frameGeometry.center())
    if not onSecond:
        window.setGeometry(screen.availableGeometry)
    if not (onSecond and window.isMaximized() and window.visible):
        window.showMaximized()
    window.raise_()


# ---------------------------------------------------------------------------
# Step icons: a pictogram of the step next to its state badge
# ---------------------------------------------------------------------------
def _stepGlyph(key, size=STEP_ICON_SIZE):
    """A small line drawing of a step (drawn at twice the size): the patient (data), two overlapping images
    (registration), the template grid (normalization), the atlas regions (regions), a bar chart (review)."""
    scale = 2
    full = size * scale
    pixmap = qt.QPixmap(full, full)
    pixmap.fill(qt.QColor(0, 0, 0, 0))
    painter = qt.QPainter(pixmap)
    try:
        painter.setRenderHint(qt.QPainter.Antialiasing)
        color = qt.QColor(STEP_GLYPH_COLOR)
        pen = qt.QPen(color)
        pen.setWidth(2 * scale)
        painter.setPen(pen)
        painter.setBrush(qt.QBrush(qt.QColor(0, 0, 0, 0)))
        m = 3 * scale                       # margin
        w = full - 2 * m
        if key == STEP_DATA:                # head and shoulders
            head = int(w * 0.42)
            painter.drawEllipse(m + (w - head) // 2, m, head, head)
            painter.drawArc(m + w // 8, m + int(w * 0.52), w - w // 4, int(w * 0.9), 0, 180 * 16)
        elif key == STEP_REG:               # two overlapping images
            side = int(w * 0.68)
            painter.drawRect(m, m, side, side)
            painter.setBrush(qt.QBrush(qt.QColor(color.red(), color.green(), color.blue(), 90)))
            painter.drawRect(m + w - side, m + w - side, side, side)
        elif key == STEP_NORM:              # the template grid
            painter.drawRoundedRect(m, m, w, w, 2 * scale, 2 * scale)
            thin = qt.QPen(color)
            thin.setWidth(scale)
            painter.setPen(thin)
            for i in (1, 2):
                painter.drawLine(m + i * w // 3, m, m + i * w // 3, m + w)
                painter.drawLine(m, m + i * w // 3, m + w, m + i * w // 3)
        elif key == STEP_REGIONS:           # a brain divided into regions
            painter.drawEllipse(m, m + w // 8, w, int(w * 0.75))
            painter.drawLine(m + w // 2, m + w // 8, m + w // 2, m + w // 8 + int(w * 0.75))
            painter.drawLine(m + w // 2, m + w // 2, m + w, m + w // 2)
            painter.drawLine(m + w // 4, m + w // 6, m + w // 4, m + int(w * 0.8))
        else:                               # review: a bar chart
            painter.setPen(qt.Qt.NoPen)
            painter.setBrush(qt.QBrush(color))
            bar = w // 5
            for i, height in enumerate((0.45, 0.85, 0.65)):
                h = int(w * height)
                painter.drawRect(m + i * (bar + bar // 2) + bar // 4, m + w - h, bar, h)
    finally:
        painter.end()
    return pixmap


def _stepIcon(key, state, number, size=STEP_ICON_SIZE):
    """The pictogram of a step and its state badge side by side (twice as wide as high)."""
    scale = 2
    full = size * scale
    pixmap = qt.QPixmap(2 * full, full)
    pixmap.fill(qt.QColor(0, 0, 0, 0))
    painter = qt.QPainter(pixmap)
    try:
        painter.drawPixmap(0, 0, _stepGlyph(key, size))
        painter.drawPixmap(full, 0, _badgePixmap(state, number, size))
    finally:
        painter.end()
    return qt.QIcon(pixmap)


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

    @staticmethod
    def _subjectHierarchyPatient(node):
        """(name, ID) of the patient item above the volume in the subject hierarchy (a DICOM load); ('', '')."""
        try:
            shNode = slicer.vtkMRMLSubjectHierarchyNode.GetSubjectHierarchyNode(slicer.mrmlScene)
            item = shNode.GetItemByDataNode(node)
            root = shNode.GetSceneItemID()
            constants = slicer.vtkMRMLSubjectHierarchyConstants
            nameKey = constants.GetDICOMPatientNameAttributeName()
            idKey = constants.GetDICOMPatientIDAttributeName()
            while item and item != root:
                name, patientId = shNode.GetItemAttribute(item, nameKey), shNode.GetItemAttribute(item, idKey)
                if name or patientId:
                    return sn.formatPatientName(name), (patientId or "").strip()
                item = shNode.GetItemParent(item)
        except Exception:
            logging.debug("Static Neuro: no patient in the subject hierarchy", exc_info=True)
        return "", ""

    @classmethod
    def patientNameAndId(cls, node):
        """(name, ID) of the patient: the values entered on the Data page, else the DICOM tags (the file, else the
        subject hierarchy), else the BIDS subject of an imported file (ID)."""
        if node is None:
            return "", ""
        name, patientId = node.GetAttribute(NAME_ATTRIBUTE), node.GetAttribute(ID_ATTRIBUTE)
        if name is None or patientId is None:
            values = cls.dicomValues(node, ("0010,0010", "0010,0020"))
            taggedName, taggedId = sn.formatPatientName(values["0010,0010"]), values["0010,0020"].strip()
            if not taggedName or not taggedId:
                hierarchyName, hierarchyId = cls._subjectHierarchyPatient(node)
                taggedName, taggedId = taggedName or hierarchyName, taggedId or hierarchyId
            taggedId = taggedId or sn.taggedSubjectId(cls.attributes(node))
            name = taggedName if name is None else name
            patientId = taggedId if patientId is None else patientId
        return name, patientId

    @staticmethod
    def setPatientNameAndId(node, name, patientId):
        if node is None:
            return
        node.SetAttribute(NAME_ATTRIBUTE, (name or "").strip())
        node.SetAttribute(ID_ATTRIBUTE, (patientId or "").strip())

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
        for node in undo.get("extras") or ():          # the CT / MR sequences normalized with the image
            if node is not None and slicer.mrmlScene.IsNodePresent(node):
                slicer.mrmlScene.RemoveNode(node)
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
               description="", method=METHOD_PET, anatomy=None, extras=None):
        """The image resampled through the normalization onto the template's field of view. The patient's CT/MR is
        normalized too with the same transform (on the template grid) whatever the method: it lies on the image
        (registered in step 2, or a PET/CT, PET/MR); so are extras, other sequences lying on the CT/MR (undo
        ["extras"]). Returns (undo, volume, None, normalized CT/MR or None)."""
        attributes = {NEURO_PREFIX + "Template": template.GetName(), NEURO_PREFIX + "Method": method,
                      NEURO_PREFIX + "Transform": description}
        undo, node = self.hardenVolume(volume, template, linearNode, action, grid, spacing, interpolation,
                                       attributes, NORMALIZED_SUFFIX)
        normalized = None
        if anatomy is not None:
            try:
                normalized = self.normalizedAnatomy(anatomy, template, linearNode)
            except Exception:
                logging.exception("Static Neuro: could not normalize the CT/MR")
        undo["extras"] = []
        for extra in extras or ():
            if extra is None or extra is volume or extra is anatomy:
                continue
            try:
                undo["extras"].append(self.normalizedAnatomy(extra, template, linearNode))
            except Exception:
                logging.exception(f"Static Neuro: could not normalize '{extra.GetName()}'")
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

    # --- the regions of a normal database and their values ----------------------------------------
    @staticmethod
    def databaseRegionArguments(folder, header):
        """Keyword arguments of loadAtlasRegions() for the region settings of a database (as the Regions step uses
        them when locked)."""
        settings = sn.regionSettings(folder, header)
        arguments = {"targetKind": settings["regionAtlas"], "referenceKind": ATLAS_BILATERAL, "referenceName": None,
                     "growMm": settings["growMm"], "erodeReferenceMm": settings["erodeReferenceMm"],
                     "growInto": settings["growInto"]}
        if settings["referenceMaskPath"]:
            arguments["referenceMaskPath"] = settings["referenceMaskPath"]
            arguments["referenceName"] = settings["referenceTitle"] or "Reference"
        elif settings["referenceAtlasRegion"]:
            arguments["referenceName"] = settings["referenceAtlasRegion"]
        return arguments

    def segmentMasks(self, segmentation, gridNode, dropped=True):
        """{segment name: boolean array on the grid of gridNode} of a segmentation (the optic chiasm left out with
        dropped)."""
        masks = {}
        shape = tuple(reversed(gridNode.GetImageData().GetDimensions()))
        for segmentId in self.segmentIds(segmentation):
            name = segmentation.GetSegmentation().GetSegment(segmentId).GetName()
            if dropped and nr.isDroppedSegment(name):
                continue
            array = slicer.util.arrayFromSegmentBinaryLabelmap(segmentation, segmentId, gridNode)
            masks[name] = np.asarray(array) > 0 if array is not None else np.zeros(shape, dtype=bool)
        return masks

    def databaseRegionMasks(self, atlas, folder, header, gridNode):
        """The regions of a database's region settings (CerebrA of the atlas folder) as masks on gridNode's grid."""
        segmentation, _ = self.loadAtlasRegions(atlas, **self.databaseRegionArguments(folder, header))
        try:
            return self.segmentMasks(segmentation, gridNode)
        finally:
            slicer.mrmlScene.RemoveNode(segmentation)

    @staticmethod
    def subjectRegionValues(folder, header, masks, progress=None):
        """{subject ID: {region: mean}} of the subjects' maps (SUVR or SUV by the map type; not smoothed) of a user
        defined database. progress(index, total, text) returns False to stop (None returned)."""
        subjects = normaldb.readSubjects(folder)
        mapKey = normaldb.subjectMapKey(header)
        values = {}
        for index, subject in enumerate(subjects):
            if progress is not None and not progress(index, len(subjects), f"{subject['id']} ({index + 1} / "
                                                                           f"{len(subjects)})"):
                return None
            relative = subject.get(mapKey)
            path = os.path.join(folder, relative) if relative else ""
            if not path or not os.path.isfile(path):
                continue
            node = slicer.util.loadVolume(path, {"show": False})
            try:
                values[subject["id"]] = nr.regionMeans(slicer.util.arrayFromVolume(node), masks)
            finally:
                slicer.mrmlScene.RemoveNode(node)
        return values

    def regionReference(self, folder, header, masks, groupKey, locked, progress=None):
        """({region: [mean, SD, n]}, source text) of the normal regional values of a group: the database's own
        (stats/regions.json, made with the mean / SD maps) when the regions are the database's (locked); else, for a
        user defined database, from its subjects on the fly (cached in <database>/cache/); else the approximation from
        the mean / SD maps. None when stopped."""
        names = list(masks)
        version = header.get("version", 1)
        mapKey = normaldb.subjectMapKey(header)
        path = nr.regionsPath(folder, header)
        if locked and path:
            document = nr.readJson(path)
            if document and document.get("maps") == mapKey:
                found, missing = nr.matchRegions(document.get("groups"), groupKey, names)
                if found and len(missing) <= len(names) // 10:
                    return found, "database regional statistics"
        if header.get("kind") == normaldb.USER_DEFINED and normaldb.readSubjects(folder):
            ids = sorted(s.get("id", "") for s in normaldb.readSubjects(folder))
            signature = nr.masksSignature(masks, extra=ids)
            cache = nr.cachePath(folder, signature, mapKey, version)
            document = nr.readJson(cache)
            if document is None:
                values = self.subjectRegionValues(folder, header, masks, progress)
                if values is None:
                    return None
                groups = normaldb.statisticsGroups(header, normaldb.readSubjects(folder))
                document = nr.regionsDocument(nr.regionStatistics(values, groups), mapKey, version,
                                              signature=signature)
                try:
                    nr.writeJson(cache, document)
                except OSError:
                    logging.warning(f"Static Neuro: could not write {cache}", exc_info=True)
            found, _ = nr.matchRegions(document.get("groups"), groupKey, names)
            if found:
                return found, "the database's subjects (these regions)"
        return None, "approximate (from the mean / SD maps)"

    # --- review: the patient compared with the database ------------------------------------------
    @staticmethod
    def resampledArray(volume, gridNode):
        """The volume's voxels resampled (linear) onto the grid of gridNode, as a float32 array (k, j, i)."""
        dims = tuple(int(x) for x in gridNode.GetImageData().GetDimensions())
        image = resampling.Resampler(_ijkToRas(volume), dims, _ijkToRas(gridNode), None, True,
                                     "Linear").resample(volume.GetImageData())
        node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "Neuro review (temporary)")
        try:
            node.SetIJKToRASMatrix(_vtkMatrix(_ijkToRas(gridNode)))
            node.SetAndObserveImageData(image)
            return np.array(slicer.util.arrayFromVolume(node), dtype=np.float32)
        finally:
            slicer.mrmlScene.RemoveNode(node)

    @staticmethod
    def dropSegments(segmentation):
        """Remove the segments the review leaves out (the optic chiasm). Returns their names."""
        removed = []
        seg = segmentation.GetSegmentation()
        for segmentId in DynamicNeuroPETLogic.segmentIds(segmentation):
            name = seg.GetSegment(segmentId).GetName()
            if nr.isDroppedSegment(name):
                seg.RemoveSegment(segmentId)
                removed.append(name)
        return removed

    def reviewData(self, volume, segmentation, folder=None, header=None, groupKey=normaldb.MASTER, atlas=None,
                   referenceSegmentId=None, locked=True, progress=None):
        """The patient (a normalized volume and its regions) compared with a normal database:
        {"grid": the grid node (the group's mean map; without a database the volume), "z": voxel z map (negative
        below the mean; t = z / tFactor(n)) or None, "n", "names", "suv" {region: SUV mean}, "values" {region:
        value of the database's map type}, "reference" {region: [mean, SD, n]}, "source", "ratio" (map type SUVR /
        Ratio), "referenceMean", "fwhm", "notes"}. None when stopped."""
        notes = []
        result = {"z": None, "n": None, "reference": {}, "source": "", "ratio": False, "referenceMean": None,
                  "fwhm": 0.0, "notes": notes}
        groups = ((header or {}).get("statistics") or {}).get("groups") or {}
        group = groups.get(groupKey) if header is not None else None
        if header is not None and not group:
            raise ValueError(f"The database '{header.get('name')}' has no mean / SD maps"
                             + (f" of the group '{groupKey}'." if groups else ": compute them in the Neuro PET "
                                "Database Manager."))
        if header is None:
            grid = volume
            values = suv = np.array(slicer.util.arrayFromVolume(volume), dtype=np.float32)
            ratio = None
            masks = self.segmentMasks(segmentation, grid)
            result.update(grid=grid, names=list(masks), suv=nr.regionMeans(suv, masks))
            result["labels"] = nr.labelMap(masks, values.shape)       # the region names at the mouse
            if referenceSegmentId:
                mask = slicer.util.arrayFromSegmentBinaryLabelmap(segmentation, referenceSegmentId, grid)
                if mask is not None and np.any(mask):
                    ratio, mean = sn.ratioMap(values, np.asarray(mask) > 0)
                    result.update(ratio=True, referenceMean=mean, values=nr.regionMeans(ratio, masks))
                else:
                    ratio = None
            result.setdefault("values", result["suv"])
            result.update(smoothed=ratio if result["ratio"] else values, union=nr.unionMask(masks, values.shape),
                          hemispheres=ssp.hemisphereMasks(masks, values.shape))
            return result
        grid = self.loadAtlasVolume(os.path.join(folder, group["mean"]), name=f"{header['name']} mean ({groupKey})")
        sdNode = self.loadAtlasVolume(os.path.join(folder, group["sd"]), name=f"{header['name']} SD ({groupKey})")
        mean = np.array(slicer.util.arrayFromVolume(grid), dtype=np.float32)
        sd = np.array(slicer.util.arrayFromVolume(sdNode), dtype=np.float32)
        if sd.shape != mean.shape:
            raise ValueError("The mean and SD maps of the database are not on the same grid.")
        suv = self.resampledArray(volume, grid)
        values = suv
        ratio = header.get("mapType") in (normaldb.MAP_SUVR, normaldb.MAP_RATIO)
        if ratio:
            maskPath = sn.referenceMaskPath(folder, header)
            if maskPath:
                maskArray, maskNode = self._loadLabels(maskPath)
                slicer.mrmlScene.RemoveNode(maskNode)
                if maskArray.shape != mean.shape:
                    raise ValueError("The database's reference mask is not on the grid of its mean map.")
                referenceMask = maskArray > 0
            elif referenceSegmentId:
                referenceMask = np.asarray(slicer.util.arrayFromSegmentBinaryLabelmap(
                    segmentation, referenceSegmentId, grid)) > 0
                notes.append("The database has no reference mask: the regions' reference region is used.")
            else:
                raise ValueError("The database compares ratio maps but has no reference mask, and the regions have "
                                 "no reference region.")
            values, referenceMean = sn.ratioMap(suv, referenceMask)
            result["referenceMean"] = referenceMean
        masks = self.segmentMasks(segmentation, grid)
        result["labels"] = nr.labelMap(masks, mean.shape)             # the region names at the mouse
        result.update(grid=grid, names=list(masks), suv=nr.regionMeans(suv, masks),
                      values=nr.regionMeans(values, masks), ratio=ratio, n=group.get("n"))
        # voxel scores: the patient smoothed like the normals, in the mask, with the SD floor
        fwhm = float((header.get("statistics") or {}).get("smoothingFwhmMm", header.get("smoothingFwhmMm")) or 0)
        result["fwhm"] = fwhm
        spacing = tuple(float(x) for x in grid.GetSpacing())[::-1]
        brain = None
        if atlas and neuro.BRAIN_MASK in atlas:
            brainArray, brainNode = self._loadLabels(atlas[neuro.BRAIN_MASK])
            slicer.mrmlScene.RemoveNode(brainNode)
            if brainArray.shape == mean.shape:
                brain = brainArray > 0
            else:
                notes.append("The MNI brain mask is not on the database grid: the score mask is the mean map's.")
        mask = nr.scoreMask(mean, brain)
        smoothed = normaldb.smoothMap(values, fwhm, spacing)
        result.update(smoothed=smoothed, union=nr.unionMask(masks, mean.shape),
                      hemispheres=ssp.hemisphereMasks(masks, mean.shape))
        result["z"] = nr.scoreMap(smoothed, mean, sd, mask, nr.SCORE_Z)
        result["brightness"] = nr.petBrightness(values)     # the patient's map (not smoothed) as brightness
        result["gm"] = None         # the MNI grey matter probability on the grid: the score slices' default mask
        if atlas and "GM" in atlas:
            try:
                result["gm"] = self.resampledArray(self.loadAtlasVolume(atlas["GM"], name="MNI ICBM152 2009c GM"), grid)
            except Exception as error:
                notes.append(f"No grey matter map ({error}): the score slices show the high normal uptake.")
        result.update(mask=mask, mean=mean)     # mean: the normal uptake (the score map shown where it is high)
        # regional normal values
        reference = self.regionReference(folder, header, masks, groupKey, locked, progress)
        if reference is None:
            return None
        found, source = reference
        if found is None:
            found = nr.approximateRegions(mean, sd, masks)
            notes.append("Regional normal values approximated from the mean / SD maps (the mean voxel SD is larger "
                         "than the SD of a regional mean: these scores are conservative).")
        result.update(reference=found, source=source)
        return result

    # --- 3D-SSP ------------------------------------------------------------------------------------
    @staticmethod
    def brainSurface(mask, gridNode, targetPoints=SSP_TARGET_POINTS):
        """The surface of a brain mask (numpy (k, j, i) on gridNode's grid; holes filled, smoothed 1 voxel, padded
        so that it is closed, the largest piece): contoured on a grid coarser by a step that gives about targetPoints
        vertices (edges at most SSP_MAX_EDGE_MM; no decimation): one closed surface, its normals consistent:
        {"points" (RAS mm), "normals" (outward, unit), "triangles"}."""
        from scipy import ndimage
        from vtk.util import numpy_support
        mask = ndimage.binary_fill_holes(np.asarray(mask, dtype=bool))
        if not mask.any():
            raise ValueError("The regions are empty: no brain surface.")
        smooth = np.pad(ndimage.gaussian_filter(mask.astype(np.float32), 1.0), 1)      # closed at the grid's edges
        ijkToRas = _ijkToRas(gridNode)
        spacing = float(np.min(np.linalg.norm(ijkToRas[:3, :3], axis=0))) or 1.0

        def contour(field):
            image = vtk.vtkImageData()
            image.SetDimensions(*field.shape[::-1])
            image.GetPointData().SetScalars(numpy_support.numpy_to_vtk(np.ascontiguousarray(field).ravel(), deep=True,
                                                                       array_type=vtk.VTK_FLOAT))
            surface = vtk.vtkFlyingEdges3D() if hasattr(vtk, "vtkFlyingEdges3D") else vtk.vtkMarchingCubes()
            surface.SetInputData(image)
            surface.SetValue(0, 0.5)
            largest = vtk.vtkPolyDataConnectivityFilter()
            largest.SetInputConnection(surface.GetOutputPort())
            largest.SetExtractionModeToLargestRegion()
            cleaned = vtk.vtkCleanPolyData()        # the points of the other pieces dropped
            cleaned.SetInputConnection(largest.GetOutputPort())
            cleaned.Update()
            return cleaned.GetOutput()

        poly = contour(smooth)
        step = 1.0
        count = poly.GetNumberOfPoints()
        if count > targetPoints:
            # vertices ~ area / step^2; a triangle's edge at most the cell's diagonal (sqrt(3) step)
            step = min(float(np.sqrt(count / float(targetPoints))), SSP_MAX_EDGE_MM / (np.sqrt(3.0) * spacing))
            if step > 1.05:
                grid = np.meshgrid(*[np.arange(0.0, n - 1 + 1e-6, step) for n in smooth.shape], indexing="ij")
                poly = contour(ndimage.map_coordinates(smooth, grid, order=1).astype(np.float32))
            else:
                step = 1.0
        # the contour's (i, j, k): the padded grid's, step apart
        matrix = ijkToRas @ np.array([[step, 0, 0, -1], [0, step, 0, -1], [0, 0, step, -1], [0, 0, 0, 1]], float)
        transform = vtk.vtkTransform()
        transform.SetMatrix(_vtkMatrix(matrix))
        moved = vtk.vtkTransformPolyDataFilter()
        moved.SetTransform(transform)
        moved.SetInputData(poly)
        normals = vtk.vtkPolyDataNormals()
        normals.SetInputConnection(moved.GetOutputPort())
        normals.ComputePointNormalsOn()
        normals.ComputeCellNormalsOff()
        normals.SplittingOff()
        normals.ConsistencyOn()
        normals.AutoOrientNormalsOn()       # outward (the surface closed and in one piece)
        normals.Update()
        poly = normals.GetOutput()
        points = numpy_support.vtk_to_numpy(poly.GetPoints().GetData()).astype(np.float64)
        pointNormals = numpy_support.vtk_to_numpy(poly.GetPointData().GetNormals()).astype(np.float64)
        polys = poly.GetPolys()
        if hasattr(polys, "GetConnectivityArray"):
            cells = numpy_support.vtk_to_numpy(polys.GetConnectivityArray()).reshape(-1, 3)
        else:
            cells = numpy_support.vtk_to_numpy(polys.GetData()).reshape(-1, 4)[:, 1:]
        out = {"points": points, "normals": pointNormals, "triangles": cells.astype(np.int64)}
        openEdges = ssp.openEdges(out["triangles"])
        if openEdges:
            logging.warning(f"Static Neuro: the 3D-SSP surface has {openEdges} open edges")
        return out

    @classmethod
    def reviewSurface(cls, mask, gridNode, targetPoints=SSP_TARGET_POINTS, hemispheres=None):
        """The 3D-SSP surface of a brain mask: the whole brain's and, for the medial views, each hemisphere's in one
        surface (ssp.combine). hemispheres: (left, right) masks of the regions' own sides (ssp.hemisphereMasks), each
        hemisphere's surface made from its own mask; None (bilateral regions): the brain's only, no medial views."""
        brain = cls.brainSurface(mask, gridNode, targetPoints)
        if hemispheres is None:
            return brain
        mask = np.asarray(mask, dtype=bool)
        halves = []
        for half in hemispheres:
            try:
                halves.append(cls.brainSurface(mask & np.asarray(half, dtype=bool), gridNode, targetPoints // 2))
            except ValueError:          # nothing on that side
                halves.append(None)
        return ssp.combine(brain, *halves)

    @staticmethod
    def surfacePolyData(surface, values=None, name="SSP"):
        """vtkPolyData of a surface dict, with the vertex values as its active scalars."""
        from vtk.util import numpy_support
        poly = vtk.vtkPolyData()
        points = vtk.vtkPoints()
        points.SetData(numpy_support.numpy_to_vtk(np.asarray(surface["points"], np.float32), deep=True))
        poly.SetPoints(points)
        triangles = np.asarray(surface["triangles"], dtype=np.int64)
        cells = np.hstack([np.full((len(triangles), 1), 3, dtype=np.int64), triangles]).ravel()
        polys = vtk.vtkCellArray()
        polys.SetCells(len(triangles), numpy_support.numpy_to_vtkIdTypeArray(cells, deep=True))
        poly.SetPolys(polys)
        normals = numpy_support.numpy_to_vtk(np.asarray(surface["normals"], np.float32), deep=True)
        normals.SetName("Normals")
        poly.GetPointData().SetNormals(normals)
        if values is not None:
            array = numpy_support.numpy_to_vtk(np.asarray(values, np.float32), deep=True)
            array.SetName(name)
            poly.GetPointData().AddArray(array)
            poly.GetPointData().SetActiveScalars(name)
        return poly

    def sspFromSubjects(self, folder, header, surface, fwhm, progress=None, mask=None):
        """{group key: {"mean", "sd", "n", "name"}} of the normals' projections on a surface: each subject's map
        (SUVR or SUV by the map type) smoothed with fwhm and projected (each ray stopped where it leaves mask, the
        union of the regions on the database's grid); cached in <database>/cache/ (key: the surface,
        the projection, the FWHM, the map type, the database version and the subjects). None when stopped."""
        subjects = normaldb.readSubjects(folder)
        mapKey = normaldb.subjectMapKey(header)
        key = ssp.signature(surface["points"], surface["normals"], ssp.DEPTH_MM, ssp.STEP_MM,
                            extra=(fwhm, mapKey, header.get("version", 1), sorted(s.get("id", "") for s in subjects),
                                   "clip-brain"))
        cache = ssp.cachePath(folder, key)
        cached = ssp.read(cache)
        if cached is not None:
            return cached[1]
        groups = {k: g for k, g in normaldb.statisticsGroups(header, subjects).items() if len(g["ids"]) >= 2}
        accumulators = {k: normaldb.RunningStatistics() for k in groups}
        for index, subject in enumerate(subjects):
            if progress is not None and not progress(index, len(subjects), f"surface projection of {subject['id']} "
                                                                           f"({index + 1} / {len(subjects)})"):
                return None
            relative = subject.get(mapKey)
            path = os.path.join(folder, relative) if relative else ""
            if not path or not os.path.isfile(path):
                continue
            node = slicer.util.loadVolume(path, {"show": False})
            try:
                smoothed = normaldb.smoothMap(slicer.util.arrayFromVolume(node), fwhm,
                                              tuple(float(x) for x in node.GetSpacing())[::-1])
                clip = mask if mask is not None and np.shape(mask) == np.shape(smoothed) else None
                if mask is not None and clip is None:
                    logging.warning(f"Static Neuro: {subject['id']} is not on the regions' grid: 3D-SSP not clipped")
                projected = ssp.project(smoothed, _ijkToRas(node), surface["points"], surface["normals"], mask=clip)
            finally:
                slicer.mrmlScene.RemoveNode(node)
            for groupKey, group in groups.items():
                if subject["id"] in group["ids"]:
                    accumulators[groupKey].add(projected)
        out = {k: {"mean": a.mean(), "sd": a.sd(), "n": a.n, "name": groups[k]["name"]}
               for k, a in accumulators.items() if a.n >= 2}
        try:
            ssp.write(cache, surface, out, {"depthMm": ssp.DEPTH_MM, "stepMm": ssp.STEP_MM, "clipToBrain": True,
                                            "fwhmMm": fwhm,
                                            "maps": mapKey, "version": header.get("version", 1)})
        except OSError:
            logging.warning(f"Static Neuro: could not write {cache}", exc_info=True)
        return out

    def reviewSsp(self, result, folder=None, header=None, groupKey=normaldb.MASTER, locked=True, progress=None):
        """3D-SSP of a reviewData() result: {"surface", "patient" (the patient's projection), "mean", "sd", "n" (the
        normals' projections of the group, Minoshima; None when not available), "voxel" (the voxel z map projected,
        the most extreme along the normal; None without a database), "source", "notes"}. The surface: the database's
        (stats/ssp.npz) when the regions are the database's, else made from the union of the regions. None when
        stopped."""
        grid = result["grid"]
        notes = []
        out = {"mean": None, "sd": None, "n": None, "voxel": None, "source": "", "notes": notes}
        surface = groups = None
        mapKey = normaldb.subjectMapKey(header) if header is not None else ""
        path = ssp.sspPath(folder, header) if header is not None else None
        outdated = noMedial = False
        medial = result.get("hemispheres") is not None      # the segmentation's regions with sides (R / L)
        if path and locked:
            read = ssp.read(path)
            if read is not None and read[2].get("maps") == mapKey and not (
                    abs(float(read[2].get("depthMm") or 0) - ssp.DEPTH_MM) < 1e-6 and read[2].get("clipToBrain")
                    and int(read[2].get("surface") or 0) >= ssp.SURFACE_VERSION):
                outdated, read = True, None         # an older projection (13.5 mm, not stopped at the brain's edge,
                #                                     an older surface)
            if read is not None and read[2].get("maps") == mapKey and medial and not ssp.hasMedial(read[0]):
                noMedial, read = True, None         # the database's regions bilateral, the segmentation's R / L
            if read is not None and read[2].get("maps") == mapKey:
                surface, groups, meta = read
                if not medial and ssp.hasPart(surface, ssp.PART_LEFT):      # bilateral regions: no medial views
                    surface, index = ssp.brainPart(surface)
                    groups = {k: dict(g, mean=np.asarray(g["mean"])[index], sd=np.asarray(g["sd"])[index])
                              for k, g in groups.items()}
                out["source"] = "the database's surface projections"
                if abs(float(meta.get("fwhmMm") or 0) - result["fwhm"]) > 1e-6:
                    notes.append("The database's surface projections were made with another smoothing.")
        if surface is None:
            surface = self.reviewSurface(result["union"], grid, hemispheres=result.get("hemispheres"))
        out["surface"] = surface
        ijkToRas = _ijkToRas(grid)
        brain = result["union"]                     # each ray stops where it leaves the brain
        out["patient"] = ssp.project(result["smoothed"], ijkToRas, surface["points"], surface["normals"], mask=brain)
        if header is None:
            return out
        if groups is None and header.get("kind") == normaldb.USER_DEFINED and normaldb.readSubjects(folder):
            groups = self.sspFromSubjects(folder, header, surface, result["fwhm"], progress, mask=brain)
            if groups is None:
                return None
            out["source"] = "the database's subjects projected on these regions"
            if outdated:
                notes.append("The database's 3D-SSP projections were made by an older version (another depth or "
                             "surface): projected from its subjects now; compute the mean / SD again in the Neuro "
                             "PET Database Manager to store them.")
            elif noMedial:
                notes.append("The database's 3D-SSP projections have no medial surfaces (its regions are "
                             "bilateral): its subjects projected on the hemispheres of these R / L regions now "
                             "(cached).")
        group = (groups or {}).get(groupKey)
        if group is not None:
            out.update(mean=group["mean"], sd=group["sd"], n=group.get("n") or result.get("n"))
        elif noMedial:
            notes.append("The database's 3D-SSP projections have no medial surfaces (its regions are bilateral) and "
                         "it has no subjects: the 3D-SSP shows the voxel score map projected.")
        elif outdated:
            notes.append("The database's 3D-SSP projections are from an older version (another depth, not stopped at "
                         "the brain's edge, or no medial surfaces): the 3D-SSP shows the voxel score map projected.")
        else:
            notes.append("No surface projections of the normals for this database: the 3D-SSP shows the voxel score "
                         "map projected.")
        if result.get("z") is not None:
            # the voxel scores of the grey matter (else of the high normal uptake): not the CSF's or the white matter's
            if result.get("gm") is not None:
                shown = nr.greyMatterMask(result["gm"], result["mask"])
            else:
                shown = nr.uptakeMask(result["mean"], result["mask"])
            out["voxel"] = ssp.project(np.where(shown, result["z"], 0.0), ijkToRas, surface["points"],
                                       surface["normals"], mode="extreme", mask=brain)
        return out

    # --- batch: many normal subjects into a database ----------------------------------------------
    BATCH_COLUMNS = ("file", "id", "status", "message", "referenceMean", "seconds")

    def batchPlan(self, header, folder=None):
        """(plan, registration settings) of a batch into the database: its normalization settings
        (staticneuro.batchNormalization; PET to PET when the database in folder has its template) and the registration
        settings they give."""
        hasTemplate = bool(folder) and bool(sn.petTemplatePath(folder, header))
        plan = sn.batchNormalization(header.get("normalization"), neuro.DEFAULT_TEMPLATE, NORMALIZATION_ENGINE,
                                     {reg.ENGINE_ITERATIVE: (NORMALIZATION_ITERATIVE_LINEAR,
                                                             NORMALIZATION_ITERATIVE_DEFORMABLE),
                                      reg.ENGINE_BRAINSFIT: (NORMALIZATION_STEPS[0], "BSpline" in NORMALIZATION_STEPS)},
                                     hasTemplate)
        settings = self.defaultSettings(plan["method"])
        if plan["engine"] == reg.ENGINE_ITERATIVE:
            settings.update(engine=reg.ENGINE_ITERATIVE, iterativeLinear=plan["linear"],
                            iterativeDeformable=plan["deformable"],
                            steps=reg.iterativeSteps(plan["linear"], plan["deformable"]))
        else:
            steps = reg.validateSteps([plan["linear"]] + (["BSpline"] if plan["deformable"] else []))
            settings.update(engine=reg.ENGINE_BRAINSFIT, steps=steps,
                            resample=[NORMALIZATION_WORKING_MM > 0] * len(steps))
        return plan, settings

    def batchAdd(self, folder, files, atlas, qc=True, progress=None, replace=False):
        """Add BIDS PET files (staticneuro.findPetFiles) of normal subjects to a user defined database, one after the
        other: each loaded, normalized to the MNI template (or the database's own template, PET to PET) with the
        database's normalization settings (no manual check), resampled onto the template grid and added
        (logic.addSubject, the database's reference mask). IDs already in the database are skipped (replace: normalized
        again and replaced); a subject that fails is logged and the batch goes on. qc: a PNG of the
        normalized PET over the MNI T1 (3 slices) in <database>/qc/. progress(index, total, text) returns False to
        stop. The log is <database>/qc/batch_<time>.csv. Returns (rows, log path, stopped)."""
        header = normaldb.readHeader(folder)
        if header.get("kind") != normaldb.USER_DEFINED:
            raise ValueError("Subjects can be added to user defined databases only.")
        gridPath = sn.referenceMaskPath(folder, header)
        if not gridPath:
            raise ValueError(f"'{header['name']}' has no reference mask (reference_mask in its folder).")
        plan, settings = self.batchPlan(header, folder)
        if plan["method"] == METHOD_PET_TEMPLATE:
            template = self.loadAtlasVolume(sn.petTemplatePath(folder, header), name=f"{header['name']} template")
        else:
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
            if normaldb.hasSubject(folder, subjectId) and not replace:
                row.update(status="skipped", message="already in the database")
            else:
                def stepProgress(step, steps, text, index=index, subjectId=subjectId):
                    return progress is None or progress(index, total, f"{subjectId} ({index + 1} / {total}): {text}")
                try:
                    added = self._batchSubject(folder, header, path, subjectId, template, mask, settings, plan,
                                               gridPath, anatomy, qcFolder, stepProgress, replace)
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
                      qcFolder, progress, replace=False):
        """One subject of batchAdd(): the row added (with "notes"), None when the registration was stopped."""
        from DynamicPETRegistration import DynamicPETRegistrationLogic
        scene = slicer.mrmlScene
        nodes = []
        try:
            volume = self.loadFromFile(path)
            nodes.append(volume)
            notes = sn.unitWarnings(self.imageInfo(volume), header)
            method = plan["method"]
            result = self.normalize(volume, template, method, None, settings, mask, progress)
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
                                              None, "Linear", description, method)
            nodes.append(normalized)
            normalized.SetAttribute(NORMALIZATION_ATTRIBUTE, json.dumps(plan))
            age, sex = self.ageAndSex(normalized)
            if age is None:
                notes.append("age not known")
            row = self.addSubject(folder, header, normalized, subjectId, gridPath, age, sex, plan, replace)
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
    NORM_HARDEN_GRIDS = NORM_HARDEN_GRIDS
    NORM_HARDEN_ACTIONS = NORM_HARDEN_ACTIONS

    def setup(self):
        self._patientTexts = {}         # volume ID: the patient's name (ID) shown on the toolbar
        self._contrastFor = None        # (CT/MR ID, (contrast, reason)): read once per CT/MR
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

    # --- workflow toolbar: the patient right of the module's name, the review tools ----------------------------
    def _stepButton(self, index, key, title, explanation):
        button = DynamicNeuroPETWidget._stepButton(self, index, key, title, explanation)
        button.setIconSize(qt.QSize(2 * STEP_ICON_SIZE, STEP_ICON_SIZE))
        return button

    def _stepIcon(self, key, state, number):
        return _stepIcon(key, state, number)

    def _buildToolbar(self):
        toolbar = DynamicNeuroPETWidget._buildToolbar(self)
        if toolbar is None:
            return None
        toolbar.setIconSize(qt.QSize(2 * STEP_ICON_SIZE, STEP_ICON_SIZE))
        actions = list(toolbar.actions())
        self.patientLabel = qt.QLabel()
        self.patientLabel.setStyleSheet("QLabel { color: #2f7ed8; font-weight: bold; padding: 0 6px; }")
        if len(actions) > 1:            # right of the module's name, before the steps
            self._patientAction = toolbar.insertWidget(actions[1], self.patientLabel)
        else:
            self._patientAction = toolbar.addWidget(self.patientLabel)
        self._patientAction.setVisible(False)
        self._reviewToolActions = [toolbar.addSeparator()]
        for text, tip, fill in (
                ("Layout", "Review layout (single or dual monitor, the 3D-SSP views)", self._fillLayoutMenu),
                ("Windows", "Score windows (slices and 3D-SSP) and the PET, 3D-SSP and fusion windows",
                 self._fillWindowMenu),
                ("Colors", "Color maps of the PET, the scores and the 3D-SSP", self._fillColorMenu)):
            button = qt.QToolButton()
            button.setText(text)
            button.toolTip = tip
            button.setToolButtonStyle(qt.Qt.ToolButtonTextOnly)
            button.setPopupMode(qt.QToolButton.InstantPopup)
            button.setFocusPolicy(qt.Qt.NoFocus)
            menu = qt.QMenu(button)
            button.setMenu(menu)
            menu.connect("aboutToShow()", lambda m=menu, f=fill: f(m))
            self._reviewToolActions.append(toolbar.addWidget(button))
        for action in self._reviewToolActions:
            action.setVisible(False)
        return toolbar

    def _updateReviewTools(self):
        for action in getattr(self, "_reviewToolActions", []):
            try:
                action.setVisible(getattr(self, "current", None) == STEP_REVIEW)
            except Exception:
                pass

    @staticmethod
    def _menuChoice(menu, title, checked, callback):
        action = menu.addAction(title)
        action.setCheckable(True)
        action.setChecked(bool(checked))
        action.connect("triggered()", callback)
        return action

    def _fillLayoutMenu(self, menu):
        menu.clear()
        current = self.layoutCombo.currentData
        for key, title in nr.LAYOUTS:
            self._menuChoice(menu, title, key == current, lambda k=key: self._pickLayout(k))

    def _pickLayout(self, key):
        index = self.layoutCombo.findData(key)
        if index >= 0:
            self.layoutCombo.setCurrentIndex(index)
            self._onLayoutPicked()

    def _setScoreWindows(self, maximum):
        """The score window of the slices and of the 3D-SSP: -maximum .. +maximum."""
        self.scoreMaximumSpin.value = maximum
        self.sspMaximumSpin.value = maximum

    def _fillWindowMenu(self, menu):
        menu.clear()
        menu.addSection("Score window (slices and 3D-SSP)")
        current = float(self.scoreMaximumSpin.value)
        for maximum in REVIEW_SCORE_WINDOWS:
            self._menuChoice(menu, f"\u00b1 {maximum:g} SD", abs(current - maximum) < 1e-6
                             and abs(float(self.sspMaximumSpin.value) - maximum) < 1e-6,
                             lambda v=maximum: self._setScoreWindows(v))
        menu.addSection("Image windows")
        reset = menu.addAction("PET: automatic (0 - 99.5th percentile)")
        reset.connect("triggered()", self._initPetWindow)
        full = menu.addAction("PET: 0 - maximum")
        full.connect("triggered()", lambda: self.petWindowRange.setValues(0.0, self.petWindowRange.maximum))
        patient = menu.addAction("3D-SSP patient values: automatic")
        patient.connect("triggered()", lambda: (self._initSspWindow(), self._showSsp()))
        patient.enabled = bool((self.reviewResult or {}).get("ssp"))
        fusion = menu.addAction("Fusion image: automatic")
        fusion.connect("triggered()", self._resetAnatomyWindow)
        fusion.enabled = self._fusionVolume() is not None

    def _resetAnatomyWindow(self):
        self._anatomyWindowFor = None
        self._onFusionChanged()

    def _fillColorMenu(self, menu):
        menu.clear()
        for title, combo, labels in (("PET", self.petColorCombo, colormaps.LABELS),
                                     ("Scores (slices)", self.scoreColorCombo, nr.SCORE_COLORMAPS),
                                     ("3D-SSP scores", self.sspColorCombo, nr.SCORE_COLORMAPS),
                                     ("3D-SSP patient values", self.sspPatientColorCombo, colormaps.LABELS)):
            sub = menu.addMenu(title)
            for label in labels:
                self._menuChoice(sub, label, combo.currentText == label,
                                 lambda c=combo, t=label: c.setCurrentIndex(max(0, c.findText(t))))
        if self.fusionCheck.enabled:
            menu.addSeparator()
            self._menuChoice(menu, "Fusion with an anatomical image", self.fusionCheck.checked,
                             lambda: setattr(self.fusionCheck, "checked", not self.fusionCheck.checked))

    # --- navigation: no CT/MR, no registration step ----------------------------------------------------------
    def moveStep(self, delta):
        """Back / Next; the CT/MR registration is skipped when there is no CT/MR."""
        keys = [k for k, _, _, _ in self.STEPS]
        index = keys.index(self.current) + delta
        while 0 <= index < len(keys) and keys[index] == STEP_REG and self.anatomy() is None:
            index += 1 if delta > 0 else -1
        if 0 <= index < len(keys):
            self.showStep(keys[index])

    # --- the CT/MR: an MR may be normalized (to the MR template), a CT not ------------------------------------
    def contrast(self):
        """(contrast, reason) of the CT/MR: the user's choice, else read from the DICOM tags (once per CT/MR)."""
        chosen = self.contrastCombo.currentData if hasattr(self, "contrastCombo") else ""
        if chosen:
            return chosen, "chosen"
        anatomy = self.anatomy()
        key = anatomy.GetID() if anatomy is not None else ""
        if self._contrastFor is None or self._contrastFor[0] != key:
            self._contrastFor = (key, self.logic.anatomyContrast(anatomy))
        return self._contrastFor[1]

    def _anatomyIsMr(self):
        """Whether the CT/MR can be normalized to the MR template: an MR (a CT fits its bone to the brain)."""
        return self.anatomy() is not None and self.contrast()[0] != "CT"

    def _updateMethodItems(self):
        """The MR to MR template method offered only with an MR; with a CT (or none) the PET/SPECT is normalized."""
        if not hasattr(self, "methodCombo"):
            return
        allowed = self._anatomyIsMr()
        index = self.methodCombo.findData(METHOD_ANATOMY)
        item = self.methodCombo.model().item(index) if index >= 0 else None
        if item is not None:
            item.setEnabled(allowed)
            item.setToolTip("" if allowed else ("Pick an MR in step 1: a CT is not normalized to the MR template (its "
                                                "bone pulls the fit); the PET/SPECT is, and the CT follows it."))
        if not allowed and self.method() == METHOD_ANATOMY and self.normResult is None:
            self.methodCombo.setCurrentIndex(self.methodCombo.findData(METHOD_PET))

    def _defaultMethod(self):
        """The MR when the study has one, else the PET/SPECT (to the MR template, or to the database's template when
        its settings say so); until the method is picked by hand."""
        self._updateMethodItems()
        if getattr(self, "_methodChosen", False) or not hasattr(self, "methodCombo") or self.normResult is not None:
            return
        wanted = METHOD_ANATOMY if self._anatomyIsMr() else METHOD_PET
        if self.methodCombo.currentData != wanted:
            self.methodCombo.setCurrentIndex(self.methodCombo.findData(wanted))

    # --- page 1: data --------------------------------------------------------------------------------
    def _buildDataPage(self):
        page = qt.QWidget()
        layout = qt.QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        patientBox = qt.QGroupBox("Patient")
        patientForm = qt.QFormLayout(patientBox)
        self.patientNameEdit = qt.QLineEdit()
        self.patientNameEdit.toolTip = ("Name of the patient (from the DICOM tags; edit it if it is missing or wrong). "
                                        "Shown on the toolbar from the next step on.")
        patientForm.addRow("Name:", self.patientNameEdit)
        self.patientIdEdit = qt.QLineEdit()
        self.patientIdEdit.toolTip = ("Patient ID (from the DICOM tags, or the BIDS subject of an imported file; edit "
                                      "it if it is missing or wrong)")
        patientForm.addRow("Patient ID:", self.patientIdEdit)
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
        patientForm.addRow("Age:", ageRow)
        box = qt.QGroupBox("Images")
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
        self.anatomySelector = self._nodeSelector(["vtkMRMLScalarVolumeNode"],
                                                  "CT or MR of the same patient (optional: registration in step 2; "
                                                  "normalized with the image in step 3, and an MR may be the one "
                                                  "normalized)")
        self.anatomySelector.hideChildNodeTypes = ["vtkMRMLLabelMapVolumeNode"]
        form.addRow("CT / MR:", self.anatomySelector)
        self.anatomyInfo = qt.QLabel()
        self.anatomyInfo.wordWrap = True
        form.addRow(self.anatomyInfo)
        self.contrastCombo = qt.QComboBox()
        for key, title in (("", "Automatic"), ("T1", "T1"), ("T2", "T2"), ("PD", "PD"), ("CT", "CT")):
            self.contrastCombo.addItem(title, key)
        self.contrastCombo.toolTip = ("Contrast of the CT/MR, read from the DICOM tags. It picks the template of "
                                      "step 3 when the MR is normalized (a CT is not: the PET/SPECT is).")
        form.addRow("Contrast:", self.contrastCombo)
        self.extraSelector = slicer.qMRMLCheckableNodeComboBox()
        self.extraSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.extraSelector.hideChildNodeTypes = ["vtkMRMLLabelMapVolumeNode"]
        self.extraSelector.setMRMLScene(slicer.mrmlScene)
        self.extraSelector.toolTip = ("Other sequences of the patient lying on the CT/MR (e.g. FLAIR, T2, the CT of a "
                                      "PET/CT next to the MR): normalized with the same transform in step 3, for the "
                                      "fusion of the review")
        form.addRow("Other sequences:", self.extraSelector)
        layout.addWidget(patientBox)
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
        self.patientNameEdit.connect("textEdited(QString)", lambda text: self.onPatientChanged())
        self.patientIdEdit.connect("textEdited(QString)", lambda text: self.onPatientChanged())
        return page

    def extraVolumes(self):
        """The other sequences picked on the Data page (not the image or the CT/MR)."""
        if not hasattr(self, "extraSelector"):
            return []
        try:
            nodes = list(self.extraSelector.checkedNodes())
        except Exception:
            return []
        return [n for n in nodes if n is not None and n is not self.volume() and n is not self.anatomy()]

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
        volume = self.volume()
        age, sex = self.logic.ageAndSex(volume)
        name, patientId = self.logic.patientNameAndId(volume)
        self._updating = True
        try:
            self.ageSpin.value = age if age is not None else -1.0
            self.sexCombo.setCurrentIndex(max(0, self.sexCombo.findData(sex or "")))
            self.patientNameEdit.text = name
            self.patientIdEdit.text = patientId
            for widget in (self.ageSpin, self.sexCombo, self.patientNameEdit, self.patientIdEdit):
                widget.enabled = volume is not None
        finally:
            self._updating = False
        if volume is not None:
            self._patientTexts[volume.GetID()] = sn.patientText(name, patientId)
        self._updatePatientLabel()

    def onPatientChanged(self):
        volume = self.volume()
        if self._updating or volume is None:
            return
        self.logic.setPatientNameAndId(volume, self.patientNameEdit.text, self.patientIdEdit.text)
        self._patientTexts[volume.GetID()] = sn.patientText(self.patientNameEdit.text, self.patientIdEdit.text)
        self._updatePatientLabel()

    def _patientText(self):
        """'Name (ID)' of the image's patient (read once per volume)."""
        volume = self.volume()
        if volume is None:
            return ""
        if volume.GetID() not in self._patientTexts:
            self._patientTexts[volume.GetID()] = sn.patientText(*self.logic.patientNameAndId(volume))
        return self._patientTexts[volume.GetID()]

    def _updatePatientLabel(self):
        """The patient's name and ID right of the module's name on the toolbar, from the second step on (under the
        step title when there is no toolbar)."""
        text = self._patientText() if getattr(self, "current", STEP_DATA) != STEP_DATA else ""
        label = getattr(self, "patientLabel", None)
        if label is not None:
            label.text = text
            label.toolTip = f"Patient: {text}" if text else ""
            self._patientAction.setVisible(bool(text))
        elif text and hasattr(self, "titleLabel") and "Patient:" not in self.titleLabel.text:
            self.titleLabel.text += f"<br><span style='color:#2f7ed8'>Patient: {text}</span>"

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
            self.anatomyInfo.text = "No CT/MR: step 2 is skipped and the PET/SPECT is normalized."
        else:
            contrast, reason = self.contrast()
            if contrast == "CT":
                self.anatomyInfo.text = (f"Recognised as <b>CT</b> ({reason}): the PET/SPECT is normalized and the CT "
                                         "follows it (a CT is not normalized to the MR template).")
            else:
                self.anatomyInfo.text = (f"Recognised as <b>{contrast}</b> ({reason})." if contrast else
                                         f"Contrast not recognised ({reason}); the T1 template is used unless you "
                                         "pick one.")

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
            self.templateNote.text = ("Pick a normal database above: its template (a mean map frozen as the "
                                      "template, else its master mean map) is the template.")
        elif not sn.petTemplatePath(folder, header):
            self.templateNote.text = (f"'{header['name']}' has no template or master mean map yet (statistics not "
                                      "computed): pick another method or database.")
        else:
            frozen = normaldb.templatePath(folder, header)
            self.templateNote.text = (f"Registered to the {'template' if frozen else 'master mean map'} of "
                                      f"'{header['name']}' (a "
                                      f"{header.get('tracer') or header.get('nuclide') or ''} template in the MNI "
                                      "2009c space).")

    def _template(self):
        if self.method() != METHOD_PET_TEMPLATE:
            return DynamicNeuroPETWidget._template(self)
        folder = self._database()
        header = self._header(folder)
        path = sn.petTemplatePath(folder, header) if header else None
        if not path:
            raise ValueError("Pick a normal database with a template (or a master mean map).")
        template = self.logic.loadAtlasVolume(path, name=f"{header['name']} template")
        self.setNode("Template", template)
        return template

    def _updateNormButtons(self):
        DynamicNeuroPETWidget._updateNormButtons(self)
        if hasattr(self, "dbCombo") and self.method() == METHOD_PET_TEMPLATE:
            folder = self._database()
            header = self._header(folder)
            self.normalizeButton.enabled = (self.volume() is not None and header is not None
                                            and bool(sn.petTemplatePath(folder, header)))

    @staticmethod
    def _normHardenText(volume, template, method):
        return (f"Resample '{volume.GetName()}' through the normalization onto the field of view of the template "
                f"'{template.GetName()}'.\nThe patient's CT/MR (and the other sequences of step 1) are normalized "
                "too with the same transform (on the template grid)."
                "\n\nOverwrite (default) replaces the image in the patient space (Revert puts it back); Save "
                "separately keeps it next to the normalized one.")

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
        if self.method() == METHOD_ANATOMY and self.anatomy() is not None and not self._anatomyIsMr():
            slicer.util.warningDisplay("A CT is not normalized to the MR template (its bone pulls the fit). Normalize "
                                       "the PET/SPECT: the CT follows it.", windowTitle="Normalization")
            return
        settings = self.normalizationSettings()
        DynamicNeuroPETWidget.onNormalize(self)
        if self.normResult is not None:
            self.normResult["settings"] = settings
        self._updateDatabaseInfo()

    def _hardenNormalization(self, volume, template, choice, description, method):
        return self.logic.harden(volume, template, self.normSession.linear, choice["action"], choice["grid"],
                                 choice["spacing"], choice["interpolation"], description, method, self.anatomy(),
                                 self.extraVolumes())

    def onNormHarden(self):
        settings = (self.normResult or {}).get("settings")
        undo = self.normUndo
        method = (self.normResult or {}).get("method")
        DynamicNeuroPETWidget.onNormHarden(self)
        volume = self.volume()
        if self.normUndo is not undo and volume is not None:
            if settings:
                volume.SetAttribute(NORMALIZATION_ATTRIBUTE, json.dumps(settings))
            notes = []
            extras = (self.normUndo.get("undo") or {}).get("extras") or []
            if extras:
                notes.append("Other sequences: " + ", ".join(f"'{n.GetName()}'" for n in extras) + ".")
            anatomy = self.anatomy()
            if (anatomy is not None and method != METHOD_ANATOMY
                    and not self.logic.petInAnatomySpace(self.normUndo["previous"], anatomy)):
                notes.append(f"'{anatomy.GetName()}' was not registered in step 2: it is taken as lying on the image "
                             "(PET/CT, PET/MR).")
            if notes:
                self.normLabel.text += "\n" + " ".join(notes)
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
            if not sn.petTemplatePath(folder, header):
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
        self._buildReviewBoxes(layout)

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

    # --- review: comparison, views and the region table -------------------------------------------
    def _buildReviewBoxes(self, layout):
        box = qt.QGroupBox("Comparison with the normal database")
        form = qt.QFormLayout(box)
        self.reviewLabel = qt.QLabel()
        self.reviewLabel.wordWrap = True
        form.addRow(self.reviewLabel)
        self.groupCombo = qt.QComboBox()
        self.groupCombo.toolTip = ("Mean / SD maps the patient is compared with: automatically the patient's age group "
                                   "(the master, all normals, when the age is not known or the group has no maps)")
        form.addRow("Normals:", self.groupCombo)
        self.scoreCombo = qt.QComboBox()
        for key, title in nr.SCORES:
            self.scoreCombo.addItem(title, key)
        self.scoreCombo.toolTip = ("z = (patient - normal mean) / SD; Crawford-Howell t = z / sqrt(1 + 1/n), for a "
                                   "small number of normals n. Negative: the patient below the normals "
                                   "(hypometabolism / hypoperfusion). Shown in the score row and the region table.")
        form.addRow("Score:", self.scoreCombo)
        self.compareButton = _button("Compare with the database")
        form.addRow(self.compareButton)
        self.compareLabel = qt.QLabel()
        self.compareLabel.wordWrap = True
        self.compareLabel.setTextInteractionFlags(qt.Qt.TextSelectableByMouse)
        form.addRow(self.compareLabel)
        layout.addWidget(box)

        viewBox = qt.QGroupBox("Display")
        view = qt.QFormLayout(viewBox)
        self.layoutCombo = qt.QComboBox()
        for key, title in nr.LAYOUTS:
            self.layoutCombo.addItem(title, key)
        self.layoutCombo.setCurrentIndex(max(0, self.layoutCombo.findData(nr.defaultLayout(_screenCount()))))
        self.layoutCombo.toolTip = ("Single monitor: the slices, the 3D-SSP and the region table together. Dual "
                                    "monitor (the default with two screens): the slices and the table here, the "
                                    "3D-SSP views in a window on the second screen. Or the slices and the table, or "
                                    "the 3D-SSP views, alone.")
        view.addRow("Layout:", self.layoutCombo)
        self.petColorCombo = qt.QComboBox()
        self.scoreColorCombo = qt.QComboBox()
        for label in colormaps.LABELS:
            self.petColorCombo.addItem(label)
        self.petColorCombo.setCurrentIndex(max(0, self.petColorCombo.findText(REVIEW_PET_COLORMAP)))
        for label in nr.SCORE_COLORMAPS:
            self.scoreColorCombo.addItem(label)
        view.addRow("PET colors:", self.petColorCombo)
        self.petWindowRange = ctk.ctkRangeWidget()
        self.petWindowRange.decimals = 2
        self.petWindowRange.toolTip = "Display range of the normalized PET / SPECT (top row)"
        view.addRow("PET window:", self.petWindowRange)
        self.scoreColorCombo.toolTip = ("Diverging colors over the score window, 0 in the middle: below the normals "
                                        "(negative) cold, above them hot. Outside the brain mask black.")
        view.addRow("Score colors:", self.scoreColorCombo)
        self.scoreMaximumSpin = qt.QDoubleSpinBox()
        self.scoreMaximumSpin.setRange(0.5, 50.0)
        self.scoreMaximumSpin.decimals = 1
        self.scoreMaximumSpin.singleStep = 0.5
        self.scoreMaximumSpin.value = nr.DEFAULT_MAXIMUM
        self.scoreMaximumSpin.prefix = "\u00b1 "
        self.scoreMaximumSpin.toolTip = "Score window, symmetric about 0: from minus to plus this value"
        view.addRow("Score window:", self.scoreMaximumSpin)
        self.scoreSideCombo = qt.QComboBox()
        for key, title in nr.SCORE_SIDES:
            self.scoreSideCombo.addItem(title, key)
        self.scoreSideCombo.toolTip = (
            "Which scores are colored (score slices and 3D-SSP): all; only those below the normals "
            "(hypometabolism / hypoperfusion); or only those below the threshold. In the last two the color scale is "
            "reversed across minus the window .. 0 (the lowest scores at its top end) and the rest is grey.")
        view.addRow("Scores colored:", self.scoreSideCombo)
        self.thresholdSpin = qt.QDoubleSpinBox()
        self.thresholdSpin.setRange(-10.0, 0.0)
        self.thresholdSpin.decimals = 1
        self.thresholdSpin.singleStep = 0.5
        self.thresholdSpin.value = nr.DEFAULT_THRESHOLD
        self.thresholdSpin.toolTip = ("Below the threshold only: the scores below this value colored, the rest grey; "
                                      "also the low score 3D-SSP views of the 16 view layouts")
        view.addRow("Threshold:", self.thresholdSpin)
        self.scoreStyleCombo = qt.QComboBox()
        for key, title in nr.SCORE_STYLES:
            self.scoreStyleCombo.addItem(title, key)
        self.scoreStyleCombo.toolTip = (
            "Score as color, PET as brightness (default): each voxel colored by its score and as bright as the "
            "patient's uptake there, with no threshold or mask (the PET's own pattern in the score's colors; over the "
            "fusion image at the score opacity). Score map: the scores alone, in a mask and with a threshold (below).")
        view.addRow("Score slices:", self.scoreStyleCombo)
        self.scoreMaskCombo = qt.QComboBox()
        for key, title in nr.SCORE_MASKS:
            self.scoreMaskCombo.addItem(title, key)
        self.scoreMaskCombo.toolTip = (
            "Where the score slices show the scores. Grey matter (default): the MNI grey matter probability, the same "
            "for every tracer (the cortex also with a striatal tracer). High normal uptake: the database mean above a "
            "fraction of its maximum. Elsewhere the PET (grey) or the MR shows.")
        view.addRow("Scores shown in:", self.scoreMaskCombo)
        self.gmSpin = qt.QDoubleSpinBox()
        self.gmSpin.setRange(0.05, 0.95)
        self.gmSpin.decimals = 2
        self.gmSpin.singleStep = 0.05
        self.gmSpin.value = nr.GM_MINIMUM
        self.gmSpin.toolTip = "Grey matter probability from which the scores are shown"
        view.addRow("Grey matter >=", self.gmSpin)
        self.uptakeSpin = qt.QSpinBox()
        self.uptakeSpin.setRange(int(round(nr.MASK_FRACTION * 100)), 95)
        self.uptakeSpin.singleStep = 5
        self.uptakeSpin.value = int(round(nr.UPTAKE_FRACTION * 100))
        self.uptakeSpin.suffix = " % of the normal maximum"
        self.uptakeSpin.toolTip = ("The score slices show only where the normals' uptake is high: the database mean "
                                   "above this fraction of its maximum (cortex, deep grey matter). White matter and "
                                   "CSF, low in the normals anyway, are hidden (black, or the MR in fusion).")
        view.addRow("Normal uptake >", self.uptakeSpin)
        self.hideBelowSpin = qt.QDoubleSpinBox()
        self.hideBelowSpin.setRange(0.0, 10.0)
        self.hideBelowSpin.decimals = 1
        self.hideBelowSpin.singleStep = 0.5
        self.hideBelowSpin.value = nr.HIDE_BELOW
        self.hideBelowSpin.toolTip = ("Scores closer to 0 than this are not colored (the PET or the MR under them "
                                      "shows): only the clusters below / above the normals. 0: all colored.")
        view.addRow("Hide |score| below:", self.hideBelowSpin)
        self.scoreBackgroundCombo = qt.QComboBox()
        self.scoreBackgroundCombo.addItem("PET (grey)", "pet")
        self.scoreBackgroundCombo.addItem("Black", "black")
        self.scoreBackgroundCombo.toolTip = ("Under the score slices (without the MR fusion): the PET in grey, or "
                                             "black")
        view.addRow("Score background:", self.scoreBackgroundCombo)
        self.outlineCheck = qt.QCheckBox("Show the region outlines")
        self.outlineCheck.checked = True
        self.outlineCheck.toolTip = (f"The regions as {REVIEW_OUTLINE_THICKNESS} px outlines on the PET and the score "
                                     "slices. Click a region in the table to show it alone for a few seconds.")
        view.addRow(self.outlineCheck)
        self.legendCheck = qt.QCheckBox("Show the color legends")
        self.legendCheck.checked = True
        self.legendCheck.toolTip = "A color legend on the left of each row (PET, scores, 3D-SSP)"
        view.addRow(self.legendCheck)
        layout.addWidget(viewBox)
        self._buildSspBox(layout)

        fusionBox = qt.QGroupBox("Fusion")
        fusion = qt.QFormLayout(fusionBox)
        self.fusionCheck = qt.QCheckBox("Show the PET and the scores over an anatomical image")
        fusion.addRow(self.fusionCheck)
        self.fusionSourceCombo = qt.QComboBox()
        self.fusionSourceCombo.toolTip = ("The patient's CT / MR and the other sequences of step 1 (normalized with "
                                          "the image in step 3), or an MNI template (T1, T2, PD)")
        fusion.addRow("Image:", self.fusionSourceCombo)
        self.fusionOpacity = ctk.ctkSliderWidget()
        self.fusionOpacity.minimum, self.fusionOpacity.maximum = 0.0, 1.0
        self.fusionOpacity.singleStep, self.fusionOpacity.decimals = 0.05, 2
        self.fusionOpacity.value = REVIEW_FUSION_OPACITY
        self.fusionOpacity.toolTip = "Opacity of the PET over the MR"
        fusion.addRow("PET opacity:", self.fusionOpacity)
        self.scoreOpacity = ctk.ctkSliderWidget()
        self.scoreOpacity.minimum, self.scoreOpacity.maximum = 0.0, 1.0
        self.scoreOpacity.singleStep, self.scoreOpacity.decimals = 0.05, 2
        self.scoreOpacity.value = REVIEW_SCORE_OPACITY
        self.scoreOpacity.toolTip = "Opacity of the z / t scores over the MR"
        fusion.addRow("Score opacity:", self.scoreOpacity)
        self.anatomyWindowRange = ctk.ctkRangeWidget()
        self.anatomyWindowRange.decimals = 1
        self.anatomyWindowRange.toolTip = "Display range of the fusion image (CT / MR / template, grey)"
        fusion.addRow("Image window:", self.anatomyWindowRange)
        layout.addWidget(fusionBox)
        settingsRow = qt.QHBoxLayout()
        self.saveDisplayButton = _button("Save display settings")
        self.saveDisplayButton.toolTip = ("Keep the display, 3D-SSP and fusion settings (colors, score windows, score "
                                          "slices, outlines, legends, opacities; not the image windows or the layout) "
                                          "as the defaults of the next sessions")
        self.restoreDisplayButton = _button("Restore defaults", "revert")
        self.restoreDisplayButton.toolTip = "The display settings back to the saved ones (the factory ones if none)"
        self.factoryDisplayButton = _button("Restore factory", "revert")
        self.factoryDisplayButton.toolTip = "The display settings back to the module's own; the saved ones forgotten"
        for button in (self.saveDisplayButton, self.restoreDisplayButton, self.factoryDisplayButton):
            settingsRow.addWidget(button)
        layout.addLayout(settingsRow)

        self.compareButton.connect("clicked()", self.onCompare)
        self.groupCombo.connect("activated(int)", lambda i: setattr(self, "_groupChosen", True))
        self.scoreCombo.connect("currentIndexChanged(int)", lambda i: self._showScores())
        self.petColorCombo.connect("currentIndexChanged(int)", lambda i: self._applyPetDisplay())
        self.petWindowRange.connect("valuesChanged(double,double)", lambda a, b: self._applyPetDisplay())
        for widget in (self.scoreColorCombo,):
            widget.connect("currentIndexChanged(int)", lambda i: self._applyScoreDisplay())
        self.scoreMaximumSpin.connect("valueChanged(double)", lambda v: self._applyScoreDisplay())
        self.scoreSideCombo.connect("currentIndexChanged(int)", lambda i: (self._applyScoreDisplay(),
                                                                         self._updateLegends()))
        self.thresholdSpin.connect("valueChanged(double)", lambda v: self._applyScoreDisplay())
        self.fusionCheck.connect("toggled(bool)", lambda on: self._onFusionChanged())
        self.fusionSourceCombo.connect("currentIndexChanged(int)", lambda i: self._onFusionChanged())
        self.fusionOpacity.connect("valueChanged(double)", lambda v: self._assignReviewViews())
        self.scoreOpacity.connect("valueChanged(double)", lambda v: self._assignReviewViews())
        self.layoutCombo.connect("activated(int)", lambda i: self._onLayoutPicked())
        self.uptakeSpin.connect("valueChanged(int)", lambda v: self._showScoreMap())
        self.gmSpin.connect("valueChanged(double)", lambda v: self._showScoreMap())
        self.hideBelowSpin.connect("valueChanged(double)", lambda v: self._showScoreMap())
        self.scoreMaskCombo.connect("currentIndexChanged(int)", lambda i: self._showScoreMap())
        self.scoreStyleCombo.connect("currentIndexChanged(int)", lambda i: (self._showScoreMap(),
                                                                          self._assignReviewViews()))
        self.scoreBackgroundCombo.connect("currentIndexChanged(int)", lambda i: self._assignReviewViews())
        self.outlineCheck.connect("toggled(bool)", lambda on: self._assignReviewViews())
        self.legendCheck.connect("toggled(bool)", lambda on: (self._updateLegends(), self._showSsp()))
        self.anatomyWindowRange.connect("valuesChanged(double,double)", lambda a, b: self._applyAnatomyDisplay())
        self.reviewResult = None
        self.sspModels = {}                 # (view set, surface part): model node
        self.scoreNode = None
        self.reviewTable = None
        self._reviewPreviousLayout = None
        self._segmentationDisplay = None
        self.regionTable = None             # QTableWidget in the layout's table view
        self._regionTableHost = None
        self._flash = None                  # (token, {segment ID: visible}) while a region is shown alone
        self._autoCompared = None
        self._factoryDisplay = self._displayValues()
        saved = self._savedDisplay()
        if saved:
            self._setDisplayValues(saved, quiet=True)
        self.saveDisplayButton.connect("clicked()", self.onSaveDisplay)
        self.restoreDisplayButton.connect("clicked()", self.onRestoreDisplay)
        self.factoryDisplayButton.connect("clicked()", self.onFactoryDisplay)

    # saved display settings
    def _displayWidgets(self):
        """{key: widget} of the review display settings that are saved (not the image windows or the layout)."""
        return {"petColors": self.petColorCombo, "scoreColors": self.scoreColorCombo,
                "scoreMaximum": self.scoreMaximumSpin, "scoreSide": self.scoreSideCombo,
                "threshold": self.thresholdSpin,
                "scoreStyle": self.scoreStyleCombo, "scoreMask": self.scoreMaskCombo, "greyMatter": self.gmSpin,
                "uptake": self.uptakeSpin, "hideBelow": self.hideBelowSpin,
                "scoreBackground": self.scoreBackgroundCombo, "outlines": self.outlineCheck,
                "legends": self.legendCheck, "sspMap": self.sspMapCombo, "sspSource": self.sspSourceCombo,
                "sspColors": self.sspColorCombo, "sspMaximum": self.sspMaximumSpin,
                "sspPatientColors": self.sspPatientColorCombo, "sspLighting": self.sspLightingCheck,
                "fusionOpacity": self.fusionOpacity, "scoreOpacity": self.scoreOpacity}

    def _displayValues(self):
        values = {}
        for key, widget in self._displayWidgets().items():
            if isinstance(widget, qt.QComboBox):
                values[key] = widget.currentText
            elif isinstance(widget, qt.QCheckBox):
                values[key] = bool(widget.checked)
            else:
                values[key] = widget.value
        return values

    def _setDisplayValues(self, values, quiet=False):
        """The display widgets set to values (unknown keys and labels skipped); quiet: without their signals."""
        for key, widget in self._displayWidgets().items():
            if key not in values:
                continue
            value = values[key]
            if quiet:
                widget.blockSignals(True)
            try:
                if isinstance(widget, qt.QComboBox):
                    index = widget.findText(str(value))
                    if index >= 0:
                        widget.setCurrentIndex(index)
                elif isinstance(widget, qt.QCheckBox):
                    widget.checked = bool(value)
                elif isinstance(widget, qt.QSpinBox):
                    widget.value = int(value)
                else:
                    widget.value = float(value)
            except (TypeError, ValueError):
                logging.warning(f"Static Neuro: display setting {key} = {value!r} skipped")
            finally:
                if quiet:
                    widget.blockSignals(False)

    @staticmethod
    def _savedDisplay():
        try:
            return json.loads(str(qt.QSettings().value(SETTINGS_DISPLAY) or "") or "{}")
        except ValueError:
            return {}

    def onSaveDisplay(self):
        qt.QSettings().setValue(SETTINGS_DISPLAY, json.dumps(self._displayValues()))
        slicer.util.showStatusMessage("Static Neuro: display settings saved as the defaults", 3000)

    def onRestoreDisplay(self):
        self._setDisplayValues(self._savedDisplay() or self._factoryDisplay)

    def onFactoryDisplay(self):
        qt.QSettings().remove(SETTINGS_DISPLAY)
        self._setDisplayValues(self._factoryDisplay)

    def _updateReviewInfo(self, volume, segmentation, header, age, sex):
        lines = [f"Image: {volume.GetName() if volume else 'none'}"
                 + (" (normalized)" if volume is not None and volume.GetAttribute(NEURO_PREFIX + "Template") else ""),
                 "Age: " + (f"{age:g}" if age is not None else "unknown") + ", sex: "
                 + (dict(sn.SEXES).get(sex) if sex else "unknown"),
                 f"Regions: {segmentation.GetName() if segmentation else 'not loaded (step 4)'}"]
        if header is not None:
            stats = header.get("statistics") or {}
            lines.append(f"Database: {header['name']} ({header.get('mapType', '')} maps"
                         + (f", smoothing {float(stats.get('smoothingFwhmMm', header.get('smoothingFwhmMm')) or 0):g} "
                            "mm" if stats.get("groups") else "") + ")")
            if not stats.get("groups"):
                lines.append(WARNING_STYLE.format("The database has no mean / SD maps: compute them in the Neuro PET "
                                                  "Database Manager."))
            elif stats.get("stale"):
                lines.append(WARNING_STYLE.format("Subjects were added after the mean / SD maps were computed."))
        else:
            lines.append("No database: the regional values (SUV, SUVR to the reference region) only.")
        result = self.reviewResult
        if result is not None and (result.get("volume") is not volume or result.get("header") != header):
            lines.append(WARNING_STYLE.format("The image or the database changed: compare again."))
        self.reviewLabel.text = "<br>".join(lines)
        self._fillGroups(header, age)
        self.groupCombo.enabled = header is not None and self.groupCombo.count > 1
        self.compareButton.text = "Compare with the database" if header is not None else "Compute the regional values"
        normalized = volume is not None and bool(volume.GetAttribute(NEURO_PREFIX + "Template"))
        self.compareButton.enabled = normalized and segmentation is not None and (
            header is None or bool((header.get("statistics") or {}).get("groups")))
        self._fillFusionSources()

    def _fillGroups(self, header, age):
        key = (json.dumps((header or {}).get("statistics"), sort_keys=True, default=str), age)
        if getattr(self, "_groupsFor", None) == key:
            return
        self._groupsFor = key
        current = self.groupCombo.currentData
        chosen = getattr(self, "_groupChosen", False)
        self.groupCombo.blockSignals(True)
        self.groupCombo.clear()
        if header is not None:
            automatic, why = nr.patientGroup(header, age)
            names = {k: (name, n) for k, name, n in nr.statisticsGroups(header)}
            title = names.get(automatic, (automatic, None))[0]
            title = normaldb.MASTER_TITLE if automatic == normaldb.MASTER else title
            self.groupCombo.addItem(f"Automatic: {title} ({why})", "")
            for k, (name, n) in names.items():
                self.groupCombo.addItem((normaldb.MASTER_TITLE if k == normaldb.MASTER else f"Age group {name}")
                                        + (f" (n = {n})" if n else ""), k)
        index = self.groupCombo.findData(current) if chosen and current else -1
        self.groupCombo.setCurrentIndex(max(0, index))
        self.groupCombo.blockSignals(False)

    def _groupKey(self, header, age):
        key = self.groupCombo.currentData if self.groupCombo.count else ""
        return key or nr.patientGroup(header, age)[0]

    def _fusionCandidates(self):
        """The patient's volumes in the template space for the fusion: the normalized CT / MR first, then the other
        normalized sequences (and any other normalized anatomical volume)."""
        first = self.node("NormalizedAnatomy")
        skip = {n.GetID() for n in (self.volume(), getattr(self, "greyPetNode", None), self.scoreNode) if n is not None}
        nodes = []
        for node in slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode"):
            if (node.GetID() in skip or node.IsA("vtkMRMLLabelMapVolumeNode") or node.GetHideFromEditors()
                    or node.GetAttribute(ATLAS_FILE_ATTRIBUTE) or not node.GetAttribute(NEURO_PREFIX + "Template")
                    or self.logic.isStaticImage(node)):
                continue
            nodes.append(node)
        nodes.sort(key=lambda n: n is not first)
        return nodes

    def _fillFusionSources(self):
        wanted = []
        for node in self._fusionCandidates():
            source = node.GetAttribute(SOURCE_ATTRIBUTE)
            what = "CT" if source and source == (self.anatomy().GetName() if self.anatomy() else None) and \
                self.contrast()[0] == "CT" else "Patient"
            wanted.append((f"node:{node.GetID()}", f"{what}: {node.GetName()}"))
        for key, title, _ in neuro.TEMPLATES:
            if key in FUSION_TEMPLATES and key in (self.atlas or {}):
                wanted.append((f"template:{key}", f"MNI template {title}"))
        if [(self.fusionSourceCombo.itemData(i), self.fusionSourceCombo.itemText(i))
                for i in range(self.fusionSourceCombo.count)] == wanted:
            return
        current = self.fusionSourceCombo.currentData
        self.fusionSourceCombo.blockSignals(True)
        self.fusionSourceCombo.clear()
        for key, title in wanted:
            self.fusionSourceCombo.addItem(title, key)
        self.fusionSourceCombo.setCurrentIndex(max(0, self.fusionSourceCombo.findData(current)))
        self.fusionSourceCombo.blockSignals(False)
        self.fusionCheck.enabled = bool(wanted)

    def _fusionVolume(self):
        if not self.fusionCheck.checked or not self.fusionCheck.enabled:
            return None
        key = self.fusionSourceCombo.currentData or ""
        if key.startswith("node:"):
            node = slicer.mrmlScene.GetNodeByID(key[len("node:"):])
            return node if node is not None and slicer.mrmlScene.IsNodePresent(node) else None
        if key.startswith("template:") and key[len("template:"):] in (self.atlas or {}):
            name = key[len("template:"):]
            return self.logic.loadAtlasVolume(self.atlas[name], name=f"MNI ICBM152 2009c {name}")
        return None

    def onCompare(self):
        volume, segmentation = self.volume(), self.node("Segmentation")
        if volume is None or segmentation is None:
            return
        folder = self._database()
        header = self._header(folder)
        removed = self.logic.dropSegments(segmentation)
        age, _ = self.logic.ageAndSex(volume)
        groupKey = self._groupKey(header, age) if header is not None else normaldb.MASTER
        referenceId = self._parameter().GetParameter(NEURO_PREFIX + "ReferenceSegment") or None
        if referenceId and segmentation.GetSegmentation().GetSegment(referenceId) is None:
            referenceId = None
        dialog = slicer.util.createProgressDialog(windowTitle="Review", labelText="Comparing...", maximum=0)

        def progress(index, total, text):
            dialog.maximum = total
            dialog.setValue(index)
            dialog.setLabelText(f"The normals: {text}")
            slicer.app.processEvents()
            return not dialog.wasCanceled

        slicer.app.processEvents()
        qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)
        try:
            result = self.logic.reviewData(volume, segmentation, folder if header is not None else None, header,
                                           groupKey, self.atlas, referenceId, self._locked(), progress)
            if result is not None:
                try:
                    result["ssp"] = self.logic.reviewSsp(result, folder, header, groupKey, self._locked(), progress)
                    if result["ssp"] is None:
                        result = None
                    else:
                        result["notes"].extend(result["ssp"]["notes"])
                except Exception as error:
                    logging.exception("Static Neuro: the 3D-SSP failed")
                    result["ssp"] = None
                    result["notes"].append(f"No 3D-SSP: {error}")
        except Exception as error:
            logging.exception("Static Neuro: the review failed")
            slicer.util.errorDisplay(f"Could not compare the patient:\n{error}", windowTitle="Review")
            return
        finally:
            qt.QApplication.restoreOverrideCursor()
            dialog.close()
        if result is None:
            self.compareLabel.text = "Stopped."
            return
        result.update(volume=volume, header=header, groupKey=groupKey, segmentation=segmentation)
        self.reviewResult = result
        self._makeScoreNode()
        self._initPetWindow()
        self._initSspWindow()
        self._showScores()
        self._enterReviewViews()
        lines = []
        if header is not None:
            group = next((name for k, name, _ in nr.statisticsGroups(header) if k == groupKey), groupKey)
            lines.append(f"Compared with {normaldb.MASTER_TITLE if groupKey == normaldb.MASTER else group} "
                         f"(n = {result.get('n')}); patient smoothed {result['fwhm']:g} mm FWHM like the normals.")
            lines.append(f"Regional normal values: {result['source']}.")
        if result.get("referenceMean"):
            lines.append(f"Reference region mean: {result['referenceMean']:.4g}.")
        if removed:
            lines.append("Left out: " + ", ".join(removed) + ".")
        lines.extend(result["notes"])
        self.compareLabel.text = " ".join(lines)
        self.refresh()

    def _autoCompare(self):
        """Compare on entering the review when there is no comparison of this image, regions and database yet (once
        per image, regions and database: not again after a failure or a stop)."""
        if getattr(self, "current", None) != STEP_REVIEW or not self.compareButton.enabled:
            return
        volume, segmentation = self.volume(), self.node("Segmentation")
        folder = self._database()
        header = self._header(folder)
        result = self.reviewResult
        if (result is not None and result.get("volume") is volume and result.get("header") == header
                and result.get("segmentation") is segmentation):
            return
        key = (volume.GetID() if volume else "", segmentation.GetID() if segmentation else "", folder,
               json.dumps(header, sort_keys=True, default=str))
        if self._autoCompared == key:
            return
        self._autoCompared = key
        self.onCompare()

    def _makeScoreNode(self):
        result = self.reviewResult
        scene = slicer.mrmlScene
        if result is None or result.get("z") is None:
            if self.scoreNode is not None and scene.IsNodePresent(self.scoreNode):
                scene.RemoveNode(self.scoreNode)
            self.scoreNode = None
            return
        if self.scoreNode is None or not scene.IsNodePresent(self.scoreNode):
            self.scoreNode = scene.AddNewNodeByClass("vtkMRMLScalarVolumeNode",
                                                     scene.GenerateUniqueName(f"{result['volume'].GetName()} score"))
            self.scoreNode.CreateDefaultDisplayNodes()
        self.scoreNode.SetIJKToRASMatrix(_vtkMatrix(_ijkToRas(result["grid"])))
        slicer.util.updateVolumeFromArray(self.scoreNode, result["z"])

    def _scoreKind(self):
        return self.scoreCombo.currentData or nr.DEFAULT_SCORE

    def _showScores(self):
        """The score map (z, or t = z / sqrt(1 + 1/n)) and the region table of the last comparison."""
        result = self.reviewResult
        if result is None:
            return
        kind = self._scoreKind()
        self._showScoreMap()
        self._updateReviewTable(kind)
        self._showSsp()

    def _showScoreMap(self):
        """The score map shown where the normals' uptake is high (the threshold of the Display box)."""
        result = self.reviewResult
        if result is None or self.scoreNode is None or result.get("z") is None:
            return
        kind = self._scoreKind()
        factor = nr.tFactor(result.get("n")) if kind == nr.SCORE_T else 1.0
        mask = result["mask"]
        masked = self._scoreStyle() == nr.STYLE_MASKED
        greyMatter = self.scoreMaskCombo.currentData == nr.MASK_GM and result.get("gm") is not None
        for widget in (self.scoreMaskCombo, self.hideBelowSpin, self.scoreBackgroundCombo):
            widget.enabled = masked
        self.gmSpin.enabled = masked and greyMatter
        self.uptakeSpin.enabled = masked and not greyMatter
        if not masked:
            pass        # the colored PET: every scored voxel (the score node for the data probe and the legend)
        elif greyMatter:
            mask = nr.greyMatterMask(result["gm"], mask, float(self.gmSpin.value))
        elif result.get("mean") is not None:
            mask = nr.uptakeMask(result["mean"], mask, self.uptakeSpin.value / 100.0)
        slicer.util.updateVolumeFromArray(self.scoreNode, nr.maskedScores(
            result["z"], mask, factor, float(self.hideBelowSpin.value) if masked else 0.0))
        self.scoreNode.SetName(f"{result['volume'].GetName()} {kind} map")
        self._applyScoreDisplay()
        self._updateLegends()

    def _updateReviewTable(self, kind):
        result = self.reviewResult
        scene = slicer.mrmlScene
        n = result.get("n")
        reference = {k: [v[0], v[1], v[2] if len(v) > 2 and v[2] is not None else n]
                     for k, v in (result.get("reference") or {}).items()}
        rows = nr.tableRows(result["values"], result["suv"], reference, kind)
        referenceKey = (result.get("header") or {}).get("referenceRegion")
        rows = [row for row in rows if not nr.isReferenceRegion(row[0], referenceKey)]     # the reference left out
        units = result["header"].get("mapType") if result.get("header") else "SUVR"
        compared = result.get("ratio") and units in (normaldb.MAP_SUVR, normaldb.MAP_RATIO, "SUVR")
        columns = ["Region", "SUVmean"] + ([units or "SUVR"] if compared else [])
        if result.get("header") is not None:
            columns += ["Normal mean", "Normal SD", "n", kind]
        table = vtk.vtkTable()
        arrays = []
        for name in columns:
            array = vtk.vtkStringArray()
            array.SetName(name)
            table.AddColumn(array)
            arrays.append(array)

        def text(value, digits=3):
            if value is None or (isinstance(value, float) and not np.isfinite(value)):
                return ""
            return f"{value:.{digits}g}" if isinstance(value, float) else str(value)

        def rounded(value, digits=3):     # the number shown, kept a number (numeric sorting)
            if value is None or (isinstance(value, float) and not np.isfinite(value)):
                return None
            return float(f"{value:.{digits}g}") if isinstance(value, float) else value

        cellRows = []
        for name, suv, value, mean, sd, count, score in rows:
            cells = [name, text(suv)] + ([text(value)] if compared else [])
            numbers = [name, rounded(suv)] + ([rounded(value)] if compared else [])
            if result.get("header") is not None:
                cells += [text(mean), text(sd), text(count), "" if score is None else f"{score:.2f}"]
                numbers += [rounded(mean), rounded(sd), count, None if score is None else round(float(score), 2)]
            for array, cell in zip(arrays, cells):
                array.InsertNextValue(cell)
            cellRows.append(numbers)
        if self.reviewTable is None or not scene.IsNodePresent(self.reviewTable):
            self.reviewTable = scene.AddNewNodeByClass("vtkMRMLTableNode", scene.GenerateUniqueName("Neuro review"))
        self.reviewTable.SetAndObserveTable(table)
        if hasattr(self.reviewTable, "SetUseColumnNameAsColumnHeader"):
            self.reviewTable.SetUseColumnNameAsColumnHeader(True)
        self.reviewTable.SetName(f"{result['volume'].GetName()} regions")
        self._regionTableData = (columns, cellRows)
        self._showTable()

    def _showTable(self):
        """The region table in the layout's table view: a sortable table with the region colors (clicking a region
        shows it alone for a few seconds); Slicer's own table view of the table node when it cannot be embedded."""
        if self.reviewTable is None:
            return
        if self._fillRegionTable():
            return
        selection = slicer.app.applicationLogic().GetSelectionNode()
        selection.SetActiveTableID(self.reviewTable.GetID())
        slicer.app.applicationLogic().PropagateTableSelection()

    @staticmethod
    def _tableViewWidget():
        """The qMRMLTableWidget of the review layout's table view, None when the layout has none."""
        layoutManager = slicer.app.layoutManager()
        if layoutManager is None or not hasattr(layoutManager, "tableWidget"):
            return None
        for index in range(getattr(layoutManager, "tableViewCount", 0)):
            widget = layoutManager.tableWidget(index)
            viewNode = widget.mrmlTableViewNode() if widget is not None else None
            if viewNode is not None and viewNode.GetSingletonTag() == nr.TABLE_VIEW:
                return widget
        return None

    def _regionTableWidget(self):
        """The QTableWidget inside the table view (made there the first time); None when it cannot be embedded."""
        host = self._tableViewWidget()
        if host is None:
            return None
        table = self.regionTable
        try:
            alive = table is not None and table.parent() is host
        except Exception:       # the C++ widget deleted with its view
            alive = False
        views = [host.tableView()] if hasattr(host, "tableView") else slicer.util.findChildren(
            host, className="qMRMLTableView")
        for child in views:
            child.setVisible(False)
        controller = host.tableController() if hasattr(host, "tableController") else None
        if controller is not None:
            controller.setVisible(False)
        if alive:
            table.setVisible(True)
            return table
        hostLayout = host.layout()
        if hostLayout is None:
            return None
        table = qt.QTableWidget(host)
        table.setEditTriggers(qt.QAbstractItemView.NoEditTriggers)
        table.setSelectionBehavior(qt.QAbstractItemView.SelectRows)
        table.setSelectionMode(qt.QAbstractItemView.SingleSelection)
        table.verticalHeader().setVisible(False)
        table.setAlternatingRowColors(True)
        table.toolTip = ("Click a column header to sort by it; click a region to show it alone on the slices for "
                         f"{REVIEW_FLASH_MS // 1000} s.")
        hostLayout.addWidget(table)
        table.connect("cellClicked(int,int)", self._onRegionClicked)
        self.regionTable = table
        self._regionTableHost = host
        return table

    def _fillRegionTable(self):
        data = getattr(self, "_regionTableData", None)
        table = self._regionTableWidget() if data is not None else None
        if table is None:
            return False
        columns, rows = data
        colors, ids = self._segmentColors()
        sortColumn = table.horizontalHeader().sortIndicatorSection() if table.sortingEnabled else -1
        sortOrder = table.horizontalHeader().sortIndicatorOrder()
        table.setSortingEnabled(False)
        table.clear()
        table.setColumnCount(len(columns) + 1)
        table.setRowCount(len(rows))
        table.setHorizontalHeaderLabels([""] + list(columns))
        for r, cells in enumerate(rows):
            name = cells[0]
            swatch = qt.QTableWidgetItem()
            color = colors.get(name)
            if color is not None:
                swatch.setBackground(qt.QBrush(qt.QColor(*(int(round(255 * c)) for c in color))))
            swatch.setData(qt.Qt.UserRole, ids.get(name, ""))
            table.setItem(r, 0, swatch)
            for c, value in enumerate(cells):
                item = qt.QTableWidgetItem()
                if value is None:
                    item.setText("")
                elif isinstance(value, str):
                    item.setText(value)
                else:
                    item.setData(qt.Qt.DisplayRole, value)      # a number: sorted by size
                if c > 0:
                    item.setTextAlignment(qt.Qt.AlignRight | qt.Qt.AlignVCenter)
                item.setData(qt.Qt.UserRole, ids.get(name, ""))
                table.setItem(r, c + 1, item)
        table.setSortingEnabled(True)
        if sortColumn >= 0:
            table.sortItems(sortColumn, sortOrder)
        table.resizeColumnsToContents()
        table.setColumnWidth(0, 14)
        qt.QTimer.singleShot(100, self._fitTableWidth)
        return True

    def _segmentColors(self):
        """({segment name: (r, g, b)}, {segment name: segment ID}) of the regions."""
        segmentation = self.node("Segmentation")
        colors, ids = {}, {}
        if segmentation is None:
            return colors, ids
        seg = segmentation.GetSegmentation()
        for segmentId in self.logic.segmentIds(segmentation):
            segment = seg.GetSegment(segmentId)
            colors[segment.GetName()] = tuple(segment.GetColor())
            ids[segment.GetName()] = segmentId
        return colors, ids

    def _fitTableWidth(self):
        """The table view as wide as the table (the slices take the rest)."""
        table = self.regionTable
        try:
            if table is None or not table.visible:
                return
            width = sum(table.columnWidth(c) for c in range(table.columnCount))
            width += 2 * table.frameWidth + table.verticalScrollBar().sizeHint.width() + 8
            child = self._regionTableHost
            splitter = child.parent() if child is not None else None
            while splitter is not None and not splitter.inherits("QSplitter"):
                child, splitter = splitter, splitter.parent()
            if splitter is None:
                return
            index = splitter.indexOf(child)
            sizes = list(splitter.sizes())
            if index < 0 or len(sizes) < 2:
                return
            total = sum(sizes)
            width = min(width, int(total * 0.6))
            others = [i for i in range(len(sizes)) if i != index]
            rest = total - width
            spread = sum(sizes[i] for i in others) or 1
            for i in others:
                sizes[i] = int(rest * sizes[i] / spread)
            sizes[index] = width
            splitter.setSizes(sizes)
        except Exception:
            logging.debug("Static Neuro: could not resize the region table", exc_info=True)

    def _onRegionClicked(self, row, column):
        """The region of the row alone on the slices for REVIEW_FLASH_MS, then the regions shown as before."""
        item = self.regionTable.item(row, 0) if self.regionTable is not None else None
        segmentId = item.data(qt.Qt.UserRole) if item is not None else ""
        segmentation = self.node("Segmentation")
        display = segmentation.GetDisplayNode() if segmentation is not None else None
        if not segmentId or display is None:
            return
        self._endFlash()
        ids = self.logic.segmentIds(segmentation)
        saved = {i: bool(display.GetSegmentVisibility(i)) for i in ids}
        token = object()
        self._flash = (token, display, saved)
        wasModifying = display.StartModify()
        for i in ids:
            display.SetSegmentVisibility(i, i == segmentId)
        display.SetVisibility2DOutline(True)        # also with the outlines hidden
        display.EndModify(wasModifying)
        qt.QTimer.singleShot(REVIEW_FLASH_MS, lambda: self._endFlash(token))
        self._jumpToRegion(segmentation, segmentId)

    def _regionPoint(self, segmentation, segmentId):
        """RAS of a point inside a region: the voxel of the compared regions nearest to its centroid, else the
        segment's center."""
        result = self.reviewResult or {}
        name = segmentation.GetSegmentation().GetSegment(segmentId).GetName()
        if result.get("labels") is not None and result.get("segmentation") is segmentation:
            labels, names = result["labels"]
            if name in names:
                point = nr.insidePoint(labels, names.index(name) + 1)
                if point is not None:
                    ijk = np.array([point[2], point[1], point[0], 1.0])
                    return (_ijkToRas(result["grid"]) @ ijk)[:3]
        if hasattr(segmentation, "GetSegmentCenterRAS"):
            return np.asarray(segmentation.GetSegmentCenterRAS(segmentId), dtype=float)
        return None

    def _jumpToRegion(self, segmentation, segmentId):
        """The review's slice views moved through a point inside the region (zoom and pan kept)."""
        try:
            ras = self._regionPoint(segmentation, segmentId)
            if ras is None:
                return
            for _, widget in self._reviewSliceWidgets():
                widget.mrmlSliceNode().JumpSliceByOffsetting(*(float(v) for v in ras))
        except Exception:
            logging.debug("Static Neuro: could not move the slices to the region", exc_info=True)

    def _endFlash(self, token=None):
        flash = self._flash
        if flash is None or (token is not None and flash[0] is not token):
            return
        self._flash = None
        _, display, saved = flash
        if not slicer.mrmlScene.IsNodePresent(display):
            return
        wasModifying = display.StartModify()
        for segmentId, visible in saved.items():
            display.SetSegmentVisibility(segmentId, visible)
        if self._segmentationDisplay is not None and self._segmentationDisplay[0] is display:
            display.SetVisibility2DOutline(bool(self.outlineCheck.checked))
        display.EndModify(wasModifying)

    # display
    @staticmethod
    def _rgbOf(label):
        """256 x 3 RGB (0..1) of a color map of colormaps.LABELS or a Slicer color table (by name)."""
        colors = colortables.table(label)
        if colors is None:
            colors = nr.colorTable(label)
        if colors is not None:
            return colors
        node = colormaps.colorNodeForLabel(label) if label in colormaps.LABELS else colormaps._builtInNode(label)
        lookup = node.GetScalarsToColors() if node is not None else None
        if lookup is None:
            return np.tile(np.linspace(0, 1, 256)[:, None], (1, 3))
        low, high = lookup.GetRange()
        colors = np.zeros((256, 3), dtype=float)
        rgb = [0.0, 0.0, 0.0]
        for i in range(256):        # GetColor(value, rgb): the output array form works in every VTK wrapping
            lookup.GetColor(low + (high - low) * i / 255.0, rgb)
            colors[i] = rgb
        return colors

    def _scoreColorNode(self):
        scene = slicer.mrmlScene
        node = scene.GetFirstNodeByName(SCORE_COLOR_NODE)
        if node is None or not node.IsA("vtkMRMLColorTableNode"):
            node = scene.CreateNodeByClass("vtkMRMLColorTableNode")
            node.UnRegister(None)
            node.SetName(SCORE_COLOR_NODE)
            node.SetTypeToUser()
            node.SetNumberOfColors(256)
            node.SetHideFromEditors(False)
            scene.AddNode(node)
        rgba, _ = self._scoreColors()
        for i, (r, g, b, a) in enumerate(rgba):
            node.SetColor(i, f"{i}", float(r), float(g), float(b), float(a))
        return node

    def _scoreSide(self):
        return self.scoreSideCombo.currentData if hasattr(self, "scoreSideCombo") else nr.SIDE_ALL

    def _scoreColors(self, label=None, maximum=None, side=None):
        """(RGBA, (low, high)) of the score colors: the score slices' (or a color map label's, a window's, a side's)."""
        return nr.scoreColors(self._rgbOf(label or self.scoreColorCombo.currentText), side or self._scoreSide(),
                              float(self.scoreMaximumSpin.value if maximum is None else maximum),
                              float(self.thresholdSpin.value))

    def _applyScoreDisplay(self):
        if self.scoreNode is None or not slicer.mrmlScene.IsNodePresent(self.scoreNode):
            return
        display = self.scoreNode.GetDisplayNode()
        if display is None:
            self.scoreNode.CreateDefaultDisplayNodes()
            display = self.scoreNode.GetDisplayNode()
        _, (low, high) = self._scoreColors()
        display.SetAndObserveColorNodeID(self._scoreColorNode().GetID())
        display.SetAutoWindowLevel(False)
        display.SetWindowLevelMinMax(low, high)
        display.SetApplyThreshold(True)         # outside the brain mask (OUTSIDE) transparent: black
        display.SetLowerThreshold(nr.OUTSIDE / 2.0)
        display.SetUpperThreshold(-nr.OUTSIDE)
        self._updateScoreColors()

    def _scoreStyle(self):
        return self.scoreStyleCombo.currentData if hasattr(self, "scoreStyleCombo") else nr.STYLE_MASKED

    def _scoreColorsNode(self):
        """The colored PET (RGB: the score's color times the PET's brightness) when shown, else None."""
        node = getattr(self, "scoreRgbNode", None)
        if self._scoreStyle() != nr.STYLE_BRIGHTNESS or node is None or not slicer.mrmlScene.IsNodePresent(node):
            return None
        return node

    def _updateScoreColors(self):
        """The colored PET of the score row: each voxel the score's color, as bright as the patient's uptake."""
        result = self.reviewResult or {}
        if (self._scoreStyle() != nr.STYLE_BRIGHTNESS or result.get("z") is None
                or result.get("brightness") is None):
            return
        scene = slicer.mrmlScene
        node = getattr(self, "scoreRgbNode", None)
        if node is None or not scene.IsNodePresent(node):
            node = scene.AddNewNodeByClass("vtkMRMLVectorVolumeNode", scene.GenerateUniqueName(
                f"{result['volume'].GetName()} score colors"))
            self.scoreRgbNode = node
        node.SetIJKToRASMatrix(_vtkMatrix(_ijkToRas(result["grid"])))
        factor = nr.tFactor(result.get("n")) if self._scoreKind() == nr.SCORE_T else 1.0
        rgba, (low, high) = self._scoreColors()
        rgb = nr.brightnessColors(np.asarray(result["z"], np.float32) / np.float32(factor), result["brightness"],
                                  rgba, high, low)
        slicer.util.updateVolumeFromArray(node, rgb)
        if node.GetDisplayNode() is None:
            node.CreateDefaultDisplayNodes()
        display = node.GetDisplayNode()
        if display is not None:     # the RGB as it is
            display.SetAutoWindowLevel(False)
            display.SetWindowLevelMinMax(0.0, 255.0)

    def _initPetWindow(self):
        volume = self.volume()
        if volume is None:
            return
        low, high = _percentiles(volume, 0.0, 99.5)
        top = float(np.nanmax(slicer.util.arrayFromVolume(volume)))
        self.petWindowRange.blockSignals(True)
        self.petWindowRange.minimum = min(0.0, low)
        self.petWindowRange.maximum = max(top, high)
        self.petWindowRange.singleStep = max(high / 200.0, 0.001)
        self.petWindowRange.setValues(max(0.0, low), high)
        self.petWindowRange.blockSignals(False)
        self._applyPetDisplay()

    def _applyPetDisplay(self):
        volume = self.volume()
        if volume is None or self.current != STEP_REVIEW and self.reviewResult is None:
            return
        display = volume.GetDisplayNode()
        if display is None:
            volume.CreateDefaultDisplayNodes()
            display = volume.GetDisplayNode()
        display.SetAndObserveColorNodeID(colormaps.colorNodeForLabel(self.petColorCombo.currentText).GetID())
        display.SetAutoWindowLevel(False)
        low, high = self.petWindowRange.minimumValue, self.petWindowRange.maximumValue
        display.SetWindowLevelMinMax(low, max(high, low + 1e-6))
        grey = getattr(self, "greyPetNode", None)
        if grey is not None and slicer.mrmlScene.IsNodePresent(grey) and grey.GetDisplayNode() is not None:
            grey.GetDisplayNode().SetWindowLevelMinMax(low, max(high, low + 1e-6))
        fused = self._fusionVolume() is not None
        display.SetApplyThreshold(fused)
        if fused:
            display.SetLowerThreshold(low + 0.02 * (high - low))
            display.SetUpperThreshold(float(self.petWindowRange.maximum) * 10.0 + 1.0)

    def _greyPet(self):
        """The PET in grey under the score slices: a second volume node sharing the PET's voxels (no copy), with the
        PET window."""
        pet = self.volume()
        if pet is None:
            return None
        scene = slicer.mrmlScene
        node = getattr(self, "greyPetNode", None)
        if node is None or not scene.IsNodePresent(node):
            node = scene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", scene.GenerateUniqueName("Neuro review PET (grey)"))
            node.SetHideFromEditors(True)
            node.CreateDefaultDisplayNodes()
            self.greyPetNode = node
        if node.GetImageData() is not pet.GetImageData():
            node.SetAndObserveImageData(pet.GetImageData())
        matrix = vtk.vtkMatrix4x4()
        pet.GetIJKToRASMatrix(matrix)
        node.SetIJKToRASMatrix(matrix)
        display = node.GetDisplayNode()
        display.SetAndObserveColorNodeID("vtkMRMLColorTableNodeGrey")
        display.SetAutoWindowLevel(False)
        low, high = self.petWindowRange.minimumValue, self.petWindowRange.maximumValue
        display.SetWindowLevelMinMax(low, max(high, low + 1e-6))
        return node

    def _onFusionChanged(self):
        anatomy = self._fusionVolume()
        if anatomy is not None and getattr(self, "_anatomyWindowFor", None) is not anatomy:
            self._anatomyWindowFor = anatomy
            low, high = _percentiles(anatomy, 1.0, 99.5)
            self.anatomyWindowRange.blockSignals(True)
            self.anatomyWindowRange.minimum = min(low, 0.0)
            self.anatomyWindowRange.maximum = high * 1.5 if high > 0 else high + 1.0
            self.anatomyWindowRange.setValues(low, high)
            self.anatomyWindowRange.blockSignals(False)
            self._applyAnatomyDisplay()
        self._applyPetDisplay()
        self._assignReviewViews()

    def _applyAnatomyDisplay(self):
        anatomy = self._fusionVolume()
        if anatomy is None:
            return
        display = anatomy.GetDisplayNode()
        if display is None:
            anatomy.CreateDefaultDisplayNodes()
            display = anatomy.GetDisplayNode()
        display.SetAndObserveColorNodeID("vtkMRMLColorTableNodeGrey")
        display.SetAutoWindowLevel(False)
        display.SetWindowLevelMinMax(self.anatomyWindowRange.minimumValue, self.anatomyWindowRange.maximumValue)

    # 3D-SSP
    def _buildSspBox(self, layout):
        box = qt.QGroupBox("3D-SSP")
        form = qt.QFormLayout(box)
        self.sspMapCombo = qt.QComboBox()
        for key, title in ssp.MAPS:
            self.sspMapCombo.addItem(title, key)
        self.sspMapCombo.toolTip = ("The score (z / t, as picked above) on the brain surface, or the patient's own "
                                    "projected values")
        form.addRow("Map:", self.sspMapCombo)
        self.sspSourceCombo = qt.QComboBox()
        for key, title in ssp.SOURCES:
            self.sspSourceCombo.addItem(title, key)
        self.sspSourceCombo.toolTip = (
            "Minoshima: the patient's maximum along the inward normal (0 - 20 mm, stopped at the brain's edge) scored against the same projection "
            "of every normal (vertex mean / SD). Voxel: the voxel score map projected (its most extreme score along the "
            "normal); used when the database has no projections of its normals (predefined).")
        form.addRow("Score from:", self.sspSourceCombo)
        self.sspColorCombo = qt.QComboBox()
        self.sspPatientColorCombo = qt.QComboBox()
        for label in nr.SCORE_COLORMAPS:
            self.sspColorCombo.addItem(label)
        for label in colormaps.LABELS:
            self.sspPatientColorCombo.addItem(label)
        self.sspPatientColorCombo.setCurrentIndex(max(0, self.sspPatientColorCombo.findText(REVIEW_SSP_PATIENT_COLORMAP)))
        form.addRow("Score colors:", self.sspColorCombo)
        self.sspMaximumSpin = qt.QDoubleSpinBox()
        self.sspMaximumSpin.setRange(0.5, 50.0)
        self.sspMaximumSpin.decimals = 1
        self.sspMaximumSpin.singleStep = 0.5
        self.sspMaximumSpin.value = nr.DEFAULT_MAXIMUM
        self.sspMaximumSpin.prefix = "\u00b1 "
        self.sspMaximumSpin.toolTip = "Score window of the surface, symmetric about 0"
        form.addRow("Score window:", self.sspMaximumSpin)
        form.addRow("Patient colors:", self.sspPatientColorCombo)
        self.sspWindowRange = ctk.ctkRangeWidget()
        self.sspWindowRange.decimals = 2
        self.sspWindowRange.toolTip = "Display range of the patient's projected values"
        form.addRow("Patient window:", self.sspWindowRange)
        self.sspLightingCheck = qt.QCheckBox("Lighting")
        self.sspLightingCheck.checked = True
        self.sspLightingCheck.toolTip = "Shaded surface; off: flat colors at full brightness (as in the original 3D-SSP)"
        form.addRow(self.sspLightingCheck)
        self.sspLightingCheck.connect("toggled(bool)", lambda on: self._showSsp())
        self.restoreViewsButton = _button("Restore 3D views")
        self.restoreViewsButton.toolTip = "The 3D-SSP views back to their default direction, zoom and position"
        form.addRow(self.restoreViewsButton)
        self.restoreViewsButton.connect("clicked()", lambda: self._setupSspViews(fit=True))
        layout.addWidget(box)
        for combo in (self.sspMapCombo, self.sspSourceCombo, self.sspColorCombo, self.sspPatientColorCombo):
            combo.connect("currentIndexChanged(int)", lambda i: self._showSsp())
        self.sspMaximumSpin.connect("valueChanged(double)", lambda v: self._showSsp())
        self.sspWindowRange.connect("valuesChanged(double,double)", lambda a, b: self._showSsp())
        self.scoreSideCombo.connect("currentIndexChanged(int)", lambda i: self._showSsp())
        self.thresholdSpin.connect("valueChanged(double)", lambda v: self._showSsp())

    def _patientUnits(self):
        """Units of the patient's map compared (and projected): the database's map type, SUVR to the reference
        region without a database, else SUV."""
        result = self.reviewResult or {}
        if not result.get("ratio"):
            return "SUV"
        header = result.get("header")
        return (header or {}).get("mapType") or "SUVR"

    def _initSspWindow(self):
        data = (self.reviewResult or {}).get("ssp")
        if not data:
            return
        index = self.sspMapCombo.findData(ssp.MAP_PATIENT)
        if index >= 0:
            self.sspMapCombo.setItemText(index, f"Patient ({self._patientUnits()}, projected, smoothed)")
        values = np.asarray(data["patient"], dtype=np.float32)
        low, high = (float(v) for v in np.percentile(values, (1.0, 99.5))) if values.size else (0.0, 1.0)
        high = high if high > low else low + 1.0
        self.sspWindowRange.blockSignals(True)
        self.sspWindowRange.minimum = min(0.0, low)
        self.sspWindowRange.maximum = max(float(values.max()), high) if values.size else high
        self.sspWindowRange.singleStep = max(high / 200.0, 0.001)
        self.sspWindowRange.setValues(low, high)
        self.sspWindowRange.blockSignals(False)

    def _sspScores(self):
        """(vertex scores, what) of the 3D-SSP (Minoshima or the voxel map by "Score from"); (None, "") without."""
        result = self.reviewResult
        data = result.get("ssp")
        kind = self._scoreKind()
        n = data.get("n") or result.get("n")
        if self.sspSourceCombo.currentData == ssp.SOURCE_MINOSHIMA and data.get("mean") is not None:
            return ssp.scores(data["patient"], data["mean"], data["sd"], n, kind), f"{kind} (Minoshima)"
        if data.get("voxel") is not None:
            factor = nr.tFactor(result.get("n")) if kind == nr.SCORE_T else 1.0
            return np.asarray(data["voxel"], np.float32) / np.float32(factor), f"{kind} (voxel map)"
        return None, ""

    def _sspValues(self, viewSet=nr.SSP_MAIN):
        """(vertex values, color table RGBA, (low, high), what) of a set of 3D-SSP views: the main one as set in the
        3D-SSP box; the low scores' the scores below the threshold (colors reversed). None for the low scores without
        scores."""
        result = self.reviewResult
        data = result.get("ssp")
        score = what = None
        if self.sspMapCombo.currentData == ssp.MAP_SCORE or viewSet == nr.SSP_LOW:
            score, what = self._sspScores()
        if score is not None:
            side = nr.SIDE_THRESHOLD if viewSet == nr.SSP_LOW else None
            colors, window = self._scoreColors(self.sspColorCombo.currentText, self.sspMaximumSpin.value, side)
            return score, colors, window, what + (f" < {self.thresholdSpin.value:g}" if viewSet == nr.SSP_LOW else "")
        if viewSet == nr.SSP_LOW:
            return None
        low, high = self.sspWindowRange.minimumValue, self.sspWindowRange.maximumValue
        colors = ssp.surfaceColors(self._rgbOf(self.sspPatientColorCombo.currentText), low, high)
        return np.asarray(data["patient"], np.float32), colors, (low, max(high, low + 1e-6)), "patient"

    def _sspColorNode(self, rgba, name=SSP_COLOR_NODE):
        """The color table of a set of 3D-SSP views, written only when its colors change and then in one modification
        (each SetColor alone fires a Modified event that updates every model using the table in every 3D view: 256
        of them made a layout change take tens of seconds)."""
        scene = slicer.mrmlScene
        rgba = np.asarray(rgba, dtype=np.float64)
        node = scene.GetFirstNodeByName(name)
        if node is None or not node.IsA("vtkMRMLColorTableNode"):
            node = scene.CreateNodeByClass("vtkMRMLColorTableNode")
            node.UnRegister(None)
            node.SetName(name)
            node.SetTypeToUser()
            node.SetNumberOfColors(len(rgba))
            scene.AddNode(node)
        written = getattr(self, "_sspColorsWritten", None)
        if written is None:
            written = self._sspColorsWritten = {}
        key = (node.GetID(), rgba.shape, rgba.tobytes())
        if written.get(name) == key:
            return node
        wasModifying = node.StartModify()
        try:
            if node.GetNumberOfColors() != len(rgba):
                node.SetNumberOfColors(len(rgba))
            for i, (r, g, b, a) in enumerate(rgba):
                node.SetColor(i, f"{i}", float(r), float(g), float(b), float(a))
        finally:
            node.EndModify(wasModifying)
        written[name] = key
        return node

    def _sspPolyData(self, viewSet, part, surface, values):
        """The polydata of one part of the surface in one set of views with its vertex values: the geometry built once
        per surface (and the part's vertex indices kept), only the values copied in when they change; (polydata,
        changed)."""
        from vtk.util import numpy_support
        cache = getattr(self, "_sspPolys", None)
        if cache is None:
            cache = self._sspPolys = {}
        found = cache.get((viewSet, part))
        if found is None or found[0] is not surface:
            if part == ssp.PART_BRAIN and surface.get("parts") is None:
                piece, index = surface, None
            else:
                piece, index = ssp.piece(surface, part)
            count = len(piece["points"])
            poly = self.logic.surfacePolyData(piece, np.zeros(count, np.float32))
            found = cache[(viewSet, part)] = (surface, poly, index, None)
        surface, poly, index, last = found
        values = np.asarray(values if index is None else values[index], dtype=np.float32)
        if last is not None and last.shape == values.shape and np.array_equal(last, values):
            return poly, False
        array = poly.GetPointData().GetArray("SSP")
        numpy_support.vtk_to_numpy(array)[:] = values
        array.Modified()
        poly.Modified()
        cache[(viewSet, part)] = (surface, poly, index, values.copy())
        return poly, True

    @staticmethod
    def _setViewNodeIds(display, viewIds):
        """The views of a display node, changed only when they differ (a change redraws the model in every view)."""
        current = [display.GetNthViewNodeID(i) for i in range(display.GetNumberOfViewNodeIDs())]
        if current == list(viewIds):
            return
        wasModifying = display.StartModify()
        display.RemoveAllViewNodeIDs()
        for viewId in viewIds:
            display.AddViewNodeID(viewId)
        display.EndModify(wasModifying)

    def _sspViewNodes(self):
        """{(view set, view key): (3D widget, view node)} of the 3D-SSP views of the layout."""
        layoutManager = slicer.app.layoutManager()
        out = {}
        if layoutManager is None:
            return out
        tags = nr.sspViewTags()
        for index in range(layoutManager.threeDViewCount):
            widget = layoutManager.threeDWidget(index)
            viewNode = widget.mrmlViewNode() if widget is not None else None
            if viewNode is not None and viewNode.GetSingletonTag() in tags:
                out[tags[viewNode.GetSingletonTag()]] = (widget, viewNode)
        return out

    def _sspModelNode(self, viewSet, part):
        """The model of one part of the surface (the brain, or a hemisphere for the medial views) in one set of
        views."""
        scene = slicer.mrmlScene
        node = self.sspModels.get((viewSet, part))
        if node is None or not scene.IsNodePresent(node):
            suffix = {ssp.PART_LEFT: " left hemisphere", ssp.PART_RIGHT: " right hemisphere"}.get(part, "")
            name = SSP_MODEL_NAME + (" low scores" if viewSet else "") + suffix
            node = scene.AddNewNodeByClass("vtkMRMLModelNode", scene.GenerateUniqueName(name))
            node.SetAttribute(SSP_MODEL_ATTRIBUTE, f"{viewSet or 'main'}/{part}")
            node.CreateDefaultDisplayNodes()
            self.sspModels[(viewSet, part)] = node
        return node

    def _dropStaleSspModels(self):
        """The 3D-SSP models in the scene that are not this widget's removed: those of a module reloaded (or an older
        version) stay in their views, drawn inside the new ones (a nested, speckled surface)."""
        mine = {node.GetID() for node in self.sspModels.values() if node is not None}
        for node in list(slicer.util.getNodesByClass("vtkMRMLModelNode")):
            name = node.GetName() or ""
            if (name.startswith(SSP_MODEL_NAME) or node.GetAttribute(SSP_MODEL_ATTRIBUTE)) and node.GetID() not in mine:
                slicer.mrmlScene.RemoveNode(node)

    def _dropSspModel(self, viewSet, part):
        getattr(self, "_sspPolys", {}).pop((viewSet, part), None)
        node = self.sspModels.pop((viewSet, part), None)
        if node is not None and slicer.mrmlScene.IsNodePresent(node):
            slicer.mrmlScene.RemoveNode(node)

    def _showSsp(self):
        result = self.reviewResult
        data = (result or {}).get("ssp")
        if not data or not hasattr(self, "sspMapCombo"):
            return
        self._dropStaleSspModels()
        views = self._sspViewNodes()
        surface = data["surface"]
        for viewSet, colorName in ((nr.SSP_MAIN, SSP_COLOR_NODE), (nr.SSP_LOW, SSP_LOW_COLOR_NODE)):
            shownViews = {key: viewNode for (vs, key), (_, viewNode) in views.items() if vs == viewSet}
            found = self._sspValues(viewSet) if viewSet == nr.SSP_MAIN or shownViews else None
            if found is None:
                for part in (ssp.PART_BRAIN, ssp.PART_LEFT, ssp.PART_RIGHT):
                    self._dropSspModel(viewSet, part)
                continue
            values, rgba, (low, high), what = found
            colorNode = self._sspColorNode(rgba, colorName)
            values = np.asarray(values, np.float32)
            models = []
            for part in (ssp.PART_BRAIN, ssp.PART_LEFT, ssp.PART_RIGHT):
                if not ssp.hasPart(surface, part):
                    self._dropSspModel(viewSet, part)
                    continue
                model = self._sspModelNode(viewSet, part)
                poly, _ = self._sspPolyData(viewSet, part, surface, values)
                if model.GetPolyData() is not poly:
                    model.SetAndObservePolyData(poly)
                display = model.GetDisplayNode()
                display.SetAndObserveColorNodeID(colorNode.GetID())
                display.SetActiveScalarName("SSP")
                display.SetScalarVisibility(True)
                display.SetScalarRangeFlag(slicer.vtkMRMLDisplayNode.UseManualScalarRange)
                display.SetScalarRange(low, high)
                display.SetBackfaceCulling(False)
                display.SetVisibility2D(False)
                if hasattr(display, "SetLighting"):         # off: full brightness, no shading (flat colors)
                    display.SetLighting(bool(self.sspLightingCheck.checked))
                shown = [viewNode.GetID() for key, viewNode in shownViews.items() if ssp.viewPart(key) == part]
                # not in the review layout: hidden (shown in the SSP views only)
                self._setViewNodeIds(display, shown or ["vtkMRMLViewNodeNone"])
                models.append(model)
            if not models:
                continue
            rows = [shownViews[key].GetID() for key in (nr.SSP_VIEWS[0][0], nr.SSP_VIEWS[nr.SSP_COLUMNS][0])
                    if key in shownViews]
            self._legend(models[0], rows, what if what != "patient" else self._patientUnits(), "%.1f")

    def _legend(self, node, viewIds, title, labelFormat):
        """A color legend of node on the left of the views viewIds (hidden with none)."""
        try:
            colorLogic = slicer.modules.colors.logic()
            if not hasattr(colorLogic, "AddDefaultColorLegendDisplayNode"):
                return
            legend = colorLogic.GetColorLegendDisplayNode(node) if hasattr(colorLogic, "GetColorLegendDisplayNode") \
                else None
            if legend is None:
                if not viewIds:
                    return
                legend = colorLogic.AddDefaultColorLegendDisplayNode(node)
            wasModifying = legend.StartModify()
            legend.RemoveAllViewNodeIDs()
            for viewId in viewIds:
                legend.AddViewNodeID(viewId)
            legend.SetVisibility(bool(viewIds) and bool(self.legendCheck.checked))
            legend.SetTitleText(title)
            legend.SetLabelFormat(labelFormat)
            legend.SetPosition(0.02, 0.5)       # left, middle
            legend.SetSize(0.1, 0.6)
            legend.EndModify(wasModifying)
        except Exception:
            logging.debug("Static Neuro: no color legend", exc_info=True)

    def _updateLegends(self):
        """One color legend per row, on the left: the PET, the scores, the 3D-SSP (each in its row's first view)."""
        if not hasattr(self, "legendCheck"):
            return
        first = {}
        for row, widget in self._reviewSliceWidgets():
            first.setdefault(row, widget.mrmlSliceNode().GetID())
        volume = self.volume()
        if volume is not None:
            self._legend(volume, [first[nr.PET_ROW]] if nr.PET_ROW in first else [], self._petUnits(volume), "%.2f")
        if self.scoreNode is not None and slicer.mrmlScene.IsNodePresent(self.scoreNode):
            self._legend(self.scoreNode, [first[nr.SCORE_ROW]] if nr.SCORE_ROW in first else [], self._scoreKind(),
                         "%.1f")

    def _petUnits(self, volume):
        units = self.logic.imageInfo(volume).get("units") or ""
        return "SUV" if not units or "suv" in units.lower() or "g/ml" in units.lower() else units

    def _syncCameras(self, enable=True):
        """Zoom and pan of the 3D-SSP views linked (each its own rotation): camera node observers."""
        for cameraNode, tag in getattr(self, "_cameraObservers", []):
            try:
                cameraNode.RemoveObserver(tag)
            except Exception:
                pass
        self._cameraObservers = []
        if not enable:
            return
        logic = slicer.modules.cameras.logic()
        for key, (_, viewNode) in self._sspViewNodes().items():
            cameraNode = logic.GetViewActiveCameraNode(viewNode)
            if cameraNode is not None:
                tag = cameraNode.AddObserver(vtk.vtkCommand.ModifiedEvent,
                                             lambda caller, event, k=key: self._onCameraMoved(k))
                self._cameraObservers.append((cameraNode, tag))

    def _sspBoundsOf(self):
        """(centre, radius) of the 3D-SSP surface's vertices, worked out once per surface; None without one."""
        data = (self.reviewResult or {}).get("ssp")
        if not data:
            return None
        surface = data["surface"]
        found = getattr(self, "_sspBounds", None)
        if found is None or found[0] is not surface:
            points = np.asarray(surface["points"], dtype=np.float64)
            center = points.mean(axis=0) if len(points) else np.zeros(3)
            radius = float(np.max(np.linalg.norm(points - center, axis=1))) if len(points) else 100.0
            found = self._sspBounds = (surface, center, radius)
        return found[1], found[2]

    def _sspCenter(self):
        bounds = self._sspBoundsOf()
        return bounds[0].copy() if bounds else np.zeros(3)

    def _onCameraMoved(self, sourceKey):
        if getattr(self, "_syncingCameras", False):
            return
        views = self._sspViewNodes()
        if sourceKey not in views:
            return
        logic = slicer.modules.cameras.logic()
        source = logic.GetViewActiveCameraNode(views[sourceKey][1])
        if source is None:
            return
        center = self._sspCenter()
        camera = source.GetCamera()
        pan = ssp.panOffset(center, camera.GetFocalPoint(), camera.GetPosition(), camera.GetViewUp())
        scale = camera.GetParallelScale()
        self._syncingCameras = True
        try:
            for key, (widget, viewNode) in views.items():
                target = logic.GetViewActiveCameraNode(viewNode)
                if key == sourceKey or target is None:
                    continue
                other = target.GetCamera()
                focal, position = ssp.pannedCamera(center, pan, other.GetFocalPoint(), other.GetPosition(),
                                                   other.GetViewUp())
                if (np.allclose(focal, other.GetFocalPoint(), atol=1e-3)
                        and abs(other.GetParallelScale() - scale) < 1e-6):
                    continue
                wasModifying = target.StartModify()
                target.SetFocalPoint(*focal)
                target.SetPosition(*position)
                target.SetParallelScale(scale)
                target.EndModify(wasModifying)
                self._clipSspCamera(widget, target)
        finally:
            self._syncingCameras = False

    def _clipSspCamera(self, widget, cameraNode):
        """The near and far planes of a 3D-SSP camera set to hold the whole surface (with the hemispheres' views
        drawn before anything was rendered in them, the planes were left where they cut the surface: the view black
        or striped until zoomed)."""
        bounds = self._sspBoundsOf()
        camera = cameraNode.GetCamera() if cameraNode is not None else None
        if bounds is None or camera is None:
            return
        center, radius = bounds
        focal = np.asarray(camera.GetFocalPoint(), dtype=np.float64)
        distance = float(np.linalg.norm(np.asarray(camera.GetPosition()) - focal))
        # the surface's bounding sphere seen from the focal point (not every vertex again at each camera change)
        reach = radius + float(np.linalg.norm(focal - center))
        margin = 20.0
        camera.SetClippingRange(max(0.1, distance - reach - margin), distance + reach + margin)
        try:
            widget.threeDView().scheduleRender()
        except AttributeError:
            pass

    def _sspSideMarkers(self, widget, viewNode, key):
        """The R / L markers on the left and right of a superior, inferior, anterior or posterior 3D-SSP view (text
        actors in its renderer, made once per view)."""
        labels = ssp.sideLabels(key)
        markers = getattr(self, "_sspMarkers", None)
        if markers is None:
            markers = self._sspMarkers = {}
        try:
            renderer = widget.threeDView().renderWindow().GetRenderers().GetFirstRenderer()
        except Exception:
            return
        old = markers.get(viewNode.GetID())
        if old is not None and (old[0] is not renderer or old[2] != labels):
            for actor in old[1]:
                old[0].RemoveViewProp(actor)
            old = None
        if old is not None or labels is None or renderer is None:
            return
        actors = []
        for text, x in zip(labels, SSP_SIDE_MARKER_X):
            actor = vtk.vtkTextActor()
            actor.SetInput(text)
            prop = actor.GetTextProperty()
            prop.SetFontSize(20)
            prop.BoldOn()
            prop.SetColor(1.0, 1.0, 1.0)
            prop.SetJustificationToCentered()
            prop.SetVerticalJustificationToCentered()
            actor.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
            actor.SetPosition(x, 0.5)
            renderer.AddViewProp(actor)
            actors.append(actor)
        markers[viewNode.GetID()] = (renderer, actors, labels)

    def _setupSspViews(self, fit=False):
        """The 3D-SSP views: black, no box, parallel projection, a box orientation marker, each camera looking at
        the surface from its side (top and bottom with the front up, the others with the top up; the medial views at
        one hemisphere from the other side), all with the same zoom and pan, and near / far planes holding the whole
        surface."""
        data = (self.reviewResult or {}).get("ssp")
        views = self._sspViewNodes()
        if not views:
            self._syncCameras(False)
            return
        self._showSsp()
        center = self._sspCenter()
        scale = None
        self._syncingCameras = True
        try:
            for key, (widget, viewNode) in views.items():
                widget.threeDController().setVisible(False)
                viewNode.SetBoxVisible(False)
                viewNode.SetAxisLabelsVisible(False)
                viewNode.SetBackgroundColor(0.0, 0.0, 0.0)
                viewNode.SetBackgroundColor2(0.0, 0.0, 0.0)
                viewNode.SetRenderMode(slicer.vtkMRMLViewNode.Orthographic)
                if hasattr(viewNode, "SetOrientationMarkerType"):      # a small box with the sides' letters
                    viewNode.SetOrientationMarkerType(slicer.vtkMRMLAbstractViewNode.OrientationMarkerTypeCube)
                    viewNode.SetOrientationMarkerSize(slicer.vtkMRMLAbstractViewNode.OrientationMarkerSizeSmall)
                if not data or not fit:
                    continue
                direction, up = ssp.CAMERAS[key[1]]
                camera = slicer.modules.cameras.logic().GetViewActiveCameraNode(viewNode)
                if camera is None:
                    continue
                camera.SetFocalPoint(*center)
                camera.SetPosition(*(center + SSP_CAMERA_DISTANCE * np.asarray(direction)))
                camera.SetViewUp(*up)
            if data and fit:
                # one zoom for all, centred: the whole surface's extent across the views (not ResetCamera, which
                # sees nothing in a view not drawn yet and centres each on its own model)
                points = np.asarray(data["surface"]["points"], dtype=np.float64)
                half = 0.5 * (points.max(axis=0) - points.min(axis=0)) if len(points) else np.full(3, 100.0)
                scale = 1.05 * float(max(half[1], half[2], half[0]))
            for key, (widget, viewNode) in views.items():
                camera = slicer.modules.cameras.logic().GetViewActiveCameraNode(viewNode)
                if camera is None:
                    continue
                if scale:
                    direction, _ = ssp.CAMERAS[key[1]]
                    camera.SetFocalPoint(*center)
                    camera.SetPosition(*(center + SSP_CAMERA_DISTANCE * np.asarray(direction)))
                    camera.SetParallelScale(scale)
                self._clipSspCamera(widget, camera)
                self._sspSideMarkers(widget, viewNode, key[1])
                widget.threeDView().forceRender()
        finally:
            self._syncingCameras = False
        self._syncCameras(True)

    # layout and views
    def _reviewLayout(self):
        return self.layoutCombo.currentData if hasattr(self, "layoutCombo") else nr.LAYOUT_SINGLE

    def _onLayoutPicked(self):
        if getattr(self, "current", None) == STEP_REVIEW and self.node("Segmentation") is not None:
            self._enterReviewViews()

    def _registerReviewLayouts(self, layoutManager):
        """The review layouts in Slicer (a description updated only when that layout is not on screen)."""
        layoutNode = layoutManager.layoutLogic().GetLayoutNode()
        attributes = _mainViewportAttributes()
        medial = self._sspMedial()
        for layoutId in nr.LAYOUT_IDS:
            description = nr.layoutXml(layoutId, attributes, medial)
            if not layoutNode.IsLayoutDescription(layoutId):
                layoutNode.AddLayoutDescription(layoutId, description)
            elif layoutNode.GetLayoutDescription(layoutId) != description:
                if layoutManager.layout == layoutId:      # with or without the medial views: left and built again
                    layoutManager.setLayout(self._reviewPreviousLayout
                                            or slicer.vtkMRMLLayoutNode.SlicerLayoutFourUpView)
                    slicer.app.processEvents()
                layoutNode.SetLayoutDescription(layoutId, description)

    def _sspMedial(self):
        """Whether the 3D-SSP has the medial views (the regions with sides, R / L): the layouts' 2 x 4 views, else
        2 x 3."""
        data = (self.reviewResult or {}).get("ssp")
        return ssp.hasMedial(data["surface"]) if data else True

    def _enterReviewViews(self):
        if self.volume() is None:
            return
        layoutManager = slicer.app.layoutManager()
        if layoutManager is None:
            return
        self._registerReviewLayouts(layoutManager)
        wanted = self._reviewLayout()
        if layoutManager.layout != wanted:
            if layoutManager.layout not in nr.LAYOUT_IDS:
                self._reviewPreviousLayout = layoutManager.layout
            layoutManager.setLayout(wanted)
            slicer.app.processEvents()
        if self.reviewResult is None:
            self._initPetWindow()
        self._assignReviewViews(fit=True)
        self._showTable()
        self._hoverNames(True)
        self._watchLayout(True)
        if wanted in nr.DUAL_LAYOUTS:
            qt.QTimer.singleShot(MONITOR2_PLACE_DELAY_MS, self._placeMonitor2)
        else:       # once the new views are drawn: their cameras and planes again
            qt.QTimer.singleShot(SSP_SETUP_DELAY_MS, lambda: self._setupSspViews(fit=True))

    def _monitor2Window(self):
        """The floating window of the 3D-SSP views (dual monitor layout); None when docked or not shown."""
        candidates = [widget.window() for widget, _ in self._sspViewNodes().values()][:1]
        candidates += [w for w in qt.QApplication.topLevelWidgets() if w.windowTitle == nr.MONITOR2_TITLE]
        for window in candidates:
            if window is None or window.inherits("QMainWindow"):
                continue
            if window.inherits("QDockWidget") and not getattr(window, "floating", True):
                continue        # docked into the main window by the user: left there
            return window
        return None

    def _placeMonitor2(self):
        layoutManager = slicer.app.layoutManager()
        if layoutManager is None or layoutManager.layout not in nr.DUAL_LAYOUTS:
            return
        try:
            window = self._monitor2Window()
            if window is not None:
                _showOnSecondScreen(window)
                qt.QTimer.singleShot(3 * MONITOR2_PLACE_DELAY_MS, lambda: self._setupSspViews(fit=True))
        except Exception:
            logging.exception("Static Neuro: could not place the 3D-SSP window; drag it to the second screen")

    # the name of the region under the mouse, on its left
    def _hoverNames(self, enable):
        crosshair = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLCrosshairNode")
        old = getattr(self, "_hoverObserver", None)
        if old is not None:
            try:
                old[0].RemoveObserver(old[1])
            except Exception:
                pass
        self._hoverObserver = None
        self._hideHoverName()
        if enable and crosshair is not None:
            tag = crosshair.AddObserver(slicer.vtkMRMLCrosshairNode.CursorPositionModifiedEvent,
                                        lambda caller, event: self._onCursorMoved(caller))
            self._hoverObserver = (crosshair, tag)

    def _hoverLabel(self, parent):
        label = getattr(self, "_hoverNameLabel", None)
        try:
            if label is not None and label.parent() is not parent:
                label.setParent(parent)
        except Exception:       # deleted with its view
            label = None
        if label is None:
            label = qt.QLabel(parent)
            label.setStyleSheet("QLabel { color: white; background-color: rgba(0, 0, 0, 170); padding: 1px 4px; "
                                "border-radius: 3px; }")
            label.setAttribute(qt.Qt.WA_TransparentForMouseEvents)
            self._hoverNameLabel = label
        return label

    def _hideHoverName(self):
        label = getattr(self, "_hoverNameLabel", None)
        try:
            if label is not None:
                label.hide()
        except Exception:
            self._hoverNameLabel = None

    def _onCursorMoved(self, crosshair):
        """The visible regions at the mouse on a review slice view, named left of the mouse."""
        try:
            xyz = [0.0, 0.0, 0.0]
            sliceNode = crosshair.GetCursorPositionXYZ(xyz)
            segmentation = self.node("Segmentation")
            display = segmentation.GetDisplayNode() if segmentation is not None else None
            names = []
            widget = None
            if sliceNode is not None and display is not None:
                layoutManager = slicer.app.layoutManager()
                widget = layoutManager.sliceWidget(sliceNode.GetLayoutName()) if layoutManager else None
            if widget is not None and any(w is widget for _, w in self._reviewSliceWidgets()):
                ras = [0.0, 0.0, 0.0]
                crosshair.GetCursorPositionRAS(ras)
                result = self.reviewResult or {}
                if result.get("labels") is not None and result.get("segmentation") is segmentation:
                    # the regions of the comparison: named also with the outlines hidden (a region hidden by a click
                    # in the table is not named)
                    labels, regionNames = result["labels"]
                    found = nr.labelAt(labels, _ijkToRas(result["grid"]), ras)
                    _, ids = self._segmentColors()
                    if found and display.GetSegmentVisibility(ids.get(regionNames[found - 1], "")):
                        names.append(regionNames[found - 1])
                elif display.GetVisibility() and display.GetVisibility2DOutline():
                    manager = widget.sliceView().displayableManagerByClassName(
                        "vtkMRMLSegmentationsDisplayableManager2D")
                    ids = vtk.vtkStringArray()
                    if manager is not None:
                        manager.GetVisibleSegmentsForPosition(ras, display, ids)
                    seg = segmentation.GetSegmentation()
                    for i in range(ids.GetNumberOfValues()):
                        segment = seg.GetSegment(ids.GetValue(i))
                        if segment is not None:
                            names.append(segment.GetName())
            if not names:
                self._hideHoverName()
                return
            view = widget.sliceView()
            label = self._hoverLabel(view)
            label.text = ", ".join(names)
            label.adjustSize()
            point = view.mapFromGlobal(qt.QCursor.pos())
            x = point.x() - label.width - 12
            if x < 2:           # no room on the left: on the right
                x = point.x() + 16
            label.move(x, max(2, point.y() - label.height // 2))
            label.show()
            label.raise_()
        except Exception:
            logging.debug("Static Neuro: no region name at the mouse", exc_info=True)
            self._hideHoverName()

    def _watchLayout(self, enable):
        """Observe the layout node: after a view is maximized (double click) and restored, Slicer builds the layout
        again; the review's view settings (no title bars, the embedded region table and its width, the 3D-SSP views,
        the legends) are then applied again."""
        old = getattr(self, "_layoutObserver", None)
        if old is not None:
            try:
                old[0].RemoveObserver(old[1])
            except Exception:
                pass
        self._layoutObserver = None
        layoutManager = slicer.app.layoutManager()
        if not enable or layoutManager is None:
            return
        layoutNode = layoutManager.layoutLogic().GetLayoutNode()
        tag = layoutNode.AddObserver(vtk.vtkCommand.ModifiedEvent, lambda caller, event: self._onLayoutModified())
        self._layoutObserver = (layoutNode, tag)

    def _onLayoutModified(self):
        if getattr(self, "_layoutRefreshPending", False):
            return
        self._layoutRefreshPending = True
        qt.QTimer.singleShot(200, self._afterLayoutModified)

    def _afterLayoutModified(self):
        self._layoutRefreshPending = False
        layoutManager = slicer.app.layoutManager()
        if (getattr(self, "current", None) != STEP_REVIEW or layoutManager is None
                or layoutManager.layout not in nr.LAYOUT_IDS):
            return
        try:
            self._assignReviewViews()
            self._showTable()
        except Exception:
            logging.exception("Static Neuro: could not restore the review views")

    def _exitReviewViews(self):
        self._watchLayout(False)
        self._hoverNames(False)
        self._endFlash()
        self._syncCameras(False)
        self._restoreSegmentationDisplay()
        layoutManager = slicer.app.layoutManager()
        if layoutManager is not None and layoutManager.layout in nr.LAYOUT_IDS and self._reviewPreviousLayout is not None:
            layoutManager.setLayout(self._reviewPreviousLayout)
        self._reviewPreviousLayout = None

    def _reviewSliceWidgets(self):
        layoutManager = slicer.app.layoutManager()
        out = []
        for row in (nr.PET_ROW, nr.SCORE_ROW):
            for orientation, _ in nr.ORIENTATIONS:
                widget = layoutManager.sliceWidget(nr.sliceViewName(row, orientation)) if layoutManager else None
                if widget is not None:
                    out.append((row, widget))
        return out

    def _assignReviewViews(self, fit=False):
        layoutManager = slicer.app.layoutManager()
        if layoutManager is None or layoutManager.layout not in nr.LAYOUT_IDS:
            return
        pet = self.volume()
        anatomy = self._fusionVolume()
        score = self.scoreNode if self.scoreNode is not None and slicer.mrmlScene.IsNodePresent(self.scoreNode) \
            else None
        sliceViews = []
        for row, widget in self._reviewSliceWidgets():
            composite = widget.mrmlSliceCompositeNode()
            composite.SetLinkedControl(False)
            top = pet if row == nr.PET_ROW else score
            colored = self._scoreColorsNode() if row == nr.SCORE_ROW and score is not None else None
            if colored is not None and anatomy is not None:     # the colored PET over the fusion image
                background, foreground, opacity = anatomy, colored, float(self.scoreOpacity.value)
            elif colored is not None:   # the colored PET; the scores on top, invisible (data probe, legend)
                background, foreground, opacity = colored, score, 0.0
            elif anatomy is not None:
                background, foreground = anatomy, top
                opacity = float((self.fusionOpacity if row == nr.PET_ROW else self.scoreOpacity).value)
            elif row == nr.SCORE_ROW and self.scoreBackgroundCombo.currentData == "pet" and score is not None:
                background, foreground, opacity = self._greyPet(), score, 1.0     # the scores over the PET in grey
            else:
                background, foreground, opacity = top, None, 0.0
            composite.SetBackgroundVolumeID(background.GetID() if background is not None else None)
            composite.SetForegroundVolumeID(foreground.GetID() if foreground is not None else None)
            composite.SetForegroundOpacity(opacity)
            composite.SetLabelVolumeID(None)
            composite.SetLinkedControl(True)        # all the review's slice views linked
            widget.sliceController().setVisible(False)
            sliceViews.append(widget.mrmlSliceNode().GetID())
            if fit:
                widget.sliceLogic().FitSliceToAll()
        self._outlineSegmentation(sliceViews)
        self._setupSspViews(fit)
        self._updateLegends()

    def _outlineSegmentation(self, viewIds):
        """The regions as outlines (no fill, REVIEW_OUTLINE_THICKNESS px) on the PET and the score slices, hidden with
        the outline check box; the earlier display restored on leaving."""
        segmentation = self.node("Segmentation")
        display = segmentation.GetDisplayNode() if segmentation is not None else None
        if display is None or not viewIds:
            return
        if self._segmentationDisplay is None or self._segmentationDisplay[0] is not display:
            self._restoreSegmentationDisplay()
            self._segmentationDisplay = (display, [display.GetNthViewNodeID(i) for i in
                                                   range(display.GetNumberOfViewNodeIDs())],
                                         display.GetVisibility2DFill(), display.GetVisibility2DOutline(),
                                         display.GetSliceIntersectionThickness())
        wasModifying = display.StartModify()
        display.SetVisibility(True)
        display.SetVisibility2DFill(False)
        display.SetVisibility2DOutline(bool(self.outlineCheck.checked) or self._flash is not None)
        if hasattr(display, "SetSliceIntersectionThickness"):
            display.SetSliceIntersectionThickness(REVIEW_OUTLINE_THICKNESS)
        display.RemoveAllViewNodeIDs()
        for viewId in viewIds:
            display.AddViewNodeID(viewId)
        display.EndModify(wasModifying)

    def _restoreSegmentationDisplay(self):
        saved, self._segmentationDisplay = self._segmentationDisplay, None
        if saved is None:
            return
        display, viewIds, fill, outline, thickness = saved
        if not slicer.mrmlScene.IsNodePresent(display):
            return
        display.RemoveAllViewNodeIDs()
        for viewId in viewIds:
            display.AddViewNodeID(viewId)
        display.SetVisibility2DFill(fill)
        display.SetVisibility2DOutline(outline)
        display.SetSliceIntersectionThickness(thickness)

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
        self._updateReviewInfo(volume, segmentation, header, age, sex)
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
        if key == STEP_REVIEW:
            if method == "exit":
                self._exitReviewViews()
            elif method == "enter" and self.node("Segmentation") is not None:
                self._enterReviewViews()
                qt.QTimer.singleShot(200, self._autoCompare)
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
        self._updatePatientLabel()
        self._updateReviewTools()

    def cleanup(self):
        self._closeRegSession()
        if getattr(self, "current", None) == STEP_REVIEW:
            self._exitReviewViews()
        DynamicNeuroPETWidget.cleanup(self)

    def onSceneClose(self, caller, event):
        DynamicNeuroPETWidget.onSceneClose(self, caller, event)
        self._closeRegSession()
        self.regResult = None
        self.regUndo = None
        self._databaseKey = None
        self._patientTexts = {}
        self._contrastFor = None
        self.reviewResult = self.scoreNode = self.reviewTable = self.greyPetNode = None
        self.sspModels = {}
        self._sspPolys = {}
        self._sspColorsWritten = {}
        self._sspBounds = None
        self.scoreRgbNode = None
        self._cameraObservers = []
        self._segmentationDisplay = None
        self._anatomyWindowFor = None
        self._flash = self._autoCompared = None
        self._regionTableData = None
        if getattr(self, "regionTable", None) is not None:
            try:
                self.regionTable.setRowCount(0)
            except Exception:
                self.regionTable = None
        if hasattr(self, "compareLabel"):
            self.compareLabel.text = ""
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
        self.setUp()
        self.test_Review()

    def test_Review(self):
        """The comparison with a (predefined, SUV) database: voxel z, regional values (approximate)."""
        import tempfile
        self.delayDisplay("Static Neuro: review")
        logic = StaticNeuroPETLogic()
        patient = np.full((10, 10, 10), 5.0, dtype=np.float32)
        patient[:, :, :5] = 3.0                         # region A below the normals
        volume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "patient")
        slicer.util.updateVolumeFromArray(volume, patient)
        labels = np.zeros((10, 10, 10), dtype=np.int16)
        labels[2:8, 2:8, 1:5] = 1
        labels[2:8, 2:8, 5:9] = 2
        labels[0, 0, 0] = 3
        labelNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode", "labels")
        slicer.util.updateVolumeFromArray(labelNode, labels)
        segmentation = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", "regions")
        slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(labelNode, segmentation)
        ids = logic.segmentIds(segmentation)
        for segmentId, name in zip(ids, ("A", "B", "Optic Chiasm")):
            segmentation.GetSegmentation().GetSegment(segmentId).SetName(name)
        self.assertEqual(logic.dropSegments(segmentation), ["Optic Chiasm"])
        # without a database: the SUV of the regions
        plain = logic.reviewData(volume, segmentation)
        self.assertAlmostEqual(plain["suv"]["A"], 3.0, places=4)
        self.assertIsNone(plain["z"])
        # a predefined SUV database: mean 5, SD 1
        root = tempfile.mkdtemp()
        header = normaldb.newHeader("review test", kind=normaldb.PREDEFINED, nuclide="F-18", tracer="FDG",
                                    mapType=normaldb.MAP_SUV)
        folder = normaldb.createDatabase(root, header)
        mapNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "map")
        for which, value in (("mean", 5.0), ("sd", 1.0)):
            slicer.util.updateVolumeFromArray(mapNode, np.full((10, 10, 10), value, dtype=np.float32))
            self.assertTrue(slicer.util.saveNode(mapNode, os.path.join(folder, normaldb.statsFileName(
                normaldb.MASTER, which, ".nrrd"))))
        header["statistics"] = {"computed": normaldb.now(), "stale": False, "groups": {normaldb.MASTER: {
            "name": normaldb.MASTER_TITLE, "n": 10, "mean": normaldb.statsFileName(normaldb.MASTER, "mean", ".nrrd"),
            "sd": normaldb.statsFileName(normaldb.MASTER, "sd", ".nrrd")}}}
        normaldb.writeHeader(folder, header)
        result = logic.reviewData(volume, segmentation, folder, normaldb.readHeader(folder))
        z = result["z"]
        self.assertAlmostEqual(float(z[5, 5, 2]), -2.0, places=3)       # (3 - 5) / 1
        self.assertAlmostEqual(float(z[5, 5, 7]), 0.0, places=3)
        self.assertTrue(nr.uptakeMask(result["mean"], result["mask"])[5, 5, 2])     # uniform normals: all shown
        self.assertAlmostEqual(result["reference"]["A"][0], 5.0, places=4)
        self.assertTrue(any("approximated" in note for note in result["notes"]))
        rows = nr.tableRows(result["values"], result["suv"], result["reference"], nr.SCORE_Z)
        self.assertAlmostEqual(dict((r[0], r[-1]) for r in rows)["A"], -2.0, places=3)
        # 3D-SSP: the surface of the regions, the voxel scores projected (no normals' projections: predefined)
        data = logic.reviewSsp(result, folder, normaldb.readHeader(folder))
        self.assertGreater(len(data["surface"]["points"]), 10)
        self.assertIsNone(data["mean"])
        self.assertLess(float(np.min(data["voxel"])), -1.5)
        self.assertAlmostEqual(float(np.max(data["patient"])), 5.0, places=2)
        poly = logic.surfacePolyData(data["surface"], data["voxel"])
        self.assertEqual(poly.GetNumberOfPoints(), len(data["surface"]["points"]))
        self.delayDisplay("Review passed")

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
        # no CT/MR: Next from the data goes to the normalization (and Back to the data)
        widget.setNode("Anatomy", None)
        widget.showStep(STEP_DATA)
        widget.moveStep(1)
        self.assertEqual(widget.current, STEP_NORM)
        widget.moveStep(-1)
        self.assertEqual(widget.current, STEP_DATA)
        # the patient's name and ID: the BIDS subject, then the values typed on the Data page
        node.SetAttribute(sn.STATIC_PREFIX + "SubjectID", "7")
        self.assertEqual(StaticNeuroPETLogic.patientNameAndId(node), ("", "sub-7"))
        StaticNeuroPETLogic.setPatientNameAndId(node, " DOE JOHN ", "123")
        self.assertEqual(StaticNeuroPETLogic.patientNameAndId(node), ("DOE JOHN", "123"))
        self.assertEqual(NORM_HARDEN_GRIDS[0][1], reg.OUTPUT_FIXED_GRID)
        self.assertEqual(NORM_HARDEN_ACTIONS[0][1], reg.OVERWRITE)
        self.assertFalse(widget._anatomyIsMr())
        for key, _, _, _ in STEPS:
            self.assertFalse(_stepIcon(key, STATE_READY, 1).isNull())
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
