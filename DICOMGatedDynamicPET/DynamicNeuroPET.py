"""
Dynamic Neuro Receptor PET (3D Slicer module) - workflow app

A step-by-step workflow for dynamic receptor PET of the brain, built on the other PET4D modules:

    1. Data                 the patient's dynamic PET series and its CT / MR (the MR contrast, T1 / T2 / PD, is read
                            from the DICOM tags)
    2. Motion correction    the Dynamic PET Motion Correction panel, on the series of step 1
    3. CT/MR registration   the Dynamic PET to CT/MR Registration panel, the series to the CT / MR
    4. Normalization        to the MNI ICBM 152 2009c template (BRAINSFit, affine then B-spline): the PET registered
                            to the template (default), or the CT / MR normalized and the same transform applied to the
                            PET. Checked in the registration layout (PET / fusion / template, checkerboard, manual
                            correction), then hardened into a new series in the template space.
    5. Regions              the CerebrA atlas (Manera et al. 2020) as a segmentation with the structure names: the
                            regions from the left / right atlas ('Putamen R', 'Putamen L') or the bilateral one
                            ('Putamen'), the reference region from either (default: the bilateral cerebellar grey
                            matter). The bilateral atlas is made on first use and saved next to CerebrA.
    6. Modeling             the Dynamic PET Modeling panel, ready with the normalized series, the regions, SRTM and
                            the cerebellar grey matter as the reference

Steps 2, 3 and 6 show the panels of those modules inside this one; the series each step makes (a motion corrected or
registered series saved separately) becomes the series of the next step. The atlases are not shipped with the
extension: pick the folder that holds them (searched with its subfolders, kept in the application settings).
"""

import logging
import os
import sys

import ctk
import numpy as np
import qt
import vtk
import slicer
from slicer.ScriptedLoadableModule import (ScriptedLoadableModule, ScriptedLoadableModuleWidget,
                                           ScriptedLoadableModuleLogic, ScriptedLoadableModuleTest)
from slicer.util import VTKObservationMixin

_MODULE_DIR = os.path.dirname(os.path.abspath(globals().get("__file__", "") or "."))
if globals().get("__file__") and _MODULE_DIR not in sys.path:
    sys.path.insert(0, _MODULE_DIR)

from DICOMGatedDynamicPETLib import neuro  # noqa: E402
from DICOMGatedDynamicPETLib import registration as reg  # noqa: E402
from DICOMGatedDynamicPETLib import resampling  # noqa: E402
from DICOMGatedDynamicPETLib.hardendialog import hardenDialog  # noqa: E402

PREFIX = "DynamicPET."
REG_PREFIX = "DynamicPET.Registration."
MC_PREFIX = "DynamicPET.MotionCorrection."
NEURO_PREFIX = "DynamicPET.Neuro."
ATLAS_FILE_ATTRIBUTE = NEURO_PREFIX + "AtlasFile"
SETTINGS_ATLAS_FOLDER = "DICOMGatedDynamicPET/Neuro/AtlasFolder"
NORMALIZED_SUFFIX = " MNI"
REGIONS_NAME = "CerebrA regions"
REFERENCE_SUFFIX = " (reference)"
ATLAS_LATERAL, ATLAS_BILATERAL = "lateral", "bilateral"
ATLAS_KINDS = ((ATLAS_LATERAL, "CerebrA, left and right separate (Putamen R / L)"),
               (ATLAS_BILATERAL, "CerebrA bilateral, sides merged (Putamen)"))
DEFAULT_TARGET = "Putamen"
DEFAULT_MODEL = "SRTM"

METHOD_PET, METHOD_ANATOMY = "pet", "anatomy"
METHODS = ((METHOD_PET, "Register the PET to the template (default)"),
           (METHOD_ANATOMY, "Normalize the CT/MR, apply the same transform to the PET"))
NORMALIZATION_STEPS = ("Affine", "BSpline")
NORMALIZATION_INITIALIZATION = {METHOD_PET: "useMoments", METHOD_ANATOMY: "useGeometryAlign"}
# Harden: the template's field of view with the PET voxel size (default), the template grid (1 mm), or custom
NORMALIZATION_GRIDS = (("Keep the PET voxel size (over the template field of view)", reg.OUTPUT_PET_SPACING),
                       ("Resample to the template grid (1 mm)", reg.OUTPUT_FIXED_GRID),
                       ("Resample to this voxel size:", reg.OUTPUT_CUSTOM))

STEP_DATA, STEP_MC, STEP_REG, STEP_NORM, STEP_REGIONS, STEP_MODEL = "data", "mc", "reg", "norm", "regions", "model"
# (key, embedded module or None, title, explanation)
STEPS = (
    (STEP_DATA, None, "Data", "Dynamic PET series and the patient's CT / MR"),
    (STEP_MC, "DynamicPETMotionCorrection", "Motion correction", "Frame-by-frame motion correction"),
    (STEP_REG, "DynamicPETRegistration", "CT/MR registration", "The series registered to the CT / MR"),
    (STEP_NORM, None, "Normalization", "To the MNI ICBM 152 2009c template"),
    (STEP_REGIONS, None, "Regions", "CerebrA atlas regions with their names"),
    (STEP_MODEL, "DynamicPETModeling", "Modeling", "Kinetic modelling of the regions (SRTM, Logan, ...)"),
)
STATE_DONE, STATE_READY, STATE_WAITING = "done", "ready", "waiting"
STATE_COLORS = {STATE_DONE: "#2e9e5b", STATE_READY: "#2f7ed8", STATE_WAITING: "#8f969c"}
BUTTON_STYLE = ("QPushButton {{ background-color: {0}; color: white; font-weight: bold; border: none; "
                "border-radius: 4px; padding: 5px 10px; }} QPushButton:disabled {{ background-color: #9aa1a7; }}")


class DynamicNeuroPET(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        parent.title = "Dynamic Neuro Receptor PET"
        parent.categories = ["Nuclear Medicine.PET4D"]
        parent.dependencies = []
        parent.contributors = ["Burak Demir"]
        parent.helpText = (
            "Workflow for dynamic receptor PET of the brain: <b>1. Data</b> (dynamic series and CT/MR), "
            "<b>2. Motion correction</b>, <b>3. CT/MR registration</b>, <b>4. Normalization</b> to the MNI ICBM 152 "
            "2009c template, <b>5. Regions</b> (CerebrA atlas) and <b>6. Modeling</b> (SRTM with the cerebellar grey "
            "matter as the reference by default). Steps 2, 3 and 6 show the panels of the PET4D modules. The atlases "
            "are not shipped: pick the folder that holds them in step 4.")
        parent.acknowledgementText = (
            "Part of the DICOM Gated / Dynamic PET extension. MNI ICBM 152 nonlinear symmetric 2009c template: Fonov "
            "VS et al., NeuroImage 2011;54:313-327 and NeuroImage 2009;47:S102 (Copyright (C) 1993-2004 Louis "
            "Collins, McConnell Brain Imaging Centre, Montreal Neurological Institute, McGill University). CerebrA: "
            "Manera AL et al., Scientific Data 2020;7:237 (CC0). Registration by BRAINSFit.")


def _vtkMatrix(array):
    return resampling.vtkMatrix(array)


def _ijkToRas(node):
    matrix = vtk.vtkMatrix4x4()
    node.GetIJKToRASMatrix(matrix)
    return resampling.numpyMatrix(matrix)


def _percentiles(node, low=1.0, high=99.5):
    array = slicer.util.arrayFromVolume(node)
    if array is None or array.size == 0:
        return 0.0, 1.0
    sample = array[::2, ::2, ::2]
    sample = sample[np.isfinite(sample)]
    if sample.size == 0:
        return 0.0, 1.0
    lower, upper = (float(v) for v in np.percentile(sample, (low, high)))
    return (lower, upper) if upper > lower else (lower, lower + 1.0)


# ---------------------------------------------------------------------------
# Logic
# ---------------------------------------------------------------------------
def defaultReferenceName(entries):
    """The cerebellar grey matter of an atlas version: the bilateral region, else the first side."""
    names = [n for _, n, _ in entries]
    for name in names:
        if name.lower() == neuro.REFERENCE_STRUCTURE.lower():
            return name
    for name in names:
        if name.lower().startswith(neuro.REFERENCE_STRUCTURE.lower()):
            return name
    return names[0] if names else None


class DynamicNeuroPETLogic(ScriptedLoadableModuleLogic):
    """Normalization and regions of the Neuro workflow; usable without the widget:

        logic = DynamicNeuroPETLogic()
        atlas = neuro.findAtlasFiles(folder)
        template = logic.loadAtlasVolume(atlas["T1"])
        result = logic.normalize(sequence, template, METHOD_PET)
        ... (a linear transform node with the chain: logic.chainNodes(result)) ...
        segmentation, referenceId = logic.loadAtlasRegions(atlas)      # makes the bilateral CerebrA on first use
    """

    # --- atlas -------------------------------------------------------------------------------------
    @staticmethod
    def atlasFolder():
        return str(qt.QSettings().value(SETTINGS_ATLAS_FOLDER) or "")

    @staticmethod
    def setAtlasFolder(folder):
        qt.QSettings().setValue(SETTINGS_ATLAS_FOLDER, folder or "")

    @staticmethod
    def loadAtlasVolume(path, labelmap=False, name=None):
        """The atlas file as a volume (label map with labelmap), reused when it is already in the scene."""
        className = "vtkMRMLLabelMapVolumeNode" if labelmap else "vtkMRMLScalarVolumeNode"
        for node in slicer.util.getNodesByClass(className):
            if node.GetAttribute(ATLAS_FILE_ATTRIBUTE) == path:
                return node
        properties = {"show": False, "labelmap": bool(labelmap)}
        if name:
            properties["name"] = name
        node = slicer.util.loadVolume(path, properties)
        if node is None:
            raise RuntimeError(f"Could not load {path}")
        node.SetAttribute(ATLAS_FILE_ATTRIBUTE, path)
        return node

    # --- anatomy -----------------------------------------------------------------------------------
    @staticmethod
    def dicomValues(node, tags):
        """{tag: value} from the DICOM file of the volume's first instance ('' when not found)."""
        out = {t: "" for t in tags}
        try:
            uids = (node.GetAttribute("DICOM.instanceUIDs") or "").split()
            if not uids or slicer.dicomDatabase is None:
                return out
            path = slicer.dicomDatabase.fileForInstance(uids[0])
            if not path:
                return out
            for tag in tags:
                out[tag] = slicer.dicomDatabase.fileValue(path, tag) or ""
        except Exception:
            logging.debug("Neuro: could not read the DICOM tags", exc_info=True)
        return out

    @classmethod
    def anatomyContrast(cls, node):
        """(contrast "CT" / "T1" / "T2" / "PD" / None, reason) of the patient's CT / MR volume."""
        if node is None:
            return None, "no CT/MR"
        values = cls.dicomValues(node, ("0008,0060", "0008,103e", "0018,1030", "0018,0024", "0018,0080",
                                        "0018,0081", "0018,0082"))
        modality = values["0008,0060"]
        if not modality:
            array = slicer.util.arrayFromVolume(node)
            if array is not None and array.size and float(np.min(array[::4, ::4, ::4])) < -500.0:
                modality = "CT"
        return neuro.classifyAnatomy(modality, (values["0008,103e"], values["0018,1030"], values["0018,0024"],
                                                node.GetName()),
                                     values["0018,0080"], values["0018,0081"], values["0018,0082"])

    # --- normalization -----------------------------------------------------------------------------
    @staticmethod
    def referenceImage(sequence):
        """The PET image registered to the template: the summed image of the series, else the frame of the highest
        activity."""
        from DynamicPETRegistration import DynamicPETRegistrationLogic
        summed = DynamicPETRegistrationLogic.summedVolumeOf(sequence)
        if summed is not None:
            return summed
        from DynamicPETMotionCorrection import DynamicPETMotionCorrectionLogic
        from DICOMGatedDynamicPETLib import motioncorrection as mc
        activities = DynamicPETMotionCorrectionLogic.frameActivities(sequence)
        return sequence.GetNthDataNode(mc.maxActivityFrame(activities))

    @staticmethod
    def defaultSettings(method=METHOD_PET):
        from DynamicPETRegistration import DynamicPETRegistrationLogic
        settings = DynamicPETRegistrationLogic.defaultSettings()
        settings["steps"] = list(NORMALIZATION_STEPS)
        settings["resample"] = reg.resampleFlags(NORMALIZATION_STEPS)
        settings["initialization"] = NORMALIZATION_INITIALIZATION[method]
        return settings

    def normalize(self, sequence, template, method=METHOD_PET, anatomy=None, settings=None, mask=None,
                  progress=None):
        """Register the PET reference image (METHOD_PET) or the CT/MR (METHOD_ANATOMY) to the template; the result
        of DynamicPETRegistrationLogic.register() (linear matrix, B-spline node, ...) plus "moving"."""
        from DynamicPETRegistration import DynamicPETRegistrationLogic
        settings = dict(settings or self.defaultSettings(method))
        if method == METHOD_ANATOMY:
            if anatomy is None:
                raise ValueError("Pick the CT/MR in step 1 to normalize it.")
            moving = anatomy
        else:
            moving = self.referenceImage(sequence)
        if moving.GetParentTransformNode() is not None:
            raise ValueError(f"'{moving.GetName()}' is under a transform. Harden or remove it first.")
        fixed = self.brainOnly(template, mask) if mask is not None else template
        try:
            result = DynamicPETRegistrationLogic().register(fixed, moving, settings, progress)
        finally:
            if fixed is not template:
                slicer.mrmlScene.RemoveNode(fixed)
        result["moving"] = moving
        result["masked"] = mask is not None
        return result

    @staticmethod
    def brainOnly(template, mask):
        """A hidden copy of the template with the voxels outside the brain mask set to 0 (skull stripped), on the
        template's grid: the registration then fits the brain, not the scalp and the neck."""
        from DynamicPETRegistration import DynamicPETRegistrationLogic
        array = slicer.util.arrayFromVolume(template).astype(np.float32)
        maskArray = slicer.util.arrayFromVolume(mask)
        if maskArray.shape != array.shape:
            raise ValueError(f"The brain mask '{mask.GetName()}' is not on the grid of '{template.GetName()}'.")
        helper = DynamicPETRegistrationLogic.helperVolume(f"{template.GetName()} brain")
        helper.SetIJKToRASMatrix(_vtkMatrix(_ijkToRas(template)))
        slicer.util.updateVolumeFromArray(helper, np.where(maskArray > 0.5, array, 0.0).astype(np.float32))
        return helper

    @staticmethod
    def petInAnatomySpace(sequence, anatomy):
        """Whether the series was registered (hardened) to this CT/MR (step 3)."""
        return bool(sequence and anatomy and sequence.GetAttribute(REG_PREFIX + "Fixed") == anatomy.GetName())

    def normalizedAnatomy(self, anatomy, template, linearNode, interpolation="Linear"):
        """A new volume: the CT/MR resampled through the normalization chain onto the template grid."""
        from DynamicPETRegistration import DynamicPETRegistrationLogic
        worldToMoving, linear = DynamicPETRegistrationLogic.worldToPet(linearNode)
        dims = template.GetImageData().GetDimensions()
        resampler = resampling.Resampler(_ijkToRas(anatomy), dims, _ijkToRas(template), worldToMoving, linear,
                                         interpolation)
        scene = slicer.mrmlScene
        node = scene.AddNewNodeByClass("vtkMRMLScalarVolumeNode",
                                       scene.GenerateUniqueName(anatomy.GetName() + NORMALIZED_SUFFIX))
        node.SetAndObserveImageData(resampler.resample(anatomy.GetImageData()))
        node.SetIJKToRASMatrix(_vtkMatrix(_ijkToRas(template)))
        node.SetAttribute(NEURO_PREFIX + "Template", template.GetName())
        node.SetAttribute(NEURO_PREFIX + "Source", anatomy.GetName())
        node.CreateDefaultDisplayNodes()
        if anatomy.GetDisplayNode() is not None:
            node.GetDisplayNode().CopyContent(anatomy.GetDisplayNode())
        return node

    def harden(self, sequence, template, linearNode, action, grid, spacing=None, interpolation="Linear",
               description="", method=METHOD_PET, anatomy=None):
        """Resample the series (and its summed image) through the normalization chain onto the template field of
        view (DynamicPETRegistrationLogic.harden); with METHOD_ANATOMY the CT/MR is normalized too. Returns (undo,
        sequence, summed, normalized CT/MR or None)."""
        from DynamicPETRegistration import DynamicPETRegistrationLogic
        regLogic = DynamicPETRegistrationLogic()
        summed = regLogic.summedVolumeOf(sequence)
        undo, newSequence, newSummed = regLogic.harden(sequence, summed, template, linearNode, action, grid, spacing,
                                                       interpolation, description)
        for node in (newSequence, newSummed):
            if node is not None:
                node.SetAttribute(NEURO_PREFIX + "Template", template.GetName())
                node.SetAttribute(NEURO_PREFIX + "Method", method)
        if action == reg.SAVE_SEPARATELY:
            name = sequence.GetName() + NORMALIZED_SUFFIX
            newSequence.SetName(slicer.mrmlScene.GenerateUniqueName(name))
            if newSummed is not None and summed is not None:
                newSummed.SetName(slicer.mrmlScene.GenerateUniqueName(summed.GetName() + NORMALIZED_SUFFIX))
        normalized = None
        if method == METHOD_ANATOMY and anatomy is not None:
            try:
                normalized = self.normalizedAnatomy(anatomy, template, linearNode)
            except Exception:
                logging.exception("Neuro: could not normalize the CT/MR")
        return undo, newSequence, newSummed, normalized

    # --- regions -----------------------------------------------------------------------------------
    @staticmethod
    def _loadLabels(path):
        """(label array (int), hidden label map node) of an atlas file; the caller removes the node."""
        node = slicer.util.loadVolume(path, {"show": False, "labelmap": True, "name": "CerebrA labels (loading)"})
        if node is None:
            raise RuntimeError(f"Could not load {path}")
        return np.rint(slicer.util.arrayFromVolume(node)).astype(np.int32), node

    def ensureBilateral(self, atlas):
        """The bilateral CerebrA (left and right merged): (labels path, CSV path). Made on first use from the CerebrA
        files and saved next to them; when that folder cannot be written, in Slicer's temporary folder (made again
        next time)."""
        labelsPath, csvPath = neuro.bilateralPaths(atlas)
        if labelsPath is None or neuro.CEREBRA_CSV not in atlas:
            raise ValueError("The CerebrA files (mni_icbm152_CerebrA_tal_nlin_sym_09c.nii, CerebrA_LabelDetails.csv) "
                             "are not in the atlas folder.")
        if os.path.isfile(labelsPath) and os.path.isfile(csvPath):
            return labelsPath, csvPath
        structures = neuro.readCerebraLabels(atlas[neuro.CEREBRA_CSV])
        _, mapping = neuro.labelTable(structures, mergeSides=True)
        array, source = self._loadLabels(atlas[neuro.CEREBRA_LABELS])
        scene = slicer.mrmlScene
        merged = scene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode", "CerebrA bilateral")
        try:
            merged.SetIJKToRASMatrix(_vtkMatrix(_ijkToRas(source)))
            slicer.util.updateVolumeFromArray(merged, neuro.remapLabels(array, mapping).astype(np.int16))
            text = neuro.bilateralCsvText(structures)
            for folder in (os.path.dirname(labelsPath), slicer.app.temporaryPath):
                labels = os.path.join(folder, os.path.basename(labelsPath))
                table = os.path.join(folder, os.path.basename(csvPath))
                try:
                    with open(table, "w", newline="", encoding="utf-8") as f:
                        f.write(text)
                    if not slicer.util.saveNode(merged, labels):
                        raise OSError(f"could not write {labels}")
                    logging.info(f"Neuro: bilateral CerebrA saved as {labels} and {table}")
                    return labels, table
                except OSError:
                    logging.warning(f"Neuro: could not save the bilateral CerebrA in {folder}", exc_info=True)
            raise RuntimeError("Could not save the bilateral CerebrA files.")
        finally:
            for node in (source, merged):
                if scene.IsNodePresent(node):
                    scene.RemoveNode(node)

    @staticmethod
    def atlasEntries(atlas, kind):
        """[(label value, segment name, (r, g, b))] of a CerebrA version: ATLAS_LATERAL ('Putamen R', 'Putamen L')
        or ATLAS_BILATERAL ('Putamen'; from the bilateral CSV when it exists, else from the CerebrA CSV)."""
        if kind == ATLAS_BILATERAL and neuro.BILATERAL_CSV in atlas and os.path.isfile(atlas[neuro.BILATERAL_CSV]):
            return neuro.readBilateralLabels(atlas[neuro.BILATERAL_CSV])
        structures = neuro.readCerebraLabels(atlas[neuro.CEREBRA_CSV])
        return neuro.labelTable(structures, mergeSides=kind == ATLAS_BILATERAL)[0]

    def atlasPaths(self, atlas, kind):
        """(labels path, entries) of a CerebrA version; the bilateral one is made first when it is missing."""
        if kind == ATLAS_BILATERAL:
            labelsPath, csvPath = self.ensureBilateral(atlas)
            return labelsPath, neuro.readBilateralLabels(csvPath)
        return atlas[neuro.CEREBRA_LABELS], self.atlasEntries(atlas, ATLAS_LATERAL)

    def loadRegions(self, labelsPath, entries, referencePath=None, referenceName=None, referenceEntries=None,
                    name=REGIONS_NAME):
        """A segmentation of the atlas labels (labelsPath, entries [(value, name, color)]) with their names. The
        reference region referenceName comes from referencePath (another version of the atlas on the same grid, e.g.
        the bilateral one, with its referenceEntries) as a segment of its own, or, when referencePath is labelsPath
        (or None), is the segment of that name. Returns (segmentation node, reference segment ID or None)."""
        scene = slicer.mrmlScene
        array, source = self._loadLabels(labelsPath)
        colorNode = labelNode = referenceNode = None
        try:
            values = {v for v, _, _ in entries}
            remapped = neuro.remapLabels(array, {v: v for v in values}).astype(np.int16)
            colorNode = scene.AddNewNodeByClass("vtkMRMLColorTableNode", scene.GenerateUniqueName("CerebrA colors"))
            colorNode.SetTypeToUser()
            top = max(values) + 1
            colorNode.SetNumberOfColors(top)
            colorNode.SetColor(0, "Background", 0.0, 0.0, 0.0, 0.0)
            for value in range(1, top):
                colorNode.SetColor(value, "", 0.0, 0.0, 0.0, 0.0)
            for value, label, (r, g, b) in entries:
                colorNode.SetColor(value, label, r, g, b, 1.0)
            labelNode = scene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode", "CerebrA labels")
            labelNode.SetIJKToRASMatrix(_vtkMatrix(_ijkToRas(source)))
            slicer.util.updateVolumeFromArray(labelNode, remapped)
            labelNode.CreateDefaultDisplayNodes()
            labelNode.GetDisplayNode().SetAndObserveColorNodeID(colorNode.GetID())
            segmentation = scene.AddNewNodeByClass("vtkMRMLSegmentationNode", scene.GenerateUniqueName(name))
            segmentation.CreateDefaultDisplayNodes()
            segmentation.SetReferenceImageGeometryParameterFromVolumeNode(labelNode)
            if not slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(labelNode, segmentation):
                scene.RemoveNode(segmentation)
                raise RuntimeError("Could not make the CerebrA segmentation")
            self._nameSegments(segmentation, entries)
            referenceId = None
            if referenceName:
                if referencePath is None or os.path.normcase(os.path.abspath(referencePath)) == \
                        os.path.normcase(os.path.abspath(labelsPath)):
                    referenceId = self.segmentIdByName(segmentation, referenceName)
                else:
                    referenceArray, referenceNode = self._loadLabels(referencePath)
                    if referenceArray.shape != array.shape:
                        raise ValueError("The reference atlas is not on the grid of the region atlas.")
                    referenceValue = self.entryValue(referenceEntries or (), referenceName)
                    if referenceValue is None:
                        raise ValueError(f"No region '{referenceName}' in the reference atlas.")
                    referenceId = self._addMaskSegment(segmentation, referenceArray == referenceValue, labelNode,
                                                       referenceName + REFERENCE_SUFFIX, (0.85, 0.85, 0.2))
            segmentation.SetAttribute(NEURO_PREFIX + "Atlas", os.path.basename(labelsPath))
            display = segmentation.GetDisplayNode()
            if display is not None:
                display.SetOpacity2DFill(0.3)
                display.SetVisibility3D(False)
        finally:
            for node in (source, labelNode, colorNode, referenceNode):
                if node is not None and scene.IsNodePresent(node):
                    scene.RemoveNode(node)
        logging.info(f"Neuro: '{segmentation.GetName()}' made from {os.path.basename(labelsPath)} "
                     f"({len(entries)} regions" + (f", reference '{referenceName}'" if referenceId else "") + ")")
        return segmentation, referenceId

    @staticmethod
    def entryValue(entries, name):
        for value, entryName, _ in entries:
            if entryName.strip().lower() == name.strip().lower():
                return value
        return None

    def loadAtlasRegions(self, atlas, targetKind=None, referenceKind=None, referenceName=None):
        """Regions of the atlas folder (neuro.findAtlasFiles): the target regions from one CerebrA version, the
        reference region from another (default: the left and right regions, the bilateral cerebellar grey matter
        as the reference). Makes the bilateral version on first use. Returns (segmentation, reference segment ID)."""
        targetKind = targetKind or ATLAS_LATERAL
        referenceKind = referenceKind or ATLAS_BILATERAL
        atlas = dict(atlas)
        atlas[neuro.BILATERAL_LABELS], atlas[neuro.BILATERAL_CSV] = self.ensureBilateral(atlas)
        labelsPath, entries = self.atlasPaths(atlas, targetKind)
        referencePath, referenceEntries = self.atlasPaths(atlas, referenceKind)
        if referenceName is None:
            referenceName = defaultReferenceName(referenceEntries)
        return self.loadRegions(labelsPath, entries, referencePath, referenceName, referenceEntries)

    @staticmethod
    def _addMaskSegment(segmentation, mask, geometryNode, name, color):
        """A segment of its own (own labelmap layer, so it may overlap the others) from a boolean array on the grid
        of geometryNode. Returns its ID."""
        scene = slicer.mrmlScene
        maskNode = scene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode", "CerebrA mask (temporary)")
        try:
            maskNode.SetIJKToRASMatrix(_vtkMatrix(_ijkToRas(geometryNode)))
            slicer.util.updateVolumeFromArray(maskNode, mask.astype(np.uint8))
            image = slicer.vtkSlicerSegmentationsModuleLogic.CreateOrientedImageDataFromVolumeNode(maskNode)
        finally:
            scene.RemoveNode(maskNode)
        segment = slicer.vtkSegment()
        segment.SetName(name)
        segment.SetColor(*color)
        segment.AddRepresentation(slicer.vtkSegmentationConverter.GetBinaryLabelmapRepresentationName(), image)
        seg = segmentation.GetSegmentation()
        before = set(DynamicNeuroPETLogic.segmentIds(segmentation))
        seg.AddSegment(segment)
        added = [i for i in DynamicNeuroPETLogic.segmentIds(segmentation) if i not in before]
        return added[0] if added else DynamicNeuroPETLogic.segmentIdByName(segmentation, name)

    @staticmethod
    def _nameSegments(segmentation, entries):
        """Names and colors from the table (older Slicer versions name imported segments 'Label_<n>')."""
        byValue = {v: (n, c) for v, n, c in entries}
        seg = segmentation.GetSegmentation()
        for segmentId in DynamicNeuroPETLogic.segmentIds(segmentation):
            segment = seg.GetSegment(segmentId)
            value = segment.GetLabelValue() if hasattr(segment, "GetLabelValue") else None
            if value not in byValue:
                digits = "".join(ch for ch in segmentId.split("_")[-1] if ch.isdigit())
                value = int(digits) if digits else None
            if value in byValue and segment.GetName() != byValue[value][0]:
                name, color = byValue[value]
                if segment.GetName().startswith("Label") or not segment.GetName():
                    segment.SetName(name)
                    segment.SetColor(*color)

    @staticmethod
    def segmentIds(segmentation):
        ids = vtk.vtkStringArray()
        segmentation.GetSegmentation().GetSegmentIDs(ids)
        return [ids.GetValue(i) for i in range(ids.GetNumberOfValues())]

    @classmethod
    def segmentIdByName(cls, segmentation, name, startswith=False):
        name = name.strip().lower()
        for segmentId in cls.segmentIds(segmentation):
            segmentName = segmentation.GetSegmentation().GetSegment(segmentId).GetName().strip().lower()
            if segmentName == name or (startswith and segmentName.startswith(name)):
                return segmentId
        return None


# ---------------------------------------------------------------------------
# Widget
# ---------------------------------------------------------------------------
def _badge(state, number, size=22):
    scale = 2
    pixmap = qt.QPixmap(size * scale, size * scale)
    pixmap.fill(qt.QColor(0, 0, 0, 0))
    painter = qt.QPainter(pixmap)
    try:
        painter.setRenderHint(qt.QPainter.Antialiasing)
        color = qt.QColor(STATE_COLORS[state])
        margin = 2 * scale
        full = size * scale
        if state == STATE_DONE:
            painter.setPen(qt.Qt.NoPen)
            painter.setBrush(qt.QBrush(color))
        else:
            pen = qt.QPen(color)
            pen.setWidth(2 * scale)
            painter.setPen(pen)
            painter.setBrush(qt.QBrush(qt.QColor(0, 0, 0, 0)))
        painter.drawEllipse(margin, margin, full - 2 * margin, full - 2 * margin)
        font = qt.QFont()
        font.setBold(True)
        font.setPixelSize(int(full * 0.55))
        painter.setFont(font)
        painter.setPen(qt.QColor("white") if state == STATE_DONE else color)
        painter.drawText(qt.QRect(0, 0, full, full), qt.Qt.AlignCenter, "✓" if state == STATE_DONE
                         else str(number))
    finally:
        painter.end()
    return qt.QIcon(pixmap)


def _button(text, color="#2f7ed8"):
    button = qt.QPushButton(text)
    button.setStyleSheet(BUTTON_STYLE.format(color))
    return button


class DynamicNeuroPETWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Step bar on top, the page of the current step below: the module's own pages (Data, Normalization, Regions)
    and the panels of the PET4D modules (Motion Correction, Registration, Modeling)."""

    def __init__(self, parent=None):
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)
        self.logic = DynamicNeuroPETLogic()
        self.current = STEP_DATA
        self.pages = {}            # step key -> page widget (own pages)
        self.embedded = {}         # step key -> widget representation of the embedded module
        self.stepButtons = {}
        self.normSession = None    # RegistrationSession of the normalization check
        self.normResult = None     # pending normalization (until Harden)
        self.normUndo = None
        self.atlas = {}
        self._updating = False
        self._entered = False

        # --- step bar
        bar = qt.QFrame()
        bar.setFrameShape(qt.QFrame.StyledPanel)
        barLayout = qt.QGridLayout(bar)
        barLayout.setContentsMargins(4, 4, 4, 4)
        barLayout.setSpacing(4)
        for index, (key, _, title, explanation) in enumerate(STEPS):
            button = qt.QToolButton()
            button.setText(title)
            button.setToolButtonStyle(qt.Qt.ToolButtonTextBesideIcon)
            button.setAutoRaise(True)
            button.setCheckable(True)
            button.setSizePolicy(qt.QSizePolicy.Expanding, qt.QSizePolicy.Fixed)
            button.toolTip = f"{index + 1}. {title}: {explanation}"
            button.connect("clicked(bool)", lambda checked, k=key: self.showStep(k))
            barLayout.addWidget(button, index // 3, index % 3)
            self.stepButtons[key] = button
        self.layout.addWidget(bar)
        self.titleLabel = qt.QLabel()
        self.titleLabel.wordWrap = True
        self.layout.addWidget(self.titleLabel)

        self.stack = qt.QStackedWidget()
        self.layout.addWidget(self.stack, 1)

        nav = qt.QHBoxLayout()
        self.backButton = qt.QPushButton("‹ Back")
        self.nextButton = _button("Next ›")
        nav.addWidget(self.backButton)
        nav.addStretch(1)
        nav.addWidget(self.nextButton)
        self.layout.addLayout(nav)
        self.backButton.connect("clicked()", lambda: self.moveStep(-1))
        self.nextButton.connect("clicked()", lambda: self.moveStep(1))

        self.pages[STEP_DATA] = self._buildDataPage()
        self.pages[STEP_NORM] = self._buildNormPage()
        self.pages[STEP_REGIONS] = self._buildRegionsPage()

        self.refreshTimer = qt.QTimer()
        self.refreshTimer.setSingleShot(True)
        self.refreshTimer.setInterval(300)
        self.refreshTimer.connect("timeout()", self.refresh)
        for event in (slicer.mrmlScene.NodeAddedEvent, slicer.mrmlScene.NodeRemovedEvent,
                      slicer.mrmlScene.EndImportEvent):
            self.addObserver(slicer.mrmlScene, event, lambda caller, event: self.refreshTimer.start())
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, lambda c, e: self.refreshTimer.start())
        self.onAtlasFolderChanged(self.logic.atlasFolder(), save=False)
        self.selectInputs()
        self.showStep(STEP_DATA, enterStep=False)

    # --- parameter node: the series and images shared by the steps ---------------------------------
    def _parameter(self):
        return self.logic.getParameterNode()

    def node(self, role):
        return self._parameter().GetNodeReference(NEURO_PREFIX + role)

    def setNode(self, role, node):
        self._parameter().SetNodeReferenceID(NEURO_PREFIX + role, node.GetID() if node is not None else None)

    def sequence(self):
        return self.node("Sequence")

    def anatomy(self):
        return self.node("Anatomy")

    # --- page 1: data --------------------------------------------------------------------------------
    @staticmethod
    def _nodeSelector(nodeTypes, toolTip):
        selector = slicer.qMRMLNodeComboBox()
        selector.nodeTypes = nodeTypes
        selector.noneEnabled = True
        selector.addEnabled = False
        selector.removeEnabled = False
        selector.renameEnabled = False
        selector.setMRMLScene(slicer.mrmlScene)
        selector.setCurrentNode(None)
        selector.toolTip = toolTip
        return selector

    def _buildDataPage(self):
        page = qt.QWidget()
        layout = qt.QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        box = qt.QGroupBox("Patient data")
        form = qt.QFormLayout(box)
        self.sequenceSelector = self._nodeSelector(["vtkMRMLSequenceNode"],
                                                   "Dynamic PET series of the patient (as loaded from DICOM: "
                                                   "'<series> dynamic Bq/ml')")
        form.addRow("Dynamic PET:", self.sequenceSelector)
        self.sequenceInfo = qt.QLabel()
        self.sequenceInfo.wordWrap = True
        form.addRow(self.sequenceInfo)
        self.anatomySelector = self._nodeSelector(["vtkMRMLScalarVolumeNode"],
                                                  "CT or MR of the same patient (registration in step 3, and the "
                                                  "normalization through the CT/MR in step 4)")
        form.addRow("CT / MR:", self.anatomySelector)
        self.anatomyInfo = qt.QLabel()
        self.anatomyInfo.wordWrap = True
        form.addRow(self.anatomyInfo)
        self.contrastCombo = qt.QComboBox()
        for key, title in (("", "Automatic"), ("T1", "T1"), ("T2", "T2"), ("PD", "PD"), ("CT", "CT")):
            self.contrastCombo.addItem(title, key)
        self.contrastCombo.toolTip = ("Contrast of the CT/MR, read from the DICOM tags (series description, "
                                      "protocol, TR / TE). It picks the template of step 4 when the CT/MR is "
                                      "normalized.")
        form.addRow("Contrast:", self.contrastCombo)
        layout.addWidget(box)
        dicomButton = qt.QPushButton("Open the DICOM browser")
        dicomButton.toolTip = "Load the dynamic PET ('dynamic Bq/ml') and the CT/MR from DICOM"
        dicomButton.connect("clicked()", lambda: slicer.util.selectModule("DICOM"))
        layout.addWidget(dicomButton)
        note = qt.QLabel("This workflow is for dynamic receptor imaging of the brain: the series is motion corrected, "
                         "registered to the CT/MR, normalized to the MNI ICBM 152 2009c template, and the CerebrA "
                         "regions are modelled (SRTM with the cerebellar grey matter as the reference by default). "
                         "Steps can be skipped.")
        note.wordWrap = True
        note.setStyleSheet("color: #7d868d")
        layout.addWidget(note)
        layout.addStretch(1)
        self.sequenceSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onDataChanged)
        self.anatomySelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onDataChanged)
        self.contrastCombo.connect("currentIndexChanged(int)", lambda i: self.onDataChanged())
        return page

    def selectInputs(self):
        if self.sequence() is None:
            from DynamicPETEditor import DynamicPETEditorLogic
            nodes = [n for n in slicer.util.getNodesByClass("vtkMRMLSequenceNode")
                     if (n.GetAttribute(PREFIX + "Kind") == "dynamic" or n.GetAttribute(PREFIX + "FrameDurationS"))
                     and DynamicPETEditorLogic.isDynamicSequence(n)]
            if nodes:
                original = [n for n in nodes if not n.GetAttribute(MC_PREFIX + "Method")
                            and not n.GetAttribute(REG_PREFIX + "Fixed")]
                self.setNode("Sequence", (original or nodes)[-1])
        if self.anatomy() is None:
            from DynamicPETRegistration import fixedCandidates
            candidates = [n for n in fixedCandidates() if not n.GetAttribute(ATLAS_FILE_ATTRIBUTE)
                          and not n.GetAttribute(NEURO_PREFIX + "Template")]
            if candidates:
                self.setNode("Anatomy", candidates[0])
        self._syncDataPage()

    def _syncDataPage(self):
        self._updating = True
        try:
            self.sequenceSelector.setCurrentNode(self.sequence())
            self.anatomySelector.setCurrentNode(self.anatomy())
        finally:
            self._updating = False
        self._updateDataInfo()

    def onDataChanged(self, *args):
        if self._updating:
            return
        sequence, anatomy = self.sequenceSelector.currentNode(), self.anatomySelector.currentNode()
        if sequence is not self.sequence():
            self.setNode("Sequence", sequence)
            self.setNode("OriginalSequence", sequence)
            self._clearNormalization()
        if anatomy is not self.anatomy():
            self.setNode("Anatomy", anatomy)
            self._clearNormalization()
        self._updateDataInfo()
        self._updateTemplateChoice()
        self.refresh()

    def contrast(self):
        """(contrast, reason) of the CT/MR: the user's choice, else read from the DICOM tags."""
        chosen = self.contrastCombo.currentData
        if chosen:
            return chosen, "chosen"
        return self.logic.anatomyContrast(self.anatomy())

    def _updateDataInfo(self):
        sequence = self.sequence()
        if sequence is None:
            self.sequenceInfo.text = "Pick the dynamic PET series."
        else:
            from DynamicPETRegistration import DynamicPETRegistrationLogic
            summed = DynamicPETRegistrationLogic.summedVolumeOf(sequence)
            done = []
            if sequence.GetAttribute(MC_PREFIX + "Method"):
                done.append("motion corrected")
            if sequence.GetAttribute(REG_PREFIX + "Fixed"):
                done.append(f"registered to '{sequence.GetAttribute(REG_PREFIX + 'Fixed')}'")
            if sequence.GetAttribute(NEURO_PREFIX + "Template"):
                done.append("normalized")
            self.sequenceInfo.text = (f"{sequence.GetNumberOfDataNodes()} frames"
                                      + (f", summed image '{summed.GetName()}'" if summed else ", no summed image")
                                      + (f"; {', '.join(done)}" if done else "") + ".")
        anatomy = self.anatomy()
        if anatomy is None:
            self.anatomyInfo.text = "No CT/MR: steps 3 and the CT/MR normalization are skipped."
        else:
            contrast, reason = self.logic.anatomyContrast(anatomy)
            self.anatomyInfo.text = (f"Recognised as <b>{contrast}</b> ({reason})." if contrast else
                                     f"Contrast not recognised ({reason}); the T1 template is used unless you pick "
                                     "one.")

    # --- page 4: normalization ---------------------------------------------------------------------
    def _buildNormPage(self):
        page = qt.QWidget()
        layout = qt.QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)

        atlasBox = qt.QGroupBox("Atlas folder")
        atlasLayout = qt.QVBoxLayout(atlasBox)
        self.atlasPath = ctk.ctkPathLineEdit()
        self.atlasPath.filters = ctk.ctkPathLineEdit.Dirs
        self.atlasPath.settingKey = ""
        self.atlasPath.toolTip = ("Folder with the MNI ICBM 152 2009c (nlin sym) NIfTI files and the CerebrA labels "
                                  "(searched with its subfolders). Kept in the application settings.")
        atlasLayout.addWidget(self.atlasPath)
        self.atlasInfo = qt.QLabel()
        self.atlasInfo.wordWrap = True
        atlasLayout.addWidget(self.atlasInfo)
        layout.addWidget(atlasBox)

        box = qt.QGroupBox("Normalization (BRAINSFit)")
        form = qt.QFormLayout(box)
        self.methodCombo = qt.QComboBox()
        for key, title in METHODS:
            self.methodCombo.addItem(title, key)
        self.methodCombo.toolTip = (
            "PET: the summed image (or the frame of the highest activity) is registered to the template. CT/MR: the "
            "CT/MR is registered to the template and the same transform moves the PET, which must already lie on "
            "the CT/MR (step 3).")
        form.addRow("Method:", self.methodCombo)
        self.templateCombo = qt.QComboBox()
        for key, title, _ in neuro.TEMPLATES:
            self.templateCombo.addItem(title, key)
        self.templateCombo.toolTip = ("Template image registered to. T1 by default; with the CT/MR method the template "
                                      "of the MR contrast (T2, PD) is picked by itself.")
        form.addRow("Template:", self.templateCombo)
        self.templateNote = qt.QLabel()
        self.templateNote.wordWrap = True
        form.addRow(self.templateNote)
        self.maskCheck = qt.QCheckBox("Skull-stripped template (brain mask)")
        self.maskCheck.checked = False
        self.maskCheck.toolTip = ("Register to the template with the voxels outside its brain mask "
                                  "(mni_icbm152_t1_tal_nlin_sym_09c_mask) set to 0, so the scalp and the neck do not "
                                  "pull the fit. Off by default.")
        form.addRow(self.maskCheck)
        self.normStepCombos = []
        for index, default in enumerate(NORMALIZATION_STEPS):
            combo = qt.QComboBox()
            if index > 0:
                combo.addItem("(none)", "")
            for name in reg.STEP_TYPES:
                combo.addItem(name, name)
            combo.setCurrentIndex(combo.findData(default))
            combo.toolTip = "Affine then B-spline by default (a B-spline only as the last step)"
            self.normStepCombos.append(combo)
            form.addRow(f"Step {index + 1}:", combo)
        self.normInitCombo = qt.QComboBox()
        for title, mode in reg.INITIALIZATIONS:
            self.normInitCombo.addItem(title, mode)
        self.normInitCombo.toolTip = ("How the first step starts. Centers of mass (default for the PET) puts the "
                                      "brain activity on the template's head; geometric centers (default for the "
                                      "CT/MR) aligns the volume centres.")
        form.addRow("Initialization:", self.normInitCombo)
        self.normalizeButton = _button("Normalize")
        self.normalizeButton.toolTip = "Register to the template and show the result in the registration layout"
        form.addRow(self.normalizeButton)
        row = qt.QHBoxLayout()
        self.normManualButton = qt.QPushButton("Correct manually")
        self.normManualButton.checkable = True
        self.normManualButton.toolTip = "Move / rotate handles on the PET, on top of the normalization"
        self.normRevertButton = qt.QPushButton("Revert")
        self.normRevertButton.toolTip = ("Before Harden: drop the normalization. After Harden: remove the normalized "
                                         "series (or put an overwritten one back).")
        self.normHardenButton = _button("Harden", "#2e9e5b")
        self.normHardenButton.toolTip = "Resample every frame (and the summed image) into the template space"
        row.addWidget(self.normManualButton)
        row.addWidget(self.normRevertButton)
        row.addWidget(self.normHardenButton)
        form.addRow(row)
        self.normLabel = qt.QLabel("Not normalized yet.")
        self.normLabel.wordWrap = True
        self.normLabel.setTextInteractionFlags(qt.Qt.TextSelectableByMouse)
        form.addRow(self.normLabel)
        layout.addWidget(box)

        viewBox = ctk.ctkCollapsibleButton()
        viewBox.text = "View"
        viewBox.collapsed = True
        view = qt.QFormLayout(viewBox)
        self.normOpacity = ctk.ctkSliderWidget()
        self.normOpacity.minimum, self.normOpacity.maximum = 0.0, 1.0
        self.normOpacity.singleStep, self.normOpacity.decimals = 0.05, 2
        self.normOpacity.value = 0.5
        view.addRow("Fusion opacity:", self.normOpacity)
        self.normCheckerCheck = qt.QCheckBox("Checkerboard column")
        view.addRow(self.normCheckerCheck)
        layout.addWidget(viewBox)
        layout.addStretch(1)

        self.atlasPath.connect("currentPathChanged(QString)", lambda path: self.onAtlasFolderChanged(path))
        self.methodCombo.connect("currentIndexChanged(int)", self.onMethodChanged)
        self.templateCombo.connect("currentIndexChanged(int)", lambda i: self._clearNormalization())
        self.normalizeButton.connect("clicked()", self.onNormalize)
        self.normManualButton.connect("toggled(bool)", lambda on: self.normSession and
                                      self.normSession.setManualHandles(on))
        self.normRevertButton.connect("clicked()", self.onNormRevert)
        self.normHardenButton.connect("clicked()", self.onNormHarden)
        self.normOpacity.connect("valueChanged(double)", lambda v: self.normSession and
                                 self.normSession.setFusionOpacity(v))
        self.normCheckerCheck.connect("toggled(bool)", lambda on: self.normSession and
                                      self.normSession.setShowCheckerboard(on))
        self.onMethodChanged()
        return page

    def onAtlasFolderChanged(self, folder, save=True):
        folder = folder or ""
        if save:
            self.logic.setAtlasFolder(folder)
        if self.atlasPath.currentPath != folder:
            self.atlasPath.blockSignals(True)
            self.atlasPath.currentPath = folder
            self.atlasPath.blockSignals(False)
        self.atlas = neuro.findAtlasFiles(folder) if folder else {}
        if not folder:
            self.atlasInfo.text = ("Pick the folder that holds the MNI ICBM 152 2009c files (e.g. "
                                   "mni_icbm152_t1_tal_nlin_sym_09c.nii) and the CerebrA labels "
                                   "(mni_icbm152_CerebrA_tal_nlin_sym_09c.nii, CerebrA_LabelDetails.csv).")
        else:
            missing = neuro.missingFiles(self.atlas)
            found = len(self.atlas)
            text = f"{found} atlas file(s) found."
            if missing:
                text += " Missing: " + ", ".join(missing) + "."
            if any(k not in self.atlas for k in neuro.REQUIRED):
                text = "<span style='color:#c0392b'>" + text + "</span>"
            self.atlasInfo.text = text
        for index in range(self.templateCombo.count):
            item = self.templateCombo.model().item(index)
            if item is not None:
                item.setEnabled(self.templateCombo.itemData(index) in self.atlas)
        self.maskCheck.enabled = neuro.BRAIN_MASK in self.atlas
        self._updateTemplateChoice()
        self.refreshTimer.start()

    def method(self):
        return self.methodCombo.currentData or METHOD_PET

    def onMethodChanged(self, *args):
        method = self.method()
        self.normInitCombo.setCurrentIndex(self.normInitCombo.findData(NORMALIZATION_INITIALIZATION[method]))
        self._updateTemplateChoice()
        self._clearNormalization()

    def _updateTemplateChoice(self):
        """T1 for the PET; the template of the MR contrast for the CT/MR method."""
        if not hasattr(self, "templateCombo"):
            return
        method = self.method()
        if method == METHOD_ANATOMY:
            contrast, reason = self.contrast()
            key = neuro.templateForContrast(contrast)
            note = (f"CT/MR recognised as {contrast} ({reason})." if contrast else f"CT/MR contrast: {reason}.")
        else:
            key, note = neuro.DEFAULT_TEMPLATE, "The PET is registered to the T1 template by default."
        if key not in self.atlas and self.atlas:
            note += f" The {key} template is not in the atlas folder, the T1 is used."
            key = neuro.DEFAULT_TEMPLATE
        index = self.templateCombo.findData(key)
        if index >= 0 and index != self.templateCombo.currentIndex:
            self.templateCombo.blockSignals(True)
            self.templateCombo.setCurrentIndex(index)
            self.templateCombo.blockSignals(False)
        self.templateNote.text = note

    def normSettings(self):
        settings = self.logic.defaultSettings(self.method())
        steps = [c.currentData for c in self.normStepCombos if c.currentData]
        settings["steps"] = reg.validateSteps(steps)
        settings["resample"] = reg.resampleFlags(settings["steps"])
        settings["initialization"] = self.normInitCombo.currentData
        return settings

    def _template(self):
        key = self.templateCombo.currentData
        path = self.atlas.get(key)
        if not path:
            raise ValueError("Pick the atlas folder: the template file "
                             f"{dict((k, s) for k, _, s in neuro.TEMPLATES)[key]}.nii was not found.")
        template = self.logic.loadAtlasVolume(path, name=f"MNI ICBM152 2009c {key}")
        self.setNode("Template", template)
        return template

    def _mask(self):
        if not self.maskCheck.checked or neuro.BRAIN_MASK not in self.atlas:
            return None
        return self.logic.loadAtlasVolume(self.atlas[neuro.BRAIN_MASK], name="MNI ICBM152 2009c brain mask")

    def onNormalize(self):
        sequence, anatomy, method = self.sequence(), self.anatomy(), self.method()
        if sequence is None:
            slicer.util.warningDisplay("Pick the dynamic PET series in step 1.", windowTitle="Normalization")
            return
        if method == METHOD_ANATOMY:
            if anatomy is None:
                slicer.util.warningDisplay("Pick the CT/MR in step 1, or register the PET to the template.",
                                           windowTitle="Normalization")
                return
            if not self.logic.petInAnatomySpace(sequence, anatomy) and not slicer.util.confirmOkCancelDisplay(
                    f"'{sequence.GetName()}' was not registered to '{anatomy.GetName()}' in step 3. The CT/MR "
                    "transform moves the PET correctly only when the PET already lies on the CT/MR (e.g. PET/MR, or "
                    "registered and hardened). Go on?", windowTitle="Normalization"):
                return
        try:
            settings = self.normSettings()
            template = self._template()
            mask = self._mask()
        except Exception as error:
            slicer.util.errorDisplay(str(error), windowTitle="Normalization")
            return
        self._clearNormalization()
        progressDialog = slicer.util.createProgressDialog(windowTitle="Normalization", labelText="Registering...",
                                                          maximum=len(settings["steps"]))

        def progress(index, total, text):
            progressDialog.setValue(index)
            progressDialog.setLabelText(f"Normalization: {text}")
            slicer.app.processEvents()
            return not progressDialog.wasCanceled

        qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)
        try:
            self.normResult = self.logic.normalize(sequence, template, method, anatomy, settings, mask, progress)
        except Exception as error:
            logging.exception("Neuro: normalization failed")
            slicer.util.errorDisplay(f"Normalization failed:\n{error}")
            self.normResult = None
            return
        finally:
            qt.QApplication.restoreOverrideCursor()
            progressDialog.close()
        self.normResult["template"] = template
        self.normResult["method"] = method
        self._openNormSession()
        lines = reg.summaryLines(self.normResult["steps"], self.normResult["linear"], self.normResult["center"],
                                 self.normResult["deformation"])
        moving = self.normResult["moving"]
        self.normLabel.text = (f"'{moving.GetName()}' registered to '{template.GetName()}'"
                               + (" (stopped)" if self.normResult["stopped"] else "") + ":\n" + "\n".join(lines)
                               + "\nCheck it in the views, correct it by hand if needed, then Harden.")
        self._updateNormButtons()

    def _openNormSession(self):
        """The registration layout (PET / fusion / template, checkerboard) with the normalization chain on the PET.
        With the CT/MR method the PET is shown too: it lies on the CT/MR, so the same chain moves it."""
        from DynamicPETRegistration import RegistrationSession, DynamicPETRegistrationWidget
        self._closeNormSession()
        sequence = self.sequence()
        if self.normResult is None or sequence is None:
            return
        DynamicPETRegistrationWidget._closeOtherModules()
        registrationWidget = getattr(slicer.modules, "DynamicPETRegistrationWidget", None)
        if registrationWidget is not None and getattr(registrationWidget, "session", None) is not None:
            registrationWidget.onClose()
        try:
            session = RegistrationSession(self.normResult["template"])
            session.fusionOpacity = self.normOpacity.value
            session.showCheckerboard = self.normCheckerCheck.checked
            session.open(self.logic.referenceImage(sequence))
            session.setRegistration(self.normResult["linear"], self.normResult["bspline"])
            session.setFixedWindow(*_percentiles(self.normResult["template"]))
            self.normSession = session
        except Exception as error:
            logging.exception("Neuro: could not open the normalization layout")
            slicer.util.errorDisplay(f"Could not open the normalization layout:\n{error}")

    def _closeNormSession(self):
        if self.normSession is not None:
            if self.normManualButton.checked:
                self.normManualButton.blockSignals(True)
                self.normManualButton.checked = False
                self.normManualButton.blockSignals(False)
            self.normSession.close()
            self.normSession = None

    def _clearNormalization(self):
        """Drop a pending (not hardened) normalization."""
        if not hasattr(self, "normLabel"):
            return
        self._closeNormSession()
        if self.normResult is not None:
            bspline = self.normResult.get("bspline")
            if bspline is not None and slicer.mrmlScene.IsNodePresent(bspline):
                slicer.mrmlScene.RemoveNode(bspline)
            self.normResult = None
            self.normLabel.text = "Not normalized yet."
        self._updateNormButtons()

    def _updateNormButtons(self):
        if not hasattr(self, "normHardenButton"):
            return
        pending = self.normResult is not None and self.normSession is not None
        self.normHardenButton.enabled = pending
        self.normManualButton.enabled = pending
        self.normRevertButton.enabled = pending or self.normUndo is not None
        self.normalizeButton.enabled = self.sequence() is not None and neuro.DEFAULT_TEMPLATE in self.atlas

    def onNormHarden(self):
        sequence = self.sequence()
        if self.normResult is None or self.normSession is None or sequence is None:
            return
        template, method = self.normResult["template"], self.normResult["method"]
        petSpacing = reg.gridSpacing(_ijkToRas(sequence.GetNthDataNode(0)))
        text = (f"Resample the {sequence.GetNumberOfDataNodes()} frames of '{sequence.GetName()}' (and its summed "
                f"image) through the normalization onto the field of view of the template '{template.GetName()}'."
                + ("\nThe CT/MR is normalized too (on the template grid)." if method == METHOD_ANATOMY else "")
                + "\n\nSave separately (default) keeps the series in the patient space next to the normalized one.")
        choice = hardenDialog("Harden normalization", text, NORMALIZATION_GRIDS, petSpacing,
                              [("Save separately", reg.SAVE_SEPARATELY), ("Overwrite", reg.OVERWRITE)],
                              slicer.util.mainWindow())
        if choice is None:
            return
        steps = reg.stepsText(self.normResult["steps"])
        description = (f"Normalized to {template.GetName()} ({steps}"
                       + (", through the CT/MR" if method == METHOD_ANATOMY else "") + ")")
        self.normManualButton.checked = False
        qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)
        try:
            undo, newSequence, newSummed, normalizedAnatomy = self.logic.harden(
                sequence, template, self.normSession.linear, choice["action"], choice["grid"], choice["spacing"],
                choice["interpolation"], description, method, self.anatomy())
        except Exception as error:
            logging.exception("Neuro: could not harden the normalization")
            slicer.util.errorDisplay(f"Could not harden the normalization:\n{error}")
            return
        finally:
            qt.QApplication.restoreOverrideCursor()
        self._clearNormalization()
        self.normUndo = {"undo": undo, "previous": sequence, "anatomy": normalizedAnatomy}
        self.setNode("Sequence", newSequence)
        self.setNode("NormalizedAnatomy", normalizedAnatomy)
        self.normLabel.text = (description + ".\n"
                               + (f"Overwrote '{sequence.GetName()}'" if choice["action"] == reg.OVERWRITE else
                                  f"Saved as '{newSequence.GetName()}'"
                                  + (f" and '{newSummed.GetName()}'" if newSummed is not None else ""))
                               + (f"; CT/MR as '{normalizedAnatomy.GetName()}'" if normalizedAnatomy else "")
                               + ". Next: load the regions.")
        self._showNormalized(newSequence, template)
        self._updateNormButtons()
        self.refresh()

    def _showNormalized(self, sequence, template):
        """The normalized summed image over the template in the standard views."""
        from DynamicPETRegistration import DynamicPETRegistrationLogic
        summed = DynamicPETRegistrationLogic.summedVolumeOf(sequence) or sequence.GetNthDataNode(0)
        try:
            slicer.app.layoutManager().setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutFourUpView)
            slicer.util.setSliceViewerLayers(background=template, foreground=summed, foregroundOpacity=0.5,
                                             label=None, fit=True)
        except Exception:
            logging.debug("Neuro: could not show the normalized series", exc_info=True)

    def onNormRevert(self):
        if self.normResult is not None:
            self._clearNormalization()
            return
        if self.normUndo is None:
            return
        from DynamicPETRegistration import DynamicPETRegistrationLogic
        record = self.normUndo
        try:
            DynamicPETRegistrationLogic().revert(record["undo"])
        except Exception as error:
            logging.exception("Neuro: could not revert the normalization")
            slicer.util.errorDisplay(f"Could not revert the normalization:\n{error}")
            return
        anatomy = record.get("anatomy")
        if anatomy is not None and slicer.mrmlScene.IsNodePresent(anatomy):
            slicer.mrmlScene.RemoveNode(anatomy)
        if slicer.mrmlScene.IsNodePresent(record["previous"]):
            self.setNode("Sequence", record["previous"])
        self.setNode("NormalizedAnatomy", None)
        self.normUndo = None
        self.normLabel.text = "Reverted: the series is in the patient space."
        self._updateNormButtons()
        self.refresh()

    # --- page 5: regions ---------------------------------------------------------------------------
    def _buildRegionsPage(self):
        page = qt.QWidget()
        layout = qt.QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        box = qt.QGroupBox("CerebrA regions")
        form = qt.QFormLayout(box)
        self.regionsInfo = qt.QLabel()
        self.regionsInfo.wordWrap = True
        form.addRow(self.regionsInfo)
        self.targetAtlasCombo = qt.QComboBox()
        self.referenceAtlasCombo = qt.QComboBox()
        for combo in (self.targetAtlasCombo, self.referenceAtlasCombo):
            for key, title in ATLAS_KINDS:
                combo.addItem(title, key)
        self.targetAtlasCombo.setCurrentIndex(self.targetAtlasCombo.findData(ATLAS_LATERAL))
        self.referenceAtlasCombo.setCurrentIndex(self.referenceAtlasCombo.findData(ATLAS_BILATERAL))
        self.targetAtlasCombo.toolTip = ("Atlas of the regions modelled: left and right separate ('Putamen R', "
                                         "'Putamen L', 102 regions) or merged ('Putamen', 51 regions). The bilateral "
                                         "version is made from CerebrA on first use and saved next to it.")
        self.referenceAtlasCombo.toolTip = ("Atlas the reference region is taken from (bilateral by default: both "
                                            "sides of the cerebellar grey matter in one region)")
        form.addRow("Region atlas:", self.targetAtlasCombo)
        form.addRow("Reference atlas:", self.referenceAtlasCombo)
        self.referenceRegionCombo = qt.QComboBox()
        self.referenceRegionCombo.toolTip = ("Reference region of the reference tissue models (no specific binding; "
                                             "the cerebellar grey matter by default)")
        form.addRow("Reference region:", self.referenceRegionCombo)
        self.loadRegionsButton = _button("Load regions and go to modeling")
        form.addRow(self.loadRegionsButton)
        self.regionsLabel = qt.QLabel()
        self.regionsLabel.wordWrap = True
        form.addRow(self.regionsLabel)
        layout.addWidget(box)
        layout.addStretch(1)
        self.referenceAtlasCombo.connect("currentIndexChanged(int)", lambda i: self._fillReferenceRegions())
        self.loadRegionsButton.connect("clicked()", self.onLoadRegions)
        return page

    def _fillReferenceRegions(self):
        """Regions of the reference atlas (from the CerebrA CSV, so they are listed before the bilateral files
        exist), the cerebellar grey matter picked."""
        current = self.referenceRegionCombo.currentText
        kind = self.referenceAtlasCombo.currentData
        try:
            entries = self.logic.atlasEntries(self.atlas, kind) if neuro.CEREBRA_CSV in self.atlas else []
        except Exception:
            logging.warning("Neuro: could not read the CerebrA labels", exc_info=True)
            entries = []
        names = sorted(n for _, n, _ in entries)
        key = (kind, tuple(names))
        if getattr(self, "_referenceKey", None) == key:
            return
        self._referenceKey = key
        self.referenceRegionCombo.clear()
        for name in names:
            self.referenceRegionCombo.addItem(name)
        index = self.referenceRegionCombo.findText(current) if current else -1
        if index < 0:
            index = self.referenceRegionCombo.findText(defaultReferenceName(entries) or "")
        self.referenceRegionCombo.setCurrentIndex(max(0, index))

    def _updateRegionsPage(self):
        self._fillReferenceRegions()
        sequence = self.sequence()
        normalized = sequence is not None and bool(sequence.GetAttribute(NEURO_PREFIX + "Template"))
        files = neuro.CEREBRA_LABELS in self.atlas and neuro.CEREBRA_CSV in self.atlas
        text = []
        if not normalized:
            text.append("<span style='color:#c0392b'>The series is not normalized yet (step 4): the CerebrA regions "
                        "lie in the template space.</span>")
        if not files:
            text.append("<span style='color:#c0392b'>mni_icbm152_CerebrA_tal_nlin_sym_09c.nii and "
                        "CerebrA_LabelDetails.csv are not in the atlas folder (step 4).</span>")
        segmentation = self.node("Segmentation")
        if segmentation is not None:
            text.append(f"Loaded: '{segmentation.GetName()}'.")
        if not text:
            text.append(f"'{sequence.GetName()}' is in the template space. Load the CerebrA regions.")
        self.regionsInfo.text = " ".join(text)
        self.loadRegionsButton.enabled = files and sequence is not None

    def onLoadRegions(self):
        sequence = self.sequence()
        if sequence is None:
            return
        if not sequence.GetAttribute(NEURO_PREFIX + "Template") and not slicer.util.confirmOkCancelDisplay(
                "The series is not normalized (step 4), so the CerebrA regions will not lie on the brain. Load them "
                "anyway?", windowTitle="Regions"):
            return
        qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)
        try:
            segmentation, referenceId = self.logic.loadAtlasRegions(
                self.atlas, self.targetAtlasCombo.currentData, self.referenceAtlasCombo.currentData,
                self.referenceRegionCombo.currentText or None)
        except Exception as error:
            logging.exception("Neuro: could not load the regions")
            slicer.util.errorDisplay(f"Could not load the CerebrA regions:\n{error}")
            return
        finally:
            qt.QApplication.restoreOverrideCursor()
        old = self.node("Segmentation")
        self.setNode("Segmentation", segmentation)
        self._parameter().SetParameter(NEURO_PREFIX + "ReferenceSegment", referenceId or "")
        self.atlas = neuro.findAtlasFiles(self.logic.atlasFolder()) or self.atlas   # the bilateral files are new
        count = len(self.logic.segmentIds(segmentation))
        reference = segmentation.GetSegmentation().GetSegment(referenceId).GetName() if referenceId else None
        self.regionsLabel.text = (f"'{segmentation.GetName()}': {count} segments"
                                  + (f", reference '{reference}'" if reference else ", no reference region") + "."
                                  + (f" (The earlier '{old.GetName()}' is still in the scene.)" if old else ""))
        self.showStep(STEP_MODEL)

    # --- embedded modules --------------------------------------------------------------------------
    @staticmethod
    def _moduleName(key):
        return next(m for k, m, _, _ in STEPS if k == key)

    def stepPanel(self, key):
        """The page of a step: an own page, or the widget representation of the embedded module (taken back into
        the stack when Slicer showed it in its own panel meanwhile)."""
        if key in self.pages:
            panel = self.pages[key]
        else:
            moduleName = self._moduleName(key)
            module = getattr(slicer.modules, moduleName.lower(), None)
            if module is None:
                raise RuntimeError(f"Module {moduleName} is not loaded")
            panel = module.widgetRepresentation()
            self.embedded[key] = panel
        if self.stack.indexOf(panel) < 0:
            self.stack.addWidget(panel)
        return panel

    def stepWidget(self, key):
        panel = self.embedded.get(key)
        if panel is None:
            return None
        try:
            return panel.self()
        except Exception:
            return None

    def _callStep(self, key, method):
        if key in self.embedded:
            widget = self.stepWidget(key)
            function = getattr(widget, method, None)
            if callable(function):
                try:
                    function()
                except Exception:
                    logging.warning(f"Neuro: {method}() of step {key} failed", exc_info=True)
        elif key == STEP_NORM and method == "exit":
            self._closeNormSession()
        elif key == STEP_NORM and method == "enter" and self.normResult is not None:
            self._openNormSession()

    def _takeSeriesFrom(self, key):
        """The series the embedded module ended with (e.g. a new series saved separately) is the workflow's."""
        widget = self.stepWidget(key)
        selector = getattr(widget, "sequenceSelector", None) if widget is not None else None
        node = selector.currentNode() if selector is not None else None
        if node is not None and node is not self.sequence():
            self.setNode("Sequence", node)
            self._clearNormalization()

    def _prepareStep(self, key):
        """Give the embedded module the workflow's series (and CT/MR, regions) before it is shown."""
        widget = self.stepWidget(key)
        sequence = self.sequence()
        if widget is None:
            return
        try:
            if sequence is not None and widget.sequenceSelector.currentNode() is not sequence:
                widget.sequenceSelector.setCurrentNode(sequence)
            if key == STEP_REG and self.anatomy() is not None and widget.fixedSelector.currentNode() is not \
                    self.anatomy():
                widget.fixedSelector.setCurrentNode(self.anatomy())
            if key == STEP_MODEL:
                self._prepareModeling(widget)
        except Exception:
            logging.warning(f"Neuro: could not set the inputs of step {key}", exc_info=True)

    def _prepareModeling(self, widget):
        segmentation = self.node("Segmentation")
        if segmentation is None:
            return
        fresh = widget.segmentationSelector.currentNode() is not segmentation
        if fresh:
            index = widget.modelCombo.findData(DEFAULT_MODEL)
            if index >= 0:
                widget.modelCombo.setCurrentIndex(index)
            widget.segmentationSelector.setCurrentNode(segmentation)
            referenceId = self._parameter().GetParameter(NEURO_PREFIX + "ReferenceSegment")
            index = widget.referenceCombo.findData(referenceId) if referenceId else -1
            if index >= 0:
                widget.referenceCombo.setCurrentIndex(index)
            target = (self.logic.segmentIdByName(segmentation, DEFAULT_TARGET)
                      or self.logic.segmentIdByName(segmentation, DEFAULT_TARGET + " R")
                      or self.logic.segmentIdByName(segmentation, DEFAULT_TARGET, startswith=True))
            index = widget.targetCombo.findData(target) if target else -1
            if index >= 0:
                widget.targetCombo.setCurrentIndex(index)

    def showStep(self, key, enterStep=True):
        previous = self.current
        if previous != key:
            if previous in self.embedded:
                self._takeSeriesFrom(previous)
            self._callStep(previous, "exit")
        try:
            panel = self.stepPanel(key)
        except Exception as error:
            logging.warning(f"Neuro: cannot show step {key}: {error}")
            key, panel = STEP_DATA, self.stepPanel(STEP_DATA)
        self.current = key
        self.stack.setCurrentWidget(panel)
        # only the shown page stays in the stack (the stack takes the height of its tallest page otherwise)
        for index in reversed(range(self.stack.count)):
            page = self.stack.widget(index)
            if page is not panel:
                self.stack.removeWidget(page)
        if key == STEP_DATA:
            self._syncDataPage()
        if key in self.embedded:
            self._prepareStep(key)
        if enterStep and slicer.util.selectedModule() == "DynamicNeuroPET":
            self._callStep(key, "enter")
        self.refresh()

    def moveStep(self, delta):
        keys = [k for k, _, _, _ in STEPS]
        index = keys.index(self.current) + delta
        if 0 <= index < len(keys):
            self.showStep(keys[index])

    # --- state ---------------------------------------------------------------------------------------
    def stepStates(self):
        sequence, anatomy = self.sequence(), self.anatomy()
        states = {}
        states[STEP_DATA] = STATE_DONE if sequence is not None else STATE_READY
        ready = STATE_READY if sequence is not None else STATE_WAITING
        states[STEP_MC] = STATE_DONE if sequence is not None and sequence.GetAttribute(MC_PREFIX + "Method") else ready
        states[STEP_REG] = (STATE_DONE if sequence is not None and sequence.GetAttribute(REG_PREFIX + "Fixed") else
                            ready if anatomy is not None else STATE_WAITING)
        normalized = sequence is not None and bool(sequence.GetAttribute(NEURO_PREFIX + "Template"))
        states[STEP_NORM] = STATE_DONE if normalized else ready
        segmentation = self.node("Segmentation")
        states[STEP_REGIONS] = (STATE_DONE if segmentation is not None else
                                STATE_READY if normalized else STATE_WAITING)
        states[STEP_MODEL] = STATE_READY if segmentation is not None else STATE_WAITING
        return states

    def refresh(self):
        if not hasattr(self, "stepButtons"):
            return
        states = self.stepStates()
        for number, (key, _, title, explanation) in enumerate(STEPS, start=1):
            button = self.stepButtons[key]
            button.setIcon(_badge(states[key], number))
            button.checked = key == self.current
            if key == self.current:
                self.titleLabel.text = f"<b>{number}. {title}</b> - {explanation}"
        keys = [k for k, _, _, _ in STEPS]
        index = keys.index(self.current)
        self.backButton.enabled = index > 0
        self.nextButton.enabled = index < len(keys) - 1
        if self.current == STEP_DATA:
            self._updateDataInfo()
        self._updateNormButtons()
        self._updateRegionsPage()

    # --- module ------------------------------------------------------------------------------------
    def enter(self):
        self._entered = True
        self.selectInputs()
        self.showStep(self.current)

    def exit(self):
        if self.current in self.embedded:
            self._takeSeriesFrom(self.current)
        self._callStep(self.current, "exit")

    def cleanup(self):
        self._closeNormSession()
        self.refreshTimer.stop()
        self.removeObservers()
        for panel in self.embedded.values():   # the panels belong to their modules
            try:
                self.stack.removeWidget(panel)
                panel.setParent(None)
            except Exception:
                pass
        self.embedded = {}
        ScriptedLoadableModuleWidget.cleanup(self)

    def onSceneClose(self, caller, event):
        self._closeNormSession()
        self.normResult = None
        self.normUndo = None
        if hasattr(self, "normLabel"):
            self.normLabel.text = "Not normalized yet."


# ---------------------------------------------------------------------------
# Test: CerebrA labels into a segmentation; contrast; a synthetic normalization
# ---------------------------------------------------------------------------
class DynamicNeuroPETTest(ScriptedLoadableModuleTest):
    def setUp(self):
        slicer.mrmlScene.Clear()

    def runTest(self):
        self.setUp()
        self.test_Regions()
        self.setUp()
        self.test_Widget()

    def test_Regions(self):
        import tempfile
        folder = tempfile.mkdtemp()
        csvPath = os.path.join(folder, neuro.CSV_NAME)
        with open(csvPath, "w") as f:
            f.write("Mindboggle ID,Label Name,RH Label,LH Labels,Notes,Dice Kappa\n"
                    "51,Putamen,21,72,,0.87\n45,Cerebellum Gray Matter,46,97,,0.83\n")
        array = np.zeros((20, 20, 20), dtype=np.int16)
        array[2:6, 2:6, 2:6] = 21
        array[2:6, 10:14, 2:6] = 72
        array[12:16, 2:6, 2:6] = 46
        array[12:16, 10:14, 2:6] = 97
        source = slicer.util.addVolumeFromArray(array, name="labels", nodeClassName="vtkMRMLLabelMapVolumeNode")
        labelsPath = os.path.join(folder, "mni_icbm152_CerebrA_tal_nlin_sym_09c.nrrd")
        slicer.util.saveNode(source, labelsPath)
        slicer.mrmlScene.RemoveNode(source)
        logic = DynamicNeuroPETLogic()
        atlas = neuro.findAtlasFiles(folder)
        self.assertNotIn(neuro.BILATERAL_LABELS, atlas)
        # default: left / right regions, the bilateral cerebellar grey matter as the reference
        segmentation, referenceId = logic.loadAtlasRegions(atlas)
        names = sorted(segmentation.GetSegmentation().GetSegment(i).GetName() for i in logic.segmentIds(segmentation))
        self.assertEqual(names, sorted(["Putamen R", "Putamen L", "Cerebellum Gray Matter R",
                                        "Cerebellum Gray Matter L", "Cerebellum Gray Matter" + REFERENCE_SUFFIX]))
        self.assertIsNotNone(referenceId)
        mask = slicer.util.arrayFromSegmentBinaryLabelmap(segmentation, referenceId)
        self.assertEqual(int(np.count_nonzero(mask)), 2 * 64)
        right = logic.segmentIdByName(segmentation, "Cerebellum Gray Matter R")
        self.assertEqual(int(np.count_nonzero(slicer.util.arrayFromSegmentBinaryLabelmap(segmentation, right))), 64)
        # the bilateral files were saved next to CerebrA
        atlas = neuro.findAtlasFiles(folder)
        self.assertIn(neuro.BILATERAL_LABELS, atlas)
        self.assertEqual([n for _, n, _ in neuro.readBilateralLabels(atlas[neuro.BILATERAL_CSV])],
                         ["Putamen", "Cerebellum Gray Matter"])
        # bilateral regions with their own reference segment
        merged, mergedRef = logic.loadAtlasRegions(atlas, ATLAS_BILATERAL, ATLAS_BILATERAL)
        names = sorted(merged.GetSegmentation().GetSegment(i).GetName() for i in logic.segmentIds(merged))
        self.assertEqual(names, ["Cerebellum Gray Matter", "Putamen"])
        self.assertEqual(merged.GetSegmentation().GetSegment(mergedRef).GetName(), "Cerebellum Gray Matter")
        # a one-sided reference from the left / right atlas
        _, sideRef = logic.loadAtlasRegions(atlas, ATLAS_BILATERAL, ATLAS_LATERAL, "Cerebellum Gray Matter L")
        self.assertIsNotNone(sideRef)
        self.delayDisplay("Regions test passed")

    def test_Widget(self):
        widget = slicer.modules.dynamicneuropet.widgetRepresentation().self()
        for key, _, _, _ in STEPS:
            widget.showStep(key, enterStep=False)
            self.assertEqual(widget.current, key)
            self.assertEqual(widget.stack.count, 1)
        widget.showStep(STEP_DATA, enterStep=False)
        self.delayDisplay("Widget test passed")
