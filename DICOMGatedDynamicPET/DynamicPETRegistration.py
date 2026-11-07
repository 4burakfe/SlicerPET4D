"""
Dynamic PET to CT/MR Registration (3D Slicer module)

Registers a dynamic PET series (as loaded by the DICOM Gated / Dynamic PET plugin) to a CT or MR with Slicer's
BRAINSFit: one reference image of the series (the summed image by default, or a frame) is registered in up to three
steps run one after the other (rigid, then affine, then B-spline by default; each step starts where the previous one
ended, the first from the geometric centres), and the same transform moves every frame and the summed image.

Layout:
    2 rows (axial on top, coronal below) x 3 columns: PET alone (left), PET over the CT/MR (fusion, middle), CT/MR
    alone (right). After a registration a checkerboard column is added on the left (2 x 4): a checkerboard made on the
    PET grid and moved by the same transform, over the PET, so the deformation can be seen.
The PET colours and window, the fusion opacity and the CT/MR window (presets) are adjustable. "Correct manually" shows
move / rotate handles (as in EasyReg) on a manual transform applied on top of the registration. Harden resamples every
frame (and the summed image) onto the CT/MR field of view and either overwrites the series or saves a new one; Revert
changes goes back.

Transforms (PET -> CT/MR, Slicer's "to parent" direction):  PET  ->  linear (all linear steps)  ->  B-spline (when the
last step is one)  ->  manual  ->  world. The module's own views show the PET through hidden helper volumes that share
its voxels, so the user's volumes are not moved until Harden.
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

from DICOMGatedDynamicPETLib import colormaps, dynamicviews  # noqa: E402
from DICOMGatedDynamicPETLib import registration as reg  # noqa: E402
from DICOMGatedDynamicPETLib import resampling  # noqa: E402
from DICOMGatedDynamicPETLib.hardendialog import hardenDialog  # noqa: E402
from DynamicPETEditor import guardViewVolumes  # noqa: E402

LAYOUT_ID = 7803                 # 2 x 3
CHECKER_LAYOUT_ID = 7804         # 2 x 4, with the checkerboard column
PREFIX = "DynamicPET."
REG_PREFIX = "DynamicPET.Registration."
HELPER_ATTRIBUTE = "DynamicPETRegistration.Helper"
REGISTERED_SUFFIX = " registered"
PET_COLORMAP = "Inverted grey"   # PET column (and under the checkerboard)
FUSION_COLORMAP = "PET-DICOM"    # PET over the CT/MR
CHECKER_COLOR_ID = "vtkMRMLColorTableNodeGreen"
DEFAULT_FUSION_OPACITY = 0.5
DEFAULT_CHECKER_OPACITY = 0.35
THRESHOLD_FRACTION = 0.03        # fusion PET voxels below 3 % of the PET window are transparent over the CT/MR
AUTO_PERCENTILE = 99.9
SUMMED = "summed"


class DynamicPETRegistration(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        parent.title = "Dynamic PET to CT/MR Registration"
        parent.categories = ["Nuclear Medicine.PET4D"]
        parent.dependencies = []
        parent.contributors = ["Burak Demir"]
        parent.helpText = (
            "Registers a dynamic PET series to a CT or MR with BRAINSFit. Pick the dynamic sequence, the reference "
            "image (the summed image by default, or a frame) and the CT/MR; the reference is registered in up to "
            "three steps (rigid, affine, B-spline by default) and the same transform moves every frame and the "
            "summed image. Check the result in the PET / fusion / CT-MR views and the checkerboard, correct it "
            "by hand if needed, then Harden (overwrite the series or save a new one).")
        parent.acknowledgementText = ("Part of the DICOM Gated / Dynamic PET extension. Registration by BRAINSFit "
                                      "(Johnson HJ et al., BRAINSFit: Mutual Information Rigid Registrations of "
                                      "Whole-Brain 3D Images, Using the Insight Toolkit, Insight Journal 2007).")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _vtkMatrix(array):
    return resampling.vtkMatrix(array)


def _numpy(matrix):
    return resampling.numpyMatrix(matrix)


def _ijkToRas(node):
    matrix = vtk.vtkMatrix4x4()
    node.GetIJKToRASMatrix(matrix)
    return _numpy(matrix)


def _percentile(array):
    sample = np.asarray(array).ravel()
    sample = sample[::4] if sample.size > 400000 else sample
    sample = sample[np.isfinite(sample)]
    return float(np.percentile(sample, AUTO_PERCENTILE)) if sample.size else 1.0


def _center(node, world=False):
    bounds = [0.0] * 6
    (node.GetRASBounds if world else node.GetBounds)(bounds)
    return [(bounds[2 * a] + bounds[2 * a + 1]) / 2.0 for a in range(3)]


def setDisplacementField(transformNode, displacement, ijkToWorld):
    """Displacement field [k, j, i, 3] (world RAS mm, from parent: the child point of world point x is x + d(x)) on a
    grid placed by ijkToWorld, as the transform from parent of a grid transform node (as in EasyReg)."""
    from vtk.util import numpy_support
    matrix = np.asarray(ijkToWorld, dtype=float)
    spacing = np.linalg.norm(matrix[:3, :3], axis=0)
    direction = np.eye(4)
    direction[:3, :3] = matrix[:3, :3] / spacing[None, :]
    nk, nj, ni = displacement.shape[:3]
    grid = vtk.vtkImageData()
    grid.SetOrigin(*matrix[:3, 3])
    grid.SetSpacing(*spacing)
    grid.SetExtent(0, ni - 1, 0, nj - 1, 0, nk - 1)
    vectors = numpy_support.numpy_to_vtk(np.ascontiguousarray(displacement, dtype=np.float64).reshape(-1, 3),
                                         deep=True, array_type=vtk.VTK_DOUBLE)
    vectors.SetNumberOfComponents(3)
    grid.GetPointData().SetScalars(vectors)
    gridTransform = slicer.vtkOrientedGridTransform()
    gridTransform.SetDisplacementGridData(grid)
    gridTransform.SetGridDirectionMatrix(_vtkMatrix(direction))
    gridTransform.SetInterpolationModeToCubic()
    transformNode.SetAndObserveTransformFromParent(gridTransform)


def _modality(node):
    """'CT', 'MR', 'PT', ... from the DICOM database for a volume loaded from DICOM, else ''."""
    try:
        uids = (node.GetAttribute("DICOM.instanceUIDs") or "").split()
        if uids and slicer.dicomDatabase:
            fileName = slicer.dicomDatabase.fileForInstance(uids[0])
            return (slicer.dicomDatabase.fileValue(fileName, "0008,0060") or "").strip().upper()
    except Exception:
        pass
    return ""


def _isPet(node):
    return any(key.startswith(PREFIX) for key in (node.GetAttributeNames() or ())) or _modality(node) == "PT"


def fixedCandidates():
    """Volumes that can be the CT/MR: shown scalar volumes that are not PET (CT and MR from DICOM first)."""
    nodes = [n for n in slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode")
             if not n.GetHideFromEditors() and not n.IsA("vtkMRMLLabelMapVolumeNode")
             and not n.GetAttribute(HELPER_ATTRIBUTE) and not _isPet(n)]
    return sorted(nodes, key=lambda n: 0 if _modality(n) in ("CT", "MR") else 1)


def _looksLikeCT(node):
    if _modality(node) == "CT":
        return True
    if _modality(node) == "MR":
        return False
    array = slicer.util.arrayFromVolume(node)
    return array.size > 0 and float(np.min(array[::4, ::4, ::4])) < -500.0


def iterativeOptions(form, gridMm=reg.DEFAULT_ITERATIVE_GRID_MM, linear=reg.DEFAULT_ITERATIVE_LINEAR,
                     deformable=reg.DEFAULT_ITERATIVE_DEFORMABLE):
    """The iterative engine's options in a form layout (also used by the Neuro normalization). Returns the widgets
    for iterativeSettings()."""
    from DICOMGatedDynamicPETLib import sitkreg as SR
    linearCombo = qt.QComboBox()
    for name in reg.ITERATIVE_LINEAR_TYPES:
        linearCombo.addItem(name, name)
    linearCombo.setCurrentIndex(linearCombo.findData(linear))
    linearCombo.toolTip = "Linear stage, run first: rigid (shifts and angles) or affine (adds scaling and shear)"
    form.addRow("Linear stage:", linearCombo)
    deformableCheck = qt.QCheckBox("Then deformable (iterative B-spline passes)")
    deformableCheck.checked = deformable
    deformableCheck.toolTip = (f"{SR.DEFAULT_DEFORMABLE_PASSES} B-spline passes after the linear stage, the control "
                               "points twice as close each pass, ending at the grid below; a pass is kept only when "
                               "it improves the match and does not fold the image")
    form.addRow(deformableCheck)
    gridCombo = qt.QComboBox()
    for title, mm in SR.GRID_SPACING_CHOICES:
        gridCombo.addItem(title.replace("; liver", ""), mm)
    index = gridCombo.findData(gridMm)
    gridCombo.setCurrentIndex(index if index >= 0 else 0)
    gridCombo.toolTip = "Control-point spacing of the last deformable pass: coarser is stiffer"
    form.addRow("Deformable grid:", gridCombo)
    spacingCombo = qt.QComboBox()
    for title, mm in SR.FINAL_SPACING_CHOICES:
        spacingCombo.addItem(title, mm)
    spacingCombo.setCurrentIndex(spacingCombo.findData(reg.DEFAULT_ITERATIVE_SPACING_MM))
    spacingCombo.toolTip = ("Resolution of the last level (the images are averaged down to it): 2 mm is plenty for "
                            "PET; finer takes longer")
    form.addRow("Final resolution:", spacingCombo)
    qualityCombo = qt.QComboBox()
    for name in SR.QUALITY_SAMPLES:
        qualityCombo.addItem(name, name)
    qualityCombo.setCurrentIndex(qualityCombo.findData(reg.DEFAULT_ITERATIVE_QUALITY))
    qualityCombo.toolTip = "Number of sample points of the metric per level"
    form.addRow("Quality:", qualityCombo)
    deformableCheck.connect("toggled(bool)", lambda on: setattr(gridCombo, "enabled", on))
    return {"linear": linearCombo, "deformable": deformableCheck, "grid": gridCombo, "spacing": spacingCombo,
            "quality": qualityCombo}


def iterativeSettings(widgets):
    """Settings of the iterative engine from the widgets of iterativeOptions()."""
    def data(combo):
        return combo.itemData(combo.currentIndex)
    return {"iterativeLinear": data(widgets["linear"]) or reg.DEFAULT_ITERATIVE_LINEAR,
            "iterativeDeformable": bool(widgets["deformable"].checked),
            "iterativeGridMm": float(data(widgets["grid"]) or reg.DEFAULT_ITERATIVE_GRID_MM),
            "iterativeSpacingMm": float(data(widgets["spacing"]) or reg.DEFAULT_ITERATIVE_SPACING_MM),
            "iterativeQuality": data(widgets["quality"]) or reg.DEFAULT_ITERATIVE_QUALITY}


# ---------------------------------------------------------------------------
# Logic
# ---------------------------------------------------------------------------
class DynamicPETRegistrationLogic(ScriptedLoadableModuleLogic):
    """Registration of a PET reference image to a CT/MR and Harden of the whole series. Settings (dict):
        steps                BRAINSFit transform types run one after the other (reg.validateSteps)
        resample             per step: register to the CT/MR resampled to the PET voxel size (reg.resampleFlags;
                             default only the B-spline)
        initialization       BRAINSFit initializeTransformMode of the first step (geometric centres by default)
        samplingPercentage, bsplineGrid    BRAINSFit details
        workingVoxelMm       voxel size of the CT/MR copy the resampled steps register to (0: the PET voxel size; a
                             CT/MR that is not finer is used as it is), see workingFixed()
        engine               reg.ENGINE_BRAINSFIT (the steps above) or reg.ENGINE_ITERATIVE (registerIterative: the
                             multi-resolution engine of EasyReg, a rigid / affine stage then iterative B-spline passes)
        iterativeLinear, iterativeDeformable, iterativeGridMm, iterativeSpacingMm, iterativeQuality
                             the iterative engine's linear stage, deformable passes on / off, control-point spacing of
                             the last pass, final working resolution (mm) and quality (sitkreg.QUALITY_SAMPLES)
        movingFunctional     the moving image is a PET (True) or an anatomical image (iterative engine intensities)
    """

    @staticmethod
    def defaultSettings():
        return {"engine": reg.DEFAULT_ENGINE,
                "steps": list(reg.DEFAULT_STEPS), "resample": reg.resampleFlags(reg.DEFAULT_STEPS),
                "initialization": reg.DEFAULT_INITIALIZATION,
                "samplingPercentage": reg.DEFAULT_SAMPLING, "bsplineGrid": reg.DEFAULT_BSPLINE_GRID,
                "workingVoxelMm": reg.DEFAULT_WORKING_VOXEL_MM,
                "iterativeLinear": reg.DEFAULT_ITERATIVE_LINEAR,
                "iterativeDeformable": reg.DEFAULT_ITERATIVE_DEFORMABLE,
                "iterativeGridMm": reg.DEFAULT_ITERATIVE_GRID_MM,
                "iterativeSpacingMm": reg.DEFAULT_ITERATIVE_SPACING_MM,
                "iterativeQuality": reg.DEFAULT_ITERATIVE_QUALITY, "movingFunctional": True}

    @staticmethod
    def helperVolume(name, source=None):
        """A hidden volume that is not saved with the scene; shares the voxels and geometry of source."""
        node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", name)
        node.SetHideFromEditors(True)
        node.SetSaveWithScene(False)
        node.SetAttribute(HELPER_ATTRIBUTE, "1")
        if source is not None:
            DynamicPETRegistrationLogic.shareVoxels(node, source)
        return node

    @staticmethod
    def shareVoxels(node, source):
        node.SetAndObserveImageData(source.GetImageData())
        node.SetIJKToRASMatrix(_vtkMatrix(_ijkToRas(source)))

    @staticmethod
    def transformNode(className, name):
        node = slicer.mrmlScene.AddNewNodeByClass(className, name)
        node.SetHideFromEditors(True)
        node.SetSaveWithScene(False)
        node.SetAttribute(HELPER_ATTRIBUTE, "1")
        return node

    @staticmethod
    def _runCli(module, parameters):
        cliNode = slicer.cli.runSync(module, None, parameters, update_display=False)
        try:
            if cliNode.GetStatus() & cliNode.ErrorsMask:
                raise RuntimeError(cliNode.GetErrorText() or "CLI failed")
        finally:
            slicer.mrmlScene.RemoveNode(cliNode)

    def registerStep(self, fixed, moving, transformType, settings, initialization):
        """One BRAINSFit run (moving -> fixed); returns the output transform node."""
        className = "vtkMRMLBSplineTransformNode" if transformType == "BSpline" else "vtkMRMLLinearTransformNode"
        output = self.transformNode(className, f"PET registration {transformType}")
        parameters = {
            "fixedVolume": fixed.GetID(),
            "movingVolume": moving.GetID(),
            "transformType": transformType,
            "initializeTransformMode": initialization,
            "costMetric": "MMI",
            "samplingPercentage": float(settings["samplingPercentage"]),
            "interpolationMode": "Linear",
            "maskProcessingMode": "NOMASK",
        }
        if transformType == "BSpline":
            grid = int(settings["bsplineGrid"])
            parameters["bsplineTransform"] = output.GetID()
            parameters["splineGridSize"] = f"{grid},{grid},{grid}"
        else:
            parameters["linearTransform"] = output.GetID()
        try:
            self._runCli(slicer.modules.brainsfit, parameters)
        except Exception:
            slicer.mrmlScene.RemoveNode(output)
            raise
        return output

    def workingFixed(self, fixed, petSource, voxelMm=None):
        """The CT/MR BRAINSFit registers to: a hidden copy smoothed and resampled to the PET voxel size (or voxelMm)
        over the same field of view, or None when the CT/MR is not finer (it is used as it is). A sub-millimetre MR
        otherwise makes the B-spline step use tens of GB of memory."""
        return self.workingCopy(fixed, reg.gridSpacing(_ijkToRas(petSource)), voxelMm)

    def workingCopy(self, volume, targetSpacing, voxelMm=None):
        """A hidden copy of volume smoothed and resampled to targetSpacing (i, j, k; or voxelMm along every axis)
        over the same field of view, never finer than the volume itself; None when the volume is not finer."""
        volumeMatrix = _ijkToRas(volume)
        volumeSpacing = reg.gridSpacing(volumeMatrix)
        spacing = reg.workingSpacing(volumeSpacing, targetSpacing, voxelMm)
        if spacing is None:
            return None
        dims, matrix = reg.outputGrid(volume.GetImageData().GetDimensions(), volumeMatrix, spacing)
        resampler = resampling.Resampler(volumeMatrix, dims, matrix)
        working = self.helperVolume(f"{volume.GetName()} registration working copy")
        working.SetAndObserveImageData(resampler.resample(volume.GetImageData(),
                                                          reg.smoothingSigmas(volumeSpacing, spacing)))
        working.SetIJKToRASMatrix(_vtkMatrix(matrix))
        logging.info(f"Dynamic PET registration: '{volume.GetName()}' registered at "
                     f"{', '.join(f'{v:.3g}' for v in spacing)} mm ({dims[0]} x {dims[1]} x {dims[2]})")
        return working

    def registerIterative(self, fixed, moving, settings, progress=None):
        """The iterative multi-resolution engine (sitkreg): moving (not under a transform) registered to fixed with a
        rigid / affine stage, then B-spline passes coarse to fine, on a working grid of a few mm (little memory).
        Returns the same dict as register(): "linear" is the linear matrix, "bspline" a hidden grid transform node with
        the deformable part (to go under the linear one), "deformation" its stats; also "notes" and "metric"."""
        from DICOMGatedDynamicPETLib import sitkreg as SR
        if fixed.GetParentTransformNode() is not None:
            raise ValueError(f"The CT/MR '{fixed.GetName()}' is under a transform. Harden or remove it first.")
        if not SR.sitkAvailable():
            raise RuntimeError("SimpleITK is not available in this Slicer.")
        defaults = self.defaultSettings()
        get = lambda key: settings.get(key, defaults[key])  # noqa: E731
        linear, deformable = get("iterativeLinear"), bool(get("iterativeDeformable"))
        steps = reg.iterativeSteps(linear, deformable)
        spacingMm = float(get("iterativeSpacingMm"))
        fixedArray = slicer.util.arrayFromVolume(fixed)
        fixedMatrix = _ijkToRas(fixed)
        fieldShape = fixedArray.shape
        # the inputs are averaged down to about the working resolution first (a sub-millimetre MR is never copied at
        # full resolution); the deformation field still covers the CT/MR's own field of view
        fixedArray, fixedWorkingMatrix = reg.poolToSpacing(fixedArray, fixedMatrix, spacingMm)
        movingArray, movingMatrix = reg.poolToSpacing(slicer.util.arrayFromVolume(moving), _ijkToRas(moving),
                                                      spacingMm)
        images = (SR.ImageInput(fixedArray, fixedWorkingMatrix, SR.imageKind(fixedArray)),
                  SR.ImageInput(movingArray, movingMatrix,
                                SR.imageKind(movingArray, functional=bool(get("movingFunctional")))))
        state = {"cancelled": False}

        def report(text):
            if progress is not None and progress(0, len(steps), text) is False:
                state["cancelled"] = True

        result = {"steps": [], "linear": np.eye(4), "bspline": None, "deformation": None, "center": _center(moving),
                  "stopped": False, "notes": [], "metric": None}
        try:
            output = SR.register(images[0], images[1], None, SR.AFFINE if linear == "Affine" else SR.RIGID,
                                 spacingMm, get("iterativeQuality"), None,
                                 reg.iterativeInitialization(get("initialization")), progress=report,
                                 isCancelled=lambda: state["cancelled"], deformable=deformable,
                                 gridSpacingMm=float(get("iterativeGridMm")),
                                 fieldGeometry=(fieldShape, fixedMatrix) if deformable else None)
        except SR.RegistrationCancelled:
            result["stopped"] = True
            return result
        result.update(linear=np.asarray(output["matrix"], dtype=float), notes=list(output["notes"]),
                      metric=(output["initialMetric"], output["finalMetric"]))
        displacement = output.get("displacement")
        if displacement is not None:
            residual = reg.residualField(displacement, output["fieldIjkToWorld"], result["linear"])
            node = self.transformNode("vtkMRMLGridTransformNode", "PET registration deformable")
            setDisplacementField(node, residual, output["fieldIjkToWorld"])
            result["bspline"] = node
            result["deformation"] = reg.fieldStats(residual)
            result["steps"] = steps
        else:
            result["steps"] = steps[:1]
        logging.info(f"Dynamic PET to CT/MR registration (iterative) of '{moving.GetName()}' to '{fixed.GetName()}' "
                     f"in {output['seconds']:.0f} s: " + " ".join(result["notes"]))
        return result

    def register(self, fixed, petSource, settings, progress=None):
        """Register petSource (a PET volume, not under a transform) to fixed (CT/MR) in the settings' steps. Every
        step starts where the previous ended: the linear steps are chained by moving the PET's geometry (lossless),
        and a B-spline step registers the PET as the linear steps left it. progress(index, total, text) returns
        False to stop. Returns {"steps", "linear" (4x4, PET -> CT/MR), "bspline" (B-spline transform node, hidden,
        from the PET after the linear steps to the CT/MR, or None), "deformation" (reg.displacementStats of the
        B-spline, or None), "center" (centre of the PET, mm), "stopped"}. With settings["engine"] ENGINE_ITERATIVE the
        iterative engine runs instead (registerIterative)."""
        if settings.get("engine") == reg.ENGINE_ITERATIVE:
            return self.registerIterative(fixed, petSource, settings, progress)
        steps = reg.validateSteps(settings["steps"])
        if fixed.GetParentTransformNode() is not None:
            raise ValueError(f"The CT/MR '{fixed.GetName()}' is under a transform. Harden or remove it first.")
        flags = reg.resampleFlags(steps, settings.get("resample"))
        working = self.workingFixed(fixed, petSource, settings.get("workingVoxelMm") or None) if any(flags) else None
        # a moving image finer than the grid it registers to (e.g. a sub-millimetre MR normalized to the 1 mm
        # template) is resampled to that grid's voxel size first: same transform, a fraction of the memory
        targetSpacing = reg.gridSpacing(_ijkToRas(working if working is not None else fixed))
        moving = self.workingCopy(petSource, targetSpacing) if settings.get("resampleMoving", True) else None
        if moving is None:
            moving = self.helperVolume("PET registration moving", petSource)
        original = _ijkToRas(moving)
        current = np.eye(4)
        result = {"steps": [], "linear": current, "bspline": None, "deformation": None,
                  "center": _center(petSource), "stopped": False}
        try:
            for index, step in enumerate(steps):
                if progress is not None and progress(index, len(steps), f"step {index + 1} of {len(steps)}: {step}") \
                        is False:
                    result["stopped"] = True
                    break
                moving.SetIJKToRASMatrix(_vtkMatrix(current @ original))
                initialization = settings.get("initialization", reg.DEFAULT_INITIALIZATION) if index == 0 else "Off"
                target = working if flags[index] and working is not None else fixed
                transform = self.registerStep(target, moving, step, settings, initialization)
                if reg.isLinearStep(step):
                    matrix = vtk.vtkMatrix4x4()
                    transform.GetMatrixTransformToParent(matrix)
                    current = _numpy(matrix) @ current
                    slicer.mrmlScene.RemoveNode(transform)
                else:
                    bounds = [0.0] * 6
                    moving.GetRASBounds(bounds)
                    points = reg.samplePoints(bounds)
                    toParent = transform.GetTransformToParent()
                    moved = [toParent.TransformPoint([float(v) for v in p]) for p in points]
                    result["bspline"] = transform
                    result["deformation"] = reg.displacementStats(points, moved)
                result["steps"].append(step)
                result["linear"] = current
        except Exception:
            if result["bspline"] is not None:
                slicer.mrmlScene.RemoveNode(result["bspline"])
            raise
        finally:
            slicer.mrmlScene.RemoveNode(moving)
            if working is not None:
                slicer.mrmlScene.RemoveNode(working)
        logging.info(f"Dynamic PET to CT/MR registration of '{petSource.GetName()}' to '{fixed.GetName()}': "
                     + "; ".join(reg.summaryLines(result["steps"], result["linear"], result["center"],
                                                  result["deformation"])))
        return result

    # --- harden / revert --------------------------------------------------------------------------
    @staticmethod
    def summedVolumeOf(sequenceNode):
        nodeID = sequenceNode.GetAttribute(PREFIX + "SummedVolumeID")
        return slicer.mrmlScene.GetNodeByID(nodeID) if nodeID else None

    @staticmethod
    def outputGeometry(fixed, petNode, grid, customSpacing=None):
        """(dims, ijkToRas) of the Harden grid: the CT/MR field of view with the voxel size of grid."""
        fixedMatrix = _ijkToRas(fixed)
        spacing = reg.outputSpacing(grid, reg.gridSpacing(_ijkToRas(petNode)), reg.gridSpacing(fixedMatrix),
                                    customSpacing)
        return reg.outputGrid(fixed.GetImageData().GetDimensions(), fixedMatrix, spacing)

    @staticmethod
    def worldToPet(linearNode):
        """(vtkGeneralTransform from the world to the PET's own coordinates, linear) of the registration chain whose
        PET end is linearNode."""
        transform = vtk.vtkGeneralTransform()
        slicer.vtkMRMLTransformNode.GetTransformBetweenNodes(None, linearNode, transform)
        return transform, bool(linearNode.IsTransformToWorldLinear())

    def harden(self, sequenceNode, summedNode, fixed, linearNode, action, grid, customSpacing=None,
               interpolation=reg.DEFAULT_INTERPOLATION, description=""):
        """Resample every frame of the sequence (and summedNode, when given) through the registration chain onto the
        Harden grid. action reg.OVERWRITE replaces the voxels and geometry of the frames and of the summed image;
        reg.SAVE_SEPARATELY makes a new sequence '<name> registered' (with its browser) and a new summed image next
        to the originals. Returns (undo record for revert(), sequence, summed)."""
        first = sequenceNode.GetNthDataNode(0)
        dims, outputMatrix = self.outputGeometry(fixed, first, grid, customSpacing)
        worldToPet, linear = self.worldToPet(linearNode)
        resamplers = {}

        def resampled(node):
            matrix = _ijkToRas(node)
            key = matrix.round(6).tobytes()
            if key not in resamplers:
                resamplers[key] = resampling.Resampler(matrix, dims, outputMatrix, worldToPet, linear, interpolation)
            return resamplers[key].resample(node.GetImageData())

        attributes = {REG_PREFIX + "Fixed": fixed.GetName(), REG_PREFIX + "Transform": description,
                      REG_PREFIX + "Interpolation": interpolation,
                      REG_PREFIX + "VoxelSizeMm": ", ".join(f"{v:.3g}" for v in reg.gridSpacing(outputMatrix))}
        count = sequenceNode.GetNumberOfDataNodes()
        if action == reg.OVERWRITE:
            changed = []
            nodes = [sequenceNode.GetNthDataNode(i) for i in range(count)]
            if summedNode is not None:
                nodes.append(summedNode)
            images = [resampled(node) for node in nodes]        # all first: a failure leaves the series as it was
            for node, image in zip(nodes, images):
                changed.append((node, node.GetImageData(), _ijkToRas(node),
                                {k: node.GetAttribute(k) for k in attributes}))
                node.SetAndObserveImageData(image)
                node.SetIJKToRASMatrix(_vtkMatrix(outputMatrix))
                for key, value in attributes.items():
                    node.SetAttribute(key, value)
            sequenceAttributes = {k: sequenceNode.GetAttribute(k) for k in attributes}
            for key, value in attributes.items():
                sequenceNode.SetAttribute(key, value)
            sequenceNode.Modified()
            self.refreshProxies(sequenceNode)
            logging.info(f"Dynamic PET to CT/MR registration: '{sequenceNode.GetName()}' overwritten ({count} "
                         f"frames{', and its summed image' if summedNode is not None else ''}) on the grid of "
                         f"'{fixed.GetName()}'")
            return {"action": action, "changed": changed, "sequence": sequenceNode,
                    "sequenceAttributes": sequenceAttributes}, sequenceNode, summedNode

        from DynamicPETMotionCorrection import DynamicPETMotionCorrectionLogic
        mcLogic = DynamicPETMotionCorrectionLogic()
        scene = slicer.mrmlScene
        name = scene.GenerateUniqueName(sequenceNode.GetName() + REGISTERED_SUFFIX)
        newSequence = scene.AddNewNodeByClass("vtkMRMLSequenceNode", name)
        newSequence.SetIndexName(sequenceNode.GetIndexName())
        newSequence.SetIndexUnit(sequenceNode.GetIndexUnit())
        newSequence.SetIndexType(sequenceNode.GetIndexType())
        for key in sequenceNode.GetAttributeNames() or ():
            if key != PREFIX + "SummedVolumeID":
                newSequence.SetAttribute(key, sequenceNode.GetAttribute(key))
        for key, value in attributes.items():
            newSequence.SetAttribute(key, value)
        newSequence.SetAttribute(REG_PREFIX + "OriginalSequence", sequenceNode.GetName())
        try:
            for i in range(count):
                source = sequenceNode.GetNthDataNode(i)
                frame = mcLogic._frameCopy(source, resampled(source))
                frame.SetIJKToRASMatrix(_vtkMatrix(outputMatrix))
                for key, value in attributes.items():
                    frame.SetAttribute(key, value)
                newSequence.SetDataNodeAtValue(frame, sequenceNode.GetNthIndexValue(i))
            browser, proxy = mcLogic._browserOf(newSequence, sequenceNode)
            newSummed = None
            if summedNode is not None:
                newSummed = scene.AddNewNodeByClass("vtkMRMLScalarVolumeNode",
                                                    scene.GenerateUniqueName(summedNode.GetName() + REGISTERED_SUFFIX))
                newSummed.SetAndObserveImageData(resampled(summedNode))
                newSummed.SetIJKToRASMatrix(_vtkMatrix(outputMatrix))
                for key in summedNode.GetAttributeNames() or ():
                    newSummed.SetAttribute(key, summedNode.GetAttribute(key))
                for key, value in attributes.items():
                    newSummed.SetAttribute(key, value)
                newSummed.SetAttribute(PREFIX + "DynamicSequence", newSequence.GetID())
                try:
                    newSummed.SetVoxelValueUnits(summedNode.GetVoxelValueUnits())
                except Exception:
                    pass
                newSummed.CreateDefaultDisplayNodes()
                if summedNode.GetDisplayNode() is not None:
                    newSummed.GetDisplayNode().CopyContent(summedNode.GetDisplayNode())
                newSequence.SetAttribute(PREFIX + "SummedVolumeID", newSummed.GetID())
        except Exception:
            scene.RemoveNode(newSequence)
            raise
        created = [n for n in (browser, proxy, newSequence, newSummed) if n is not None]
        mcLogic._placeNextTo(sequenceNode, created)
        logging.info(f"Dynamic PET to CT/MR registration: '{name}' made from '{sequenceNode.GetName()}' on the grid "
                     f"of '{fixed.GetName()}'")
        return {"action": action, "created": created, "sequence": sequenceNode}, newSequence, newSummed

    def revert(self, undo):
        """Undo harden(): remove the new series, or put the overwritten voxels, geometry and tags back."""
        if undo["action"] == reg.OVERWRITE:
            for node, image, matrix, attributes in undo["changed"]:
                if not slicer.mrmlScene.IsNodePresent(node) and not self._inSequence(undo["sequence"], node):
                    continue
                node.SetAndObserveImageData(image)
                node.SetIJKToRASMatrix(_vtkMatrix(matrix))
                for key, value in attributes.items():
                    node.SetAttribute(key, value) if value is not None else node.RemoveAttribute(key)
            sequence = undo["sequence"]
            for key, value in undo["sequenceAttributes"].items():
                sequence.SetAttribute(key, value) if value is not None else sequence.RemoveAttribute(key)
            sequence.Modified()
            self.refreshProxies(sequence)
            return
        for node in undo["created"]:
            if slicer.mrmlScene.IsNodePresent(node):
                slicer.mrmlScene.RemoveNode(node)

    @staticmethod
    def _inSequence(sequence, node):
        return any(sequence.GetNthDataNode(i) is node for i in range(sequence.GetNumberOfDataNodes()))

    @staticmethod
    def refreshProxies(sequenceNode):
        try:
            logic = slicer.modules.sequences.logic()
            browsers = vtk.vtkCollection()
            logic.GetBrowserNodesForSequenceNode(sequenceNode, browsers)
            for b in range(browsers.GetNumberOfItems()):
                logic.UpdateProxyNodesFromSequences(browsers.GetItemAsObject(b))
        except Exception:
            logging.warning("Could not refresh the sequence browser", exc_info=True)


# ---------------------------------------------------------------------------
# Session: layout, helper volumes and transforms
# ---------------------------------------------------------------------------
class RegistrationSession:
    """The registration layout of one PET reference and one CT/MR: PET alone, PET over the CT/MR, CT/MR alone (and the
    checkerboard over the PET), in an axial and a coronal row; the transform chain (linear, B-spline, manual) that
    moves the PET helpers."""

    def __init__(self, fixed):
        self.fixed = fixed
        self.petView = None         # PET column (and under the checkerboard)
        self.petFusion = None       # PET over the CT/MR
        self.fixedHelper = None     # CT/MR in grey with the module's window
        self.checkers = {}          # orientation -> checkerboard on the PET grid, in columns across that row's slices
        self.linear = None          # linear transform of the automatic steps
        self.bspline = None         # B-spline of the last step (owned by the widget's result)
        self.manual = None          # manual correction (handles)
        self.observers = []
        self.guarded = False
        self.previousLayout = None
        self.showCheckerboard = False
        self.linked = True
        self.petColor = PET_COLORMAP
        self.fusionColor = FUSION_COLORMAP
        self.fusionOpacity = DEFAULT_FUSION_OPACITY
        self.checkerOpacity = DEFAULT_CHECKER_OPACITY
        self.checkerMm = reg.DEFAULT_CHECKER_MM
        self.petWindow = (0.0, 1.0)
        self.petMaximum = 1.0
        self.fixedWindow = (0.0, 1.0)
        self.fixedRange = (0.0, 1.0)
        self.onManualChanged = None   # callback() when the manual transform changes (handles)
        self.manualCenter = None

    # --- open / close ---------------------------------------------------------------------------
    def open(self, petSource):
        logic = DynamicPETRegistrationLogic
        self.linear = logic.transformNode("vtkMRMLLinearTransformNode", "PET registration (automatic)")
        self.manual = logic.transformNode("vtkMRMLLinearTransformNode", "PET registration (manual)")
        self.linear.SetAndObserveTransformNodeID(self.manual.GetID())
        tag = self.manual.AddObserver(slicer.vtkMRMLTransformableNode.TransformModifiedEvent, self._onManualModified)
        self.observers.append((self.manual, tag))
        self.fixedHelper = logic.helperVolume(f"{self.fixed.GetName()} registration view", self.fixed)
        self.fixedHelper.CreateDefaultDisplayNodes()
        display = self.fixedHelper.GetDisplayNode()
        display.SetAndObserveColorNodeID("vtkMRMLColorTableNodeGrey")
        display.SetInterpolate(True)
        array = slicer.util.arrayFromVolume(self.fixed)
        self.fixedRange = (float(np.nanmin(array)), float(np.nanmax(array))) if array.size else (0.0, 1.0)
        self.setPetSource(petSource)
        self._setLayout()

    def close(self):
        for node, tag in self.observers:
            node.RemoveObserver(tag)
        self.observers = []
        self.guarded = False
        self.setManualHandles(False)
        layoutManager = slicer.app.layoutManager()
        if layoutManager.layout in (LAYOUT_ID, CHECKER_LAYOUT_ID):
            self._link(False)
            if self.previousLayout is not None:
                layoutManager.setLayout(self.previousLayout)
        if self.bspline is not None and slicer.mrmlScene.IsNodePresent(self.bspline):
            self.bspline.SetAndObserveTransformNodeID(None)
        for node in [self.petView, self.petFusion, self.fixedHelper, self.linear, self.manual] + \
                list(self.checkers.values()):
            if node is not None and slicer.mrmlScene.IsNodePresent(node):
                slicer.mrmlScene.RemoveNode(node)
        self.petView = self.petFusion = self.fixedHelper = self.linear = self.manual = None
        self.checkers = {}
        self.bspline = None

    # --- volumes --------------------------------------------------------------------------------
    def setPetSource(self, source):
        """The PET reference shown (and registered): its voxels shared by the PET, fusion and checkerboard helpers,
        all under the registration chain."""
        logic = DynamicPETRegistrationLogic
        old = [self.petView, self.petFusion]
        self.petView = logic.helperVolume(f"{source.GetName()} registration PET", source)
        self.petFusion = logic.helperVolume(f"{source.GetName()} registration fusion", source)
        array = slicer.util.arrayFromVolume(source)
        self.petMaximum = float(np.nanmax(array)) if array.size else 1.0
        self.petWindow = (0.0, _percentile(array))
        for node in (self.petView, self.petFusion):
            node.CreateDefaultDisplayNodes()
            node.GetDisplayNode().SetInterpolate(True)
            node.SetAndObserveTransformNodeID(self.linear.GetID())
        self.setPetColor(self.petColor)
        self.setFusionColor(self.fusionColor)
        self.setPetWindow(*self.petWindow)
        self._makeChecker(source)
        self._assignVolumes()
        for node in old:
            if node is not None and slicer.mrmlScene.IsNodePresent(node):
                slicer.mrmlScene.RemoveNode(node)

    def _makeChecker(self, source):
        """One checkerboard per row, on the PET grid: columns along the grid axis closest to the row's slice normal
        (S for axial, A for coronal), so every slice of the row shows the same flat 2D checkerboard, bent only by the
        transform."""
        old = list(self.checkers.values())
        self.checkers = {}
        dims = source.GetImageData().GetDimensions()
        matrix = _ijkToRas(source)
        spacing = reg.gridSpacing(matrix)
        for orientation, _ in reg.ROWS:
            column = reg.closestAxis(matrix, reg.SLICE_NORMALS[orientation])
            checker = DynamicPETRegistrationLogic.helperVolume(
                f"{source.GetName()} registration checkerboard {orientation.lower()}")
            slicer.util.updateVolumeFromArray(checker, reg.checkerboard(dims[::-1], spacing, self.checkerMm, column))
            checker.SetIJKToRASMatrix(_vtkMatrix(matrix))
            checker.CreateDefaultDisplayNodes()
            display = checker.GetDisplayNode()
            display.SetAndObserveColorNodeID(CHECKER_COLOR_ID)
            display.AutoWindowLevelOff()
            display.SetWindowLevelMinMax(0.0, 1.0)
            display.SetApplyThreshold(True)           # the 0 squares are transparent
            display.SetLowerThreshold(0.5)
            display.SetUpperThreshold(10.0)
            display.SetInterpolate(False)
            checker.SetAndObserveTransformNodeID(self.linear.GetID())
            self.checkers[orientation] = checker
        for node in old:
            if slicer.mrmlScene.IsNodePresent(node):
                slicer.mrmlScene.RemoveNode(node)

    def setCheckerSize(self, squareMm):
        self.checkerMm = float(squareMm)
        if self.petView is not None:
            self._makeChecker(self.petView)
            self._assignVolumes()

    # --- transforms -----------------------------------------------------------------------------
    def setRegistration(self, linear, bspline=None):
        """The automatic result: the linear matrix and the B-spline node (or None); identity / None clears it."""
        self.linear.SetMatrixTransformToParent(_vtkMatrix(linear if linear is not None else np.eye(4)))
        if self.bspline is not None and self.bspline is not bspline and slicer.mrmlScene.IsNodePresent(self.bspline):
            self.bspline.SetAndObserveTransformNodeID(None)
        self.bspline = bspline
        if bspline is not None:
            bspline.SetAndObserveTransformNodeID(self.manual.GetID())
            self.linear.SetAndObserveTransformNodeID(bspline.GetID())
        else:
            self.linear.SetAndObserveTransformNodeID(self.manual.GetID())

    def manualMatrix(self):
        matrix = vtk.vtkMatrix4x4()
        self.manual.GetMatrixTransformToParent(matrix)
        return _numpy(matrix)

    def resetManual(self):
        self.manual.SetMatrixTransformToParent(vtk.vtkMatrix4x4())
        if self.manualCenter is not None:
            self._centerHandles()

    def setManualHandles(self, on):
        """Move / rotate handles (as in EasyReg) of the manual transform in the slice views, centred on the PET."""
        if self.manual is None:
            return
        self.manual.CreateDefaultDisplayNodes()
        display = self.manual.GetDisplayNode()
        if display is None:
            return
        if on:
            self.manualCenter = _center(self.petView, world=True)
            self._centerHandles()
            if hasattr(display, "SetEditorScalingEnabled"):
                display.SetEditorScalingEnabled(False)
        display.SetEditorVisibility(bool(on))
        if hasattr(display, "SetEditorSliceIntersectionVisibility"):
            display.SetEditorSliceIntersectionVisibility(bool(on))
        if on:
            for widget in self._allWidgets():
                widget.mrmlSliceNode().JumpSliceByCentering(*self.manualCenter)

    def _centerHandles(self):
        """The handles (and the rotations) at the centre of the PET: Slicer draws them at the transform's centre of
        transformation, in its own coordinates (EasyReg's centreHandles)."""
        if not hasattr(self.manual, "SetCenterOfTransformation") or self.manualCenter is None:
            return
        local = np.linalg.inv(self.manualMatrix()) @ np.append(np.asarray(self.manualCenter, float), 1.0)
        self.manual.SetCenterOfTransformation([float(v) for v in local[:3]])

    def _onManualModified(self, caller=None, event=None):
        if self.onManualChanged:
            self.onManualChanged()

    # --- display --------------------------------------------------------------------------------
    def setPetColor(self, label):
        self.petColor = label
        if self.petView is not None:
            self.petView.GetDisplayNode().SetAndObserveColorNodeID(colormaps.colorNodeForLabel(label).GetID())

    def setFusionColor(self, label):
        self.fusionColor = label
        if self.petFusion is not None:
            self.petFusion.GetDisplayNode().SetAndObserveColorNodeID(colormaps.colorNodeForLabel(label).GetID())

    def setPetWindow(self, lower, upper):
        if upper <= lower:
            return
        self.petWindow = (float(lower), float(upper))
        for node in (self.petView, self.petFusion):
            if node is None:
                continue
            display = node.GetDisplayNode()
            display.AutoWindowLevelOff()
            display.SetWindowLevelMinMax(lower, upper)
        if self.petFusion is not None:      # below the window (or 3 % of it): transparent over the CT/MR
            display = self.petFusion.GetDisplayNode()
            display.SetApplyThreshold(True)
            display.SetLowerThreshold(max(lower, THRESHOLD_FRACTION * upper))
            display.SetUpperThreshold(max(abs(upper), self.petMaximum, 1.0) * 1e6)

    def setFixedWindow(self, lower, upper):
        if upper <= lower or self.fixedHelper is None:
            return
        self.fixedWindow = (float(lower), float(upper))
        display = self.fixedHelper.GetDisplayNode()
        display.AutoWindowLevelOff()
        display.SetWindowLevelMinMax(lower, upper)

    def setFusionOpacity(self, opacity):
        self.fusionOpacity = float(opacity)
        self._setOpacity("Fusion", self.fusionOpacity)

    def setCheckerOpacity(self, opacity):
        self.checkerOpacity = float(opacity)
        self._setOpacity("Checker", self.checkerOpacity)

    def _setOpacity(self, column, opacity):
        self._link(False)
        try:
            for orientation, _ in reg.ROWS:
                widget = self._widget(reg.viewName(orientation, column))
                if widget is not None:
                    widget.mrmlSliceCompositeNode().SetForegroundOpacity(opacity)
        finally:
            self._link(self.linked)

    # --- layout and views -----------------------------------------------------------------------
    def setShowCheckerboard(self, show):
        self.showCheckerboard = bool(show)
        if self.fixedHelper is not None:
            self._setLayout()

    def _setLayout(self):
        layoutManager = slicer.app.layoutManager()
        layoutNode = layoutManager.layoutLogic().GetLayoutNode()
        for layoutID, xml in ((LAYOUT_ID, reg.layoutXml(False)), (CHECKER_LAYOUT_ID, reg.layoutXml(True))):
            if layoutNode.IsLayoutDescription(layoutID):
                if layoutManager.layout != layoutID:
                    layoutNode.SetLayoutDescription(layoutID, xml)
            else:
                layoutNode.AddLayoutDescription(layoutID, xml)
        current = layoutManager.layout
        if current not in (LAYOUT_ID, CHECKER_LAYOUT_ID):
            self.previousLayout = current
        wanted = CHECKER_LAYOUT_ID if self.showCheckerboard else LAYOUT_ID
        if current != wanted:
            self._unguard()
            layoutManager.setLayout(wanted)
            slicer.app.processEvents()
        self._assignVolumes(fit=True)

    @staticmethod
    def _widget(name):
        return slicer.app.layoutManager().sliceWidget(name)

    def _views(self):
        """(orientation, column, slice widget) of the views of the current layout."""
        out = []
        for orientation, _ in reg.ROWS:
            for column in reg.columns(self.showCheckerboard):
                widget = self._widget(reg.viewName(orientation, column))
                if widget is not None:
                    out.append((orientation, column, widget))
        return out

    def _allWidgets(self):
        return [w for _, _, w in self._views()]

    def _link(self, linked):
        for widget in self._allWidgets():
            composite = widget.mrmlSliceCompositeNode()
            composite.SetLinkedControl(linked)
            composite.SetHotLinkedControl(linked)

    def setLinked(self, linked):
        self.linked = bool(linked)
        self._link(self.linked)

    def _expected(self, column, orientation):
        """(background, foreground, opacity) of a view."""
        if column == "Checker":
            return self.petView, self.checkers.get(orientation), self.checkerOpacity
        if column == "PET":
            return self.petView, None, 0.0
        if column == "Fusion":
            return self.fixedHelper, self.petFusion, self.fusionOpacity
        return self.fixedHelper, None, 0.0

    def _unguard(self):
        for node, tag in [o for o in self.observers if o[0] is not self.manual]:
            node.RemoveObserver(tag)
        self.observers = [o for o in self.observers if o[0] is self.manual]
        self.guarded = False

    def _assignVolumes(self, fit=False):
        if self.petView is None or self.fixedHelper is None:
            return
        self._unguard()
        self._link(False)
        try:
            for orientation, column, widget in self._views():
                background, foreground, opacity = self._expected(column, orientation)
                composite = widget.mrmlSliceCompositeNode()
                composite.SetBackgroundVolumeID(background.GetID() if background else None)
                composite.SetForegroundVolumeID(foreground.GetID() if foreground else None)
                composite.SetForegroundOpacity(opacity)
                composite.SetLabelVolumeID(None)
                widget.mrmlSliceNode().SetOrientation(orientation)
                if fit:
                    widget.sliceLogic().FitSliceToAll()
        finally:
            self._link(self.linked)
        if fit:     # every view on the centre of the CT/MR, at the zoom of the CT/MR view of its row
            center = _center(self.fixed, world=True)
            for orientation, _ in reg.ROWS:
                anat = self._widget(reg.viewName(orientation, "Anat"))
                if anat is not None:
                    anat.sliceController().fitSliceToBackground()
            for widget in self._allWidgets():
                widget.mrmlSliceNode().JumpSliceByCentering(*center)
        for orientation, column, widget in self._views():
            self.observers.append(guardViewVolumes(widget, lambda column=column, orientation=orientation: tuple(
                n.GetID() if n is not None else None for n in self._expected(column, orientation)[:2]) + (None,)))
        self.guarded = True


# ---------------------------------------------------------------------------
# Module panel
# ---------------------------------------------------------------------------
class DynamicPETRegistrationWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    def __init__(self, parent=None):
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)
        self.session = None
        self.logic = DynamicPETRegistrationLogic()
        self.result = None           # logic.register() result, pending until Harden
        self.undo = None             # harden() record, for Revert
        self.hardened = False
        self._updating = False

        # --- dynamic series and reference
        inputBox = qt.QGroupBox("Dynamic PET")
        form = qt.QFormLayout(inputBox)
        self.sequenceSelector = slicer.qMRMLNodeComboBox()
        self.sequenceSelector.nodeTypes = ["vtkMRMLSequenceNode"]
        self.sequenceSelector.noneEnabled = True
        self.sequenceSelector.addEnabled = False
        self.sequenceSelector.removeEnabled = False
        self.sequenceSelector.renameEnabled = False
        self.sequenceSelector.setMRMLScene(slicer.mrmlScene)
        self.sequenceSelector.toolTip = "Dynamic PET sequence (as loaded from DICOM: '<series> dynamic Bq/ml')"
        form.addRow("Dynamic sequence:", self.sequenceSelector)
        self.referenceCombo = qt.QComboBox()
        self.referenceCombo.toolTip = ("Image registered to the CT/MR: the summed image of the series (default, when "
                                       "there is one) or a frame. The same transform moves every frame and the "
                                       "summed image.")
        form.addRow("Reference frame:", self.referenceCombo)
        self.infoLabel = qt.QLabel()
        self.infoLabel.wordWrap = True
        form.addRow(self.infoLabel)
        self.fixedSelector = slicer.qMRMLNodeComboBox()
        self.fixedSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.fixedSelector.noneEnabled = True
        self.fixedSelector.addEnabled = False
        self.fixedSelector.removeEnabled = False
        self.fixedSelector.renameEnabled = False
        self.fixedSelector.setMRMLScene(slicer.mrmlScene)
        self.fixedSelector.setCurrentNode(None)
        self.fixedSelector.toolTip = "CT or MR the PET is registered to (it stays where it is)"
        form.addRow("CT / MR:", self.fixedSelector)
        self.layout.addWidget(inputBox)

        # --- registration
        regBox = qt.QGroupBox("Registration")
        regForm = qt.QFormLayout(regBox)
        self.engineCombo = qt.QComboBox()
        for title, key in reg.ENGINES:
            self.engineCombo.addItem(title, key)
        self.engineCombo.setCurrentIndex(self.engineCombo.findData(reg.DEFAULT_ENGINE))
        self.engineCombo.toolTip = (
            "BRAINSFit: the steps below, at the CT/MR resolution (a B-spline on a fine MR can take many GB). "
            "Iterative multi-resolution (as in EasyReg): a rigid (or affine) stage, then B-spline passes from coarse "
            "to fine control points, each kept only when it improves the match, on a working grid of a few mm: "
            "little memory.")
        regForm.addRow("Method:", self.engineCombo)
        self.brainsfitBox = qt.QWidget()
        brainsForm = qt.QFormLayout(self.brainsfitBox)
        brainsForm.setContentsMargins(0, 0, 0, 0)
        self.iterativeBox = qt.QWidget()
        iterativeForm = qt.QFormLayout(self.iterativeBox)
        iterativeForm.setContentsMargins(0, 0, 0, 0)
        self.iterativeWidgets = iterativeOptions(iterativeForm)
        self.stepCombos = []
        self.resampleChecks = []
        for index in range(reg.MAX_STEPS):
            default = reg.DEFAULT_STEPS[index] if index < len(reg.DEFAULT_STEPS) else ""
            combo = qt.QComboBox()
            if index > 0:
                combo.addItem("(none)", "")
            for name in reg.STEP_TYPES:
                combo.addItem(name, name)
            combo.setCurrentIndex(combo.findData(default))
            combo.toolTip = ("Steps run one after the other, each from where the previous ended. Rigid: shifts and "
                             "angles. Affine: adds scaling and shear. BSpline: local deformation (last step only).")
            check = qt.QCheckBox("Resample CT/MR to PET")
            check.checked = reg.RESAMPLE_BY_DEFAULT.get(default, True)
            check.toolTip = ("Register this step to a copy of the CT/MR smoothed and resampled to the PET voxel size "
                             "(Advanced: Register CT/MR at). On by default: a sub-millimetre MR at full resolution "
                             "takes many GB of memory, and the PET resolution limits the match anyway.")
            row = qt.QHBoxLayout()
            row.addWidget(combo, 1)
            row.addWidget(check)
            self.stepCombos.append(combo)
            self.resampleChecks.append(check)
            brainsForm.addRow(f"Step {index + 1}:", row)
        self.initializationCombo = qt.QComboBox()
        for title, mode in reg.INITIALIZATIONS:
            self.initializationCombo.addItem(title, mode)
        self.initializationCombo.setCurrentIndex(self.initializationCombo.findData(reg.DEFAULT_INITIALIZATION))
        self.initializationCombo.toolTip = ("How the first step starts: geometric centers aligns the centers of the "
                                            "PET and CT/MR volumes (as Motion Correction); centers of mass aligns "
                                            "the intensities; none starts from the scanner coordinates (PET/CT of "
                                            "the same scanner).")
        advanced = ctk.ctkCollapsibleButton()
        advanced.text = "Advanced"
        advanced.collapsed = True
        advancedForm = qt.QFormLayout(advanced)
        self.samplingSpin = qt.QDoubleSpinBox()
        self.samplingSpin.setRange(0.001, 1.0)
        self.samplingSpin.setDecimals(3)
        self.samplingSpin.setSingleStep(0.01)
        self.samplingSpin.value = reg.DEFAULT_SAMPLING
        self.samplingSpin.toolTip = "BRAINSFit sampling percentage (fraction of the voxels used by the metric)"
        advancedForm.addRow("Sampling fraction:", self.samplingSpin)
        self.gridSpin = qt.QSpinBox()
        self.gridSpin.setRange(3, 20)
        self.gridSpin.value = reg.DEFAULT_BSPLINE_GRID
        self.gridSpin.toolTip = "B-spline control grid size along each axis"
        advancedForm.addRow("B-spline grid:", self.gridSpin)
        self.workingVoxelSpin = qt.QDoubleSpinBox()
        self.workingVoxelSpin.setRange(0.0, 10.0)
        self.workingVoxelSpin.setDecimals(1)
        self.workingVoxelSpin.setSingleStep(0.5)
        self.workingVoxelSpin.setSuffix(" mm")
        self.workingVoxelSpin.specialValueText = "PET voxel size"
        self.workingVoxelSpin.value = reg.DEFAULT_WORKING_VOXEL_MM
        self.workingVoxelSpin.toolTip = (
            "Steps with 'Resample CT/MR to PET' register to the CT/MR smoothed and resampled to this voxel size (the "
            "views and Harden use it as it is). Default: the PET voxel size, which is all the PET can match; a sub-millimetre MR at "
            "full resolution makes the B-spline step take many GB of memory. A CT/MR that is not finer is used as "
            "it is.")
        advancedForm.addRow("Register CT/MR at:", self.workingVoxelSpin)
        brainsForm.addRow(advanced)
        self.stepsNote = qt.QLabel()
        self.stepsNote.wordWrap = True
        brainsForm.addRow(self.stepsNote)
        regForm.addRow(self.brainsfitBox)
        regForm.addRow(self.iterativeBox)
        regForm.addRow("Initialization:", self.initializationCombo)
        self.registerButton = qt.QPushButton("Register")
        self.registerButton.toolTip = "Register the reference to the CT/MR in the steps above"
        regForm.addRow(self.registerButton)
        manualRow = qt.QHBoxLayout()
        self.manualButton = qt.QPushButton("Correct manually")
        self.manualButton.checkable = True
        self.manualButton.toolTip = ("Move / rotate handles on the PET in the slice views (as in EasyReg), applied "
                                     "on top of the registration; rotations turn about the centre of the PET")
        self.resetManualButton = qt.QPushButton("Reset manual")
        self.resetManualButton.toolTip = "Drop the manual correction"
        manualRow.addWidget(self.manualButton, 2)
        manualRow.addWidget(self.resetManualButton, 1)
        regForm.addRow(manualRow)
        row = qt.QHBoxLayout()
        self.revertButton = qt.QPushButton("Revert changes")
        self.revertButton.toolTip = ("Before hardening: drop the registration and the manual correction. After "
                                     "hardening: put the overwritten series back, or remove the new one.")
        self.hardenButton = qt.QPushButton("Harden")
        self.hardenButton.toolTip = ("Resample every frame and the summed image onto the CT/MR field of view: "
                                     "overwrite the series or save a new one")
        row.addWidget(self.revertButton)
        row.addWidget(self.hardenButton)
        regForm.addRow(row)
        self.layout.addWidget(regBox)

        # --- applied transform
        transformBox = qt.QGroupBox("Applied transform")
        transformLayout = qt.QVBoxLayout(transformBox)
        self.transformLabel = qt.QLabel("Not registered yet.")
        self.transformLabel.wordWrap = True
        self.transformLabel.setTextInteractionFlags(qt.Qt.TextSelectableByMouse)
        transformLayout.addWidget(self.transformLabel)
        self.layout.addWidget(transformBox)

        # --- view
        viewBox = qt.QGroupBox("View")
        view = qt.QFormLayout(viewBox)
        self.petColorCombo = qt.QComboBox()
        self.fusionColorCombo = qt.QComboBox()
        for combo, default in ((self.petColorCombo, PET_COLORMAP), (self.fusionColorCombo, FUSION_COLORMAP)):
            for label in colormaps.LABELS:
                combo.addItem(label)
            combo.setCurrentIndex(max(0, combo.findText(default)))
        self.petColorCombo.toolTip = "Color map of the PET column (and of the PET under the checkerboard)"
        self.fusionColorCombo.toolTip = "Color map of the PET over the CT/MR"
        view.addRow("PET colors:", self.petColorCombo)
        view.addRow("Fusion colors:", self.fusionColorCombo)
        self.petWindowRange = ctk.ctkRangeWidget()
        self.petWindowRange.decimals = 2
        self.petWindowRange.toolTip = ("Display range of the PET (both columns); in the fusion, voxels below it are "
                                       "transparent")
        view.addRow("PET window:", self.petWindowRange)
        self.opacitySlider = ctk.ctkSliderWidget()
        self.opacitySlider.minimum, self.opacitySlider.maximum = 0.0, 1.0
        self.opacitySlider.singleStep, self.opacitySlider.decimals = 0.05, 2
        self.opacitySlider.value = DEFAULT_FUSION_OPACITY
        self.opacitySlider.toolTip = "Opacity of the PET over the CT/MR"
        view.addRow("Fusion opacity:", self.opacitySlider)
        self.fixedPresetCombo = qt.QComboBox()
        self.fixedPresetCombo.addItem(reg.AUTO_WINDOW)
        for title, _, _ in reg.CT_WINDOWS:
            self.fixedPresetCombo.addItem(title)
        self.fixedPresetCombo.toolTip = "Window of the CT/MR: a CT preset or automatic (MR)"
        view.addRow("CT/MR preset:", self.fixedPresetCombo)
        self.fixedWindowRange = ctk.ctkRangeWidget()
        self.fixedWindowRange.decimals = 1
        self.fixedWindowRange.toolTip = "Display range of the CT/MR (grey)"
        view.addRow("CT/MR window:", self.fixedWindowRange)
        self.autoWindowButton = qt.QPushButton("Auto windows")
        self.autoWindowButton.toolTip = "PET 0 - 99.9th percentile, CT/MR by its preset"
        view.addRow(self.autoWindowButton)
        self.checkerCheck = qt.QCheckBox("Checkerboard column")
        self.checkerCheck.checked = False
        self.checkerCheck.toolTip = ("A checkerboard made on the PET grid and moved by the same transform, over the "
                                     "PET (turned on after a registration)")
        view.addRow(self.checkerCheck)
        self.checkerSizeSpin = qt.QDoubleSpinBox()
        self.checkerSizeSpin.setRange(2.0, 200.0)
        self.checkerSizeSpin.setDecimals(0)
        self.checkerSizeSpin.setSingleStep(5.0)
        self.checkerSizeSpin.setSuffix(" mm")
        self.checkerSizeSpin.value = reg.DEFAULT_CHECKER_MM
        view.addRow("Checkerboard squares:", self.checkerSizeSpin)
        self.checkerOpacitySlider = ctk.ctkSliderWidget()
        self.checkerOpacitySlider.minimum, self.checkerOpacitySlider.maximum = 0.0, 1.0
        self.checkerOpacitySlider.singleStep, self.checkerOpacitySlider.decimals = 0.05, 2
        self.checkerOpacitySlider.value = DEFAULT_CHECKER_OPACITY
        view.addRow("Checkerboard opacity:", self.checkerOpacitySlider)
        self.linkCheck = qt.QCheckBox("Link the views of each row (slice, zoom, pan)")
        self.linkCheck.checked = True
        view.addRow(self.linkCheck)
        buttons = qt.QHBoxLayout()
        self.openButton = qt.QPushButton("Show layout")
        self.closeButton = qt.QPushButton("Close layout")
        self.closeButton.enabled = False
        buttons.addWidget(self.openButton)
        buttons.addWidget(self.closeButton)
        view.addRow(buttons)
        self.layout.addWidget(viewBox)
        self.layout.addStretch(1)

        self.referenceCombo.connect("currentIndexChanged(int)", self.onReferenceChanged)
        self.fixedSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onFixedChanged)
        for combo, check in zip(self.stepCombos, self.resampleChecks):
            combo.connect("currentIndexChanged(int)", lambda i, combo=combo, check=check: self.onStepTypeChanged(
                combo, check))
        self.registerButton.connect("clicked()", self.onRegister)
        self.engineCombo.connect("currentIndexChanged(int)", self.onEngineChanged)
        self.manualButton.connect("toggled(bool)", self.onManualToggled)
        self.resetManualButton.connect("clicked()", self.onResetManual)
        self.revertButton.connect("clicked()", self.onRevert)
        self.hardenButton.connect("clicked()", self.onHarden)
        self.petColorCombo.connect("currentIndexChanged(int)", lambda i: self.session and self.session.setPetColor(
            self.petColorCombo.currentText))
        self.fusionColorCombo.connect("currentIndexChanged(int)", lambda i: self.session and
                                      self.session.setFusionColor(self.fusionColorCombo.currentText))
        self.petWindowRange.connect("valuesChanged(double,double)",
                                    lambda lo, hi: self.session and self.session.setPetWindow(lo, hi))
        self.fixedWindowRange.connect("valuesChanged(double,double)",
                                      lambda lo, hi: self.session and self.session.setFixedWindow(lo, hi))
        self.fixedPresetCombo.connect("currentIndexChanged(int)", self.onFixedPreset)
        self.opacitySlider.connect("valueChanged(double)", lambda v: self.session and
                                   self.session.setFusionOpacity(v))
        self.checkerOpacitySlider.connect("valueChanged(double)", lambda v: self.session and
                                          self.session.setCheckerOpacity(v))
        self.checkerSizeSpin.connect("valueChanged(double)", lambda v: self.session and
                                     self.session.setCheckerSize(v))
        self.checkerCheck.connect("toggled(bool)", lambda on: self.session and self.session.setShowCheckerboard(on))
        self.linkCheck.connect("toggled(bool)", lambda on: self.session and self.session.setLinked(on))
        self.autoWindowButton.connect("clicked()", self.onAutoWindows)
        self.openButton.connect("clicked()", self.onOpen)
        self.closeButton.connect("clicked()", self.onClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.NodeRemovedEvent, self.onNodeRemoved)
        self.onEngineChanged()
        self.selectInputs()
        self.onSequenceChanged(self.sequenceSelector.currentNode(), openLayout=False)
        self.sequenceSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onSequenceChanged)
        self._ready = True           # from now on picking the images opens the layout
        self._updateButtons()

    # --- settings ------------------------------------------------------------------------------
    def steps(self):
        return [combo.itemData(combo.currentIndex) for combo in self.stepCombos]

    def resampleFlags(self):
        """The resample check of each step that is not (none)."""
        return [check.checked for combo, check in zip(self.stepCombos, self.resampleChecks)
                if combo.itemData(combo.currentIndex)]

    def onStepTypeChanged(self, combo, check):
        """A new step type gets its default resample check (on by default)."""
        step = combo.itemData(combo.currentIndex)
        check.enabled = bool(step)
        if step:
            check.checked = reg.RESAMPLE_BY_DEFAULT[step]
        self.onStepsChanged()

    def engine(self):
        return self.engineCombo.itemData(self.engineCombo.currentIndex) or reg.DEFAULT_ENGINE

    def onEngineChanged(self, *args):
        iterative = self.engine() == reg.ENGINE_ITERATIVE
        self.brainsfitBox.visible = not iterative
        self.iterativeBox.visible = iterative
        self.onStepsChanged()

    def settings(self):
        if self.engine() == reg.ENGINE_ITERATIVE:
            settings = DynamicPETRegistrationLogic.defaultSettings()
            settings.update(iterativeSettings(self.iterativeWidgets), engine=reg.ENGINE_ITERATIVE,
                            initialization=self.initializationCombo.itemData(self.initializationCombo.currentIndex)
                            or reg.DEFAULT_INITIALIZATION)
            settings["steps"] = reg.iterativeSteps(settings["iterativeLinear"], settings["iterativeDeformable"])
            return settings
        return {"engine": reg.ENGINE_BRAINSFIT, "steps": reg.validateSteps(self.steps()),
                "resample": self.resampleFlags(),
                "initialization": self.initializationCombo.itemData(self.initializationCombo.currentIndex) or
                reg.DEFAULT_INITIALIZATION,
                "samplingPercentage": self.samplingSpin.value, "bsplineGrid": self.gridSpin.value,
                "workingVoxelMm": self.workingVoxelSpin.value}

    def _stepsValid(self):
        if self.engine() == reg.ENGINE_ITERATIVE:
            return True
        try:
            reg.validateSteps(self.steps())
            return True
        except ValueError:
            return False

    def onStepsChanged(self, *args):
        try:
            steps = reg.validateSteps(self.steps())
            self.stepsNote.text = f"Runs: {reg.stepsText(steps)}."
            valid = True
        except ValueError as error:
            self.stepsNote.text = str(error)
            valid = False
        if hasattr(self, "registerButton"):
            self.registerButton.enabled = (valid or self.engine() == reg.ENGINE_ITERATIVE) and self._inputsReady()

    # --- inputs --------------------------------------------------------------------------------
    def selectInputs(self):
        nodes = slicer.util.getNodesByClass("vtkMRMLSequenceNode")
        dynamic = [n for n in nodes if n.GetAttribute(PREFIX + "FrameDurationS") or n.GetAttribute(PREFIX + "Kind")]
        if dynamic and self.sequenceSelector.currentNode() is None:
            self.sequenceSelector.setCurrentNode(dynamic[-1])
        if self.fixedSelector.currentNode() is None:
            candidates = fixedCandidates()
            if candidates:
                self.fixedSelector.setCurrentNode(candidates[0])

    def _inputsReady(self):
        return self.sequenceSelector.currentNode() is not None and self.fixedSelector.currentNode() is not None \
            and self.referenceSource() is not None

    def referenceSource(self):
        """The PET volume registered: the summed image or a frame of the sequence."""
        sequence = self.sequenceSelector.currentNode()
        if sequence is None or self.referenceCombo.currentIndex < 0:
            return None
        data = self.referenceCombo.itemData(self.referenceCombo.currentIndex)
        if data == SUMMED:
            return self.logic.summedVolumeOf(sequence)
        try:
            return sequence.GetNthDataNode(int(data))
        except (TypeError, ValueError):
            return None

    def referenceText(self):
        return self.referenceCombo.currentText

    def onSequenceChanged(self, node, openLayout=True):
        from DynamicPETEditor import DynamicPETEditorLogic
        self._clearResult()
        self._updating = True
        try:
            self.referenceCombo.clear()
            valid = node is not None and DynamicPETEditorLogic.isDynamicSequence(node)
            if valid:
                summed = self.logic.summedVolumeOf(node)
                if summed is not None:
                    self.referenceCombo.addItem(f"Summed image ({summed.GetName()})", SUMMED)
                for i, interval in enumerate(DynamicPETEditorLogic.frameIntervals(node)):
                    self.referenceCombo.addItem(dynamicviews.frameLabel(interval), str(i))
                if summed is None:
                    from DynamicPETMotionCorrection import DynamicPETMotionCorrectionLogic
                    from DICOMGatedDynamicPETLib import motioncorrection as mc
                    activities = DynamicPETMotionCorrectionLogic.frameActivities(node)
                    self.referenceCombo.setCurrentIndex(mc.maxActivityFrame(activities))
                else:
                    self.referenceCombo.setCurrentIndex(0)
        finally:
            self._updating = False
        self._updateInfo()
        self._updateButtons()
        if not valid:
            self.onClose()
            return
        if openLayout or self.session is not None:
            self.onOpen()

    def _updateInfo(self):
        node = self.sequenceSelector.currentNode()
        if node is None:
            self.infoLabel.text = "Pick a dynamic PET sequence."
            return
        summed = self.logic.summedVolumeOf(node)
        self.infoLabel.text = (f"{node.GetNumberOfDataNodes()} frames"
                               + (f"; the summed image '{summed.GetName()}' moves with them." if summed is not None
                                  else "; no summed image linked to the series."))

    def onReferenceChanged(self, index):
        if self._updating:
            return
        self._clearResult()
        if self.session is not None and self.referenceSource() is not None:
            self.session.setPetSource(self.referenceSource())
            self._syncWindowWidgets()
        self._updateButtons()

    def onFixedChanged(self, node):
        self._clearResult()
        if node is not None:
            self._updating = True
            try:
                self.fixedPresetCombo.setCurrentIndex(1 if _looksLikeCT(node) else 0)
            finally:
                self._updating = False
        if getattr(self, "_ready", False) and (self.session is not None or node is not None):
            self.onOpen()
        self._updateButtons()

    # --- layout --------------------------------------------------------------------------------
    def onOpen(self):
        sequence, fixed, source = self.sequenceSelector.currentNode(), self.fixedSelector.currentNode(), \
            self.referenceSource()
        if sequence is None or fixed is None or source is None:
            return
        self._closeOtherModules()
        result = self.result
        self._closeSession()
        try:
            self.session = RegistrationSession(fixed)
            self.session.petColor = self.petColorCombo.currentText
            self.session.fusionColor = self.fusionColorCombo.currentText
            self.session.fusionOpacity = self.opacitySlider.value
            self.session.checkerOpacity = self.checkerOpacitySlider.value
            self.session.checkerMm = self.checkerSizeSpin.value
            self.session.linked = self.linkCheck.checked
            self.session.showCheckerboard = self.checkerCheck.checked
            self.session.onManualChanged = self._updateTransformText
            self.session.open(source)
            if result is not None:
                self.session.setRegistration(result["linear"], result["bspline"])
            self._applyFixedPreset()
        except Exception as error:
            logging.exception("Could not open Dynamic PET to CT/MR Registration")
            slicer.util.errorDisplay(f"Could not open the registration layout:\n{error}")
            self._closeSession()
        self.closeButton.enabled = self.session is not None
        self._syncWindowWidgets()
        self._updateButtons()

    def _closeSession(self):
        if self.session is not None:
            if self.manualButton.checked:
                self.manualButton.blockSignals(True)
                self.manualButton.checked = False
                self.manualButton.blockSignals(False)
            self.session.close()
            self.session = None

    def onClose(self):
        self._closeSession()
        if hasattr(self, "closeButton"):
            self.closeButton.enabled = False
            self._updateButtons()

    @staticmethod
    def _closeOtherModules():
        """Review/Edit and Motion Correction take the main layout with a timeline dock: close them first."""
        for name in ("DynamicPETEditorWidget", "DynamicPETMotionCorrectionWidget"):
            try:
                widget = getattr(slicer.modules, name, None)
                if widget is not None and getattr(widget, "session", None) is not None:
                    widget.onClose()
            except Exception:
                logging.warning(f"Could not close {name}", exc_info=True)

    # --- windows -------------------------------------------------------------------------------
    def _fixedPresetWindow(self):
        index = self.fixedPresetCombo.currentIndex
        if index >= 1:
            _, lower, upper = reg.CT_WINDOWS[index - 1]
            return lower, upper
        fixed = self.fixedSelector.currentNode()
        return reg.percentileWindow(slicer.util.arrayFromVolume(fixed)) if fixed is not None else (0.0, 1.0)

    def _applyFixedPreset(self):
        if self.session is not None:
            self.session.setFixedWindow(*self._fixedPresetWindow())

    def onFixedPreset(self, index):
        if self._updating:
            return
        self._applyFixedPreset()
        self._syncWindowWidgets()

    def onAutoWindows(self):
        if self.session is None:
            return
        source = self.referenceSource()
        if source is not None:
            self.session.setPetWindow(0.0, _percentile(slicer.util.arrayFromVolume(source)))
        self._applyFixedPreset()
        self._syncWindowWidgets()

    def _syncWindowWidgets(self):
        session = self.session
        for widget in (self.petWindowRange, self.fixedWindowRange, self.autoWindowButton):
            widget.enabled = session is not None
        if session is None:
            return
        for widget, low, high, values in (
                (self.petWindowRange, min(0.0, session.petWindow[0]), max(session.petMaximum, session.petWindow[1]),
                 session.petWindow),
                (self.fixedWindowRange, min(session.fixedRange[0], session.fixedWindow[0]),
                 max(session.fixedRange[1], session.fixedWindow[1]), session.fixedWindow)):
            widget.blockSignals(True)
            try:
                top = max(high, low + 1e-6)
                widget.singleStep = max((top - low) / 1000.0, 1e-4)
                widget.setRange(low, top)
                widget.setValues(*values)
            finally:
                widget.blockSignals(False)

    # --- registration --------------------------------------------------------------------------
    def _clearResult(self):
        """Forget the pending registration (after Harden: also the chance to revert it)."""
        if self.result is not None and self.result.get("bspline") is not None and \
                slicer.mrmlScene.IsNodePresent(self.result["bspline"]):
            if self.session is not None:
                self.session.setRegistration(None, None)
            slicer.mrmlScene.RemoveNode(self.result["bspline"])
        self.result = None
        self.undo = None
        self.hardened = False
        if getattr(self, "session", None) is not None:
            self.session.setRegistration(None, None)
            self.session.resetManual()
            self.session.setShowCheckerboard(False)
        if hasattr(self, "checkerCheck"):
            self.checkerCheck.blockSignals(True)
            self.checkerCheck.checked = False
            self.checkerCheck.blockSignals(False)
            self._updateTransformText()
            self._updateButtons()

    def _updateButtons(self):
        if not hasattr(self, "hardenButton"):
            return
        ready = self._inputsReady()
        stepsOk = self._stepsValid()
        self.registerButton.enabled = ready and stepsOk and not self.hardened
        self.openButton.enabled = ready
        self.manualButton.enabled = self.session is not None and not self.hardened
        self.resetManualButton.enabled = self.session is not None and not self.hardened
        self.hardenButton.enabled = self.session is not None and not self.hardened and \
            (self.result is not None or self._hasManual())
        self.revertButton.enabled = (self.result is not None or self._hasManual() or self.undo is not None)

    def _hasManual(self):
        return self.session is not None and self.session.manual is not None and \
            not np.allclose(self.session.manualMatrix(), np.eye(4), atol=1e-9)

    def _updateTransformText(self):
        if not hasattr(self, "transformLabel"):
            return
        result = self.result
        manual = self.session.manualMatrix() if self.session is not None and self.session.manual is not None \
            else None
        if result is None and (manual is None or np.allclose(manual, np.eye(4), atol=1e-9)):
            self.transformLabel.text = "Not registered yet." if not self.hardened else self.transformLabel.text
            self._updateButtons()
            return
        steps = result["steps"] if result else []
        linear = result["linear"] if result else np.eye(4)
        center = result["center"] if result else (_center(self.referenceSource()) if self.referenceSource() else
                                                  (0.0, 0.0, 0.0))
        lines = reg.summaryLines(steps, linear, center, result["deformation"] if result else None, manual)
        if not result:
            lines[0] = "Manual correction only"
        self.transformLabel.text = "\n".join(lines)
        self._updateButtons()

    def onRegister(self):
        if self.session is None:
            self.onOpen()
        if self.session is None:
            return
        try:
            settings = self.settings()
        except ValueError as error:
            slicer.util.errorDisplay(str(error))
            return
        fixed, source = self.fixedSelector.currentNode(), self.referenceSource()
        steps = settings["steps"]
        dialog = slicer.util.createProgressDialog(windowTitle="Dynamic PET to CT/MR Registration",
                                                  labelText="Registering...", maximum=len(steps))

        def progress(index, total, text):
            dialog.setValue(index)
            dialog.setLabelText(f"Registering: {text}")
            slicer.app.processEvents()
            return not dialog.wasCanceled

        self._clearResult()
        qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)
        try:
            result = self.logic.register(fixed, source, settings, progress)
        except Exception as error:
            logging.exception("Dynamic PET to CT/MR registration failed")
            slicer.util.errorDisplay(f"Registration failed:\n{error}")
            return
        finally:
            qt.QApplication.restoreOverrideCursor()
            dialog.close()
        if not result["steps"]:
            self.transformLabel.text = "Stopped before the first step."
            return
        self.result = result
        self.session.setRegistration(result["linear"], result["bspline"])
        self.checkerCheck.checked = True          # 2 x 4 with the checkerboard column
        self._updateTransformText()
        if result["stopped"]:
            self.transformLabel.text += "\nStopped before the last step."
        if result.get("notes"):
            self.transformLabel.text += "\n" + " ".join(result["notes"])

    def onManualToggled(self, on):
        if self.session is None:
            self.manualButton.checked = False
            return
        self.session.setManualHandles(on)

    def onResetManual(self):
        if self.session is not None:
            self.session.resetManual()
            self._updateTransformText()

    def _transformDescription(self):
        return "; ".join(self.transformLabel.text.split("\n"))

    def onHarden(self):
        sequence, fixed = self.sequenceSelector.currentNode(), self.fixedSelector.currentNode()
        if self.session is None or sequence is None or fixed is None or self.hardened:
            return
        summed = self.logic.summedVolumeOf(sequence)
        petSpacing = reg.gridSpacing(_ijkToRas(sequence.GetNthDataNode(0)))
        text = (f"Resample the {sequence.GetNumberOfDataNodes()} frames of '{sequence.GetName()}'"
                + (f" and the summed image '{summed.GetName()}'" if summed is not None else "")
                + f" through the registration onto the field of view of '{fixed.GetName()}'.\n\n"
                "Overwrite replaces the series (Revert changes puts it back until you pick another series or close "
                "the scene). Save separately makes a new series next to it. Save the scene to keep the result.")
        choice = hardenDialog("Harden registration", text, reg.OUTPUT_GRIDS, petSpacing,
                              [("Overwrite", reg.OVERWRITE), ("Save separately", reg.SAVE_SEPARATELY)],
                              slicer.util.mainWindow())
        if choice is None:
            return
        description = self._transformDescription()
        self.manualButton.checked = False
        qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)
        try:
            undo, newSequence, newSummed = self.logic.harden(
                sequence, summed, fixed, self.session.linear, choice["action"], choice["grid"], choice["spacing"],
                choice["interpolation"], description)
        except Exception as error:
            logging.exception("Could not harden the registration")
            slicer.util.errorDisplay(f"Could not harden the registration:\n{error}")
            return
        finally:
            qt.QApplication.restoreOverrideCursor()
        referenceIndex = self.referenceCombo.currentIndex
        bspline = self.result["bspline"] if self.result else None
        self.result = None
        if self.session is not None:      # the series is in the CT/MR space now: the views show it as it is
            self.session.setRegistration(None, None)
            self.session.resetManual()
        if bspline is not None and slicer.mrmlScene.IsNodePresent(bspline):
            slicer.mrmlScene.RemoveNode(bspline)
        if newSequence is not sequence:
            self.sequenceSelector.setCurrentNode(newSequence)
        else:
            self.onSequenceChanged(sequence)
        if 0 <= referenceIndex < self.referenceCombo.count:
            self.referenceCombo.setCurrentIndex(referenceIndex)
        self.undo, self.hardened = undo, True
        self.transformLabel.text = (description + "\n" + (
            f"Overwrote '{sequence.GetName()}'" if choice["action"] == reg.OVERWRITE else
            f"Saved as '{newSequence.GetName()}'" + (f" and '{newSummed.GetName()}'" if newSummed else ""))
            + f" ({choice['interpolation'].lower()}, voxels "
            + ", ".join(f"{v:.3g}" for v in reg.gridSpacing(_ijkToRas(newSequence.GetNthDataNode(0)))) + " mm).")
        self._updateButtons()

    def onRevert(self):
        if self.undo is not None:
            undo = self.undo
            original = undo["sequence"]
            present = slicer.mrmlScene.IsNodePresent(original)
            try:
                if undo["action"] == reg.SAVE_SEPARATELY and present:
                    self.sequenceSelector.setCurrentNode(original)    # before the new series is removed
                self.logic.revert(undo)
                self.undo, self.hardened = None, False
                if undo["action"] == reg.OVERWRITE and present:      # the views show the series as it was
                    if self.sequenceSelector.currentNode() is original:
                        self.onSequenceChanged(original)
                    else:
                        self.sequenceSelector.setCurrentNode(original)
            except Exception as error:
                logging.exception("Could not revert the registration")
                slicer.util.errorDisplay(f"Could not revert the registration:\n{error}")
                return
        self._clearResult()
        self.transformLabel.text = "Reverted: the PET is as it was."

    # --- scene ---------------------------------------------------------------------------------
    def enter(self):
        self.selectInputs()
        if self.session is None and self._inputsReady():
            self.onOpen()

    def exit(self):
        """Leaving the module: close the layout (the pending registration is kept for when you come back)."""
        if self.session is not None:
            self.onClose()
            slicer.app.layoutManager().setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutFourUpView)

    def cleanup(self):
        self.onClose()
        self.removeObservers()
        ScriptedLoadableModuleWidget.cleanup(self)

    def onSceneClose(self, caller, event):
        self.onClose()
        self.result = None
        self.undo = None
        self.hardened = False

    @vtk.calldata_type(vtk.VTK_OBJECT)
    def onNodeRemoved(self, caller, event, node):
        if self.session is not None and node in (self.sequenceSelector.currentNode(), self.fixedSelector.currentNode()):
            qt.QTimer.singleShot(0, self.onClose)


# ---------------------------------------------------------------------------
# Test: a synthetic CT and a dynamic PET of the same phantom, the PET moved
# ---------------------------------------------------------------------------
class DynamicPETRegistrationTest(ScriptedLoadableModuleTest):
    SPACING = 4.0
    SHIFT_VOXELS = 2            # the PET is 8 mm off along R

    def setUp(self):
        slicer.mrmlScene.Clear(0)

    def runTest(self):
        self.setUp()
        self.test_Helpers()
        self.setUp()
        self.test_Register()

    def test_Helpers(self):
        self.assertEqual(reg.validateSteps(["Rigid", "BSpline"]), ["Rigid", "BSpline"])
        self.assertEqual(reg.resampleFlags(["Rigid", "BSpline"]), [False, True])
        with self.assertRaises(ValueError):
            reg.validateSteps(["BSpline", "Rigid"])

    @staticmethod
    def phantom(shape=(40, 48, 48)):
        from DynamicPETMotionCorrection import DynamicPETMotionCorrectionTest
        return DynamicPETMotionCorrectionTest.phantom(shape)

    def makeData(self):
        base = self.phantom()
        ct = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "Synthetic CT")
        slicer.util.updateVolumeFromArray(ct, (base / 10.0 - 1000.0 * (base < 1.0)).astype(np.float32))
        ct.SetSpacing(self.SPACING, self.SPACING, self.SPACING)
        ct.SetIJKToRASDirections(1, 0, 0, 0, 1, 0, 0, 0, 1)
        moved = np.roll(base, self.SHIFT_VOXELS, axis=2)
        sequence = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSequenceNode", "Synthetic dynamic Bq/ml")
        sequence.SetIndexName("Time")
        sequence.SetIndexUnit("s")
        sequence.SetAttribute(PREFIX + "Kind", "dynamic")
        for f in range(3):
            node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", f"F{f + 1}")
            slicer.util.updateVolumeFromArray(node, moved * (f + 1))
            node.SetSpacing(self.SPACING, self.SPACING, self.SPACING)
            node.SetIJKToRASDirections(1, 0, 0, 0, 1, 0, 0, 0, 1)
            node.SetAttribute(PREFIX + "Frame", str(f + 1))
            node.SetAttribute(PREFIX + "StartFromStudyStartS", str(60.0 * f))
            node.SetAttribute(PREFIX + "DurationS", "60")
            sequence.SetDataNodeAtValue(node, str(60.0 * f + 30.0))
            slicer.mrmlScene.RemoveNode(node)
        summed = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "Synthetic summed")
        slicer.util.updateVolumeFromArray(summed, moved * 2.0)
        summed.SetSpacing(self.SPACING, self.SPACING, self.SPACING)
        summed.SetIJKToRASDirections(1, 0, 0, 0, 1, 0, 0, 0, 1)
        sequence.SetAttribute(PREFIX + "SummedVolumeID", summed.GetID())
        return ct, sequence, summed

    def test_Register(self):
        ct, sequence, summed = self.makeData()
        logic = DynamicPETRegistrationLogic()
        settings = dict(logic.defaultSettings(), engine=reg.ENGINE_BRAINSFIT, steps=["Rigid"],
                        initialization="Off")
        result = logic.register(ct, summed, settings)
        self.assertEqual(result["steps"], ["Rigid"])
        shift = reg.linearSummary(result["linear"], result["center"])["shift"]
        self.assertAlmostEqual(shift[0], -self.SHIFT_VOXELS * self.SPACING, delta=1.5)
        self.assertLess(max(abs(shift[1]), abs(shift[2])), 1.5)
        self.assertEqual([n for n in slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode")
                          if n.GetAttribute(HELPER_ATTRIBUTE)], [])

        # a CT finer than the PET is registered through a coarser working copy (removed afterwards)
        working = logic.workingFixed(ct, summed, 8.0)
        self.assertEqual(slicer.util.arrayFromVolume(working).shape, (20, 24, 24))
        slicer.mrmlScene.RemoveNode(working)
        self.assertIsNone(logic.workingFixed(ct, summed))          # same voxel size as the PET: used as it is

        # a rigid + B-spline run (geometric centres) also finishes and reports the deformation
        both = logic.register(ct, summed, dict(logic.defaultSettings(), engine=reg.ENGINE_BRAINSFIT,
                                               steps=["Rigid", "BSpline"], resample=None, workingVoxelMm=8.0))
        self.assertIsNotNone(both["bspline"])
        self.assertIsNotNone(both["deformation"])
        slicer.mrmlScene.RemoveNode(both["bspline"])

        # the layout: PET, fusion and CT/MR columns; the checkerboard column after a registration
        session = RegistrationSession(ct)
        session.open(summed)
        try:
            layoutManager = slicer.app.layoutManager()
            self.assertEqual(layoutManager.layout, LAYOUT_ID)
            fusion = layoutManager.sliceWidget(reg.viewName("Axial", "Fusion")).mrmlSliceCompositeNode()
            self.assertEqual(fusion.GetBackgroundVolumeID(), session.fixedHelper.GetID())
            self.assertEqual(fusion.GetForegroundVolumeID(), session.petFusion.GetID())
            session.setRegistration(result["linear"])
            session.setShowCheckerboard(True)
            self.assertEqual(layoutManager.layout, CHECKER_LAYOUT_ID)
            checker = layoutManager.sliceWidget(reg.viewName("Coronal", "Checker")).mrmlSliceCompositeNode()
            self.assertEqual(checker.GetForegroundVolumeID(), session.checkers["Coronal"].GetID())
            self.assertEqual(session.checkers["Axial"].GetParentTransformNode(), session.linear)
            # axial: columns along S (k here), coronal: along A (j)
            axial = slicer.util.arrayFromVolume(session.checkers["Axial"])
            coronal = slicer.util.arrayFromVolume(session.checkers["Coronal"])
            np.testing.assert_array_equal(axial[0], axial[-1])
            np.testing.assert_array_equal(coronal[:, 0, :], coronal[:, -1, :])
            # the manual handles move the PET on top of the registration
            session.setManualHandles(True)
            manual = np.eye(4)
            manual[:3, 3] = (0.0, 3.0, 0.0)
            session.manual.SetMatrixTransformToParent(_vtkMatrix(manual))
            moved = _center(session.petView, world=True)
            session.resetManual()
            self.assertAlmostEqual(moved[1] - _center(session.petView, world=True)[1], 3.0, places=3)
            session.setManualHandles(False)
            session.setFixedWindow(-160.0, 240.0)
            self.assertEqual(session.fixedHelper.GetDisplayNode().GetWindowLevelMax(), 240.0)

            # harden, saved separately: every frame and the summed image on the CT grid, on the CT
            undo, newSequence, newSummed = logic.harden(sequence, summed, ct, session.linear, reg.SAVE_SEPARATELY,
                                                        reg.OUTPUT_PET_SPACING)
            self.assertEqual(newSequence.GetNumberOfDataNodes(), 3)
            self.assertEqual(newSequence.GetAttribute(PREFIX + "SummedVolumeID"), newSummed.GetID())
            target = self.phantom()
            registered = slicer.util.arrayFromVolume(newSummed)
            before = np.abs(slicer.util.arrayFromVolume(summed) - 2.0 * target).mean()
            self.assertLess(np.abs(registered - 2.0 * target)[:, :, 4:-4].mean(), 0.5 * before)
            frame = slicer.util.arrayFromVolume(newSequence.GetNthDataNode(2))
            frameBefore = np.abs(slicer.util.arrayFromVolume(sequence.GetNthDataNode(2)) - 3.0 * target).mean()
            self.assertLess(np.abs(frame - 3.0 * target)[:, :, 4:-4].mean(), 0.5 * frameBefore)
            logic.revert(undo)
            self.assertFalse(slicer.mrmlScene.IsNodePresent(newSequence))

            # a custom voxel size, overwriting the series; revert puts it back
            original = slicer.util.arrayFromVolume(sequence.GetNthDataNode(0)).copy()
            undo, same, _ = logic.harden(sequence, summed, ct, session.linear, reg.OVERWRITE, reg.OUTPUT_CUSTOM,
                                         (8.0, 8.0, 8.0), "Nearest neighbour")
            self.assertIs(same, sequence)
            np.testing.assert_allclose(reg.gridSpacing(_ijkToRas(sequence.GetNthDataNode(0))), (8.0, 8.0, 8.0))
            self.assertEqual(slicer.util.arrayFromVolume(summed).shape, (20, 24, 24))
            logic.revert(undo)
            np.testing.assert_array_equal(slicer.util.arrayFromVolume(sequence.GetNthDataNode(0)), original)
            self.delayDisplay("Dynamic PET to CT/MR Registration open")
        finally:
            session.close()
        helpers = [n for n in slicer.util.getNodesByClass("vtkMRMLNode") if n.GetAttribute(HELPER_ATTRIBUTE)]
        self.assertEqual(helpers, [])
