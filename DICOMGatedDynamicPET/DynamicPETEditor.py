"""
Dynamic PET Review/Edit (3D Slicer module, internally DynamicPETEditor)

Opens a dynamic PET sequence (as loaded by the DICOM Gated / Dynamic PET plugin) in its own layout:
    upper 2/3   a large axial view of the current frame on the left and a near-square grid of small axial
                views on the right, one per time frame, all at the same slice position
    lower 1/3   a timeline: a time ruler from the study start over one block per frame, as wide as its
                duration; a two-handle time slider (or the mouse wheel) zooms into a time window; click a
                frame to show it, Ctrl / Shift + click to select frames, Left / Right to step, play / pause
Each view shows its frame number and start - end time from the study start in its top left corner.
The module panel sets the window (the same display range for all frames, or each frame on its own), the
color map and fusion: an anatomical background volume (CT / MR) under the PET in every view, with a PET opacity.
Spherical ROIs placed in the large view give time-activity curves, drawn on the timeline over the frame blocks.
Edit mode (timeline bar): Add frame (a static or dynamic PET series placed by its clock time, decay correction
aligned for Bq/ml), Merge (duration weighted mean of neighbouring frames), Split (a merged frame back into its
frames), Remove, Discard changes, Save as... (a new sequence) and Overwrite original. Edits are shown at once on an
edited timeline (edited frames in red) over the original one (locked, half height), and in the views; the original
series is not changed until it is overwritten.
"""

import datetime

import os
import sys
import time
import logging

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

from DICOMGatedDynamicPETLib import colormaps, dynamicviews, frameedit as fe, roicurves  # noqa: E402
from DICOMGatedDynamicPETLib.timeline import FrameTimeline  # noqa: E402

LAYOUT_ID = 7801
PREFIX = "DynamicPET."
TIMING_KEYS = ("Frame", "StartFromStudyStartS", "StartFromScanStartS", "DurationS", "StartDateTime",
               "StartFromInjectionS")
EDITOR_ATTRIBUTE = "DynamicPETEditor.FrameVolume"
COMMON_WINDOW, FRAME_WINDOW = "common", "frame"
AUTO_PERCENTILE = 99.9
ROI_ATTRIBUTE = "ReviewEdit.SphereROI"
ROI_RADIUS_ATTRIBUTE = "ReviewEdit.RadiusMm"
ROI_COLOR_ATTRIBUTE = "ReviewEdit.Color"
ROI_NODE_CLASS = "vtkMRMLMarkupsFiducialNode"
ROI_DEFAULT_RADIUS_MM = 10.0
ROI_DEBOUNCE_MS = 100


class DynamicPETEditor(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        parent.title = "Dynamic PET Review/Edit"
        parent.categories = ["Nuclear Medicine.PET4D"]
        parent.dependencies = []
        parent.contributors = ["Burak Demir"]
        parent.helpText = ("Review a dynamic PET series frame by frame: a large view of the current frame, small "
                           "views of all frames at the same slice and a timeline from the study start. Pick the "
                           "dynamic sequence and press Open Review/Edit. Spherical ROIs placed in the large view "
                           "give time-activity curves on the timeline.")
        parent.acknowledgementText = "Part of the DICOM Gated / Dynamic PET extension."


# ---------------------------------------------------------------------------
# Logic
# ---------------------------------------------------------------------------
class DynamicPETEditorLogic(ScriptedLoadableModuleLogic):
    @staticmethod
    def isDynamicSequence(sequenceNode):
        return bool(sequenceNode and sequenceNode.GetNumberOfDataNodes() >= 1
                    and sequenceNode.GetNthDataNode(0) is not None
                    and sequenceNode.GetNthDataNode(0).IsA("vtkMRMLScalarVolumeNode"))

    @staticmethod
    def frameIntervals(sequenceNode):
        """dynamicviews.frameIntervals of the frames of a sequence (from the DynamicPET.* attributes)."""
        rows = []
        for i in range(sequenceNode.GetNumberOfDataNodes()):
            node = sequenceNode.GetNthDataNode(i)
            row = {key: node.GetAttribute(PREFIX + key) for key in TIMING_KEYS}
            row["IndexValue"] = sequenceNode.GetNthIndexValue(i)
            rows.append(row)
        return dynamicviews.frameIntervals(rows)

    @staticmethod
    def shownFrames(count, maxViews):
        """Frame indexes that get a small view: all of them, or maxViews spread evenly over the series."""
        if count <= maxViews:
            return list(range(count))
        return sorted({int(round(v)) for v in np.linspace(0, count - 1, maxViews)})

    @staticmethod
    def isStaticPet(volumeNode):
        return any(node is volumeNode for node, kind, _ in petCandidates() if kind == "static")

    @staticmethod
    def browserFor(sequenceNode):
        logic = slicer.modules.sequences.logic()
        browser = logic.GetFirstBrowserNodeForSequenceNode(sequenceNode)
        if browser is None:
            browser = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSequenceBrowserNode",
                                                         f"{sequenceNode.GetName()} browser")
            logic.AddSynchronizedNode(sequenceNode, None, browser)
            if hasattr(browser, "SetAndObservePrimarySequenceNodeID"):
                browser.SetAndObservePrimarySequenceNodeID(sequenceNode.GetID())
            else:
                browser.SetAndObserveMasterSequenceNodeID(sequenceNode.GetID())
        return browser


def guardViewVolumes(widget, expected):
    """Keep the volumes of a slice view as expected() says, (background, foreground, label) node IDs (None: empty):
    Slicer puts a newly loaded volume into every slice view, also into the editor's frame grid. Returns the
    (composite node, observer tag) to remove on close."""
    composite = widget.mrmlSliceCompositeNode()

    def restore(caller=None, event=None):
        wanted = expected()
        current = (composite.GetBackgroundVolumeID(), composite.GetForegroundVolumeID(),
                   composite.GetLabelVolumeID())
        if tuple(x or None for x in current) != tuple(wanted):
            composite.SetBackgroundVolumeID(wanted[0])
            composite.SetForegroundVolumeID(wanted[1])
            composite.SetLabelVolumeID(wanted[2])

    restore()
    return composite, composite.AddObserver(vtk.vtkCommand.ModifiedEvent, restore)


# ---------------------------------------------------------------------------
# Editing: the PET series that can be added, their description, resampling and the saved series
# ---------------------------------------------------------------------------
STATIC_PREFIX = "StaticPET."
GATED_PREFIX = "GatedPET."
SUMMED_PREFIX_KEYS = ("SummedVolumeID", "FrameTimingTableID")
LONG_GAP_S = 60 * 60.0           # warn when the series are more than an hour apart
EDIT_ATTRIBUTE = "DynamicPETEditor.Edit"
# per-frame tags that do not hold for an edited frame (they are written again or removed)
FRAME_TIMING_TAGS = ("Frame", "StartDateTime", "StartFromInjectionS", "StartFromStudyStartS", "StartFromScanStartS",
                     "DurationS", "DurationMs", "MidFromInjectionS", "MidFromScanStartS", "EndFromInjectionS",
                     "FrameReferenceTimeMs", "DecayFactor", "StartSource", "LoaderDecayFactor", "FrameScale")


def _isHidden(node):
    return bool(node.GetHideFromEditors()) or node.GetAttribute(EDITOR_ATTRIBUTE) == "1"


def _isForeignStaticPet(node):
    """A volume loaded from DICOM by another plugin that is static PET: modality PT, Series Type not GATED or
    DYNAMIC, no gates or time slices (from the DICOM database)."""
    uids = (node.GetAttribute("DICOM.instanceUIDs") or "").split()
    if not uids or not getattr(slicer, "dicomDatabase", None):
        return False
    try:
        db = slicer.dicomDatabase
        if (db.instanceValue(uids[0], "0008,0060") or "").strip() != "PT":
            return False
        seriesType = (db.instanceValue(uids[0], "0054,1000") or "").upper()
        counts = [fe.number(db.instanceValue(uids[0], tag)) for tag in ("0054,0071", "0054,0101")]
        return "GATED" not in seriesType and "DYNAMIC" not in seriesType and all(not c or c <= 1 for c in counts)
    except Exception:
        return False


def petCandidates(exclude=None):
    """PET series that Add frame accepts, as (node, "static" / "dynamic", label): dynamic PET sequences and static
    PET volumes (loaded by this extension, or by another plugin from DICOM PT); no gated PET, no summed or derived
    volumes, no sequence proxies."""
    out = []
    for sequence in slicer.util.getNodesByClass("vtkMRMLSequenceNode"):
        if sequence is exclude or _isHidden(sequence) or sequence.GetNumberOfDataNodes() < 1:
            continue
        first = sequence.GetNthDataNode(0)
        if first is not None and (first.GetAttribute(PREFIX + "Kind") or "").upper() == "DYNAMIC":
            out.append((sequence, "dynamic", f"{sequence.GetName()} (dynamic)"))
    for volume in slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode"):
        if _isHidden(volume) or isinstance(volume, slicer.vtkMRMLLabelMapVolumeNode):
            continue
        names = list(volume.GetAttributeNames() or ())
        if any(n.startswith(PREFIX) or n.startswith(GATED_PREFIX) or n.startswith("MPI.") for n in names):
            continue        # dynamic frames / proxies / summed images, gated PET
        if (volume.GetAttribute(STATIC_PREFIX + "Kind") or "").upper() == "STATIC" or \
                _isForeignStaticPet(volume):
            out.append((volume, "static", f"{volume.GetName()} (static)"))
    return out


def _dicomAttributes(node):
    """The StaticPET.* tags of a static PET loaded by another plugin, from its DICOM files: {key: text}; raises
    ValueError when it cannot be read or is not static PET."""
    import pydicom
    from DICOMGatedDynamicPETLib import petseries
    uids = (node.GetAttribute("DICOM.instanceUIDs") or "").split()
    files = [slicer.dicomDatabase.fileForInstance(uid) for uid in uids] if uids else []
    files = [f for f in files if f]
    if not files:
        raise ValueError(f"'{node.GetName()}' has no DICOM files in the database to read its tags from.")
    datasets = [pydicom.dcmread(f, stop_before_pixels=True) for f in files]
    if petseries.datasetKind(datasets[0]) != petseries.STATIC:
        raise ValueError(f"'{node.GetName()}' is not a static PET series.")
    attributes = petseries.acquisitionInfo(datasets)
    attributes.update(petseries.suvAttributes(petseries.suvInfo(datasets)))
    timing = petseries.frameTiming([datasets], datasets)[0]
    for key in ("StartDateTime", "DurationS", "StartFromInjectionS"):
        if timing.get(key) is not None:
            attributes[key] = str(timing[key])
    attributes["Kind"] = petseries.STATIC
    attributes["Units"] = _foreignUnits(node)
    return {k: str(v) for k, v in attributes.items() if v not in (None, "")}


def _foreignUnits(node):
    """Units of a volume loaded by another plugin (PET SUV plugin: SUVbw), from its voxel units or its name."""
    texts = [node.GetName()]
    try:
        units = node.GetVoxelValueUnits()
        if units is not None:
            texts += [units.GetCodeValue() or "", units.GetCodeMeaning() or ""]
    except Exception:
        pass
    text = " ".join(texts).lower()
    for key, value in (("lbm", "SUVlbm"), ("bsa", "SUVbsa"), ("suv", "SUVbw"), ("bq", "Bq/ml")):
        if key in text:
            return value
    return ""


def seriesTags(node, kind):
    """{key: text} of the series tags (without prefix) of a dynamic sequence (its first frame) or a static
    volume."""
    if kind == "dynamic":
        first = node.GetNthDataNode(0)
        return {n[len(PREFIX):]: first.GetAttribute(n) for n in (first.GetAttributeNames() or ())
                if n.startswith(PREFIX)}
    if (node.GetAttribute(STATIC_PREFIX + "Kind") or "").upper() == "STATIC":
        tags = {n[len(STATIC_PREFIX):]: node.GetAttribute(n) for n in (node.GetAttributeNames() or ())
                if n.startswith(STATIC_PREFIX)}
        if not tags.get("StartDateTime") or not tags.get("DurationS") or not tags.get("PatientID"):
            try:     # loaded before these tags were kept: read them from its DICOM files
                for key, value in _dicomAttributes(node).items():
                    if not tags.get(key):
                        tags[key] = value
            except Exception:
                logging.info(f"Review/Edit: no DICOM tags for '{node.GetName()}'", exc_info=True)
        return tags
    return _dicomAttributes(node)


def decayStatus(tags):
    """(decay corrected, reference datetime) from the series tags: the Decay Correction DateTime when given, the
    injection for ADMIN, else the scan start (the SUV reference, the first frame start when the loader decay
    corrected the frames)."""
    correction = (tags.get("DecayCorrection") or "").upper()
    loader = (tags.get("LoaderDecayCorrection") or "").startswith("frames decay corrected")
    if correction == "NONE" and not loader:
        return False, None
    reference = fe.parseIso(tags.get("EditDecayReference"))
    if reference is None and correction != "NONE":
        reference = fe.parseIso(tags.get("DecayCorrectionDateTime"))
    if reference is None and correction == "ADMIN":
        reference = fe.parseIso(tags.get("InjectionTime"))
    if reference is None:
        reference = fe.parseIso(tags.get("ScanStart"))
    if reference is None and loader:
        text = tags.get("LoaderDecayCorrection") or ""
        reference = fe.parseIso(text[text.rfind("(") + 1:text.rfind(")")] if "(" in text else None)
    return True, reference


def describeSeries(node, kind, tags=None, frames=None):
    """frameedit.seriesInfo of a PET series; frames: [{"start": datetime, "duration": s}] (default: its own)."""
    tags = tags if tags is not None else seriesTags(node, kind)
    if frames is None and kind == "dynamic":
        frames = []
        for i in range(node.GetNumberOfDataNodes()):
            frameNode = node.GetNthDataNode(i)
            frames.append({"start": fe.parseIso(frameNode.GetAttribute(PREFIX + "StartDateTime")),
                           "duration": fe.number(frameNode.GetAttribute(PREFIX + "DurationS"))})
    elif frames is None:
        frames = [{"start": fe.parseIso(tags.get("StartDateTime")), "duration": fe.number(tags.get("DurationS"))}]
    corrected, reference = decayStatus(tags)
    return fe.seriesInfo(
        name=node.GetName(), kind=kind, units=tags.get("Units"), frames=frames,
        injection=fe.parseIso(tags.get("InjectionTime")), halfLife=fe.number(tags.get("RadionuclideHalfLifeS")),
        radionuclide=tags.get("Radionuclide"), decayCorrection=(tags.get("DecayCorrection") or "").upper(),
        decayCorrected=corrected, decayReference=reference, patientName=tags.get("PatientName"),
        patientID=tags.get("PatientID"), manufacturer=tags.get("Manufacturer"), model=tags.get("ScannerModel"),
        serial=tags.get("DeviceSerialNumber"), station=tags.get("StationName"))


def _vtkMatrix(node):
    matrix = vtk.vtkMatrix4x4()
    node.GetIJKToRASMatrix(matrix)
    return matrix


def _array(node):
    from vtk.util import numpy_support
    image = node.GetImageData()
    scalars = image.GetPointData().GetScalars() if image is not None else None
    if scalars is None:
        return None
    return numpy_support.vtk_to_numpy(scalars).reshape(image.GetDimensions()[::-1])


def sameGeometry(node, dims, ijkToRas, tolerance=1e-3):
    image = node.GetImageData()
    if image is None or tuple(image.GetDimensions()) != tuple(dims):
        return False
    return np.allclose(slicer.util.arrayFromVTKMatrix(_vtkMatrix(node)), ijkToRas, atol=tolerance)


def resampleToGrid(node, dims, ijkToRas):
    """The image of a volume node resampled (linear, 0 outside) on another grid (dims i, j, k and IJK to RAS),
    following the node's linear parent transform: a float32 array kji."""
    from vtk.util import numpy_support
    sourceToWorld = np.eye(4)
    parent = node.GetParentTransformNode()
    if parent is not None:
        if not parent.IsTransformToWorldLinear():
            raise ValueError(f"'{node.GetName()}' is under a non linear transform; harden it first.")
        matrix = vtk.vtkMatrix4x4()
        parent.GetMatrixTransformToWorld(matrix)
        sourceToWorld = slicer.util.arrayFromVTKMatrix(matrix)
    sourceIjkToWorld = sourceToWorld @ slicer.util.arrayFromVTKMatrix(_vtkMatrix(node))
    axes = np.linalg.inv(sourceIjkToWorld) @ np.asarray(ijkToRas, dtype=float)
    image = vtk.vtkImageData()
    image.ShallowCopy(node.GetImageData())
    image.SetOrigin(0.0, 0.0, 0.0)
    image.SetSpacing(1.0, 1.0, 1.0)
    reslice = vtk.vtkImageReslice()
    reslice.SetInputData(image)
    reslice.SetResliceAxes(slicer.util.vtkMatrixFromArray(axes))
    reslice.SetOutputOrigin(0.0, 0.0, 0.0)
    reslice.SetOutputSpacing(1.0, 1.0, 1.0)
    reslice.SetOutputExtent(0, dims[0] - 1, 0, dims[1] - 1, 0, dims[2] - 1)
    reslice.SetOutputScalarType(vtk.VTK_FLOAT)
    reslice.SetInterpolationModeToLinear()
    reslice.SetBackgroundLevel(0.0)
    reslice.Update()
    output = reslice.GetOutput()
    return numpy_support.vtk_to_numpy(output.GetPointData().GetScalars()).reshape(tuple(dims)[::-1]).copy()


def imageFromArray(array):
    from vtk.util import numpy_support
    array = np.ascontiguousarray(array, dtype=np.float32)
    image = vtk.vtkImageData()
    image.SetDimensions(array.shape[2], array.shape[1], array.shape[0])
    image.GetPointData().SetScalars(numpy_support.numpy_to_vtk(array.ravel(), deep=True, array_type=vtk.VTK_FLOAT))
    return image


# ---------------------------------------------------------------------------
# Spherical ROIs: a point in the large view, a sphere around it and its time-activity curve
# ---------------------------------------------------------------------------
class SphereROIs:
    """The spherical ROIs of an open session. Each is a fiducial node with one point (it stays in the scene, and
    is taken up again by the next session) and a sphere model that follows it; both show in the large view only.
    A roi is {"node", "source", "model", "tags", "values"}, values being its mean per frame (empty until placed).
    Curves are computed from the frames of the sequence, shortly after the point or the radius last changed."""

    def __init__(self, session):
        self.session = session
        self.rois = []
        self.pending = None           # roi waiting for its click in the view
        self.dirty = []               # rois whose curve is out of date
        self.observers = []
        self.onChanged = None         # callback() when a ROI is added, placed, removed or resized
        self.timer = qt.QTimer()
        self.timer.setSingleShot(True)
        self.timer.setInterval(ROI_DEBOUNCE_MS)
        self.timer.connect("timeout()", self.flush)

    def open(self):
        interaction = slicer.app.applicationLogic().GetInteractionNode()
        proxy = self.session.proxyNode
        self.observers = [
            (slicer.mrmlScene, slicer.mrmlScene.AddObserver(slicer.mrmlScene.NodeRemovedEvent, self.onNodeRemoved)),
            (interaction, interaction.AddObserver(interaction.EndPlacementEvent, self.onEndPlacement)),
            (proxy, proxy.AddObserver(slicer.vtkMRMLTransformableNode.TransformModifiedEvent, self.onVolumeMoved))]
        for node in slicer.util.getNodesByClass(ROI_NODE_CLASS):
            if node.GetAttribute(ROI_ATTRIBUTE) == "1":
                self._adopt(node)
        self.flush()

    def close(self):
        """Removes the sphere models and the observers; the fiducial nodes stay in the scene."""
        self.timer.stop()
        self.cancelPlacement()
        for node, tag in self.observers:
            node.RemoveObserver(tag)
        self.observers = []
        for roi in self.rois:
            self._release(roi)
        self.rois = []
        self.dirty = []

    # --- adding and removing ----------------------------------------------------------------------
    def add(self, radiusMm, centerRas=None):
        """A new ROI at centerRas, or, without it, waiting for a click in the large view."""
        self.cancelPlacement()
        node = slicer.mrmlScene.AddNewNodeByClass(ROI_NODE_CLASS, self._newName())
        node.SetMaximumNumberOfControlPoints(1)
        node.SetControlPointLabelFormat("%N")
        node.SetAttribute(ROI_ATTRIBUTE, "1")
        node.SetAttribute(ROI_RADIUS_ATTRIBUTE, f"{radiusMm:g}")
        roi = self._adopt(node)
        if centerRas is not None:
            node.AddControlPointWorld(vtk.vtkVector3d(*centerRas))
        else:
            self.pending = roi
            logic = slicer.app.applicationLogic()
            logic.GetSelectionNode().SetReferenceActivePlaceNodeClassName(ROI_NODE_CLASS)
            logic.GetSelectionNode().SetActivePlaceNodeID(node.GetID())
            logic.GetInteractionNode().SetPlaceModePersistence(0)
            logic.GetInteractionNode().SetCurrentInteractionMode(slicer.vtkMRMLInteractionNode.Place)
        self._changed()
        return roi

    def remove(self, roi):
        slicer.mrmlScene.RemoveNode(roi["node"])    # onNodeRemoved does the rest

    def removeAll(self):
        for roi in list(self.rois):
            self.remove(roi)

    def cancelPlacement(self):
        roi, self.pending = self.pending, None
        if roi is None:
            return
        interaction = slicer.app.applicationLogic().GetInteractionNode()
        if interaction.GetCurrentInteractionMode() == slicer.vtkMRMLInteractionNode.Place:
            interaction.SetCurrentInteractionMode(slicer.vtkMRMLInteractionNode.ViewTransform)
        if self.center(roi) is None:
            self.remove(roi)

    def onEndPlacement(self, caller=None, event=None):
        roi, self.pending = self.pending, None
        if roi is not None:      # the view may set the point just after this event
            qt.QTimer.singleShot(0, lambda: self._finishPlacement(roi))

    def _finishPlacement(self, roi):
        if not any(roi is r for r in self.rois):
            return
        if self.center(roi) is None:
            self.remove(roi)     # right click / Esc: no click in the view
        else:
            self._changed()

    @vtk.calldata_type(vtk.VTK_OBJECT)
    def onNodeRemoved(self, caller, event, node):
        roi = self._roiOf(node)
        if roi is None:
            return
        self.rois = [r for r in self.rois if r is not roi]
        self.dirty = [r for r in self.dirty if r is not roi]
        if self.pending is roi:
            self.pending = None
        self._release(roi)
        self._showCurves()
        self._changed()

    def _roiOf(self, node):
        return next((roi for roi in self.rois if roi["node"] is node), None)

    def _newName(self):
        names = {node.GetName() for node in slicer.util.getNodesByClass(ROI_NODE_CLASS)}
        number = 1
        while f"ROI {number}" in names:
            number += 1
        return f"ROI {number}"

    def _adopt(self, node):
        """Set up a ROI node: its color and display, its sphere and the observers."""
        color = node.GetAttribute(ROI_COLOR_ATTRIBUTE)
        if not color:
            color = roicurves.nextColor([roi["node"].GetAttribute(ROI_COLOR_ATTRIBUTE) for roi in self.rois])
            node.SetAttribute(ROI_COLOR_ATTRIBUTE, color)
        rgb = roicurves.colorRgb(color)
        viewID = self.session.mainViewNodeID()
        node.CreateDefaultDisplayNodes()
        display = node.GetDisplayNode()
        display.SetColor(*rgb)
        display.SetSelectedColor(*rgb)
        display.SetActiveColor(*rgb)
        display.SetGlyphType(slicer.vtkMRMLMarkupsDisplayNode.Cross2D)    # keeps the voxels in view
        display.SetGlyphScale(1.5)
        display.SetTextScale(2.0)
        display.SetPropertiesLabelVisibility(False)
        display.RemoveAllViewNodeIDs()
        display.AddViewNodeID(viewID)
        source = vtk.vtkSphereSource()
        source.SetThetaResolution(48)
        source.SetPhiResolution(32)
        model = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", f"{node.GetName()} sphere")
        model.SetHideFromEditors(True)
        model.SetSaveWithScene(False)
        model.SetPolyDataConnection(source.GetOutputPort())
        model.CreateDefaultDisplayNodes()
        modelDisplay = model.GetDisplayNode()
        modelDisplay.SetColor(*rgb)
        modelDisplay.SetVisibility2D(True)
        modelDisplay.SetSliceIntersectionThickness(2)
        modelDisplay.RemoveAllViewNodeIDs()
        modelDisplay.AddViewNodeID(viewID)
        events = (slicer.vtkMRMLMarkupsNode.PointModifiedEvent, slicer.vtkMRMLMarkupsNode.PointPositionDefinedEvent,
                  slicer.vtkMRMLMarkupsNode.PointRemovedEvent, slicer.vtkMRMLTransformableNode.TransformModifiedEvent)
        roi = {"node": node, "source": source, "model": model, "values": [],
               "tags": [node.AddObserver(event, self.onPointModified) for event in events]}
        self.rois.append(roi)
        self._updateSphere(roi)
        self._schedule(roi)
        return roi

    def _release(self, roi):
        for tag in roi["tags"]:
            roi["node"].RemoveObserver(tag)
        roi["tags"] = []
        if slicer.mrmlScene.IsNodePresent(roi["model"]):
            slicer.mrmlScene.RemoveNode(roi["model"])

    # --- sphere and curve -------------------------------------------------------------------------
    def radius(self, roi):
        return float(roi["node"].GetAttribute(ROI_RADIUS_ATTRIBUTE) or ROI_DEFAULT_RADIUS_MM)

    def setRadius(self, roi, radiusMm):
        roi["node"].SetAttribute(ROI_RADIUS_ATTRIBUTE, f"{radiusMm:g}")
        self._updateSphere(roi)
        self._schedule(roi)
        self._changed()

    def center(self, roi):
        """The point in world RAS; None until it is placed (the point following the mouse does not count)."""
        node = roi["node"]
        if node.GetNumberOfDefinedControlPoints() < 1:
            return None
        point = [0.0, 0.0, 0.0]
        node.GetNthControlPointPositionWorld(0, point)
        return point

    def onPointModified(self, caller, event):
        roi = self._roiOf(caller)
        if roi is not None:
            self._updateSphere(roi)
            self._schedule(roi)

    def onVolumeMoved(self, caller=None, event=None):
        for roi in self.rois:
            self._schedule(roi)

    def _updateSphere(self, roi):
        center = self.center(roi)
        roi["model"].GetDisplayNode().SetVisibility(center is not None)
        if center is not None:
            roi["source"].SetCenter(center)
            roi["source"].SetRadius(self.radius(roi))
            roi["source"].Update()

    def _schedule(self, roi):
        if not any(roi is r for r in self.dirty):
            self.dirty.append(roi)
        self.timer.start()

    def flush(self):
        """Compute the curves that are out of date now, and show them."""
        self.timer.stop()
        dirty, self.dirty = self.dirty, []
        if not dirty:
            return
        for roi in dirty:
            center = self.center(roi)
            roi["values"] = [] if center is None else roicurves.sphereCurve(
                self._frames(), self.session.worldToVolume(center), self.radius(roi))
        self._showCurves()

    def _frames(self):
        """(array_kji, ijkToRas) of every frame shown (the edited frames while editing), or None for a frame
        without an image."""
        return self.session.frameArrays()

    def refreshAll(self):
        """The frames changed (an edit): every curve again."""
        for roi in self.rois:
            self._schedule(roi)
        self.flush()

    def units(self):
        sequence = self.session.sequenceNode
        first = sequence.GetNthDataNode(0)
        return first.GetAttribute(PREFIX + "Units") or sequence.GetAttribute(PREFIX + "Units") or ""

    def _showCurves(self):
        if self.session.timeline is not None:
            self.session.timeline.setCurves([{"name": roi["node"].GetName(), "values": roi["values"],
                                              "color": roi["node"].GetAttribute(ROI_COLOR_ATTRIBUTE)}
                                             for roi in self.rois], self.units())

    def _changed(self):
        if self.onChanged:
            self.onChanged()


# ---------------------------------------------------------------------------
# Editor session: layout, frame views, overlays, timeline dock
# ---------------------------------------------------------------------------
class DynamicPETEditorSession:
    def __init__(self, sequenceNode, maxViews=64, showOverlays=True, syncSlices=True):
        self.sequenceNode = sequenceNode
        self.maxViews = maxViews
        self.showOverlays = showOverlays
        self.syncSlices = syncSlices
        self.browserNode = None
        self.proxyNode = None
        self.intervals = []           # frames shown: the edited frames while editing, else the original ones
        self.originalIntervals = []
        self.viewFrames = []          # frame index shown in each small view
        self.frameVolumes = []
        self.overlays = {}            # view name -> (sliceView, renderer, text actor)
        self.observers = []
        self.viewObservers = []       # guards of the empty grid views (rebuilt with the grid)
        self.dock = None
        self.timeline = None
        self.originalTimeline = None
        self.previousLayout = None
        self.frameStats = []          # (auto upper = 99.9th percentile, maximum, minimum) per frame
        self.windowMode = COMMON_WINDOW
        self.window = (0.0, 1.0)
        self.onWindowChanged = None   # callback(lower, upper) when the window changes in a view (mouse)
        self._applyingWindow = False
        self.colorNodeID = None
        self.fusionVolume = None
        self.fusionOpacity = 0.5
        self.playTimer = None
        self.playSpeed = 1.0          # acquisition seconds per real second
        self.playTime = None
        self._lastTick = None
        self.rois = None              # SphereROIs
        # editing
        self.edit = None              # frameedit.FrameEdit once something was edited
        self.editArrays = {}          # atom key of an added frame -> (array kji, ijkToRas)
        self.editVolume = None        # current edited frame, in the large view while editing
        self.editIndex = 0
        self._imageCache = {}
        self.temporarySequence = False   # a static volume opened as a one-frame series (Save as only)
        self.onEditsChanged = None    # callback() after every edit
        self.onSaved = None           # callback(sequence node) after Save as / Overwrite
        self._candidateTimer = None

    # --- open / close ---------------------------------------------------------------------------
    def open(self):
        logic = DynamicPETEditorLogic()
        self.browserNode = logic.browserFor(self.sequenceNode)
        self.proxyNode = self.browserNode.GetProxyNode(self.sequenceNode)
        self.originalIntervals = logic.frameIntervals(self.sequenceNode)
        self.intervals = list(self.originalIntervals)
        self.viewFrames = logic.shownFrames(len(self.intervals), self.maxViews)
        self._computeFrameStats()
        self._makeFrameVolumes()
        self._setLayout()
        self._setupViews()
        self._makeDock()
        self._observe()
        proxyDisplay = self._proxyDisplay()
        if proxyDisplay is not None:
            self.window = (proxyDisplay.GetWindowLevelMin(), proxyDisplay.GetWindowLevelMax())
            self.colorNodeID = proxyDisplay.GetColorNodeID()
        else:
            self.window = self.autoWindow()
        self.onBrowserModified()
        self.rois = SphereROIs(self)
        self.rois.open()

    def close(self):
        if self.rois is not None:
            self.rois.close()
            self.rois = None
        if self.playTimer is not None:
            self.playTimer.stop()
            self.playTimer = None
        if self._candidateTimer is not None:
            self._candidateTimer.stop()
        for node, tag in self.observers + self.viewObservers:
            if node is not None:
                node.RemoveObserver(tag)
        self.observers, self.viewObservers = [], []
        self._removeOverlays()
        if slicer.app.layoutManager().layout == LAYOUT_ID:
            self.setLinked(False)
        if self.dock is not None:
            slicer.util.mainWindow().removeDockWidget(self.dock)
            self.dock.deleteLater()
            self.dock = None
        layoutManager = slicer.app.layoutManager()
        if self.previousLayout is not None and layoutManager.layout == LAYOUT_ID:
            layoutManager.setLayout(self.previousLayout)
        self._removeFrameVolumes()
        if self.editVolume is not None and slicer.mrmlScene.IsNodePresent(self.editVolume):
            slicer.mrmlScene.RemoveNode(self.editVolume)
        self.editVolume = None
        self._imageCache = {}
        self.editArrays = {}
        if self.temporarySequence:
            for node in (self.browserNode, self.proxyNode, self.sequenceNode):
                if node is not None and slicer.mrmlScene.IsNodePresent(node):
                    slicer.mrmlScene.RemoveNode(node)

    def _removeOverlays(self):
        for sliceView, renderer, actor in self.overlays.values():
            renderer.RemoveActor2D(actor)
            sliceView.scheduleRender()
        self.overlays = {}

    def _removeFrameVolumes(self):
        for node in self.frameVolumes:
            if slicer.mrmlScene.IsNodePresent(node):
                slicer.mrmlScene.RemoveNode(node)
        self.frameVolumes = []

    # --- frames shown ----------------------------------------------------------------------------
    def isEditing(self):
        return self.edit is not None and self.edit.isEdited()

    def frameCount(self):
        return len(self.intervals)

    def _atomArray(self, key):
        if key in self.editArrays:
            return self.editArrays[key][0]
        return _array(self.sequenceNode.GetNthDataNode(int(key[1:]) - 1))

    def _atomMatrix(self, key):
        if key in self.editArrays:
            return self.editArrays[key][1]
        return slicer.util.arrayFromVTKMatrix(_vtkMatrix(self.sequenceNode.GetNthDataNode(int(key[1:]) - 1)))

    def _atomGrid(self, key):
        array = self._atomArray(key)
        return (None if array is None else array.shape), np.round(self._atomMatrix(key), 4).tobytes()

    def _frameData(self, index):
        """(vtkImageData, vtkMatrix4x4 IJK to RAS) of a frame shown. While editing, an edited frame is computed
        (duration weighted mean of its frames, decay alignment) once and kept; untouched frames share the
        original's voxels."""
        original = index if not self.isEditing() else self.edit.isOriginalFrame(index)
        if original is not None:
            node = self.sequenceNode.GetNthDataNode(original)
            return node.GetImageData(), _vtkMatrix(node)
        keys = tuple(self.edit.frames[index])
        cacheKey = (keys, tuple(round(self.edit.scale(k), 12) for k in keys))
        if cacheKey not in self._imageCache:
            array = self.edit.frameArray(index, self._atomArray)
            self._imageCache[cacheKey] = (imageFromArray(array), slicer.util.vtkMatrixFromArray(
                self._atomMatrix(keys[0])))
        return self._imageCache[cacheKey]

    def frameArrays(self):
        """(array_kji, ijkToRas) of every frame shown, or None for a frame without an image."""
        from vtk.util import numpy_support
        for i in range(self.frameCount()):
            image, matrix = self._frameData(i)
            scalars = image.GetPointData().GetScalars() if image is not None else None
            if scalars is None:
                yield None
                continue
            yield (numpy_support.vtk_to_numpy(scalars).reshape(image.GetDimensions()[::-1]),
                   slicer.util.arrayFromVTKMatrix(matrix))

    # --- window and color --------------------------------------------------------------------------
    def _computeFrameStats(self):
        from vtk.util import numpy_support
        self.frameStats = []
        for i in range(self.frameCount()):
            image, _ = self._frameData(i)
            scalars = image.GetPointData().GetScalars() if image is not None else None
            if scalars is None:
                self.frameStats.append((1.0, 1.0, 0.0))
                continue
            values = numpy_support.vtk_to_numpy(scalars)
            sample = values[::4] if values.size > 400000 else values
            self.frameStats.append((float(np.percentile(sample, AUTO_PERCENTILE)), float(values.max()),
                                    float(values.min())))

    def valueRange(self):
        """(lowest, highest) voxel value over all frames, for the window slider."""
        if not self.frameStats:
            return 0.0, 1.0
        return min(0.0, min(s[2] for s in self.frameStats)), max(s[1] for s in self.frameStats)

    def autoWindow(self):
        """0 to the highest 99.9th percentile of the frames."""
        return 0.0, max([s[0] for s in self.frameStats] or [1.0])

    def _proxyDisplay(self):
        return self.proxyNode.GetDisplayNode() if self.proxyNode is not None else None

    def mainVolume(self):
        """The volume of the large view: the current edited frame while editing, else the sequence's proxy."""
        return self.editVolume if self.isEditing() and self.editVolume is not None else self.proxyNode

    def _mainDisplay(self):
        volume = self.mainVolume()
        return volume.GetDisplayNode() if volume is not None else None

    def _displays(self):
        """(display node, frame index or None for the current frame) of every editor volume."""
        out = [(self._proxyDisplay(), None)]
        if self.editVolume is not None:
            out.append((self.editVolume.GetDisplayNode(), None))
        out += [(volume.GetDisplayNode(), index) for volume, index in zip(self.frameVolumes, self.viewFrames)]
        return [(display, index) for display, index in out if display is not None]

    def setWindowMode(self, mode):
        self.windowMode = mode
        self.applyWindow()

    def setCommonWindow(self, lower, upper):
        if upper <= lower:
            return
        self.window = (float(lower), float(upper))
        if self.windowMode == COMMON_WINDOW:
            self.applyWindow()

    def applyWindow(self):
        current = self.currentIndex()
        self._applyingWindow = True
        try:
            for display, index in self._displays():
                if self.windowMode == COMMON_WINDOW:
                    lower, upper = self.window
                else:
                    frame = current if index is None else index
                    lower = 0.0
                    upper = self.frameStats[frame][0] if 0 <= frame < len(self.frameStats) else self.window[1]
                display.AutoWindowLevelOff()
                display.SetWindowLevelMinMax(lower, upper if upper > lower else lower + 1e-6)
        finally:
            self._applyingWindow = False

    def setColorNodeID(self, colorNodeID):
        if not colorNodeID:
            return
        self.colorNodeID = colorNodeID
        for display, _ in self._displays():
            display.SetAndObserveColorNodeID(colorNodeID)

    def onProxyDisplayModified(self, caller=None, event=None):
        """Window changed in the large view (mouse drag): pass it to every frame view."""
        if self._applyingWindow or self.windowMode != COMMON_WINDOW:
            return
        display = self._mainDisplay()
        if display is None or caller is None or caller.GetID() != display.GetID():
            return
        window = (display.GetWindowLevelMin(), display.GetWindowLevelMax())
        if max(abs(window[0] - self.window[0]), abs(window[1] - self.window[1])) < 1e-9:
            return
        self.window = window
        self.applyWindow()
        if self.onWindowChanged:
            self.onWindowChanged(*window)

    # --- fusion ------------------------------------------------------------------------------------
    def _viewVolumes(self):
        """(slice widget, PET volume shown in it) of every editor view."""
        pairs = [(self._sliceWidget(dynamicviews.MAIN_VIEW), self.mainVolume())]
        pairs += [(self._sliceWidget(dynamicviews.frameViewName(v)), volume)
                  for v, volume in enumerate(self.frameVolumes)]
        return [(widget, volume) for widget, volume in pairs if widget is not None and volume is not None]

    def editorVolumes(self):
        return [v for v in (self.proxyNode, self.editVolume, *self.frameVolumes) if v is not None]

    def setFusion(self, backgroundVolume, opacity=None):
        """Show backgroundVolume (None: no fusion) under the PET of every view, the PET as foreground with the
        given opacity. The views are unlinked meanwhile, as a linked view would pass its volumes to the others."""
        self.fusionVolume = backgroundVolume
        if opacity is not None:
            self.fusionOpacity = opacity
        linked = self.syncSlices
        self.setLinked(False)
        try:
            for widget, petVolume in self._viewVolumes():
                composite = widget.mrmlSliceCompositeNode()
                if backgroundVolume is not None:
                    composite.SetBackgroundVolumeID(backgroundVolume.GetID())
                    composite.SetForegroundVolumeID(petVolume.GetID())
                    composite.SetForegroundOpacity(self.fusionOpacity)
                else:
                    composite.SetBackgroundVolumeID(petVolume.GetID())
                    composite.SetForegroundVolumeID(None)
        finally:
            self.setLinked(linked)

    def setFusionOpacity(self, opacity):
        self.fusionOpacity = opacity
        if self.fusionVolume is None:
            return
        linked = self.syncSlices
        self.setLinked(False)
        try:
            for widget, _ in self._viewVolumes():
                widget.mrmlSliceCompositeNode().SetForegroundOpacity(opacity)
        finally:
            self.setLinked(linked)

    # --- frame volumes ---------------------------------------------------------------------------
    def _makeFrameVolumes(self):
        """A hidden volume per small view, sharing the voxels of its frame."""
        proxyDisplay = self.proxyNode.GetDisplayNode() if self.proxyNode else None
        for index in self.viewFrames:
            volume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode",
                                                        f"{self.sequenceNode.GetName()} view F{index + 1}")
            volume.SetHideFromEditors(True)
            volume.SetSaveWithScene(False)
            volume.SetAttribute(EDITOR_ATTRIBUTE, "1")
            image, matrix = self._frameData(index)
            volume.SetAndObserveImageData(image)
            volume.SetIJKToRASMatrix(matrix)
            volume.CreateDefaultDisplayNodes()
            display = volume.GetDisplayNode()
            if proxyDisplay is not None:
                display.SetAndObserveColorNodeID(self.colorNodeID or proxyDisplay.GetColorNodeID())
                display.AutoWindowLevelOff()
                display.SetWindowLevelMinMax(proxyDisplay.GetWindowLevelMin(), proxyDisplay.GetWindowLevelMax())
                display.SetInterpolate(proxyDisplay.GetInterpolate())
            self.frameVolumes.append(volume)

    def _ensureEditVolume(self):
        if self.editVolume is not None:
            return
        volume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode",
                                                    f"{self.sequenceNode.GetName()} (current edited frame)")
        volume.SetHideFromEditors(True)
        volume.SetSaveWithScene(False)
        volume.SetAttribute(EDITOR_ATTRIBUTE, "1")
        volume.CreateDefaultDisplayNodes()
        proxyDisplay = self._proxyDisplay()
        if proxyDisplay is not None:
            volume.GetDisplayNode().CopyContent(proxyDisplay)
        parent = self.proxyNode.GetParentTransformNode() if self.proxyNode is not None else None
        if parent is not None:
            volume.SetAndObserveTransformNodeID(parent.GetID())
        display = volume.GetDisplayNode()
        self.observers.append((display, display.AddObserver(vtk.vtkCommand.ModifiedEvent,
                                                            self.onProxyDisplayModified)))
        self.editVolume = volume

    # --- layout and views ------------------------------------------------------------------------
    def _setLayout(self):
        layoutManager = slicer.app.layoutManager()
        layoutNode = layoutManager.layoutLogic().GetLayoutNode()
        xml = dynamicviews.layoutXml(len(self.viewFrames))
        current = layoutManager.layout
        if current != LAYOUT_ID:
            self.previousLayout = current
        if layoutNode.IsLayoutDescription(LAYOUT_ID):
            if current == LAYOUT_ID:   # leave and come back so the new grid is built
                layoutManager.setLayout(self.previousLayout or slicer.vtkMRMLLayoutNode.SlicerLayoutFourUpView)
            layoutNode.SetLayoutDescription(LAYOUT_ID, xml)
        else:
            layoutNode.AddLayoutDescription(LAYOUT_ID, xml)
        layoutManager.setLayout(LAYOUT_ID)
        slicer.app.processEvents()

    def _sliceWidget(self, name):
        return slicer.app.layoutManager().sliceWidget(name)

    def mainViewNodeID(self):
        return self._sliceWidget(dynamicviews.MAIN_VIEW).mrmlSliceNode().GetID()

    def worldToVolume(self, point):
        """A world RAS point in the volume's own RAS, when the current-frame volume is under a transform."""
        parent = self.proxyNode.GetParentTransformNode() if self.proxyNode is not None else None
        if parent is None:
            return point
        transform = vtk.vtkGeneralTransform()
        slicer.vtkMRMLTransformNode.GetTransformBetweenNodes(None, parent, transform)
        return transform.TransformPoint(point)

    def _setupViews(self, fit=True):
        main = self._sliceWidget(dynamicviews.MAIN_VIEW)
        mainVolume = self.mainVolume()
        main.mrmlSliceCompositeNode().SetBackgroundVolumeID(mainVolume.GetID() if mainVolume else None)
        main.mrmlSliceCompositeNode().SetForegroundVolumeID(None)
        main.mrmlSliceCompositeNode().SetLabelVolumeID(None)
        if fit:
            main.sliceLogic().FitSliceToAll()
        self._addOverlay(dynamicviews.MAIN_VIEW, main, 15)
        for v, volume in enumerate(self.frameVolumes):
            name = dynamicviews.frameViewName(v)
            widget = self._sliceWidget(name)
            if widget is None:
                continue
            widget.sliceController().setVisible(False)    # small views: image only
            composite = widget.mrmlSliceCompositeNode()
            composite.SetBackgroundVolumeID(volume.GetID())
            composite.SetForegroundVolumeID(None)
            composite.SetLabelVolumeID(None)
            widget.sliceLogic().FitSliceToAll()
            self._addOverlay(name, widget, 10)
        for e in range(dynamicviews.emptyViewCount(len(self.frameVolumes))):
            widget = self._sliceWidget(dynamicviews.emptyViewName(e))
            if widget is None:
                continue
            widget.sliceController().setVisible(False)    # placeholder keeping the grid cells equal, kept empty
            self.viewObservers.append(guardViewVolumes(widget, lambda: (None, None, None)))
        self.setLinked(self.syncSlices)

    def _rebuildViews(self):
        """Show the frames of self.intervals: new images in the small views, a new grid when the number of views
        changes, the edited frame or the proxy in the large view."""
        viewFrames = DynamicPETEditorLogic.shownFrames(self.frameCount(), self.maxViews)
        if len(viewFrames) == len(self.viewFrames) and self.frameVolumes:
            self.viewFrames = viewFrames
            for volume, index in zip(self.frameVolumes, viewFrames):
                image, matrix = self._frameData(index)
                volume.SetAndObserveImageData(image)
                volume.SetIJKToRASMatrix(matrix)
        else:
            mainSlice = self._sliceWidget(dynamicviews.MAIN_VIEW).mrmlSliceNode()
            saved = vtk.vtkMatrix4x4()
            saved.DeepCopy(mainSlice.GetSliceToRAS())
            fov = mainSlice.GetFieldOfView()
            for node, tag in self.viewObservers:
                node.RemoveObserver(tag)
            self.viewObservers = []
            self._removeOverlays()
            self.setLinked(False)
            self._removeFrameVolumes()
            self.viewFrames = viewFrames
            self._makeFrameVolumes()
            self._setLayout()
            self._setupViews(fit=False)
            mainSlice = self._sliceWidget(dynamicviews.MAIN_VIEW).mrmlSliceNode()
            mainSlice.GetSliceToRAS().DeepCopy(saved)
            mainSlice.SetFieldOfView(*fov)
            mainSlice.UpdateMatrices()
        self.setFusion(self.fusionVolume)
        self.setColorNodeID(self.colorNodeID)
        self.applyWindow()
        self.refreshViews()

    def refreshViews(self):
        """Fit the field of view of every editor view to its size on screen and render it: a rebuilt grid gives the
        views new sizes, and a field of view kept from before would stretch the image. Done now and again once the
        layout has settled."""
        self._fitViewAspects()
        for delay in (0, 150):
            qt.QTimer.singleShot(delay, self._fitViewAspects)

    def _fitViewAspects(self):
        if slicer.app.layoutManager().layout != LAYOUT_ID:
            return
        for widget in self._editorWidgets():
            view = widget.sliceView()
            width, height = int(view.width), int(view.height)
            node = widget.mrmlSliceNode()
            if width > 1 and height > 1:
                fov = node.GetFieldOfView()
                if fov[0] > 0 and abs(fov[1] / fov[0] - height / width) > 1e-3:
                    node.SetFieldOfView(fov[0], fov[0] * height / width, fov[2])
            view.scheduleRender()

    def _addOverlay(self, name, widget, fontSize):
        sliceView = widget.sliceView()
        renderer = sliceView.renderWindow().GetRenderers().GetFirstRenderer()
        actor = vtk.vtkTextActor()
        prop = actor.GetTextProperty()
        prop.SetFontSize(fontSize)
        prop.SetColor(1.0, 1.0, 1.0)
        prop.SetShadow(True)
        prop.SetJustificationToLeft()
        prop.SetVerticalJustificationToTop()
        actor.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
        actor.GetPositionCoordinate().SetValue(0.03, 0.96)
        actor.SetVisibility(self.showOverlays)
        renderer.AddActor2D(actor)
        self.overlays[name] = (sliceView, renderer, actor)

    def setOverlaysVisible(self, visible):
        self.showOverlays = visible
        for sliceView, _, actor in self.overlays.values():
            actor.SetVisibility(visible)
            sliceView.scheduleRender()

    def _updateOverlays(self, current):
        editing = self.isEditing()
        for v, frameIndex in enumerate(self.viewFrames):
            overlay = self.overlays.get(dynamicviews.frameViewName(v))
            if overlay is None or frameIndex >= len(self.intervals):
                continue
            sliceView, _, actor = overlay
            edited = editing and self.intervals[frameIndex].get("edited")
            actor.SetInput(dynamicviews.frameLabel(self.intervals[frameIndex]) + ("  (edited)" if edited else ""))
            highlight = frameIndex == current
            color = (1.0, 0.85, 0.2) if highlight else ((1.0, 0.45, 0.4) if edited else (1.0, 1.0, 1.0))
            actor.GetTextProperty().SetColor(*color)
            actor.GetTextProperty().SetBold(highlight)
            sliceView.scheduleRender()
        overlay = self.overlays.get(dynamicviews.MAIN_VIEW)
        if overlay is not None and 0 <= current < len(self.intervals):
            sliceView, _, actor = overlay
            text = f"{dynamicviews.frameLabel(self.intervals[current])}   ({current + 1} / {len(self.intervals)})"
            if editing:
                text += "   edited series" + (f": {self.intervals[current]['sources']}"
                                              if self.intervals[current].get("edited") else "")
            actor.SetInput(text)
            sliceView.scheduleRender()

    # --- sync ------------------------------------------------------------------------------------
    def _observe(self):
        self.observers.append((self.browserNode, self.browserNode.AddObserver(vtk.vtkCommand.ModifiedEvent,
                                                                              self.onBrowserModified)))
        proxyDisplay = self._proxyDisplay()
        if proxyDisplay is not None:
            self.observers.append((proxyDisplay, proxyDisplay.AddObserver(vtk.vtkCommand.ModifiedEvent,
                                                                          self.onProxyDisplayModified)))
        for event in (slicer.mrmlScene.NodeAddedEvent, slicer.mrmlScene.NodeRemovedEvent):
            self.observers.append((slicer.mrmlScene, slicer.mrmlScene.AddObserver(event, self._scheduleCandidates)))

    def _editorWidgets(self):
        names = [dynamicviews.MAIN_VIEW] + [dynamicviews.frameViewName(v) for v in range(len(self.frameVolumes))]
        return [w for w in (self._sliceWidget(name) for name in names) if w is not None]

    def setLinked(self, linked):
        """Slicer's view link on all editor views: orientation, slice, zoom and pan follow whichever view is
        changed. The editor views are in their own view group, so the usual Red / Yellow / Green views are not
        affected. (A linked view also passes on a change of its background volume, so the frame volumes are
        set before linking.)"""
        self.syncSlices = linked
        for widget in self._editorWidgets():
            composite = widget.mrmlSliceCompositeNode()
            composite.SetLinkedControl(linked)
            composite.SetHotLinkedControl(linked)

    def currentIndex(self):
        if self.isEditing():
            return self.editIndex
        return self.browserNode.GetSelectedItemNumber() if self.browserNode else -1

    def setCurrentIndex(self, index):
        if not (0 <= index < len(self.intervals)):
            return
        if self.isEditing():
            self.editIndex = index
            self._ensureEditVolume()
            image, matrix = self._frameData(index)
            self.editVolume.SetAndObserveImageData(image)
            self.editVolume.SetIJKToRASMatrix(matrix)
            self.onBrowserModified()
        elif self.browserNode is not None:
            self.browserNode.SetSelectedItemNumber(index)

    def onBrowserModified(self, caller=None, event=None):
        current = self.currentIndex()
        if self.windowMode == FRAME_WINDOW:
            self.applyWindow()
        self._updateOverlays(current)
        if self.timeline is not None:
            self.timeline.setCurrentIndex(current)
        if self.originalTimeline is not None and self.isEditing() and 0 <= current < self.frameCount():
            originals = [int(k[1:]) - 1 for k in self.edit.frames[current] if k.startswith("o")]
            self.originalTimeline.setHighlighted(originals)
        self._updateToolbar(current)

    # --- timeline dock ---------------------------------------------------------------------------
    def _makeDock(self):
        mainWindow = slicer.util.mainWindow()
        self.dock = qt.QDockWidget("Dynamic PET Review/Edit timeline")
        self.dock.setObjectName("DynamicPETEditorTimeline")
        self.dock.setFeatures(qt.QDockWidget.DockWidgetMovable | qt.QDockWidget.DockWidgetFloatable)
        container = qt.QWidget()
        layout = qt.QVBoxLayout(container)
        layout.setContentsMargins(6, 4, 6, 4)

        toolbar = qt.QHBoxLayout()
        self.previousButton = qt.QToolButton()
        self.previousButton.text = "<"
        self.previousButton.toolTip = "Previous frame"
        self.playButton = qt.QToolButton()
        self.playButton.text = "Play"
        self.playButton.checkable = True
        self.playButton.toolTip = ("Play / pause in acquisition time: every frame stays as long as its duration, "
                                   "scaled by the speed; plays the time window and loops")
        self.nextButton = qt.QToolButton()
        self.nextButton.text = ">"
        self.nextButton.toolTip = "Next frame"
        self.slowerButton = qt.QToolButton()
        self.slowerButton.text = "Slower"
        self.slowerButton.toolTip = "Half the playback speed"
        self.speedLabel = qt.QLabel()
        self.speedLabel.toolTip = "Playback speed: acquisition time played per real second"
        self.fasterButton = qt.QToolButton()
        self.fasterButton.text = "Faster"
        self.fasterButton.toolTip = "Double the playback speed"
        for widget in (self.previousButton, self.playButton, self.nextButton, self.slowerButton, self.speedLabel,
                       self.fasterButton):
            toolbar.addWidget(widget)
        self.frameLabel = qt.QLabel()
        self.frameLabel.setMinimumWidth(320)
        toolbar.addWidget(self.frameLabel)
        toolbar.addStretch(1)
        self.selectionLabel = qt.QLabel()
        toolbar.addWidget(self.selectionLabel)
        layout.addLayout(toolbar)
        layout.addLayout(self._makeEditRow())

        self.timeline = FrameTimeline()
        self.timeline.setFrames(self.intervals)
        self.timeline.onCurrentFrameChanged = self._pickFrame
        self.timeline.onSelectionChanged = self._onSelectionChanged
        self.timeline.onViewRangeChanged = self._onTimelineZoomed
        layout.addWidget(self.timeline, 2)
        self.originalTimeline = FrameTimeline()
        self.originalTimeline.setFrames(self.originalIntervals)
        self.originalTimeline.setCurrentIndex(-1)
        self.originalTimeline.setLocked(True)
        self.originalTimeline.setCompact(True)
        self.originalTimeline.setTrackTitle("Original")
        self.originalTimeline.onViewRangeChanged = self._onTimelineZoomed
        self.originalTimeline.toolTip = "The original series (locked while there are edits)"
        self.originalTimeline.setVisible(False)
        layout.addWidget(self.originalTimeline, 1)

        # time window: two-handle slider with spin boxes, like a volume's display range
        zoomRow = qt.QHBoxLayout()
        zoomRow.addWidget(qt.QLabel("Time window:"))
        start, end = self.timeline.fullRange()
        self.rangeWidget = ctk.ctkRangeWidget()
        self.rangeWidget.decimals = 1
        self.rangeWidget.singleStep = 1.0
        self.rangeWidget.suffix = " s"
        self.rangeWidget.setRange(start, end)
        self.rangeWidget.setValues(start, end)
        self.rangeWidget.toolTip = ("Part of the series shown in the timeline, seconds from the study start (the "
                                    "mouse wheel over the timeline zooms too)")
        zoomRow.addWidget(self.rangeWidget, 1)
        self.windowLabel = qt.QLabel()
        self.windowLabel.setMinimumWidth(150)
        zoomRow.addWidget(self.windowLabel)
        self.showAllButton = qt.QPushButton("Show all")
        self.showAllButton.toolTip = "Show the whole series in the timeline"
        zoomRow.addWidget(self.showAllButton)
        layout.addLayout(zoomRow)
        self.dock.setWidget(container)

        self.previousButton.connect("clicked()", lambda: self._pickFrame(max(0, self.currentIndex() - 1)))
        self.nextButton.connect("clicked()", lambda: self._pickFrame(min(self.frameCount() - 1,
                                                                              self.currentIndex() + 1)))
        self.playButton.connect("toggled(bool)", self._onPlay)
        self.slowerButton.connect("clicked()", lambda: self.setPlaySpeed(self.playSpeed / 2.0))
        self.fasterButton.connect("clicked()", lambda: self.setPlaySpeed(self.playSpeed * 2.0))
        self.playTimer = qt.QTimer()
        self.playTimer.setInterval(40)
        self.playTimer.connect("timeout()", self._onPlayTick)
        self.setPlaySpeed(dynamicviews.defaultPlaySpeed(self.intervals))   # whole series in 60 s
        self.rangeWidget.connect("valuesChanged(double,double)", self._onTimeWindowChanged)
        self.showAllButton.connect("clicked()", lambda: self.rangeWidget.setValues(*self.timeline.fullRange()))
        self._onTimeWindowChanged(start, end)

        mainWindow.addDockWidget(qt.Qt.BottomDockWidgetArea, self.dock)
        try:
            mainWindow.resizeDocks([self.dock], [max(160, int(mainWindow.height / 3))], qt.Qt.Vertical)
        except Exception:
            self.dock.setMinimumHeight(max(160, int(mainWindow.height / 3)))
        self._onSelectionChanged([])
        self.refreshCandidates()

    def _makeEditRow(self):
        row = qt.QHBoxLayout()
        row.addWidget(qt.QLabel("Add frame:"))
        self.addCombo = qt.QComboBox()
        self.addCombo.setMinimumContentsLength(22)
        self.addCombo.setSizeAdjustPolicy(qt.QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.addCombo.toolTip = ("PET series to add: static PET volumes and dynamic PET sequences (not gated). Their "
                                 "frames are placed by their acquisition clock time.")
        row.addWidget(self.addCombo)
        self.addButton = qt.QPushButton("Add...")
        self.addButton.toolTip = "Compare the series (injection, acquisition, frames, decay correction) and add it"
        row.addWidget(self.addButton)
        row.addSpacing(12)
        self.mergeButton = qt.QPushButton("Merge")
        self.mergeButton.toolTip = "Merge the selected neighbouring frames into one (duration weighted mean)"
        self.splitButton = qt.QPushButton("Split")
        self.splitButton.toolTip = "Split the selected merged frames back into the frames they were made of"
        self.removeButton = qt.QPushButton("Remove")
        self.removeButton.toolTip = "Remove the selected frames from the edited series"
        self.discardButton = qt.QPushButton("Discard changes")
        self.discardButton.toolTip = "Drop every edit: back to the original series"
        self.undoButton = qt.QToolButton()
        self.undoButton.text = "Undo"
        self.redoButton = qt.QToolButton()
        self.redoButton.text = "Redo"
        for button in (self.mergeButton, self.splitButton, self.removeButton, self.discardButton, self.undoButton,
                       self.redoButton):
            row.addWidget(button)
        row.addStretch(1)
        self.saveAsButton = qt.QPushButton("Save as...")
        self.saveAsButton.toolTip = ("Save the edited series as a new dynamic sequence (with its frame timing table "
                                     "and summed volume); the original stays")
        self.overwriteButton = qt.QPushButton("Overwrite original")
        self.overwriteButton.toolTip = "Replace the frames of the original sequence (and its summed volume) by the edit"
        self.exportButton = qt.QPushButton("Export to DICOM...")
        self.exportButton.toolTip = ("Write the series as a new DICOM series in the study of the original one and add "
                                     "it to the DICOM browser (save the edits first)")
        row.addWidget(self.saveAsButton)
        row.addWidget(self.overwriteButton)
        row.addWidget(self.exportButton)
        self.addButton.connect("clicked()", self.onAddClicked)
        self.mergeButton.connect("clicked()", self.onMergeClicked)
        self.splitButton.connect("clicked()", self.onSplitClicked)
        self.removeButton.connect("clicked()", self.onRemoveClicked)
        self.discardButton.connect("clicked()", self.onDiscardClicked)
        self.undoButton.connect("clicked()", self.undo)
        self.redoButton.connect("clicked()", self.redo)
        self.shortcuts = []
        for keys, action in (("Ctrl+Z", self.undo), ("Ctrl+Y", self.redo), ("Ctrl+Shift+Z", self.redo)):
            shortcut = qt.QShortcut(qt.QKeySequence(keys), self.dock)
            shortcut.setContext(qt.Qt.WidgetWithChildrenShortcut)
            shortcut.connect("activated()", action)
            self.shortcuts.append(shortcut)
        self.saveAsButton.connect("clicked()", self.onSaveAsClicked)
        self.exportButton.connect("clicked()", self.onExportClicked)
        self.overwriteButton.connect("clicked()", self.onOverwriteClicked)
        return row

    def _onTimeWindowChanged(self, start, end):
        self.timeline.setViewRange(start, end)
        if self.originalTimeline is not None:
            self.originalTimeline.setViewRange(start, end)
        self.windowLabel.text = f"{dynamicviews.formatTime(start)} - {dynamicviews.formatTime(end)}"

    def _onTimelineZoomed(self, start, end):
        self.rangeWidget.setValues(start, end)

    def _updateExtent(self):
        """Both timelines and the time window slider span the original and the edited frames."""
        frames = list(self.intervals) + list(self.originalIntervals)
        start, end = min(f["start"] for f in frames), max(f["end"] for f in frames)
        for timeline in (self.timeline, self.originalTimeline):
            timeline.setExtent(start, end)
        low, high = self.rangeWidget.minimumValue, self.rangeWidget.maximumValue
        wasFull = abs(low - self.rangeWidget.minimum) < 1e-6 and abs(high - self.rangeWidget.maximum) < 1e-6
        self.rangeWidget.setRange(start, end)
        if wasFull:
            self.rangeWidget.setValues(start, end)
        else:
            self._onTimeWindowChanged(self.rangeWidget.minimumValue, self.rangeWidget.maximumValue)

    # --- playback in acquisition time -------------------------------------------------------------
    def setPlaySpeed(self, speed):
        self.playSpeed = min(max(speed, 0.01), 1e5)
        self.speedLabel.text = dynamicviews.speedText(self.playSpeed)

    def _onPlay(self, playing):
        self.playButton.text = "Pause" if playing else "Play"
        if self.playTimer is None:
            return
        if playing:
            start, end = self.timeline.viewStart, self.timeline.viewEnd
            current = self.currentIndex()
            t = self.intervals[current]["start"] if 0 <= current < len(self.intervals) else start
            self.playTime = t if start <= t < end else start
            self._lastTick = time.monotonic()
            self.playTimer.start()
        else:
            self.playTimer.stop()
            self.playTime = None
            self.timeline.setPlayTime(None)

    def _pickFrame(self, index):
        """The user picked a frame: show it, and continue playing from its start."""
        self.setCurrentIndex(index)
        if self.playTime is not None and 0 <= index < len(self.intervals):
            self.playTime = self.intervals[index]["start"]
            self.timeline.setPlayTime(self.playTime)

    def _onPlayTick(self):
        """Advance the playback time by the real time elapsed x speed; show the frame acquired at that time.
        Plays the timeline's time window and loops."""
        now = time.monotonic()
        elapsed, self._lastTick = now - self._lastTick, now
        start, end = self.timeline.viewStart, self.timeline.viewEnd
        self.playTime += elapsed * self.playSpeed
        if self.playTime >= end or self.playTime < start:
            self.playTime = start
        index = dynamicviews.frameAt(self.intervals, self.playTime)
        if index != self.currentIndex():
            self.setCurrentIndex(index)
        self.timeline.setPlayTime(self.playTime)

    def _onSelectionChanged(self, indexes):
        self._updateEditButtons()
        if not indexes:
            self.selectionLabel.text = "Ctrl / Shift + click to select frames"
            return
        first, last = self.intervals[indexes[0]], self.intervals[indexes[-1]]
        total = sum(self.intervals[i]["duration"] for i in indexes)
        self.selectionLabel.text = (f"{len(indexes)} selected: {dynamicviews.formatTime(first['start'])} - "
                                    f"{dynamicviews.formatTime(last['end'])} ({dynamicviews.formatTime(total)})")

    def _updateToolbar(self, current):
        if self.dock is None or not (0 <= current < len(self.intervals)):
            return
        frame = self.intervals[current]
        text = (f"Frame {frame['number']} ({current + 1} of {len(self.intervals)}):  "
                f"{dynamicviews.frameTimes(self.intervals, current)},  "
                f"duration {dynamicviews.formatTime(frame['duration'])}")
        if frame.get("edited"):
            text += f"  [{frame['sources']}]"
        self.frameLabel.text = text

    # --- editing --------------------------------------------------------------------------------
    def selection(self):
        return self.timeline.selectedIndexes() if self.timeline is not None else []

    def _ensureEdit(self):
        if self.edit is None:
            self.edit = fe.FrameEdit([{"key": f"o{i + 1}", "start": iv["start"], "duration": iv["duration"],
                                       "label": f"F{iv['number']}"} for i, iv in enumerate(self.originalIntervals)])
        return self.edit

    def _applyEdits(self, select=None):
        """Show the edit: intervals, views, timelines, curves."""
        editing = self.isEditing()
        current = self.currentIndex()
        if editing:
            self._ensureEditVolume()
            live = {(tuple(f), tuple(round(self.edit.scale(k), 12) for k in f)) for f in self.edit.frames}
            self._imageCache = {k: v for k, v in self._imageCache.items() if k in live}
            self.intervals = self.edit.intervals(self.originalIntervals[0] if self.originalIntervals else None)
        else:
            self._imageCache = {}
            self.intervals = list(self.originalIntervals)
        self._computeFrameStats()
        self.timeline.setFrames(self.intervals, keepView=True)
        self.timeline.setTrackTitle("Edited" if editing else "Frames")
        self.originalTimeline.setVisible(editing)
        self.originalTimeline.setHighlighted([])
        self._updateExtent()
        self._rebuildViews()
        index = select[0] if select else min(max(current, 0), self.frameCount() - 1)
        self.editIndex = index
        if editing:
            self.setCurrentIndex(index)
        else:
            self.browserNode.SetSelectedItemNumber(index)
            self.onBrowserModified()
        self.timeline.setSelection(select or [])
        self.timeline.setCurrentIndex(index)
        self._onSelectionChanged(self.timeline.selectedIndexes())
        if self.rois is not None:
            self.rois.refreshAll()
        if self.onEditsChanged:
            self.onEditsChanged()

    def _updateEditButtons(self):
        if self.dock is None:
            return
        selected = self.selection()
        editing = self.isEditing()
        self.mergeButton.enabled = len(selected) > 1 and fe.FrameEdit.consecutive(selected)
        self.splitButton.enabled = editing and self.edit.canSplit(selected)
        self.removeButton.enabled = bool(selected) and len(selected) < self.frameCount()
        self.discardButton.enabled = editing
        canUndo, canRedo = self.edit is not None and self.edit.canUndo(), self.edit is not None and self.edit.canRedo()
        self.undoButton.enabled, self.redoButton.enabled = canUndo, canRedo
        self.undoButton.toolTip = f"Undo {self.edit.undoText()} (Ctrl+Z)" if canUndo else "Nothing to undo"
        self.redoButton.toolTip = f"Redo {self.edit.redoText()} (Ctrl+Y)" if canRedo else "Nothing to redo"
        self.saveAsButton.enabled = editing
        self.overwriteButton.enabled = editing and not self.temporarySequence
        self.exportButton.enabled = not editing and not self.temporarySequence
        self.exportButton.toolTip = (
            "Save the edits first (Save as... or Overwrite original), then export the saved series" if editing else
            "Save the series first (Save as...)" if self.temporarySequence else
            "Write the series as a new DICOM series in the study of the original one and add it to the DICOM browser")
        self.overwriteButton.toolTip = (
            "The series was opened from a static volume: use Save as..." if self.temporarySequence else
            "Replace the frames of the original sequence (and its summed volume) by the edit")

    def mergeFrames(self, indexes):
        edit = self._ensureEdit()
        grids = {self._atomGrid(k) for i in indexes for k in edit.frames[i]}
        if len(grids) > 1:
            raise fe.EditError("The selected frames are on different image grids (an added series that was not "
                               "resampled); they cannot be merged.")
        merged = edit.merge(indexes)
        self._applyEdits([merged])
        return merged

    def splitFrames(self, indexes):
        if self.edit is None:
            raise fe.EditError("Select a merged frame to split.")
        out = self.edit.split(indexes)
        self._applyEdits(out)
        return out

    def removeFrames(self, indexes):
        self._ensureEdit().remove(indexes)
        self._applyEdits([])

    def discardEdits(self):
        """Back to the original series (Undo brings the edits back)."""
        if self.edit is not None:
            self.edit.discard()
        self._applyEdits([])

    def undo(self):
        label = self.edit.undo() if self.edit is not None else None
        if label is not None:
            self._applyEdits([])
        return label

    def redo(self):
        label = self.edit.redo() if self.edit is not None else None
        if label is not None:
            self._applyEdits([])
        return label

    def _grid(self):
        first = self.sequenceNode.GetNthDataNode(0)
        return first.GetImageData().GetDimensions(), slicer.util.arrayFromVTKMatrix(_vtkMatrix(first))

    def baseDescription(self):
        """frameedit.seriesInfo of the edited series (its frames as they are now)."""
        tags = seriesTags(self.sequenceNode, "dynamic")
        clock = self.originalIntervals[0].get("clockAtZero") if self.originalIntervals else None
        frames = [{"start": clock + datetime.timedelta(seconds=f["start"]) if clock is not None else None,
                   "duration": f["duration"]} for f in self.intervals]
        info = describeSeries(self.sequenceNode, "dynamic", tags, frames)
        if self.edit is not None and self.edit.reference is not None:
            info["decayReference"] = self.edit.reference
        return info

    def prepareAdd(self, node, kind):
        """Everything the Add frame dialog shows and addSeries needs: {"comparison" (frameedit.compareSeries),
        "added" (its description), "atoms", "frameNodes", "resampleNeeded"}."""
        added = describeSeries(node, kind)
        comparison = fe.compareSeries(self.baseDescription(), added, LONG_GAP_S)
        frameNodes = [node.GetNthDataNode(i) for i in range(node.GetNumberOfDataNodes())] if kind == "dynamic" \
            else [node]
        clock = self.originalIntervals[0].get("clockAtZero") if self.originalIntervals else None
        atoms = []
        if clock is not None and not comparison["errors"]:
            for n, frame in enumerate(added["frames"]):
                label = f"{node.GetName()} F{n + 1}" if kind == "dynamic" else node.GetName()
                atoms.append({"start": (frame["start"] - clock).total_seconds(), "duration": frame["duration"],
                              "label": label})
            overlaps = self._ensureEdit().overlaps(atoms)
            if overlaps:
                comparison["errors"].append("The added frames overlap the edited series in time: " + ", ".join(
                    f"{label} with frame {n}" for label, n in overlaps[:6]) + ".")
        dims, ijkToRas = self._grid()
        resampleNeeded = any(not sameGeometry(f, dims, ijkToRas) or f.GetParentTransformNode() is not None
                             for f in frameNodes)
        return {"comparison": comparison, "added": added, "atoms": atoms, "frameNodes": frameNodes,
                "resampleNeeded": resampleNeeded, "node": node}

    def addSeries(self, node, kind, resample=True, prepared=None):
        """Add the frames of a PET series (prepareAdd); resample: onto the grid of the edited series when the
        geometry differs. Returns the indexes of the new frames."""
        prepared = prepared or self.prepareAdd(node, kind)
        comparison = prepared["comparison"]
        if comparison["errors"]:
            raise fe.EditError("\n".join(comparison["errors"]))
        edit = self._ensureEdit()
        dims, ijkToRas = self._grid()
        arrays = []
        for frameNode in prepared["frameNodes"]:
            if prepared["resampleNeeded"] and resample:
                arrays.append((resampleToGrid(frameNode, dims, ijkToRas), ijkToRas))
            else:
                array = _array(frameNode)
                if array is None:
                    raise fe.EditError(f"'{frameNode.GetName()}' has no image.")
                arrays.append((np.array(array, dtype=np.float32),
                               slicer.util.arrayFromVTKMatrix(_vtkMatrix(frameNode))))
        for key, data in zip(edit.nextAddedKeys(len(arrays)), arrays):
            self.editArrays[key] = data
        indexes = edit.add(node.GetName(), prepared["atoms"], comparison["scaleBase"], comparison["scaleAdded"],
                           comparison["reference"])
        self._applyEdits(indexes)
        return indexes

    # --- candidates of Add frame ------------------------------------------------------------------
    def _scheduleCandidates(self, caller=None, event=None):
        if self._candidateTimer is None:
            self._candidateTimer = qt.QTimer()
            self._candidateTimer.setSingleShot(True)
            self._candidateTimer.setInterval(300)
            self._candidateTimer.connect("timeout()", self.refreshCandidates)
        self._candidateTimer.start()

    def refreshCandidates(self):
        if self.dock is None:
            return
        current = self.addCombo.itemData(self.addCombo.currentIndex) if self.addCombo.count else None
        self.addCombo.clear()
        exclude = {self.sequenceNode.GetID()}
        source = self.sequenceNode.GetAttribute(PREFIX + "OpenedFromStatic")
        if source:
            exclude.add(source)
        for node, kind, label in petCandidates(self.sequenceNode):
            if node.GetID() not in exclude:
                self.addCombo.addItem(label, f"{kind}|{node.GetID()}")
        if current:
            index = self.addCombo.findData(current)
            if index >= 0:
                self.addCombo.setCurrentIndex(index)
        self.addButton.enabled = self.addCombo.count > 0
        if not self.addCombo.count:
            self.addCombo.addItem("(no other PET series in the scene)")
            self.addCombo.setItemData(0, None)

    def selectedCandidate(self):
        data = self.addCombo.itemData(self.addCombo.currentIndex) if self.addCombo.count else None
        if not data:
            return None, None
        kind, nodeID = str(data).split("|", 1)
        return slicer.mrmlScene.GetNodeByID(nodeID), kind

    # --- buttons (with the dialogs) ----------------------------------------------------------------
    def _guard(self, action, title):
        try:
            return action()
        except fe.EditError as error:
            slicer.util.warningDisplay(str(error), windowTitle=title)
        except Exception as error:
            logging.exception(f"Review/Edit: {title} failed")
            slicer.util.errorDisplay(f"{title} failed:\n{error}", windowTitle=title)
        return None

    def onAddClicked(self):
        node, kind = self.selectedCandidate()
        if node is None:
            return
        prepared = self._guard(lambda: self.prepareAdd(node, kind), "Add frame")
        if prepared is None or not addFrameDialog(self, prepared):
            return
        resample = True
        if prepared["resampleNeeded"]:
            answer = qt.QMessageBox.question(
                slicer.util.mainWindow(), "Add frame: image grid",
                f"'{node.GetName()}' is on another image grid (size, voxel size, position or orientation) than the "
                "edited series, or under a transform.\n\nResample it onto the grid of the edited series (linear "
                "interpolation, no registration)?\n\nYes: resampled; it can then be merged with other frames.\n"
                "No: kept on its own grid (e.g. to register it later); it cannot be merged with frames on another "
                "grid.", qt.QMessageBox.Yes | qt.QMessageBox.No | qt.QMessageBox.Cancel, qt.QMessageBox.Yes)
            if answer == qt.QMessageBox.Cancel:
                return
            resample = answer == qt.QMessageBox.Yes
        indexes = self._guard(lambda: self.addSeries(node, kind, resample, prepared), "Add frame")
        if indexes:
            slicer.util.infoDisplay(
                f"{len(indexes)} frame(s) of '{node.GetName()}' added (red on the edited timeline).\n\n"
                "Check the registration: the added frames come from another acquisition, so the patient may have "
                "moved (compare them with the other frames, with fusion or the ROI curves). If they are not "
                "aligned, save the edited series and correct it in Dynamic PET Motion Correction.",
                windowTitle="Add frame: check the registration")

    def onMergeClicked(self):
        indexes = self.selection()
        edit = self._ensureEdit()
        if len(indexes) > 1 and fe.FrameEdit.consecutive(indexes):
            gap = edit.largestGap(indexes)
            if gap > fe.GAP_TOLERANCE_S and not slicer.util.confirmYesNoDisplay(
                    f"There are {fe.formatDuration(gap)} between the selected frames. Merge them anyway? The merged "
                    "frame spans the gap; its duration is the sum of the acquisition times.", windowTitle="Merge"):
                return
        self._guard(lambda: self.mergeFrames(indexes), "Merge")

    def onSplitClicked(self):
        self._guard(lambda: self.splitFrames(self.selection()), "Split")

    def onRemoveClicked(self):
        self._guard(lambda: self.removeFrames(self.selection()), "Remove")

    def onDiscardClicked(self):
        if self.isEditing() and slicer.util.confirmYesNoDisplay(
                "Discard every change and go back to the original series? (Undo brings them back.)",
                windowTitle="Discard changes"):
            self.discardEdits()

    def onSaveAsClicked(self):
        if not self.isEditing():
            return
        base = self.sequenceNode.GetName()
        name = qt.QInputDialog.getText(slicer.util.mainWindow(), "Save as", "Name of the new dynamic series:",
                                       qt.QLineEdit.Normal, slicer.mrmlScene.GenerateUniqueName(f"{base} edited"))
        if not name or not str(name).strip():
            return
        sequence = self._guard(lambda: saveEditedSeries(self, str(name).strip()), "Save as")
        if sequence is not None and self.onSaved:
            self.onSaved(sequence)

    def onExportClicked(self):
        if not self.isEditing() and not self.temporarySequence:
            exportSequenceDialog(self.sequenceNode)

    def onOverwriteClicked(self):
        if not self.isEditing() or self.temporarySequence:
            return
        if not slicer.util.confirmYesNoDisplay(
                f"Overwrite '{self.sequenceNode.GetName()}' with the edited frames? Its frames, frame timing table "
                "and summed volume are replaced; this cannot be undone (Save as... keeps the original).",
                windowTitle="Overwrite original"):
            return
        sequence = self._guard(lambda: saveEditedSeries(self, overwrite=True), "Overwrite original")
        if sequence is not None and self.onSaved:
            self.onSaved(sequence)


def addFrameDialog(session, prepared):
    """The Add frame dialog: both series side by side, the decay correction, errors and warnings. True: add."""
    comparison = prepared["comparison"]
    dialog = qt.QDialog(slicer.util.mainWindow())
    dialog.setWindowTitle("Add frame")
    dialog.setMinimumWidth(720)
    layout = qt.QVBoxLayout(dialog)
    layout.addWidget(qt.QLabel(f"Add '{prepared['node'].GetName()}' to '{session.sequenceNode.GetName()}':"))
    rows = comparison["rows"]
    table = qt.QTableWidget(len(rows), 2)
    table.setHorizontalHeaderLabels(["Edited series", "Added series"])
    table.setVerticalHeaderLabels([r[0] for r in rows])
    table.setEditTriggers(qt.QAbstractItemView.NoEditTriggers)
    table.horizontalHeader().setSectionResizeMode(qt.QHeaderView.Stretch)
    for r, (_, base, added) in enumerate(rows):
        for c, text in enumerate((base, added)):
            item = qt.QTableWidgetItem(str(text))
            if base != added and r > 0:
                item.setForeground(qt.QBrush(qt.QColor(230, 150, 40)))
            table.setItem(r, c, item)
    table.setMinimumHeight(min(420, 30 + 26 * len(rows)))
    layout.addWidget(table)

    def note(text, color=None):
        label = qt.QLabel(text)
        label.wordWrap = True
        if color:
            label.setStyleSheet(f"color: {color}; font-weight: bold;")
        layout.addWidget(label)

    if comparison["note"]:
        note("Decay correction: " + comparison["note"])
    for text in comparison["errors"]:
        note("Cannot add: " + text, "#e04040")
    for text in comparison["warnings"]:
        note("Warning: " + text, "#e09020")
    if prepared["resampleNeeded"] and not comparison["errors"]:
        note("The image grid differs: you will be asked whether to resample it.")
    if not comparison["errors"]:
        note(f"{len(prepared['atoms'])} frame(s) will be placed by their acquisition clock time.")
    buttons = qt.QDialogButtonBox()
    addButton = buttons.addButton("Add" if not comparison["warnings"] else "Add anyway",
                                  qt.QDialogButtonBox.AcceptRole)
    addButton.enabled = not comparison["errors"]
    buttons.addButton(qt.QDialogButtonBox.Cancel)
    buttons.connect("accepted()", dialog.accept)
    buttons.connect("rejected()", dialog.reject)
    layout.addWidget(buttons)
    return bool(dialog.exec_()) and not comparison["errors"]


def staticAsSequence(volume):
    """A static PET volume as a one-frame dynamic sequence (hidden, not saved; removed when Review/Edit closes),
    so that frames can be added to it. Returns the sequence."""
    tags = seriesTags(volume, "static")
    if fe.parseIso(tags.get("StartDateTime")) is None or fe.number(tags.get("DurationS")) is None:
        raise ValueError(f"The acquisition time or duration of '{volume.GetName()}' is unknown.")
    scene = slicer.mrmlScene
    sequence = scene.AddNewNodeByClass("vtkMRMLSequenceNode", scene.GenerateUniqueName(f"{volume.GetName()} dynamic"))
    sequence.SetHideFromEditors(True)
    sequence.SetSaveWithScene(False)
    sequence.SetIndexName("Time")
    sequence.SetIndexUnit("s")
    sequence.SetIndexType(sequence.NumericIndex)
    frame = slicer.vtkMRMLScalarVolumeNode()
    frame.SetName(f"{volume.GetName()} F1")
    for key, value in tags.items():
        frame.SetAttribute(PREFIX + key, str(value))
    duration = fe.number(tags.get("DurationS"))
    frame.SetAttribute(PREFIX + "Kind", "DYNAMIC")
    frame.SetAttribute(PREFIX + "Frame", "1")
    frame.SetAttribute(PREFIX + "StartFromStudyStartS", "0")
    frame.SetAttribute(PREFIX + "DurationMs", f"{duration * 1000.0:g}")
    frame.SetAttribute("DICOM.Modality", "PT")
    if volume.GetAttribute("DICOM.instanceUIDs"):
        frame.SetAttribute("DICOM.instanceUIDs", volume.GetAttribute("DICOM.instanceUIDs"))
    frame.SetIJKToRASMatrix(_vtkMatrix(volume))
    image = vtk.vtkImageData()
    image.DeepCopy(volume.GetImageData())
    frame.SetAndObserveImageData(image)
    try:
        frame.SetVoxelValueUnits(volume.GetVoxelValueUnits())
    except Exception:
        pass
    sequence.SetDataNodeAtValue(frame, f"{duration / 2.0:g}")
    sequence.SetAttribute("DICOM.Modality", "PT")
    sequence.SetAttribute(PREFIX + "OpenedFromStatic", volume.GetID())
    sequence.SetAttribute(PREFIX + "SumWeighting", "duration")
    sequence.SetAttribute(PREFIX + "StudyStartDateTime", tags.get("StartDateTime"))
    browser = DynamicPETEditorLogic.browserFor(sequence)
    browser.SetHideFromEditors(True)
    browser.SetSaveWithScene(False)
    proxy = browser.GetProxyNode(sequence)
    if proxy is not None:
        proxy.SetHideFromEditors(True)
        proxy.SetSaveWithScene(False)
        proxy.SetAttribute(EDITOR_ATTRIBUTE, "1")
        proxy.CreateDefaultDisplayNodes()
        sourceDisplay = volume.GetDisplayNode()
        if sourceDisplay is not None:
            proxy.GetDisplayNode().CopyContent(sourceDisplay)
    return sequence


def _indexValues(rows):
    """Sequence index of each frame: seconds from the scan start to the frame middle, else frame numbers (as the
    loader)."""
    values = [row["MidFromScanStartS"] for row in rows]
    if any(v is None for v in values) or len({round(v, 3) for v in values}) != len(values):
        values = [row["MidFromInjectionS"] for row in rows]
    if any(v is None for v in values) or len({round(v, 3) for v in values}) != len(values):
        return [str(row["Frame"]) for row in rows]
    return [f"{v:.3f}".rstrip("0").rstrip(".") for v in values]


def editedFrameNodes(session, baseName):
    """The frames of the edited series as volume nodes (not in the scene) with their tags, and the timing rows."""
    from DICOMGatedDynamicPETLib import petseries
    edit = session.edit
    template = session.sequenceNode.GetNthDataNode(0)
    tags = seriesTags(session.sequenceNode, "dynamic")
    corrected, reference = decayStatus(tags)
    if edit.reference is not None:
        reference = edit.reference
    clock = session.originalIntervals[0].get("clockAtZero")
    rows = edit.timingRows(clock, fe.parseIso(tags.get("InjectionTime")),
                           reference or fe.parseIso(tags.get("ScanStart")))
    nodes = []
    for i, row in enumerate(rows):
        original = edit.isOriginalFrame(i)
        source = session.sequenceNode.GetNthDataNode(original) if original is not None else template
        node = slicer.vtkMRMLScalarVolumeNode()
        node.SetName(f"{baseName} F{i + 1}")
        for key in source.GetAttributeNames() or ():
            node.SetAttribute(key, source.GetAttribute(key))
        for key in FRAME_TIMING_TAGS:
            node.RemoveAttribute(PREFIX + key)
        for key, value in row.items():
            if value is not None:
                node.SetAttribute(PREFIX + key, str(value))
        node.SetAttribute(PREFIX + "Frames", str(len(rows)))
        if original is None:
            node.SetAttribute(PREFIX + "EditSources", edit.frameSources(i))
            uids = (node.GetAttribute("DICOM.instanceUIDs") or "").split()
            if uids:     # the study the DICOM export puts the series in
                node.SetAttribute(PREFIX + "TemplateInstanceUID", uids[0])
            node.RemoveAttribute("DICOM.instanceUIDs")
        if edit.reference is not None:
            node.SetAttribute(PREFIX + "EditDecayReference", edit.reference.isoformat())
            node.SetAttribute(PREFIX + "DecayReference", "Review/Edit: frames decay corrected to "
                              + edit.reference.isoformat())
        image, matrix = session._frameData(i)
        copy = vtk.vtkImageData()
        copy.DeepCopy(image)
        node.SetAndObserveImageData(copy)
        node.SetIJKToRASMatrix(matrix)
        try:
            node.SetVoxelValueUnits(template.GetVoxelValueUnits())
        except Exception:
            pass
        nodes.append(node)
    return nodes, rows, tags, petseries


def _summedArray(sequence, rows, tags, petseries, weighting):
    arrays = [_array(sequence.GetNthDataNode(i)) for i in range(sequence.GetNumberOfDataNodes())]
    if any(a is None for a in arrays) or len({a.shape for a in arrays}) != 1:
        return None, None
    corrected, _ = decayStatus(tags)
    weights, text = petseries.summationWeights([row["DurationS"] for row in rows],
                                               [row["StartFromStudyStartS"] for row in rows],
                                               fe.number(tags.get("RadionuclideHalfLifeS")), weighting, corrected)
    return petseries.weightedMean(arrays, weights).astype(np.float32), text


def saveEditedSeries(session, name=None, overwrite=False):
    """Save the edit: a new dynamic sequence `name` with its browser, frame timing table and summed volume (Save
    as), or the frames of the original sequence replaced, with its table and summed volume (overwrite). Returns
    the sequence."""
    from DICOMGatedDynamicPET import frameTimingTable
    scene = slicer.mrmlScene
    original = session.sequenceNode
    baseName = original.GetName() if overwrite else scene.GenerateUniqueName(name)
    nodes, rows, tags, petseries = editedFrameNodes(session, baseName)
    weighting = original.GetAttribute(PREFIX + "SumWeighting") or petseries.SUM_DURATION
    oldSummed = scene.GetNodeByID(original.GetAttribute(PREFIX + "SummedVolumeID") or "")
    oldTable = scene.GetNodeByID(original.GetAttribute(PREFIX + "FrameTimingTableID") or "")
    if overwrite:
        sequence = original
        sequence.RemoveAllDataNodes()
    else:
        sequence = scene.AddNewNodeByClass("vtkMRMLSequenceNode", baseName)
        sequence.SetIndexName(original.GetIndexName() or "Time")
        sequence.SetIndexUnit(original.GetIndexUnit() or "s")
        sequence.SetIndexType(original.GetIndexType())
        for key in original.GetAttributeNames() or ():
            if key not in (PREFIX + k for k in SUMMED_PREFIX_KEYS) and key != PREFIX + "OpenedFromStatic":
                sequence.SetAttribute(key, original.GetAttribute(key))
        sequence.SetAttribute(PREFIX + "EditedFrom", original.GetName())
    for node, value in zip(nodes, _indexValues(rows)):
        sequence.SetDataNodeAtValue(node, value)
    for key, value in petseries.timingLists(rows).items():
        sequence.SetAttribute(PREFIX + key, value)
    sequence.SetAttribute(PREFIX + "Frames", str(len(rows)))
    sequence.SetAttribute(PREFIX + "SumWeighting", weighting)
    if rows and rows[0]["StartDateTime"]:
        sequence.SetAttribute(PREFIX + "StudyStartDateTime", rows[0]["StartDateTime"])
    if session.edit.reference is not None:
        sequence.SetAttribute(PREFIX + "EditDecayReference", session.edit.reference.isoformat())
    sequence.SetAttribute(PREFIX + "EditHistory", "; ".join(f"F{r['Frame']}: {r['StartSource'][13:]}" for r in rows))

    table = frameTimingTable(f"{baseName} frame timing", rows)
    table.SetAttribute(PREFIX + "SequenceNodeID", sequence.GetID())
    sequence.SetAttribute(PREFIX + "FrameTimingTableID", table.GetID())
    if overwrite and oldTable is not None:
        table.SetName(oldTable.GetName())
        scene.RemoveNode(oldTable)

    array, text = _summedArray(sequence, rows, tags, petseries, weighting)
    summed = None
    if array is not None and (oldSummed is not None or not overwrite):
        if overwrite and oldSummed is not None:
            summed = oldSummed
        else:
            summedName = baseName.replace(" dynamic ", " summed ", 1) if " dynamic " in baseName else \
                f"{baseName} summed"
            summed = scene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", scene.GenerateUniqueName(summedName))
            template = oldSummed or sequence.GetNthDataNode(0)
            for key in template.GetAttributeNames() or ():
                if not key.startswith("DICOM.instanceUIDs"):
                    summed.SetAttribute(key, template.GetAttribute(key))
            for key in FRAME_TIMING_TAGS:
                summed.RemoveAttribute(PREFIX + key)
            try:
                summed.SetVoxelValueUnits(sequence.GetNthDataNode(0).GetVoxelValueUnits())
            except Exception:
                pass
        slicer.util.updateVolumeFromArray(summed, array)
        summed.SetIJKToRASMatrix(_vtkMatrix(sequence.GetNthDataNode(0)))
        summed.SetAttribute(PREFIX + "StaticFrom", f"{len(rows)} edited frames summed, {text}")
        summed.SetAttribute(PREFIX + "SummedFrames", str(len(rows)))
        summed.SetAttribute(PREFIX + "SumWeighting", weighting)
        summed.SetAttribute(PREFIX + "DurationS", f"{sum(r['DurationS'] for r in rows):.6g}")
        for key, value in (("StartFromInjectionS", rows[0]["StartFromInjectionS"]),
                           ("EndFromInjectionS", rows[-1]["EndFromInjectionS"]),
                           ("StartDateTime", rows[0]["StartDateTime"])):
            if value is not None:
                summed.SetAttribute(PREFIX + key, str(value))
        summed.SetAttribute(PREFIX + "DynamicSequence", sequence.GetID())
        sequence.SetAttribute(PREFIX + "SummedVolumeID", summed.GetID())
        summed.CreateDefaultDisplayNodes()
        if oldSummed is not None and oldSummed is not summed and oldSummed.GetDisplayNode() is not None:
            summed.GetDisplayNode().CopyContent(oldSummed.GetDisplayNode())

    created = [n for n in (table, summed) if n is not None]
    if overwrite:
        logic = slicer.modules.sequences.logic()
        browsers = vtk.vtkCollection()
        logic.GetBrowserNodesForSequenceNode(sequence, browsers)
        for b in range(browsers.GetNumberOfItems()):
            browser = browsers.GetItemAsObject(b)
            browser.SetSelectedItemNumber(min(browser.GetSelectedItemNumber(), len(rows) - 1))
            logic.UpdateProxyNodesFromSequences(browser)
    else:
        browser = DynamicPETEditorLogic.browserFor(sequence)
        browser.SetPlaybackLooped(True)
        proxy = browser.GetProxyNode(sequence)
        if proxy is not None:
            proxy.SetName(f"{baseName} (current frame)")
            proxy.CreateDefaultDisplayNodes()
            if session._proxyDisplay() is not None:
                proxy.GetDisplayNode().CopyContent(session._proxyDisplay())
        created += [n for n in (browser, proxy, sequence) if n is not None]
        source = original
        if session.temporarySequence:
            source = scene.GetNodeByID(original.GetAttribute(PREFIX + "OpenedFromStatic") or "") or original
        _placeNextTo(source, created)
    logging.info(f"Dynamic PET Review/Edit: '{sequence.GetName()}' {'overwritten' if overwrite else 'saved'} with "
                 f"{len(rows)} frames")
    return sequence


# ---------------------------------------------------------------------------
# DICOM export (also used by Dynamic PET Motion Correction)
# ---------------------------------------------------------------------------
def templateInstanceUID(sequence):
    """An instance UID of the original DICOM series of a dynamic sequence (its study is the export's), or None."""
    nodes = [sequence.GetNthDataNode(i) for i in range(sequence.GetNumberOfDataNodes())]
    summed = slicer.mrmlScene.GetNodeByID(sequence.GetAttribute(PREFIX + "SummedVolumeID") or "")
    for node in nodes + [summed, sequence]:
        if node is None:
            continue
        uids = (node.GetAttribute("DICOM.instanceUIDs") or "").split()
        uid = uids[0] if uids else node.GetAttribute(PREFIX + "TemplateInstanceUID")
        if uid:
            return uid
    return None


def _nextSeriesNumber(studyUID, fallback=900):
    """One above the highest Series Number of the study in the DICOM database."""
    try:
        db = slicer.dicomDatabase
        numbers = []
        for series in db.seriesForStudy(studyUID):
            instances = db.instancesForSeries(series) if series else []
            if instances:
                number = fe.number(db.instanceValue(instances[0], "0020,0011"))
                if number is not None:
                    numbers.append(int(number))
        return max(numbers) + 1 if numbers else fallback
    except Exception:
        return fallback


def exportSequenceToDicom(sequence, description, progress=None):
    """Write a dynamic PET sequence as a new DICOM series in the study of its original series and add it to the
    DICOM database (the DICOM browser lists it). Returns (SeriesInstanceUID, number of files)."""
    import pydicom
    import shutil
    import tempfile
    from DICOMGatedDynamicPETLib import dicomexport
    db = getattr(slicer, "dicomDatabase", None)
    if db is None or not db.isOpen:
        raise dicomexport.ExportError("The DICOM database is not open.")
    uid = templateInstanceUID(sequence)
    path = db.fileForInstance(uid) if uid else ""
    if not path:
        raise dicomexport.ExportError(f"The original DICOM series of '{sequence.GetName()}' is not in the DICOM "
                                      "database, so the study to put it in is unknown.")
    template = pydicom.dcmread(path, stop_before_pixels=True)
    tags = seriesTags(sequence, "dynamic")
    corrected, reference = decayStatus(tags)
    correction = (tags.get("DecayCorrection") or "START").upper()
    if not corrected:
        correction = "NONE"
    elif correction == "NONE" or tags.get("EditDecayReference"):
        correction = "START"       # decay corrected by the loader or aligned by Review/Edit
    frames, rows, notes = [], [], []
    matrix = None
    for i in range(sequence.GetNumberOfDataNodes()):
        node = sequence.GetNthDataNode(i)
        array = _array(node)
        if array is None:
            raise dicomexport.ExportError(f"Frame {i + 1} has no image.")
        frameMatrix = slicer.util.arrayFromVTKMatrix(_vtkMatrix(node))
        if matrix is None:
            matrix = frameMatrix
        elif array.shape != frames[0].shape or not np.allclose(frameMatrix, matrix, atol=1e-3):
            raise dicomexport.ExportError("The frames are on different image grids; resample them first.")
        frames.append(array)
        rows.append({"StartDateTime": node.GetAttribute(PREFIX + "StartDateTime"),
                     "DurationS": node.GetAttribute(PREFIX + "DurationS")})
        notes.append(node.GetAttribute(PREFIX + "EditSources") or
                     node.GetAttribute(PREFIX + "MotionCorrection.Status"))
    derivation = "Dynamic PET (PET4D)"
    if sequence.GetAttribute(PREFIX + "EditedFrom"):
        derivation += f", edited in Review/Edit from {sequence.GetAttribute(PREFIX + 'EditedFrom')}"
    if sequence.GetAttribute(PREFIX + "MotionCorrection.Method"):
        derivation += f", motion corrected ({sequence.GetAttribute(PREFIX + 'MotionCorrection.Method')})"
    datasets = dicomexport.buildDatasets(
        frames, matrix, rows, template, tags.get("Units"), description,
        _nextSeriesNumber(str(template.StudyInstanceUID)), correction,
        reference if correction == "START" else None, fe.number(tags.get("RadionuclideHalfLifeS")),
        derivation=derivation, frameNotes=[f"{derivation}: {n}" if n else None for n in notes])
    folder = tempfile.mkdtemp(prefix="PET4Dexport", dir=slicer.app.temporaryPath)
    try:
        if progress:
            progress("Writing the DICOM files...")
        paths = dicomexport.writeDatasets(folder, datasets)
        if progress:
            progress("Adding the series to the DICOM database...")
        from DICOMLib import DICOMUtils
        DICOMUtils.importDicom(folder, db, copyFiles=True)
    finally:
        shutil.rmtree(folder, ignore_errors=True)
    seriesUID = str(datasets[0].SeriesInstanceUID)
    sequence.SetAttribute(PREFIX + "ExportedSeriesInstanceUID", seriesUID)
    logging.info(f"Dynamic PET: '{sequence.GetName()}' exported to DICOM as '{description}' ({len(paths)} images, "
                 f"series {seriesUID}, study {template.StudyInstanceUID})")
    return seriesUID, len(paths)


def exportSequenceDialog(sequence):
    """Ask for the series description, export and tell the user. Returns the SeriesInstanceUID or None."""
    from DICOMGatedDynamicPETLib import dicomexport
    description = qt.QInputDialog.getText(
        slicer.util.mainWindow(), "Export to DICOM",
        "Series description of the new DICOM series (in the study of the original series):", qt.QLineEdit.Normal,
        sequence.GetName()[:64])
    if not description or not str(description).strip():
        return None
    dialog = slicer.util.createProgressDialog(labelText="Exporting to DICOM...", maximum=0)
    try:
        def progress(text):
            dialog.labelText = text
            slicer.app.processEvents()
        seriesUID, count = exportSequenceToDicom(sequence, str(description).strip()[:64], progress)
    except dicomexport.ExportError as error:
        dialog.close()
        slicer.util.warningDisplay(str(error), windowTitle="Export to DICOM")
        return None
    except Exception as error:
        dialog.close()
        logging.exception("DICOM export failed")
        slicer.util.errorDisplay(f"DICOM export failed:\n{error}", windowTitle="Export to DICOM")
        return None
    dialog.close()
    exportDoneDialog(sequence, str(description).strip()[:64], seriesUID, count)
    return seriesUID


def exportedLocation(seriesUID):
    """Where the DICOM database keeps an exported series: (folder of its files, list of files, database folder)."""
    import os
    db = slicer.dicomDatabase
    try:
        files = [str(f) for f in db.filesForSeries(seriesUID)]
    except Exception:
        files = []
    folder = ""
    if files:
        try:
            folder = os.path.commonpath([os.path.dirname(f) for f in files])
        except ValueError:
            folder = os.path.dirname(files[0])
    return folder, files, str(getattr(db, "databaseDirectory", "") or "")


def exportDoneDialog(sequence, description, seriesUID, count):
    """Tell exactly where and how the series was saved, with buttons to open the folder or copy its path."""
    import html
    import os
    folder, files, databaseFolder = exportedLocation(seriesUID)
    study = patient = seriesNumber = ""
    try:
        import pydicom
        if files:
            ds = pydicom.dcmread(files[0], stop_before_pixels=True)
            patient = f"{ds.get('PatientName', '')} ({ds.get('PatientID', '')})"
            study = " ".join(str(v) for v in (ds.get("StudyDescription", ""), ds.get("StudyDate", "")) if v)
            study = study or str(ds.get("StudyInstanceUID", ""))
            seriesNumber = str(ds.get("SeriesNumber", ""))
    except Exception:
        logging.debug("Could not read the exported series back", exc_info=True)
    lines = [
        f"<b>'{html.escape(sequence.GetName())}'</b> was exported to DICOM.",
        "",
        f"<b>Series:</b> {html.escape(description)}" + (f" (series number {seriesNumber})" if seriesNumber else ""),
        f"<b>Images:</b> {count} PET Image Storage files (one per slice and frame)",
        f"<b>Patient:</b> {html.escape(patient)}" if patient else None,
        f"<b>Study:</b> {html.escape(study)} (the study of the original series)" if study else None,
        f"<b>Series Instance UID:</b> {seriesUID}",
        "",
        "The files were copied into the Slicer DICOM database and the series is listed in the DICOM browser.",
        f"<b>Folder:</b> {folder}" if folder else "<b>Folder:</b> (not found in the database)",
        f"<b>DICOM database:</b> {databaseFolder}" if databaseFolder else None,
        "",
        "To get a copy in another folder: DICOM browser, right click the series, Export to file system.",
    ]
    box = qt.QMessageBox(slicer.util.mainWindow())
    box.setWindowTitle("Export to DICOM")
    box.setIcon(qt.QMessageBox.Information)
    box.setTextFormat(qt.Qt.RichText)
    box.setText("<br>".join(line for line in lines if line is not None))
    box.setDetailedText("\n".join(files))
    openButton = box.addButton("Open folder", qt.QMessageBox.ActionRole) if folder and os.path.isdir(folder) else None
    copyButton = box.addButton("Copy path", qt.QMessageBox.ActionRole) if folder else None
    box.addButton(qt.QMessageBox.Ok)
    if openButton is not None:
        openButton.connect("clicked()", lambda: qt.QDesktopServices.openUrl(qt.QUrl.fromLocalFile(folder)))
    if copyButton is not None:
        copyButton.connect("clicked()", lambda: qt.QApplication.clipboard().setText(folder))
    box.exec_()
    return box


def _placeNextTo(sourceNode, nodes):
    try:
        sh = slicer.vtkMRMLSubjectHierarchyNode.GetSubjectHierarchyNode(slicer.mrmlScene)
        sourceItem = sh.GetItemByDataNode(sourceNode)
        parent = sh.GetItemParent(sourceItem) if sourceItem else 0
        if not parent:
            return
        for node in nodes:
            item = sh.GetItemByDataNode(node)
            if item:
                sh.SetItemParent(item, parent)
    except Exception:
        logging.warning("Could not place the edited series in the subject hierarchy", exc_info=True)


# ---------------------------------------------------------------------------
# Module panel
# ---------------------------------------------------------------------------
class DynamicPETEditorWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    def __init__(self, parent=None):
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)
        self.session = None
        self.logic = DynamicPETEditorLogic()

        inputBox = qt.QGroupBox("Dynamic series")
        form = qt.QFormLayout(inputBox)
        self.sequenceSelector = slicer.qMRMLNodeComboBox()
        self.sequenceSelector.nodeTypes = ["vtkMRMLSequenceNode", "vtkMRMLScalarVolumeNode"]
        self.sequenceSelector.noneEnabled = True
        self.sequenceSelector.addEnabled = False
        self.sequenceSelector.removeEnabled = False
        self.sequenceSelector.renameEnabled = False
        self.sequenceSelector.setMRMLScene(slicer.mrmlScene)
        self.sequenceSelector.toolTip = ("Dynamic PET sequence (as loaded from DICOM: '<series> dynamic Bq/ml'), or a "
                                         "static PET volume to build a dynamic series from (Add frame, Save as)")
        form.addRow("Series:", self.sequenceSelector)
        self.infoLabel = qt.QLabel()
        self.infoLabel.wordWrap = True
        form.addRow(self.infoLabel)
        self.layout.addWidget(inputBox)

        optionsBox = qt.QGroupBox("Display")
        options = qt.QFormLayout(optionsBox)
        self.maxViewsSpin = qt.QSpinBox()
        self.maxViewsSpin.setRange(4, 100)
        self.maxViewsSpin.value = 64
        self.maxViewsSpin.toolTip = "Most small frame views; longer series show frames spread evenly over the series"
        options.addRow("Frame views (max):", self.maxViewsSpin)
        self.overlayCheck = qt.QCheckBox("Frame times on the views")
        self.overlayCheck.checked = True
        options.addRow(self.overlayCheck)
        self.syncCheck = qt.QCheckBox("Link views (orientation, slice, zoom, pan)")
        self.syncCheck.checked = True
        self.syncCheck.toolTip = "Changing any view changes all of them (the same as each view's link button)"
        options.addRow(self.syncCheck)
        self.layout.addWidget(optionsBox)

        self.openButton = qt.QPushButton("Open Review/Edit")
        self.windowBox = self._makeWindowBox()
        self.fusionBox = self._makeFusionBox()
        self.roiBox = self._makeRoiBox()
        self.openButton.toolTip = "Switch to the Dynamic PET Review/Edit layout with the timeline"
        self.closeButton = qt.QPushButton("Close Review/Edit")
        self.closeButton.enabled = False
        buttons = qt.QHBoxLayout()
        buttons.addWidget(self.openButton)
        buttons.addWidget(self.closeButton)
        self.layout.addLayout(buttons)
        self.layout.addWidget(self.windowBox)
        self.layout.addWidget(self.fusionBox)
        self.layout.addWidget(self.roiBox)

        operationsBox = qt.QGroupBox("Frame editing")
        operations = qt.QVBoxLayout(operationsBox)
        note = qt.QLabel("On the timeline bar: Add frame (static or dynamic PET, placed by acquisition time), Merge, "
                         "Split, Remove and Discard changes; the edited timeline shows the edits in red over the "
                         "original one. Save as... makes a new dynamic sequence, Overwrite original replaces the "
                         "frames of the series.")
        note.wordWrap = True
        operations.addWidget(note)
        self.layout.addWidget(operationsBox)
        self.layout.addStretch(1)

        self.sequenceSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onSequenceChanged)
        self.openButton.connect("clicked()", self.onOpen)
        self.closeButton.connect("clicked()", lambda: self.onClose(confirm=True))
        self.overlayCheck.connect("toggled(bool)", self.onOverlayToggled)
        self.syncCheck.connect("toggled(bool)", self.onSyncToggled)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.NodeRemovedEvent, self.onNodeRemoved)
        self.selectDynamicSequence()
        self.onSequenceChanged(self.sequenceSelector.currentNode())

    def _makeWindowBox(self):
        box = qt.QGroupBox("Window")
        form = qt.QFormLayout(box)
        self.windowModeCombo = qt.QComboBox()
        self.windowModeCombo.addItem("Same for all frames", COMMON_WINDOW)
        self.windowModeCombo.addItem("Each frame automatic (0 - 99.9th percentile)", FRAME_WINDOW)
        self.windowModeCombo.toolTip = ("Same: one display range for every frame, so frames compare directly. "
                                        "Each frame: every frame scaled to its own range, so late, low-count "
                                        "frames stay visible.")
        form.addRow("Mode:", self.windowModeCombo)
        self.windowRange = ctk.ctkRangeWidget()
        self.windowRange.decimals = 2
        self.windowRange.toolTip = "Display range (min / max) of all frames; dragging in the large view changes it too"
        form.addRow("Range:", self.windowRange)
        self.autoWindowButton = qt.QPushButton("Auto")
        self.autoWindowButton.toolTip = "0 to the highest 99.9th percentile over the frames"
        form.addRow(self.autoWindowButton)
        self.colorSelector = qt.QComboBox()
        for label in colormaps.LABELS:
            self.colorSelector.addItem(label)
        self.colorSelector.toolTip = "Color map of all views (the Tarvos / MPI color maps and Slicer's PET maps)"
        form.addRow("Color map:", self.colorSelector)
        box.enabled = False

        self.windowModeCombo.connect("currentIndexChanged(int)", self.onWindowModeChanged)
        self.windowRange.connect("valuesChanged(double,double)", self.onWindowRangeChanged)
        self.autoWindowButton.connect("clicked()", self.onAutoWindow)
        self.colorSelector.connect("currentIndexChanged(int)", self.onColorChanged)
        return box

    def _makeFusionBox(self):
        box = qt.QGroupBox("Fusion")
        form = qt.QFormLayout(box)
        self.fusionSelector = slicer.qMRMLNodeComboBox()
        self.fusionSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.fusionSelector.noneEnabled = True
        self.fusionSelector.noneDisplay = "None (PET only)"
        self.fusionSelector.addEnabled = False
        self.fusionSelector.removeEnabled = False
        self.fusionSelector.renameEnabled = False
        self.fusionSelector.setMRMLScene(slicer.mrmlScene)
        self.fusionSelector.setCurrentNode(None)
        self.fusionSelector.toolTip = "Anatomical volume (CT, MR) shown under the PET in every view"
        form.addRow("Background:", self.fusionSelector)
        self.opacitySlider = ctk.ctkSliderWidget()
        self.opacitySlider.minimum = 0.0
        self.opacitySlider.maximum = 1.0
        self.opacitySlider.singleStep = 0.05
        self.opacitySlider.decimals = 2
        self.opacitySlider.value = 0.5
        self.opacitySlider.toolTip = "PET opacity over the background"
        form.addRow("PET opacity:", self.opacitySlider)
        box.enabled = False
        self.fusionSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onFusionVolumeChanged)
        self.opacitySlider.connect("valueChanged(double)", self.onFusionOpacityChanged)
        return box

    def _makeRoiBox(self):
        box = qt.QGroupBox("ROIs")
        layout = qt.QVBoxLayout(box)
        row = qt.QHBoxLayout()
        self.addRoiButton = qt.QPushButton("Add spherical ROI")
        self.addRoiButton.toolTip = "Then click in the large view; its time-activity curve shows on the timeline"
        row.addWidget(self.addRoiButton, 1)
        row.addWidget(qt.QLabel("Radius:"))
        self.roiRadiusSpin = qt.QDoubleSpinBox()
        self.roiRadiusSpin.setRange(0.5, 200.0)
        self.roiRadiusSpin.setDecimals(1)
        self.roiRadiusSpin.setSingleStep(1.0)
        self.roiRadiusSpin.setSuffix(" mm")
        self.roiRadiusSpin.value = ROI_DEFAULT_RADIUS_MM
        self.roiRadiusSpin.setKeyboardTracking(False)
        self.roiRadiusSpin.toolTip = "Radius of new ROIs and of the selected ROI"
        row.addWidget(self.roiRadiusSpin)
        layout.addLayout(row)
        self.roiTable = qt.QTableWidget(0, 3)
        self.roiTable.setHorizontalHeaderLabels(["", "ROI", "Radius"])
        self.roiTable.verticalHeader().hide()
        self.roiTable.setSelectionBehavior(qt.QAbstractItemView.SelectRows)
        self.roiTable.setSelectionMode(qt.QAbstractItemView.SingleSelection)
        self.roiTable.setEditTriggers(qt.QAbstractItemView.NoEditTriggers)
        self.roiTable.horizontalHeader().setSectionResizeMode(1, qt.QHeaderView.Stretch)
        self.roiTable.setColumnWidth(0, 24)
        self.roiTable.setColumnWidth(2, 70)
        self.roiTable.setMaximumHeight(150)
        layout.addWidget(self.roiTable)
        row = qt.QHBoxLayout()
        self.removeRoiButton = qt.QPushButton("Remove")
        self.removeRoiButton.toolTip = "Remove the selected ROI"
        self.removeAllRoisButton = qt.QPushButton("Remove all")
        row.addWidget(self.removeRoiButton)
        row.addWidget(self.removeAllRoisButton)
        layout.addLayout(row)
        box.enabled = False
        self.selectedRoiNode = None

        self.addRoiButton.connect("clicked()", self.onAddRoi)
        self.roiRadiusSpin.connect("valueChanged(double)", self.onRoiRadiusChanged)
        self.roiTable.connect("itemSelectionChanged()", self.onRoiSelectionChanged)
        self.removeRoiButton.connect("clicked()", self.onRemoveRoi)
        self.removeAllRoisButton.connect("clicked()", self.onRemoveAllRois)
        return box

    def _rois(self):
        return self.session.rois if self.session is not None else None

    def _selectedRoi(self):
        rois = self._rois()
        return rois._roiOf(self.selectedRoiNode) if rois is not None else None

    def _refreshRoiTable(self):
        rois = self._rois()
        items = rois.rois if rois is not None else []
        self.roiTable.blockSignals(True)
        try:
            self.roiTable.clearContents()
            self.roiTable.setRowCount(len(items))
            selected = self._selectedRoi()
            for row, roi in enumerate(items):
                swatch = qt.QTableWidgetItem()
                swatch.setBackground(qt.QBrush(qt.QColor(roi["node"].GetAttribute(ROI_COLOR_ATTRIBUTE))))
                self.roiTable.setItem(row, 0, swatch)
                name = roi["node"].GetName()
                if rois.center(roi) is None:
                    name += " (click in the large view)"
                self.roiTable.setItem(row, 1, qt.QTableWidgetItem(name))
                self.roiTable.setItem(row, 2, qt.QTableWidgetItem(f"{rois.radius(roi):g} mm"))
                if roi is selected:
                    self.roiTable.selectRow(row)
            if selected is None:
                self.roiTable.clearSelection()
        finally:
            self.roiTable.blockSignals(False)
        self.removeRoiButton.enabled = selected is not None
        self.removeAllRoisButton.enabled = bool(items)

    def onAddRoi(self):
        rois = self._rois()
        if rois is not None:
            self.selectedRoiNode = rois.add(self.roiRadiusSpin.value)["node"]
            self._refreshRoiTable()

    def onRoiSelectionChanged(self):
        rois, indexes = self._rois(), self.roiTable.selectedIndexes()
        roi = rois.rois[indexes[0].row()] if rois is not None and indexes else None
        self.selectedRoiNode = roi["node"] if roi is not None else None
        self.removeRoiButton.enabled = roi is not None
        if roi is not None:
            self.roiRadiusSpin.blockSignals(True)
            self.roiRadiusSpin.value = rois.radius(roi)
            self.roiRadiusSpin.blockSignals(False)

    def onRoiRadiusChanged(self, value):
        roi = self._selectedRoi()
        if roi is not None:
            self._rois().setRadius(roi, value)

    def onRemoveRoi(self):
        roi = self._selectedRoi()
        if roi is not None:
            self._rois().remove(roi)

    def onRemoveAllRois(self):
        if self._rois() is not None:
            self._rois().removeAll()

    def onFusionVolumeChanged(self, node):
        if self.session is None:
            return
        if node is not None and node in self.session.editorVolumes():
            slicer.util.warningDisplay("Pick an anatomical volume, not the dynamic PET itself.")
            self.fusionSelector.setCurrentNode(None)
            return
        self.session.setFusion(node, self.opacitySlider.value)

    def onFusionOpacityChanged(self, value):
        if self.session is not None:
            self.session.setFusionOpacity(value)

    def _setupWindowBox(self):
        session = self.session
        self.windowBox.enabled = session is not None
        if session is None:
            return
        lowest, highest = session.valueRange()
        widgets = (self.windowRange, self.windowModeCombo, self.colorSelector)
        for widget in widgets:
            widget.blockSignals(True)
        try:
            self.windowRange.singleStep = max((highest - lowest) / 1000.0, 1e-4)
            self.windowRange.setRange(lowest, highest)
            self.windowRange.setValues(*session.window)
            self.windowModeCombo.setCurrentIndex(0)
        finally:
            for widget in widgets:
                widget.blockSignals(False)
        self.onColorChanged(self.colorSelector.currentIndex)
        self.windowRange.enabled = True
        self.autoWindowButton.enabled = True
        session.onWindowChanged = self._onViewWindowChanged

    def _onViewWindowChanged(self, lower, upper):
        self.windowRange.blockSignals(True)
        self.windowRange.setValues(lower, upper)
        self.windowRange.blockSignals(False)

    def onWindowModeChanged(self, index):
        if self.session is None:
            return
        mode = self.windowModeCombo.itemData(index)
        self.windowRange.enabled = mode == COMMON_WINDOW
        self.autoWindowButton.enabled = mode == COMMON_WINDOW
        self.session.setWindowMode(mode)

    def onWindowRangeChanged(self, lower, upper):
        if self.session is not None:
            self.session.setCommonWindow(lower, upper)

    def onAutoWindow(self):
        if self.session is not None:
            self.windowRange.setValues(*self.session.autoWindow())

    def onColorChanged(self, index):
        if self.session is not None and index >= 0:
            self.session.setColorNodeID(colormaps.colorNodeForLabel(self.colorSelector.itemText(index)).GetID())

    def enter(self):
        if self.sequenceSelector.currentNode() is None:
            self.selectDynamicSequence()

    def cleanup(self):
        self.onClose()
        self.removeObservers()
        ScriptedLoadableModuleWidget.cleanup(self)

    def selectDynamicSequence(self):
        """Pick the most recent sequence loaded as dynamic PET."""
        nodes = slicer.util.getNodesByClass("vtkMRMLSequenceNode")
        dynamic = [n for n in nodes if n.GetAttribute(PREFIX + "FrameDurationS") or n.GetAttribute(PREFIX + "Kind")]
        if dynamic:
            self.sequenceSelector.setCurrentNode(dynamic[-1])

    def onSequenceChanged(self, node):
        if node is None:
            self.infoLabel.text = "Pick a dynamic PET sequence."
            self.openButton.enabled = False
            return
        if node.IsA("vtkMRMLScalarVolumeNode"):
            if self.logic.isStaticPet(node):
                self.infoLabel.text = ("Static PET: opens as a one-frame series; add static or dynamic PET to it and "
                                       "Save as a new dynamic series.")
                self.openButton.enabled = True
            else:
                self.infoLabel.text = "Not a static PET volume (pick a dynamic PET sequence or a static PET)."
                self.openButton.enabled = False
            return
        if not self.logic.isDynamicSequence(node):
            self.infoLabel.text = "Not a sequence of volumes."
            self.openButton.enabled = False
            return
        intervals = self.logic.frameIntervals(node)
        base = intervals[0].get("timeBase", "study start")
        self.infoLabel.text = (f"{len(intervals)} frames, {dynamicviews.formatTime(intervals[0]['start'])} - "
                               f"{dynamicviews.formatTime(intervals[-1]['end'])} from the {base}"
                               if base != "frame order" else f"{len(intervals)} frames (no frame timing found)")
        self.openButton.enabled = True

    def onOpen(self):
        node = self.sequenceSelector.currentNode()
        if node is None:
            return
        if not self.onClose(confirm=True):
            return
        # Motion Correction takes the main layout and a timeline dock too: close it first
        motionCorrection = getattr(slicer.modules, "DynamicPETMotionCorrectionWidget", None)
        if motionCorrection is not None and getattr(motionCorrection, "session", None) is not None:
            motionCorrection.onClose()
        temporary = None
        try:
            if node.IsA("vtkMRMLScalarVolumeNode"):
                temporary = node = staticAsSequence(node)
            self.session = DynamicPETEditorSession(node, self.maxViewsSpin.value, self.overlayCheck.checked,
                                                   self.syncCheck.checked)
            self.session.temporarySequence = temporary is not None
            self.session.open()
        except Exception as error:
            logging.exception("Could not open Dynamic PET Review/Edit")
            slicer.util.errorDisplay(f"Could not open Dynamic PET Review/Edit:\n{error}")
            if self.session is not None:
                self.session.close()
            elif temporary is not None and slicer.mrmlScene.IsNodePresent(temporary):
                slicer.mrmlScene.RemoveNode(temporary)
            self.session = None
        self.closeButton.enabled = self.session is not None
        self._setupWindowBox()
        self.fusionBox.enabled = self.session is not None
        if self.session is not None and self.fusionSelector.currentNode() is not None:
            self.onFusionVolumeChanged(self.fusionSelector.currentNode())
        self.roiBox.enabled = self.session is not None
        if self.session is not None:
            self.session.rois.onChanged = self._refreshRoiTable
            self.session.onEditsChanged = self._onEditsChanged
            self.session.onSaved = lambda sequence: qt.QTimer.singleShot(0, lambda: self._reopen(sequence))
            self.session._updateEditButtons()
        self._refreshRoiTable()

    def _onEditsChanged(self):
        """The frames shown changed: the window slider spans their values."""
        if self.session is None:
            return
        lowest, highest = self.session.valueRange()
        self.windowRange.blockSignals(True)
        try:
            self.windowRange.setRange(min(lowest, self.session.window[0]), max(highest, self.session.window[1]))
            self.windowRange.setValues(*self.session.window)
        finally:
            self.windowRange.blockSignals(False)

    def _reopen(self, sequence):
        """After Save as / Overwrite: open the saved series, so the timeline is a plain one again."""
        self.onClose()
        if slicer.mrmlScene.IsNodePresent(sequence):
            self.sequenceSelector.setCurrentNode(sequence)
            self.onOpen()

    def onClose(self, confirm=False):
        """Close the session; confirm: ask first when there are unsaved edits. False when the user keeps it open."""
        if self.session is not None and confirm and self.session.isEditing() and \
                not slicer.app.testingEnabled() and not slicer.util.confirmYesNoDisplay(
                    "The edits of the frames are not saved. Close and discard them?", windowTitle="Review/Edit"):
            return False
        if self.session is not None:
            self.session.close()
            self.session = None
        self.closeButton.enabled = False
        if hasattr(self, "windowBox"):
            self.windowBox.enabled = False
        if hasattr(self, "fusionBox"):
            self.fusionBox.enabled = False
        if hasattr(self, "roiBox"):
            self.roiBox.enabled = False
            self.selectedRoiNode = None
            self._refreshRoiTable()
        return True

    def onOverlayToggled(self, checked):
        if self.session is not None:
            self.session.setOverlaysVisible(checked)

    def onSyncToggled(self, checked):
        if self.session is not None:
            self.session.setLinked(checked)

    def onSceneClose(self, caller, event):
        self.onClose()

    @vtk.calldata_type(vtk.VTK_OBJECT)
    def onNodeRemoved(self, caller, event, node):
        if self.session is not None and node in (self.session.sequenceNode, self.session.browserNode):
            qt.QTimer.singleShot(0, self.onClose)


# ---------------------------------------------------------------------------
# Test: a synthetic dynamic series loaded through the plugin, then opened in the editor
# ---------------------------------------------------------------------------
class DynamicPETEditorTest(ScriptedLoadableModuleTest):
    def setUp(self):
        slicer.mrmlScene.Clear(0)

    def runTest(self):
        self.setUp()
        self.test_OpenEditor()
        self.setUp()
        self.test_EditFrames()
        self.setUp()
        self.test_StartFromStatic()
        self.setUp()
        self.test_ExportDicom()

    def test_OpenEditor(self):
        import tempfile
        from DICOMGatedDynamicPETLib import petseries
        from DICOMGatedDynamicPET import DICOMGatedDynamicPETPluginClass
        durations = [10.0, 10.0, 30.0, 60.0, 120.0]
        files, _ = petseries.writeSyntheticSeries(tempfile.mkdtemp(prefix="petseries"), kind=petseries.DYNAMIC,
                                                  frames=5, frameDurationsS=durations)
        DICOMGatedDynamicPETPluginClass().loadSeries(files, "Synthetic dynamic Bq/ml", petseries.UNITS_ACTIVITY,
                                                     summed=False)
        sequence = slicer.util.getNodesByClass("vtkMRMLSequenceNode")[0]
        intervals = DynamicPETEditorLogic.frameIntervals(sequence)
        self.assertEqual([round(i["start"]) for i in intervals], [0, 10, 20, 50, 110])
        session = DynamicPETEditorSession(sequence)
        session.open()
        try:
            layoutManager = slicer.app.layoutManager()
            self.assertEqual(layoutManager.layout, LAYOUT_ID)
            for v in range(5):
                self.assertIsNotNone(layoutManager.sliceWidget(dynamicviews.frameViewName(v)))
            session.setCurrentIndex(2)
            slicer.app.processEvents()
            self.assertEqual(session.timeline.currentIndex, 2)
            actor = session.overlays[dynamicviews.MAIN_VIEW][2]
            self.assertTrue(actor.GetInput().startswith("F3  0:20 - 0:50"))
            self._checkROIs(session)
            self.delayDisplay("Dynamic PET Review/Edit open")
        finally:
            session.close()
        self.assertEqual(len(slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode")), 1)   # only the proxy
        self.assertEqual(len(slicer.util.getNodesByClass("vtkMRMLModelNode")), self.modelsBefore)
        self.assertEqual(len(slicer.util.getNodesByClass(ROI_NODE_CLASS)), 1)              # the ROI stays
        session = DynamicPETEditorSession(sequence)
        session.open()
        try:
            self.assertEqual(len(session.rois.rois), 1)                                    # and is taken up again
            self.assertEqual(session.rois.radius(session.rois.rois[0]), 6.0)
            self.assertEqual(len(session.timeline.curves[0]["values"]), 5)
        finally:
            session.close()

    def _checkROIs(self, session):
        sequence, rois = session.sequenceNode, session.rois
        self.modelsBefore = len(slicer.util.getNodesByClass("vtkMRMLModelNode"))
        bounds = [0.0] * 6
        session.proxyNode.GetRASBounds(bounds)
        center = [(bounds[2 * a] + bounds[2 * a + 1]) / 2.0 for a in range(3)]
        roi = rois.add(50.0, center)
        rois.flush()
        self.assertEqual(session.timeline.curves[0]["name"], roi["node"].GetName())
        self.assertEqual(len(roi["values"]), sequence.GetNumberOfDataNodes())
        self.assertEqual(len(session.timeline.curves[0]["values"]), 5)
        self.assertEqual(session.timeline.curveUnits, "Bq/ml")
        for i, value in enumerate(roi["values"]):
            matrix = vtk.vtkMatrix4x4()
            sequence.GetNthDataNode(i).GetIJKToRASMatrix(matrix)
            array = slicer.util.arrayFromVolume(sequence.GetNthDataNode(i))
            expected = roicurves.sphereMean(array, slicer.util.arrayFromVTKMatrix(matrix), center, 50.0)
            self.assertAlmostEqual(value, expected)
        rois.setRadius(roi, 6.0)
        rois.flush()
        self.assertEqual(roi["source"].GetRadius(), 6.0)
        viewID = session.mainViewNodeID()
        self.assertEqual(roi["node"].GetDisplayNode().GetNthViewNodeID(0), viewID)
        self.assertEqual(roi["model"].GetDisplayNode().GetNthViewNodeID(0), viewID)
        self.assertFalse(session.timeline.grab().isNull())
        # a click that never comes leaves no node behind
        pending = rois.add(5.0)
        self.assertIsNotNone(rois.pending)
        rois.cancelPlacement()
        self.assertEqual(len(rois.rois), 1)
        self.assertFalse(slicer.mrmlScene.IsNodePresent(pending["node"]))

    def _loadEditSeries(self):
        """A dynamic series (4 frames from 10:00, injection 10:00) and a static one at 11:00 (same injection)."""
        import tempfile
        from DICOMGatedDynamicPETLib import petseries
        from DICOMGatedDynamicPET import DICOMGatedDynamicPETPluginClass
        plugin = DICOMGatedDynamicPETPluginClass()
        start = datetime.datetime(2026, 1, 1, 10, 0, 0)
        files, _ = petseries.writeSyntheticSeries(tempfile.mkdtemp(prefix="petseries"), kind=petseries.DYNAMIC,
                                                  frames=4, frameDurationsS=[10.0, 30.0, 60.0, 120.0])
        summed = plugin.loadSeries(files, "Edit dynamic Bq/ml", petseries.UNITS_ACTIVITY, summed=True)
        sequence = slicer.mrmlScene.GetNodeByID(summed.GetAttribute(PREFIX + "DynamicSequence"))
        files, _ = petseries.writeSyntheticSeries(tempfile.mkdtemp(prefix="petseries"), kind=petseries.STATIC,
                                                  seriesStart=start + datetime.timedelta(hours=1), injection=start)
        static = plugin.loadSeries(files, "Late static Bq/ml", petseries.UNITS_ACTIVITY)
        return sequence, summed, static

    def test_EditFrames(self):
        sequence, summed, static = self._loadEditSeries()
        self.assertEqual(static.GetAttribute("StaticPET.DurationS"), "600")
        self.assertEqual(static.GetAttribute("StaticPET.Manufacturer"), "Synthetic")
        self.assertEqual([kind for _, kind, _ in petCandidates(sequence)], ["static"])
        session = DynamicPETEditorSession(sequence)
        session.open()
        try:
            self.assertEqual(session.addCombo.count, 1)
            # merge F1 + F2: duration weighted mean, red on the edited timeline, the original untouched
            session.mergeFrames([0, 1])
            self.assertTrue(session.isEditing())
            self.assertEqual(session.frameCount(), 3)
            self.assertFalse(session.originalTimeline.isHidden())
            self.assertTrue(session.timeline.frames[0]["edited"])
            merged = session.frameArrays()
            array, _ = next(merged)
            self.assertAlmostEqual(float(array.mean()), (1000.0 * 10 + 2000.0 * 30) / 40, places=2)
            self.assertEqual(sequence.GetNumberOfDataNodes(), 4)
            self.assertEqual(session.mainVolume(), session.editVolume)
            # add the static series: one hour later, decay corrected to its own start -> aligned to 10:00
            prepared = session.prepareAdd(static, "static")
            self.assertFalse(prepared["comparison"]["errors"])
            self.assertFalse(prepared["comparison"]["warnings"])                  # 56 min apart, same patient
            self.assertFalse(prepared["resampleNeeded"])
            self.assertEqual(session.addSeries(static, "static", prepared=prepared), [3])
            self.assertAlmostEqual(session.intervals[3]["start"], 3600.0)
            values = list(session.frameArrays())
            self.assertAlmostEqual(float(values[3][0].mean()), 1000.0 * 2.0 ** (3600.0 / 6586.2), places=1)
            # split gives F1 and F2 back; remove the last original frame
            session.splitFrames([0])
            self.assertEqual(session.frameCount(), 5)
            session.removeFrames([3])
            self.assertEqual(session.frameCount(), 4)
            # save as: a new sequence; the original keeps its 4 frames
            saved = saveEditedSeries(session, "Edit saved")
            self.assertEqual(saved.GetNumberOfDataNodes(), 4)
            self.assertEqual(sequence.GetNumberOfDataNodes(), 4)
            starts = [float(saved.GetNthDataNode(i).GetAttribute(PREFIX + "StartFromStudyStartS")) for i in range(4)]
            self.assertEqual([round(t) for t in starts], [0, 10, 40, 3600])
            self.assertEqual(saved.GetNthDataNode(3).GetAttribute(PREFIX + "DurationS"), "600")
            self.assertIn("added", saved.GetNthDataNode(3).GetAttribute(PREFIX + "EditSources"))
            table = slicer.mrmlScene.GetNodeByID(saved.GetAttribute(PREFIX + "FrameTimingTableID"))
            self.assertEqual(table.GetNumberOfRows(), 4)
            newSummed = slicer.mrmlScene.GetNodeByID(saved.GetAttribute(PREFIX + "SummedVolumeID"))
            self.assertIsNotNone(newSummed)
            self.assertIsNot(newSummed, summed)
            # discard, undo it, redo it
            session.discardEdits()
            self.assertFalse(session.isEditing())
            self.assertEqual(session.undo(), "Discard changes")
            self.assertEqual(session.frameCount(), 4)
            self.assertTrue(session.isEditing())
            self.assertEqual(session.undo(), "Remove")
            self.assertEqual(session.frameCount(), 5)
            session.redo()
            session.redo()
            self.assertFalse(session.isEditing())
            self.assertEqual(session.frameCount(), 4)
            self.assertEqual(session.mainVolume(), session.proxyNode)
            # overwrite the original with a merge
            session.mergeFrames([2, 3])
            saveEditedSeries(session, overwrite=True)
            self.assertEqual(sequence.GetNumberOfDataNodes(), 3)
            self.assertEqual(sequence.GetNthDataNode(2).GetAttribute(PREFIX + "DurationS"), "180")
            self.assertEqual(slicer.mrmlScene.GetNodeByID(sequence.GetAttribute(PREFIX + "SummedVolumeID")), summed)
        finally:
            session.close()
        self.assertFalse([n for n in slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode")
                          if n.GetAttribute(EDITOR_ATTRIBUTE) == "1"])

    def test_StartFromStatic(self):
        sequence, summed, static = self._loadEditSeries()
        temporary = staticAsSequence(static)
        session = DynamicPETEditorSession(temporary)
        session.temporarySequence = True
        session.open()
        try:
            self.assertEqual(session.frameCount(), 1)
            self.assertIn(sequence, [node for node, _, _ in petCandidates(temporary)])
            prepared = session.prepareAdd(sequence, "dynamic")
            self.assertFalse(prepared["comparison"]["errors"])
            self.assertEqual(session.addSeries(sequence, "dynamic", prepared=prepared), [0, 1, 2, 3])
            self.assertFalse(session.overwriteButton.enabled)
            # the dynamic series starts first: the static frame is decay corrected to 10:00
            self.assertAlmostEqual(session.edit.scale("o1"), 2.0 ** (3600.0 / 6586.2), places=6)
            saved = saveEditedSeries(session, "From static")
            self.assertEqual(saved.GetNumberOfDataNodes(), 5)
        finally:
            session.close()
        self.assertFalse(slicer.mrmlScene.IsNodePresent(temporary))

    def test_ExportDicom(self):
        import tempfile
        from DICOMLib import DICOMUtils
        from DICOMGatedDynamicPETLib import petseries
        from DICOMGatedDynamicPET import DICOMGatedDynamicPETPluginClass
        plugin = DICOMGatedDynamicPETPluginClass()
        with DICOMUtils.TemporaryDICOMDatabase() as db:
            folder = tempfile.mkdtemp(prefix="petseries")
            files, _ = petseries.writeSyntheticSeries(folder, kind=petseries.DYNAMIC, frames=3,
                                                      frameDurationsS=[10.0, 30.0, 60.0])
            DICOMUtils.importDicom(folder, db)
            summed = plugin.loadSeries(files, "Export dynamic Bq/ml", petseries.UNITS_ACTIVITY, summed=True)
            sequence = slicer.mrmlScene.GetNodeByID(summed.GetAttribute(PREFIX + "DynamicSequence"))
            session = DynamicPETEditorSession(sequence)
            session.open()
            try:
                session.mergeFrames([1, 2])
                saved = saveEditedSeries(session, "Export merged")
            finally:
                session.close()
            seriesUID, count = exportSequenceToDicom(saved, "Export merged")
            self.assertEqual(count, 2 * 6)
            studies = db.studiesForPatient(db.patients()[0])
            self.assertEqual(len(studies), 1)                       # the same study
            self.assertIn(seriesUID, db.seriesForStudy(studies[0]))
            exported = db.filesForSeries(seriesUID)
            folder, files, _ = exportedLocation(seriesUID)              # what the done dialog shows
            self.assertEqual(len(files), 2 * 6)
            self.assertTrue(os.path.isdir(folder) and all(f.startswith(folder) for f in files))
            again = plugin.loadSeries(exported, "Reloaded dynamic Bq/ml", petseries.UNITS_ACTIVITY, summed=True)
            reloaded = slicer.mrmlScene.GetNodeByID(again.GetAttribute(PREFIX + "DynamicSequence"))
            self.assertEqual(reloaded.GetNumberOfDataNodes(), 2)
            for i in range(2):
                np.testing.assert_allclose(slicer.util.arrayFromVolume(reloaded.GetNthDataNode(i)),
                                           slicer.util.arrayFromVolume(saved.GetNthDataNode(i)), rtol=1e-3)
            self.assertEqual(reloaded.GetNthDataNode(1).GetAttribute(PREFIX + "DurationS"), "90")
