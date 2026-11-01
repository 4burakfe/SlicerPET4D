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
Frame operations (summation, reframing, ...) are placeholders for now.
"""

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

from DICOMGatedDynamicPETLib import colormaps, dynamicviews, roicurves  # noqa: E402
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
PLANNED_OPERATIONS = (
    ("Sum selected", "Sum the selected frames into one static volume (duration weighted)"),
    ("Merge", "Merge the selected neighbouring frames into one frame"),
    ("Split", "Split the current frame"),
    ("Reframe...", "Rebin the series into a new framing scheme"),
    ("Remove", "Remove the selected frames"),
)


class DynamicPETEditor(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        parent.title = "Dynamic PET Review/Edit"
        parent.categories = ["Nuclear Medicine"]
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
        return bool(sequenceNode and sequenceNode.GetNumberOfDataNodes() > 1
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
        """(array_kji, ijkToRas) of every frame of the sequence, or None for a frame without an image."""
        from vtk.util import numpy_support
        sequence = self.session.sequenceNode
        matrix = vtk.vtkMatrix4x4()
        for i in range(sequence.GetNumberOfDataNodes()):
            node = sequence.GetNthDataNode(i)
            image = node.GetImageData()
            scalars = image.GetPointData().GetScalars() if image is not None else None
            if scalars is None:
                yield None
                continue
            node.GetIJKToRASMatrix(matrix)
            yield (numpy_support.vtk_to_numpy(scalars).reshape(image.GetDimensions()[::-1]),
                   slicer.util.arrayFromVTKMatrix(matrix))

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
        self.intervals = []
        self.viewFrames = []          # frame index shown in each small view
        self.frameVolumes = []
        self.overlays = {}            # view name -> (sliceView, renderer, text actor)
        self.observers = []
        self.dock = None
        self.timeline = None
        self.previousLayout = None
        self.frameStats = []          # (auto upper = 99.9th percentile, maximum, minimum) per frame
        self.windowMode = COMMON_WINDOW
        self.window = (0.0, 1.0)
        self.onWindowChanged = None   # callback(lower, upper) when the window changes in a view (mouse)
        self._applyingWindow = False
        self.fusionVolume = None
        self.fusionOpacity = 0.5
        self.playTimer = None
        self.playSpeed = 1.0          # acquisition seconds per real second
        self.playTime = None
        self._lastTick = None
        self.rois = None              # SphereROIs

    # --- open / close ---------------------------------------------------------------------------
    def open(self):
        logic = DynamicPETEditorLogic()
        self.browserNode = logic.browserFor(self.sequenceNode)
        self.proxyNode = self.browserNode.GetProxyNode(self.sequenceNode)
        self.intervals = logic.frameIntervals(self.sequenceNode)
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
        for node, tag in self.observers:
            if node is not None:
                node.RemoveObserver(tag)
        self.observers = []
        for sliceView, renderer, actor in self.overlays.values():
            renderer.RemoveActor2D(actor)
            sliceView.scheduleRender()
        self.overlays = {}
        if slicer.app.layoutManager().layout == LAYOUT_ID:
            self.setLinked(False)
        if self.dock is not None:
            slicer.util.mainWindow().removeDockWidget(self.dock)
            self.dock.deleteLater()
            self.dock = None
        layoutManager = slicer.app.layoutManager()
        if self.previousLayout is not None and layoutManager.layout == LAYOUT_ID:
            layoutManager.setLayout(self.previousLayout)
        for node in self.frameVolumes:
            if slicer.mrmlScene.IsNodePresent(node):
                slicer.mrmlScene.RemoveNode(node)
        self.frameVolumes = []

    # --- window and color --------------------------------------------------------------------------
    def _computeFrameStats(self):
        from vtk.util import numpy_support
        self.frameStats = []
        for i in range(self.sequenceNode.GetNumberOfDataNodes()):
            image = self.sequenceNode.GetNthDataNode(i).GetImageData()
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

    def _displays(self):
        """(display node, frame index or None for the current-frame proxy) of every editor volume."""
        out = [(self._proxyDisplay(), None)]
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
        for display, _ in self._displays():
            display.SetAndObserveColorNodeID(colorNodeID)

    def onProxyDisplayModified(self, caller=None, event=None):
        """Window changed in the large view (mouse drag): pass it to every frame view."""
        if self._applyingWindow or self.windowMode != COMMON_WINDOW:
            return
        display = self._proxyDisplay()
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
        pairs = [(self._sliceWidget(dynamicviews.MAIN_VIEW), self.proxyNode)]
        pairs += [(self._sliceWidget(dynamicviews.frameViewName(v)), volume)
                  for v, volume in enumerate(self.frameVolumes)]
        return [(widget, volume) for widget, volume in pairs if widget is not None and volume is not None]

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
        """A hidden volume per small view, sharing the voxels of its frame in the sequence."""
        proxyDisplay = self.proxyNode.GetDisplayNode() if self.proxyNode else None
        for index in self.viewFrames:
            source = self.sequenceNode.GetNthDataNode(index)
            volume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode",
                                                        f"{self.sequenceNode.GetName()} view F{index + 1}")
            volume.SetHideFromEditors(True)
            volume.SetSaveWithScene(False)
            volume.SetAttribute(EDITOR_ATTRIBUTE, "1")
            volume.SetAndObserveImageData(source.GetImageData())
            matrix = vtk.vtkMatrix4x4()
            source.GetIJKToRASMatrix(matrix)
            volume.SetIJKToRASMatrix(matrix)
            volume.CreateDefaultDisplayNodes()
            display = volume.GetDisplayNode()
            if proxyDisplay is not None:
                display.SetAndObserveColorNodeID(proxyDisplay.GetColorNodeID())
                display.AutoWindowLevelOff()
                display.SetWindowLevelMinMax(proxyDisplay.GetWindowLevelMin(), proxyDisplay.GetWindowLevelMax())
                display.SetInterpolate(proxyDisplay.GetInterpolate())
            self.frameVolumes.append(volume)

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

    def _setupViews(self):
        main = self._sliceWidget(dynamicviews.MAIN_VIEW)
        main.mrmlSliceCompositeNode().SetBackgroundVolumeID(self.proxyNode.GetID() if self.proxyNode else None)
        main.mrmlSliceCompositeNode().SetForegroundVolumeID(None)
        main.mrmlSliceCompositeNode().SetLabelVolumeID(None)
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
            widget.sliceController().setVisible(False)    # placeholder keeping the grid cells equal
            composite = widget.mrmlSliceCompositeNode()
            composite.SetBackgroundVolumeID(None)
            composite.SetForegroundVolumeID(None)
            composite.SetLabelVolumeID(None)
        self.setLinked(self.syncSlices)

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
        for v, frameIndex in enumerate(self.viewFrames):
            overlay = self.overlays.get(dynamicviews.frameViewName(v))
            if overlay is None:
                continue
            sliceView, _, actor = overlay
            actor.SetInput(dynamicviews.frameLabel(self.intervals[frameIndex]))
            highlight = frameIndex == current
            actor.GetTextProperty().SetColor(*((1.0, 0.85, 0.2) if highlight else (1.0, 1.0, 1.0)))
            actor.GetTextProperty().SetBold(highlight)
            sliceView.scheduleRender()
        overlay = self.overlays.get(dynamicviews.MAIN_VIEW)
        if overlay is not None and 0 <= current < len(self.intervals):
            sliceView, _, actor = overlay
            actor.SetInput(f"{dynamicviews.frameLabel(self.intervals[current])}   ({current + 1} / {len(self.intervals)})")
            sliceView.scheduleRender()

    # --- sync ------------------------------------------------------------------------------------
    def _observe(self):
        self.observers.append((self.browserNode, self.browserNode.AddObserver(vtk.vtkCommand.ModifiedEvent,
                                                                              self.onBrowserModified)))
        proxyDisplay = self._proxyDisplay()
        if proxyDisplay is not None:
            self.observers.append((proxyDisplay, proxyDisplay.AddObserver(vtk.vtkCommand.ModifiedEvent,
                                                                          self.onProxyDisplayModified)))

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
        return self.browserNode.GetSelectedItemNumber() if self.browserNode else -1

    def setCurrentIndex(self, index):
        if self.browserNode is not None and 0 <= index < len(self.intervals):
            self.browserNode.SetSelectedItemNumber(index)

    def onBrowserModified(self, caller=None, event=None):
        current = self.currentIndex()
        if self.windowMode == FRAME_WINDOW:
            self.applyWindow()
        self._updateOverlays(current)
        if self.timeline is not None:
            self.timeline.setCurrentIndex(current)
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
        for title, tip in PLANNED_OPERATIONS:
            button = qt.QPushButton(title)
            button.enabled = False
            button.toolTip = tip + " (coming soon)"
            toolbar.addWidget(button)
        layout.addLayout(toolbar)

        self.timeline = FrameTimeline()
        self.timeline.setFrames(self.intervals)
        self.timeline.onCurrentFrameChanged = self._pickFrame
        self.timeline.onSelectionChanged = self._onSelectionChanged
        self.timeline.onViewRangeChanged = self._onTimelineZoomed
        layout.addWidget(self.timeline, 1)

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
        self.nextButton.connect("clicked()", lambda: self._pickFrame(min(len(self.intervals) - 1,
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

    def _onTimeWindowChanged(self, start, end):
        self.timeline.setViewRange(start, end)
        self.windowLabel.text = f"{dynamicviews.formatTime(start)} - {dynamicviews.formatTime(end)}"

    def _onTimelineZoomed(self, start, end):
        self.rangeWidget.setValues(start, end)

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
        self.frameLabel.text = (f"Frame {frame['number']} ({current + 1} of {len(self.intervals)}):  "
                                f"{dynamicviews.frameTimes(self.intervals, current)},  "
                                f"duration {dynamicviews.formatTime(frame['duration'])}")


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
        self.sequenceSelector.nodeTypes = ["vtkMRMLSequenceNode"]
        self.sequenceSelector.noneEnabled = True
        self.sequenceSelector.addEnabled = False
        self.sequenceSelector.removeEnabled = False
        self.sequenceSelector.renameEnabled = False
        self.sequenceSelector.setMRMLScene(slicer.mrmlScene)
        self.sequenceSelector.toolTip = "Dynamic PET sequence (as loaded from DICOM: '<series> dynamic Bq/ml')"
        form.addRow("Sequence:", self.sequenceSelector)
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

        operationsBox = qt.QGroupBox("Frame operations")
        operations = qt.QVBoxLayout(operationsBox)
        note = qt.QLabel("Summation, merging, splitting and reframing are coming next.")
        note.wordWrap = True
        operations.addWidget(note)
        self.layout.addWidget(operationsBox)
        self.layout.addStretch(1)

        self.sequenceSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onSequenceChanged)
        self.openButton.connect("clicked()", self.onOpen)
        self.closeButton.connect("clicked()", self.onClose)
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
        if node is not None and node in (self.session.proxyNode, *self.session.frameVolumes):
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
        if not self.logic.isDynamicSequence(node):
            self.infoLabel.text = "Not a sequence of volumes with more than one frame."
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
        self.onClose()
        try:
            self.session = DynamicPETEditorSession(node, self.maxViewsSpin.value, self.overlayCheck.checked,
                                                   self.syncCheck.checked)
            self.session.open()
        except Exception as error:
            logging.exception("Could not open Dynamic PET Review/Edit")
            slicer.util.errorDisplay(f"Could not open Dynamic PET Review/Edit:\n{error}")
            if self.session is not None:
                self.session.close()
            self.session = None
        self.closeButton.enabled = self.session is not None
        self._setupWindowBox()
        self.fusionBox.enabled = self.session is not None
        if self.session is not None and self.fusionSelector.currentNode() is not None:
            self.onFusionVolumeChanged(self.fusionSelector.currentNode())
        self.roiBox.enabled = self.session is not None
        if self.session is not None:
            self.session.rois.onChanged = self._refreshRoiTable
        self._refreshRoiTable()

    def onClose(self):
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
