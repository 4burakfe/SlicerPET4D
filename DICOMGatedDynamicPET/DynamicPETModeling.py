"""
Dynamic PET Modeling (3D Slicer module)

Kinetic modelling of a dynamic PET series (as loaded by the DICOM Gated / Dynamic PET plugin) on the segments of a
segmentation: compartment models (1TC, 2TC reversible / irreversible) and the graphical Patlak and Logan plots with a
plasma input (an image-derived blood pool segment, or a plasma curve from a CSV file), and the reference region
methods SRTM, Logan (reference) and SUVR. Pick a target segment for the preview and a reference segment (the blood
pool for the plasma input models, a region without specific binding such as the cerebellum for the reference
methods). Preview fits the target only; Fit all segments fits every segment into the results table (also a table
node in the scene, and a CSV export). Voxel-wise parametric maps are made for 1TC (basis functions), Patlak, Logan,
Logan (reference), SRTM (basis functions) and SUVR.

Layout:
    upper half    the summed image (or the middle frame) with the target and reference segments on the left (1/3),
                  a near-square grid of the frames on the right (2/3), as in Review/Edit
    lower half    a dock: the time-activity curves of the segments and the fitted model on the left (the target and
                  the reference thick, the other segments faint), the results on the right

The modelling itself is in DICOMGatedDynamicPETLib.kinetics (numpy only) and DynamicPETModelingLogic can be used
without the widget, e.g. from another module or the Python console:

    import DynamicPETModeling
    logic = DynamicPETModeling.DynamicPETModelingLogic()
    analysis = logic.prepare(sequenceNode, segmentationNode)                      # TACs of every segment
    results = logic.fitSegments(analysis, "SRTM", referenceId="Cerebellum")       # {segment ID: result}
    results["Striatum"]["values"]["BPnd"]
    logic.exportCsv("/tmp/srtm.csv", analysis, results, "SRTM")
    maps = logic.parametricMaps(analysis, "SRTM", referenceId="Cerebellum")     # {name: volume node}
"""

import logging
import math
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
from DICOMGatedDynamicPETLib import kinetics as kin  # noqa: E402
from DICOMGatedDynamicPETLib.tacplot import TacPlot  # noqa: E402
from DynamicPETEditor import guardViewVolumes  # noqa: E402

LAYOUT_ID = 7805
MAIN_VIEW = "DynKMMain"
FRAME_VIEW_PREFIX = "DynKMF"
EMPTY_VIEW_PREFIX = "DynKMEmpty"
VIEW_GROUP = 80
MAIN_SHARE = 34                 # % of the width of the summed image view (the frames get the rest)
PREFIX = "DynamicPET."
KM_PREFIX = "DynamicPET.Modeling."
HELPER_ATTRIBUTE = "DynamicPETModeling.Helper"
DEFAULT_ORIENTATION = "Axial"
FRAME_COLORMAP = "PET-DICOM"
SUMMED_COLORMAP = "Inverted grey"
MAP_COLORMAP = "PET-Rainbow2"
AUTO_PERCENTILE = 99.9
TIME_FIRST_FRAME, TIME_INJECTION, TIME_MANUAL = "first", "injection", "manual"
TIME_ZEROS = ((TIME_FIRST_FRAME, "Start of the first frame"), (TIME_INJECTION, "Injection time (from the PET)"),
              (TIME_MANUAL, "Injection at (s from the first frame):"))
INPUT_SEGMENT, INPUT_FILE = "segment", "file"
DEFAULT_MAP_THRESHOLD = 5.0     # % of the 99.9th percentile of the summed activity
MAP_CHUNK = 200000              # voxels per step of the parametric maps


class DynamicPETModeling(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        parent.title = "Dynamic PET Modeling"
        parent.categories = ["Nuclear Medicine.PET4D"]
        parent.dependencies = []
        parent.contributors = ["Burak Demir"]
        parent.helpText = (
            "Kinetic modelling of a dynamic PET series on the segments of a segmentation: 1TC, 2TC (reversible or "
            "irreversible), Patlak and Logan with a plasma input (image-derived blood pool segment or a plasma curve "
            "file), and SRTM, Logan (reference) and SUVR with a reference region. Preview fits the target segment, "
            "Fit all segments fills the results table; export the results and the curves as CSV, or make voxel-wise "
            "parametric maps. The functions can be called from Python through DynamicPETModelingLogic.")
        parent.acknowledgementText = (
            "Part of the DICOM Gated / Dynamic PET extension. SRTM: Lammertsma AA, Hume SP, NeuroImage 1996; basis "
            "functions: Gunn RN et al., NeuroImage 1997; Patlak CS et al., J Cereb Blood Flow Metab 1983; Logan J et "
            "al., J Cereb Blood Flow Metab 1990 and 1996.")


# ---------------------------------------------------------------------------
# Logic (no widgets: usable from other modules and scripts)
# ---------------------------------------------------------------------------
class Analysis:
    """A dynamic series, its frame timing relative to time zero and the TACs of the segments of a segmentation:
        sequence, segmentation       the nodes
        intervals                    dynamicviews.frameIntervals (s on the study time axis)
        timeZeroS                    time zero on that axis; frames: kinetics.Frames relative to it
        units                        units of the frames (Bq/ml, SUVbw, ...)
        segments                     {segment ID: {"name", "color" ('#rrggbb'), "values" (frame means), "voxels"}}
    """

    def __init__(self, sequence, segmentation, intervals, timeZeroS, frames, units, segments):
        self.sequence, self.segmentation = sequence, segmentation
        self.intervals, self.timeZeroS, self.frames = intervals, timeZeroS, frames
        self.units, self.segments = units, segments

    def curve(self, segmentId):
        try:
            return self.segments[segmentId]["values"]
        except KeyError:
            raise ValueError(f"Segment {segmentId!r} is not in the segmentation") from None

    def name(self, segmentId):
        return self.segments[segmentId]["name"] if segmentId in self.segments else str(segmentId)


class DynamicPETModelingLogic(ScriptedLoadableModuleLogic):
    """Kinetic modelling of the segments of a dynamic series. Options of the fits are those of
    kinetics.defaultOptions (weighting, parameters, tStar, autoTStar, maxError, k2ref, window)."""

    # --- series --------------------------------------------------------------------------------
    @staticmethod
    def isDynamicSequence(sequence):
        from DynamicPETEditor import DynamicPETEditorLogic
        return DynamicPETEditorLogic.isDynamicSequence(sequence) and sequence.GetNumberOfDataNodes() >= 2

    @staticmethod
    def intervals(sequence):
        from DynamicPETEditor import DynamicPETEditorLogic
        return DynamicPETEditorLogic.frameIntervals(sequence)

    @staticmethod
    def units(sequence):
        node = sequence.GetNthDataNode(0) if sequence.GetNumberOfDataNodes() else None
        return (sequence.GetAttribute(PREFIX + "Units") or (node.GetAttribute(PREFIX + "Units") if node else None)
                or "")

    @staticmethod
    def injectionTimeS(intervals):
        """Injection on the study time axis (s), None when the series does not say."""
        return intervals[0].get("injectionAt") if intervals else None

    @classmethod
    def timeZeroS(cls, intervals, mode=TIME_FIRST_FRAME, injectionOffsetS=0.0):
        """Time zero on the study time axis (s): the first frame's start, the injection time of the PET, or the
        first frame's start + injectionOffsetS (negative: the injection was before the first frame)."""
        first = intervals[0]["start"]
        if mode == TIME_INJECTION:
            injection = cls.injectionTimeS(intervals)
            if injection is None:
                raise ValueError("The series has no injection time; pick another time zero")
            return float(injection)
        if mode == TIME_MANUAL:
            return first + float(injectionOffsetS or 0.0)
        return first

    @staticmethod
    def summedVolume(sequence):
        nodeID = sequence.GetAttribute(PREFIX + "SummedVolumeID")
        node = slicer.mrmlScene.GetNodeByID(nodeID) if nodeID else None
        return node if node is not None and node.IsA("vtkMRMLScalarVolumeNode") else None

    @staticmethod
    def frameArray(sequence, index):
        return slicer.util.arrayFromVolume(sequence.GetNthDataNode(index))

    @staticmethod
    def _geometryKey(node):
        matrix = vtk.vtkMatrix4x4()
        node.GetIJKToRASMatrix(matrix)
        return (tuple(node.GetImageData().GetDimensions()),
                tuple(round(matrix.GetElement(r, c), 6) for r in range(3) for c in range(4)))

    @staticmethod
    def helperVolume(name, source):
        """A hidden volume (not saved with the scene) sharing the voxels and geometry of source."""
        node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", name)
        node.SetHideFromEditors(True)
        node.SetSaveWithScene(False)
        node.SetAttribute(HELPER_ATTRIBUTE, "1")
        node.SetAndObserveImageData(source.GetImageData())
        matrix = vtk.vtkMatrix4x4()
        source.GetIJKToRASMatrix(matrix)
        node.SetIJKToRASMatrix(matrix)
        return node

    # --- segments ------------------------------------------------------------------------------
    @staticmethod
    def segmentIds(segmentation):
        ids = vtk.vtkStringArray()
        segmentation.GetSegmentation().GetSegmentIDs(ids)
        return [ids.GetValue(i) for i in range(ids.GetNumberOfValues())]

    @classmethod
    def segmentInfo(cls, segmentation):
        """[(segment ID, name, '#rrggbb')] of a segmentation."""
        out = []
        for segmentId in cls.segmentIds(segmentation):
            segment = segmentation.GetSegmentation().GetSegment(segmentId)
            color = "#" + "".join(f"{int(round(max(0.0, min(1.0, c)) * 255)):02x}" for c in segment.GetColor())
            out.append((segmentId, segment.GetName(), color))
        return out

    def segmentIndexes(self, segmentation, referenceVolume, segmentIds=None):
        """{segment ID: flat voxel indexes} of the segments on the grid of referenceVolume."""
        out = {}
        labelNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode", "DynamicPETModeling mask")
        labelNode.SetHideFromEditors(True)
        try:
            segmentationsLogic = slicer.modules.segmentations.logic()
            for segmentId in segmentIds or self.segmentIds(segmentation):
                ids = vtk.vtkStringArray()
                ids.InsertNextValue(segmentId)
                if not segmentationsLogic.ExportSegmentsToLabelmapNode(segmentation, ids, labelNode, referenceVolume):
                    raise RuntimeError(f"Could not get segment {segmentId} on the PET grid")
                mask = slicer.util.arrayFromVolume(labelNode)
                out[segmentId] = np.flatnonzero(mask.ravel() > 0) if mask is not None else np.array([], int)
        finally:
            slicer.mrmlScene.RemoveNode(labelNode)
        return out

    def segmentCurves(self, sequence, segmentation, segmentIds=None, progress=None):
        """{segment ID: frame means} (NaN where a segment has no voxel) and {segment ID: voxel count}. Frames with
        another grid get the segments again on their own grid."""
        count = sequence.GetNumberOfDataNodes()
        segmentIds = list(segmentIds or self.segmentIds(segmentation))
        curves = {s: np.full(count, np.nan) for s in segmentIds}
        voxels = {s: 0 for s in segmentIds}
        cache = {}
        for f in range(count):
            node = sequence.GetNthDataNode(f)
            key = self._geometryKey(node)
            if key not in cache:
                helper = self.helperVolume("DynamicPETModeling geometry", node)
                try:
                    cache[key] = self.segmentIndexes(segmentation, helper, segmentIds)
                finally:
                    slicer.mrmlScene.RemoveNode(helper)
            flat = slicer.util.arrayFromVolume(node).ravel()
            for segmentId, indexes in cache[key].items():
                if indexes.size:
                    curves[segmentId][f] = float(np.mean(flat[indexes]))
                    voxels[segmentId] = max(voxels[segmentId], int(indexes.size))
            if progress is not None and progress(f + 1, count) is False:
                break
        return curves, voxels

    def prepare(self, sequence, segmentation, timeZero=TIME_FIRST_FRAME, injectionOffsetS=0.0, segmentIds=None,
                progress=None, curves=None):
        """Analysis of a series and a segmentation (TACs of its segments; pass curves {ID: frame means} to skip
        reading them again)."""
        if not self.isDynamicSequence(sequence):
            raise ValueError("Pick a dynamic PET sequence with at least two frames")
        if segmentation is None:
            raise ValueError("Pick a segmentation")
        intervals = self.intervals(sequence)
        zero = self.timeZeroS(intervals, timeZero, injectionOffsetS)
        frames = kin.Frames.fromIntervals(intervals, zero)
        voxels = {}
        if curves is None:
            curves, voxels = self.segmentCurves(sequence, segmentation, segmentIds, progress)
        segments = {}
        for segmentId, name, color in self.segmentInfo(segmentation):
            if segmentId in curves:
                segments[segmentId] = {"name": name, "color": color, "values": np.asarray(curves[segmentId]),
                                       "voxels": voxels.get(segmentId)}
        return Analysis(sequence, segmentation, intervals, zero, frames, self.units(sequence), segments)

    def withTimeZero(self, analysis, timeZero=TIME_FIRST_FRAME, injectionOffsetS=0.0):
        """The same analysis with another time zero (the TACs are kept)."""
        zero = self.timeZeroS(analysis.intervals, timeZero, injectionOffsetS)
        return Analysis(analysis.sequence, analysis.segmentation, analysis.intervals, zero,
                        kin.Frames.fromIntervals(analysis.intervals, zero), analysis.units, analysis.segments)

    # --- inputs --------------------------------------------------------------------------------
    @staticmethod
    def readPlasmaCsv(path, timeUnit="s"):
        """(times in minutes from time zero, values) of a plasma curve file (time, value columns)."""
        with open(path, encoding="utf-8-sig") as stream:
            return kin.readCurveCsv(stream.read(), timeUnit)

    @staticmethod
    def inputs(analysis, modelKey, referenceId=None, plasma=None):
        """(inputFunction, reference, blood) for kinetics.fit: plasma models take the plasma curve ((times min,
        values), from readPlasmaCsv) when given, else the reference segment as an image-derived input; the blood
        segment, when there is one, is also the whole blood of vB. Reference models take the reference segment."""
        spec = kin.model(modelKey)
        frames = analysis.frames
        segment = None
        if referenceId is not None:
            values = analysis.curve(referenceId)
            if not np.all(np.isfinite(values[frames.valid])):
                raise ValueError(f"Segment '{analysis.name(referenceId)}' is empty (no voxel on the PET grid)")
            segment = kin.FrameInput(frames, values)
        if spec.inputKind == "plasma":
            if plasma is not None:
                times, values = plasma
                return kin.SampledInput(frames, times, values), None, (segment.values if segment else None)
            if segment is None:
                raise ValueError(f"{spec.title} needs a blood pool segment or a plasma curve")
            return segment, None, None
        if segment is None:
            raise ValueError(f"{spec.title} needs a reference segment")
        return None, segment, None

    # --- fits ----------------------------------------------------------------------------------
    def fitSegment(self, analysis, modelKey, targetId, referenceId=None, options=None, plasma=None):
        """kinetics.fit of one segment (see kinetics.fit for the result)."""
        inputFunction, reference, blood = self.inputs(analysis, modelKey, referenceId, plasma)
        tissue = analysis.curve(targetId)
        if not np.any(np.isfinite(tissue)):
            raise ValueError(f"Segment '{analysis.name(targetId)}' is empty (no voxel on the PET grid)")
        return kin.fit(modelKey, analysis.frames, tissue, inputFunction, reference, blood, options)

    def fitSegments(self, analysis, modelKey, referenceId=None, options=None, plasma=None, segmentIds=None,
                    progress=None):
        """{segment ID: result} of every segment (but the reference) or of segmentIds; a segment that cannot be
        fitted gets {"error": text}."""
        inputFunction, reference, blood = self.inputs(analysis, modelKey, referenceId, plasma)
        ids = [s for s in (segmentIds or analysis.segments) if s != referenceId]
        results = {}
        for i, segmentId in enumerate(ids):
            tissue = analysis.curve(segmentId)
            try:
                if not np.any(np.isfinite(tissue)):
                    raise ValueError("empty segment")
                results[segmentId] = kin.fit(modelKey, analysis.frames, tissue, inputFunction, reference, blood,
                                             options)
            except Exception as error:
                logging.warning(f"Dynamic PET Modeling: {analysis.name(segmentId)}: {error}")
                results[segmentId] = {"model": modelKey, "error": str(error), "values": {}, "metrics": {},
                                      "warnings": [str(error)]}
            if progress is not None and progress(i + 1, len(ids), analysis.name(segmentId)) is False:
                break
        return results

    # --- outputs -------------------------------------------------------------------------------
    @staticmethod
    def resultRows(analysis, results):
        return [(analysis.name(s), r, (analysis.segments.get(s, {}).get("voxels") or "",)) for s, r in results.items()]

    def resultsTable(self, analysis, results, modelKey, name=None, tableNode=None):
        """A table node of the results (one row per segment): region, voxels, the model's values, the fit metrics."""
        name = name or f"{analysis.sequence.GetName()} {modelKey} results"
        if tableNode is None:
            tableNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTableNode", name)
        table = tableNode.GetTable()
        table.Initialize()
        columns = kin.resultColumns(modelKey)
        region = vtk.vtkStringArray()
        region.SetName("Region")
        voxelColumn = vtk.vtkIntArray()
        voxelColumn.SetName("Voxels")
        arrays = []
        for column in columns:
            array = vtk.vtkDoubleArray()
            array.SetName(kin.columnTitle(column))
            arrays.append(array)
        warnings = vtk.vtkStringArray()
        warnings.SetName("Warnings")
        for segmentId, result in results.items():
            region.InsertNextValue(analysis.name(segmentId))
            voxelColumn.InsertNextValue(int(analysis.segments.get(segmentId, {}).get("voxels") or 0))
            for array, value in zip(arrays, kin.resultRow(result)):
                array.InsertNextValue(float("nan") if value is None else float(value))
            warnings.InsertNextValue("; ".join(result.get("warnings") or []))
        for array in [region, voxelColumn] + arrays + [warnings]:
            table.AddColumn(array)
        tableNode.SetAttribute(KM_PREFIX + "Model", modelKey)
        tableNode.SetAttribute(KM_PREFIX + "Sequence", analysis.sequence.GetID())
        tableNode.SetAttribute(KM_PREFIX + "TimeZeroS", f"{analysis.timeZeroS:.6g}")
        tableNode.Modified()
        return tableNode

    def exportCsv(self, path, analysis, results, modelKey, tacs=True):
        """Results as CSV at path; with tacs, the TACs (and the fitted curves) next to it as <name>_TACs.csv.
        Returns the paths written."""
        with open(path, "w", encoding="utf-8", newline="") as stream:
            stream.write(kin.resultsCsv(self.resultRows(analysis, results), modelKey, extraColumns=("Voxels",)))
        written = [path]
        if tacs:
            curves = {seg["name"]: seg["values"] for seg in analysis.segments.values()}
            for segmentId, result in results.items():
                if result.get("fitted") is not None:
                    curves[f"{analysis.name(segmentId)} fit ({modelKey})"] = result["fitted"]
            tacPath = os.path.splitext(path)[0] + "_TACs.csv"
            with open(tacPath, "w", encoding="utf-8", newline="") as stream:
                stream.write(kin.tacCsv(analysis.frames, curves, analysis.units))
            written.append(tacPath)
        return written

    # --- parametric maps -----------------------------------------------------------------------
    def mapMask(self, analysis, thresholdPercent=DEFAULT_MAP_THRESHOLD, insideSegments=False):
        """Flat voxel indexes of the parametric maps: voxels whose duration weighted mean activity is at least
        thresholdPercent % of its 99.9th percentile (optionally only inside the segments)."""
        sequence, frames = analysis.sequence, analysis.frames
        total = None
        for f in range(sequence.GetNumberOfDataNodes()):
            if not frames.valid[f]:
                continue
            array = self.frameArray(sequence, f).astype(np.float32).ravel()
            total = array * frames.duration[f] if total is None else total + array * frames.duration[f]
        mean = total / float(frames.duration.sum())
        sample = mean[::max(1, mean.size // 400000)]
        upper = float(np.percentile(sample[np.isfinite(sample)], AUTO_PERCENTILE)) if sample.size else 0.0
        keep = np.isfinite(mean) & (mean >= upper * float(thresholdPercent) / 100.0) & (mean > 0)
        if insideSegments:
            first = sequence.GetNthDataNode(0)
            helper = self.helperVolume("DynamicPETModeling geometry", first)
            try:
                inside = np.zeros(mean.size, dtype=bool)
                for indexes in self.segmentIndexes(analysis.segmentation, helper).values():
                    inside[indexes] = True
            finally:
                slicer.mrmlScene.RemoveNode(helper)
            keep &= inside
        return np.flatnonzero(keep)

    def parametricMaps(self, analysis, modelKey, referenceId=None, options=None, plasma=None,
                       thresholdPercent=DEFAULT_MAP_THRESHOLD, insideSegments=False, progress=None, names=None):
        """Voxel-wise maps of a model (kinetics.VOXEL_MAPS) as new volumes on the grid of the first frame:
        {map name: volume node}. Every frame must have the grid of the first one."""
        if modelKey not in kin.VOXEL_MAPS:
            raise ValueError(f"{kin.model(modelKey).title} has no voxel-wise version")
        sequence = analysis.sequence
        first = sequence.GetNthDataNode(0)
        key = self._geometryKey(first)
        if any(self._geometryKey(sequence.GetNthDataNode(f)) != key for f in range(sequence.GetNumberOfDataNodes())):
            raise ValueError("The frames do not share one grid; resample or harden them first")
        inputFunction, reference, blood = self.inputs(analysis, modelKey, referenceId, plasma)
        indexes = self.mapMask(analysis, thresholdPercent, insideSegments)
        shape = slicer.util.arrayFromVolume(first).shape
        arrays = [self.frameArray(sequence, f).ravel() for f in range(sequence.GetNumberOfDataNodes())]
        outputs = {name: np.zeros(int(np.prod(shape)), dtype=np.float32) for name in kin.VOXEL_MAPS[modelKey]}
        chunks = max(1, int(math.ceil(indexes.size / float(MAP_CHUNK))))
        for c in range(chunks):
            part = indexes[c * MAP_CHUNK:(c + 1) * MAP_CHUNK]
            data = np.stack([a[part] for a in arrays]).astype(float)
            values = kin.parametricMaps(modelKey, analysis.frames, data, inputFunction, reference, blood, options)
            for name, array in values.items():
                outputs[name][part] = array
            if progress is not None and progress(c + 1, chunks) is False:
                return {}
        nodes = {}
        matrix = vtk.vtkMatrix4x4()
        first.GetIJKToRASMatrix(matrix)
        for name, flat in outputs.items():
            nodeName = (names or {}).get(name) or f"{sequence.GetName()} {modelKey} {name}"
            node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", slicer.mrmlScene.GenerateUniqueName(
                nodeName))
            slicer.util.updateVolumeFromArray(node, flat.reshape(shape))
            node.SetIJKToRASMatrix(matrix)
            node.SetAttribute(KM_PREFIX + "Model", modelKey)
            node.SetAttribute(KM_PREFIX + "Map", name)
            node.SetAttribute(KM_PREFIX + "Sequence", sequence.GetID())
            node.SetAttribute(KM_PREFIX + "TimeZeroS", f"{analysis.timeZeroS:.6g}")
            if referenceId is not None:
                node.SetAttribute(KM_PREFIX + "Reference", analysis.name(referenceId))
            opts = options or {}
            if kin.model(modelKey).method == "graphical":
                node.SetAttribute(KM_PREFIX + "TStarMin", f"{float(opts.get('tStar', kin.DEFAULT_TSTAR_MIN)):g}")
            node.CreateDefaultDisplayNodes()
            display = node.GetDisplayNode()
            display.SetAndObserveColorNodeID(colormaps.colorNodeForLabel(MAP_COLORMAP).GetID())
            values = flat[indexes] if indexes.size else flat
            low, high = (np.percentile(values, [1, 99]) if values.size else (0.0, 1.0))
            display.AutoWindowLevelOff()
            display.SetWindowLevelMinMax(float(min(0.0, low)), float(high if high > low else low + 1.0))
            nodes[name] = node
        return nodes


# ---------------------------------------------------------------------------
# Session: layout, views and the curve / results dock
# ---------------------------------------------------------------------------
def _percentile(array):
    sample = array.ravel()
    sample = sample[::4] if sample.size > 400000 else sample
    sample = sample[np.isfinite(sample)]
    return float(np.percentile(sample, AUTO_PERCENTILE)) if sample.size else 1.0


class ModelingSession:
    """The modelling layout of one series: the summed image (or the middle frame) with the target and reference
    segments in the large view, the frames in the grid, the TAC plot and the results table in a dock."""

    def __init__(self, sequence, maxViews=64, orientation=DEFAULT_ORIENTATION, frameColor=FRAME_COLORMAP,
                 summedColor=SUMMED_COLORMAP, frameWindows=False):
        self.sequence = sequence
        self.maxViews = maxViews
        self.orientation = orientation
        self.frameColor = frameColor
        self.summedColor = summedColor
        self.frameWindows = frameWindows
        self.intervals = []
        self.viewFrames = []
        self.frameVolumes = []
        self.mainVolume = None
        self.mainText = ""
        self.mapVolume = None
        self.overlays = {}
        self.observers = []
        self.annotatedViews = []
        self.previousLayout = None
        self.dock = None
        self.plot = None
        self.table = None
        self.graphicalRadio = None
        self.tacRadio = None
        self.onRowClicked = None
        self.segmentation = None
        self.segmentState = None        # (display node, view IDs, {segment ID: visible}) to restore
        self.uppers = []

    # --- open / close ----------------------------------------------------------------------------
    def open(self):
        from DynamicPETEditor import DynamicPETEditorLogic
        self.intervals = DynamicPETEditorLogic.frameIntervals(self.sequence)
        self.viewFrames = DynamicPETEditorLogic.shownFrames(len(self.intervals), self.maxViews)
        arrays = [slicer.util.arrayFromVolume(self.sequence.GetNthDataNode(i)) for i in self.viewFrames]
        self.uppers = [_percentile(a) for a in arrays]
        self._makeVolumes()
        self._setLayout()
        self._makeDock()
        self._setupViews()

    def close(self):
        self.setSegmentation(None)
        for node, tag in self.observers:
            node.RemoveObserver(tag)
        self.observers = []
        for sliceView, renderer, actor in self.overlays.values():
            renderer.RemoveActor2D(actor)
            sliceView.scheduleRender()
        self.overlays = {}
        for sliceView in self.annotatedViews:
            sliceView.cornerAnnotation().SetVisibility(True)
            sliceView.scheduleRender()
        self.annotatedViews = []
        layoutManager = slicer.app.layoutManager()
        if layoutManager.layout == LAYOUT_ID:
            self._link(False)
        if self.dock is not None:
            slicer.util.mainWindow().removeDockWidget(self.dock)
            self.dock.deleteLater()
            self.dock = self.plot = self.table = None
        if self.previousLayout is not None and layoutManager.layout == LAYOUT_ID:
            layoutManager.setLayout(self.previousLayout)
        for node in self.frameVolumes + [self.mainVolume]:
            if node is not None and slicer.mrmlScene.IsNodePresent(node):
                slicer.mrmlScene.RemoveNode(node)
        self.frameVolumes, self.mainVolume = [], None

    # --- volumes --------------------------------------------------------------------------------
    def _makeVolumes(self):
        for index in self.viewFrames:
            volume = DynamicPETModelingLogic.helperVolume(f"{self.sequence.GetName()} KM view F{index + 1}",
                                                          self.sequence.GetNthDataNode(index))
            volume.CreateDefaultDisplayNodes()
            volume.GetDisplayNode().SetInterpolate(True)
            self.frameVolumes.append(volume)
        summed = DynamicPETModelingLogic.summedVolume(self.sequence)
        if summed is not None:
            source, self.mainText = summed, "Summed"
        else:
            middle = len(self.intervals) // 2
            source = self.sequence.GetNthDataNode(middle)
            self.mainText = f"Middle frame {dynamicviews.frameLabel(self.intervals[middle])}"
        self.mainVolume = DynamicPETModelingLogic.helperVolume(f"{self.sequence.GetName()} KM main", source)
        self.mainVolume.CreateDefaultDisplayNodes()
        display = self.mainVolume.GetDisplayNode()
        display.SetInterpolate(True)
        display.AutoWindowLevelOff()
        display.SetWindowLevelMinMax(0.0, _percentile(slicer.util.arrayFromVolume(self.mainVolume)))
        self.setColors(self.frameColor, self.summedColor)

    def setColors(self, frameColor=None, summedColor=None):
        if frameColor:
            self.frameColor = frameColor
        if summedColor:
            self.summedColor = summedColor
        frameID = colormaps.colorNodeForLabel(self.frameColor).GetID()
        for volume in self.frameVolumes:
            volume.GetDisplayNode().SetAndObserveColorNodeID(frameID)
        if self.mainVolume is not None:
            self.mainVolume.GetDisplayNode().SetAndObserveColorNodeID(
                colormaps.colorNodeForLabel(self.summedColor).GetID())
        self.applyFrameWindows()

    def setFrameWindows(self, perFrame):
        self.frameWindows = bool(perFrame)
        self.applyFrameWindows()

    def applyFrameWindows(self):
        common = max(self.uppers or [1.0])
        for volume, upper in zip(self.frameVolumes, self.uppers):
            display = volume.GetDisplayNode()
            display.AutoWindowLevelOff()
            display.SetWindowLevelMinMax(0.0, (upper if self.frameWindows else common) or 1.0)

    # --- layout and views -------------------------------------------------------------------------
    def _setLayout(self):
        layoutManager = slicer.app.layoutManager()
        layoutNode = layoutManager.layoutLogic().GetLayoutNode()
        xml = dynamicviews.layoutXml(len(self.viewFrames), mainShare=MAIN_SHARE, orientation=self.orientation,
                                     mainView=MAIN_VIEW, framePrefix=FRAME_VIEW_PREFIX, emptyPrefix=EMPTY_VIEW_PREFIX,
                                     viewGroup=VIEW_GROUP, mainLabel="S")
        current = layoutManager.layout
        if current != LAYOUT_ID:
            self.previousLayout = current
        if layoutNode.IsLayoutDescription(LAYOUT_ID):
            if current == LAYOUT_ID:
                layoutManager.setLayout(self.previousLayout or slicer.vtkMRMLLayoutNode.SlicerLayoutFourUpView)
            layoutNode.SetLayoutDescription(LAYOUT_ID, xml)
        else:
            layoutNode.AddLayoutDescription(LAYOUT_ID, xml)
        layoutManager.setLayout(LAYOUT_ID)
        slicer.app.processEvents()

    @staticmethod
    def _widget(name):
        return slicer.app.layoutManager().sliceWidget(name)

    def _frameWidgets(self):
        return [(self._widget(dynamicviews.frameViewName(v, FRAME_VIEW_PREFIX)), volume)
                for v, volume in enumerate(self.frameVolumes)]

    def _allWidgets(self):
        widgets = [self._widget(MAIN_VIEW)] + [w for w, _ in self._frameWidgets()]
        return [w for w in widgets if w is not None]

    def _link(self, linked):
        for widget in self._allWidgets():
            composite = widget.mrmlSliceCompositeNode()
            composite.SetLinkedControl(linked)
            composite.SetHotLinkedControl(linked)

    def mainSliceNodeID(self):
        widget = self._widget(MAIN_VIEW)
        return widget.mrmlSliceNode().GetID() if widget is not None else None

    def _setupViews(self):
        self._link(False)
        try:
            main = self._widget(MAIN_VIEW)
            if main is not None:
                self._addOverlay(MAIN_VIEW, main, 14)
                self.observers.append(guardViewVolumes(main, self._mainLayers))
            for v, (widget, volume) in enumerate(self._frameWidgets()):
                if widget is None:
                    continue
                widget.sliceController().setVisible(False)
                self._addOverlay(dynamicviews.frameViewName(v, FRAME_VIEW_PREFIX), widget, 10)
                self.observers.append(guardViewVolumes(widget, lambda volume=volume: (volume.GetID(), None, None)))
            for e in range(dynamicviews.emptyViewCount(len(self.frameVolumes))):
                widget = self._widget(dynamicviews.emptyViewName(e, EMPTY_VIEW_PREFIX))
                if widget is not None:
                    widget.sliceController().setVisible(False)
                    self.observers.append(guardViewVolumes(widget, lambda: (None, None, None)))
                    self.annotatedViews.append(widget.sliceView())
            self.annotatedViews += [w.sliceView() for w in self._allWidgets()]
            for sliceView in self.annotatedViews:
                sliceView.cornerAnnotation().SetVisibility(False)
            self.setOrientation(self.orientation, link=False)
        finally:
            self._link(True)
        self._updateOverlays()

    def _mainLayers(self):
        return (self.mainVolume.GetID() if self.mainVolume else None,
                self.mapVolume.GetID() if self.mapVolume is not None and slicer.mrmlScene.IsNodePresent(
                    self.mapVolume) else None, None)

    def setMapVolume(self, volume, opacity=0.6):
        """A parametric map over the summed image in the large view (None: none)."""
        self.mapVolume = volume
        main = self._widget(MAIN_VIEW)
        if main is not None:
            self._link(False)       # a linked view would pass the map on to the frame views
            try:
                composite = main.mrmlSliceCompositeNode()
                composite.SetForegroundVolumeID(volume.GetID() if volume is not None else None)
                composite.SetForegroundOpacity(opacity)
            finally:
                self._link(True)
        self._updateOverlays()

    def setOrientation(self, orientation, link=True):
        self.orientation = orientation
        if link:
            self._link(False)
        try:
            for widget in self._allWidgets():
                widget.mrmlSliceNode().SetOrientation(orientation)
                widget.sliceLogic().FitSliceToAll()
        finally:
            if link:
                self._link(True)

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
        renderer.AddActor2D(actor)
        self.overlays[name] = (sliceView, renderer, actor)

    def _updateOverlays(self):
        for v, index in enumerate(self.viewFrames):
            overlay = self.overlays.get(dynamicviews.frameViewName(v, FRAME_VIEW_PREFIX))
            if overlay is not None:
                overlay[2].SetInput(dynamicviews.frameLabel(self.intervals[index]))
                overlay[0].scheduleRender()
        overlay = self.overlays.get(MAIN_VIEW)
        if overlay is not None:
            text = self.mainText
            if self.mapVolume is not None:
                text += f"\n{self.mapVolume.GetName()}"
            overlay[2].SetInput(text)
            overlay[0].scheduleRender()

    # --- segments in the large view -----------------------------------------------------------
    def setSegmentation(self, segmentation, shownIds=()):
        """Show the segments shownIds (target, reference) of the segmentation in the large view only; the
        segmentation's own view and segment visibility are restored on close or when another one is picked."""
        if self.segmentState is not None:
            display, viewIDs, visible = self.segmentState
            if slicer.mrmlScene.IsNodePresent(display):
                display.RemoveAllViewNodeIDs()
                for viewID in viewIDs:
                    display.AddViewNodeID(viewID)
                for segmentId, shown in visible.items():
                    display.SetSegmentVisibility(segmentId, shown)
            self.segmentState = None
        self.segmentation = segmentation
        if segmentation is None:
            return
        segmentation.CreateDefaultDisplayNodes()
        display = segmentation.GetDisplayNode()
        if display is None:
            return
        viewIDs = [display.GetNthViewNodeID(i) for i in range(display.GetNumberOfViewNodeIDs())]
        ids = DynamicPETModelingLogic.segmentIds(segmentation)
        self.segmentState = (display, viewIDs, {s: bool(display.GetSegmentVisibility(s)) for s in ids})
        mainID = self.mainSliceNodeID()
        display.RemoveAllViewNodeIDs()
        if mainID:
            display.AddViewNodeID(mainID)
        self.showSegments(shownIds)

    def showSegments(self, shownIds):
        if self.segmentation is None or self.segmentation.GetDisplayNode() is None:
            return
        display = self.segmentation.GetDisplayNode()
        shown = {s for s in shownIds if s}
        for segmentId in DynamicPETModelingLogic.segmentIds(self.segmentation):
            display.SetSegmentVisibility(segmentId, segmentId in shown)

    # --- dock -------------------------------------------------------------------------------------
    def _makeDock(self):
        mainWindow = slicer.util.mainWindow()
        self.dock = qt.QDockWidget("Dynamic PET Modeling curves and results")
        self.dock.setObjectName("DynamicPETModelingDock")
        self.dock.setFeatures(qt.QDockWidget.DockWidgetMovable | qt.QDockWidget.DockWidgetFloatable)
        splitter = qt.QSplitter(qt.Qt.Horizontal)
        left = qt.QWidget()
        leftLayout = qt.QVBoxLayout(left)
        leftLayout.setContentsMargins(2, 2, 2, 2)
        toolbar = qt.QHBoxLayout()
        self.tacRadio = qt.QRadioButton("Time-activity curves")
        self.tacRadio.checked = True
        self.graphicalRadio = qt.QRadioButton("Patlak / Logan plot")
        self.graphicalRadio.enabled = False
        self.graphicalRadio.toolTip = "The linear plot of the target (Patlak / Logan methods, after a fit)"
        resetButton = qt.QToolButton()
        resetButton.text = "Reset zoom"
        resetButton.toolTip = "Wheel: zoom time, Ctrl + wheel: zoom values, double click: reset"
        toolbar.addWidget(self.tacRadio)
        toolbar.addWidget(self.graphicalRadio)
        toolbar.addStretch(1)
        toolbar.addWidget(resetButton)
        leftLayout.addLayout(toolbar)
        self.plot = TacPlot()
        leftLayout.addWidget(self.plot, 1)
        self.table = qt.QTableWidget()
        self.table.setEditTriggers(qt.QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(qt.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(qt.QAbstractItemView.SingleSelection)
        self.table.toolTip = "Results of the fits; click a row to make that segment the target"
        self.table.verticalHeader().setVisible(False)
        splitter.addWidget(left)
        splitter.addWidget(self.table)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        self.dock.setWidget(splitter)
        self.tacRadio.connect("toggled(bool)", lambda checked: self.plot.showGraphical(not checked))
        resetButton.connect("clicked()", self.plot.resetZoom)
        self.table.connect("cellClicked(int,int)", self._onCellClicked)
        mainWindow.addDockWidget(qt.Qt.BottomDockWidgetArea, self.dock)
        height = max(220, int(mainWindow.height / 2))
        try:
            mainWindow.resizeDocks([self.dock], [height], qt.Qt.Vertical)
        except Exception:
            self.dock.setMinimumHeight(height)

    def _onCellClicked(self, row, column):
        item = self.table.item(row, 0)
        if item is not None and self.onRowClicked:
            self.onRowClicked(item.data(qt.Qt.UserRole))

    def setTable(self, header, rows, boldRow=None):
        """rows: (segment ID, [texts], color). boldRow: segment ID of the target."""
        self.table.clear()
        self.table.setColumnCount(len(header))
        self.table.setRowCount(len(rows))
        self.table.setHorizontalHeaderLabels(header)
        for r, (segmentId, texts, color, role) in enumerate(rows):
            for c, text in enumerate(texts):
                item = qt.QTableWidgetItem(text)
                if c == 0:
                    item.setData(qt.Qt.UserRole, segmentId)
                    pixmap = qt.QPixmap(10, 10)
                    pixmap.fill(qt.QColor(color))
                    item.setIcon(qt.QIcon(pixmap))
                else:
                    item.setTextAlignment(qt.Qt.AlignRight | qt.Qt.AlignVCenter)
                if segmentId == boldRow or role:
                    font = item.font()
                    font.setBold(segmentId == boldRow)
                    font.setItalic(bool(role) and segmentId != boldRow)
                    item.setFont(font)
                self.table.setItem(r, c, item)
        self.table.resizeColumnsToContents()


# ---------------------------------------------------------------------------
# Widget
# ---------------------------------------------------------------------------
class DynamicPETModelingWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    def __init__(self, parent=None):
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)
        self.logic = DynamicPETModelingLogic()
        self.session = None
        self.analysis = None              # TACs of the segments (time zero applied)
        self.results = {}                 # {segment ID: result} of the current model and settings
        self.resultsModel = None
        self.observedSegmentation = None
        self.parameterRows = {}           # model key -> parameter list shown in the table
        self._updating = False
        self._computing = False           # reading the curves (the segmentation may convert representations)
        self._segmentsChanged = False
        self.segmentTimer = qt.QTimer()
        self.segmentTimer.singleShot = True
        self.segmentTimer.interval = 600
        self.segmentTimer.connect("timeout()", self.onSegmentsEdited)

        # --- inputs
        inputBox = qt.QGroupBox("Inputs")
        form = qt.QFormLayout(inputBox)
        self.sequenceSelector = self._nodeSelector(["vtkMRMLSequenceNode"],
                                                   "Dynamic PET sequence (as loaded from DICOM: '<series> dynamic "
                                                   "Bq/ml')")
        form.addRow("Dynamic series:", self.sequenceSelector)
        self.segmentationSelector = self._nodeSelector(["vtkMRMLSegmentationNode"],
                                                       "Segmentation with the regions (target, reference / blood pool)")
        form.addRow("Segmentation:", self.segmentationSelector)
        self.targetCombo = qt.QComboBox()
        self.targetCombo.toolTip = "Segment shown and fitted by Preview (e.g. a region rich in receptors)"
        form.addRow("Target segment:", self.targetCombo)
        self.referenceLabel = qt.QLabel("Reference segment:")
        self.referenceCombo = qt.QComboBox()
        form.addRow(self.referenceLabel, self.referenceCombo)
        self.inputCombo = qt.QComboBox()
        self.inputCombo.addItem("Blood pool segment (image-derived)", INPUT_SEGMENT)
        self.inputCombo.addItem("Plasma curve from a file", INPUT_FILE)
        self.inputCombo.toolTip = ("Input function of the plasma input models: the mean of the blood pool segment "
                                   "(whole blood, no plasma correction), or a plasma curve read from a CSV file")
        self.inputLabel = qt.QLabel("Input function:")
        form.addRow(self.inputLabel, self.inputCombo)
        self.plasmaRow = qt.QWidget()
        plasmaLayout = qt.QHBoxLayout(self.plasmaRow)
        plasmaLayout.setContentsMargins(0, 0, 0, 0)
        self.plasmaPath = ctk.ctkPathLineEdit()
        self.plasmaPath.filters = ctk.ctkPathLineEdit.Files
        self.plasmaPath.nameFilters = ["CSV / text files (*.csv *.tsv *.txt)", "All files (*)"]
        self.plasmaPath.toolTip = ("Two columns: time, plasma activity (same units as the images); a header line is "
                                   "skipped. The times are from time zero (the time zero chosen below).")
        self.plasmaUnitCombo = qt.QComboBox()
        self.plasmaUnitCombo.addItem("s", "s")
        self.plasmaUnitCombo.addItem("min", "min")
        self.plasmaUnitCombo.toolTip = "Unit of the times in the file (a header naming min or s wins)"
        plasmaLayout.addWidget(self.plasmaPath, 1)
        plasmaLayout.addWidget(self.plasmaUnitCombo)
        self.plasmaFileLabel = qt.QLabel("Plasma file:")
        form.addRow(self.plasmaFileLabel, self.plasmaRow)
        timeRow = qt.QHBoxLayout()
        self.timeZeroCombo = qt.QComboBox()
        for key, title in TIME_ZEROS:
            self.timeZeroCombo.addItem(title, key)
        self.timeZeroCombo.toolTip = ("Time zero of the models (t = 0 of the curves): the start of the first frame "
                                      "(default), the injection time in the PET tags, or an injection time you enter "
                                      "in seconds from the start of the first frame (negative: before the image)")
        self.injectionSpin = qt.QDoubleSpinBox()
        self.injectionSpin.setRange(-36000.0, 36000.0)
        self.injectionSpin.decimals = 1
        self.injectionSpin.suffix = " s"
        self.injectionSpin.toolTip = "Injection time from the start of the first frame (negative: before the image)"
        timeRow.addWidget(self.timeZeroCombo, 1)
        timeRow.addWidget(self.injectionSpin)
        form.addRow("Time zero:", timeRow)
        self.infoLabel = qt.QLabel()
        self.infoLabel.wordWrap = True
        form.addRow(self.infoLabel)
        self.layout.addWidget(inputBox)

        # --- model
        modelBox = qt.QGroupBox("Model")
        modelForm = qt.QFormLayout(modelBox)
        self.modelCombo = qt.QComboBox()
        for key in kin.MODEL_ORDER:
            self.modelCombo.addItem(kin.MODELS[key].title, key)
        modelForm.addRow("Method:", self.modelCombo)
        self.modelDescription = qt.QLabel()
        self.modelDescription.wordWrap = True
        modelForm.addRow(self.modelDescription)
        self.weightingCombo = qt.QComboBox()
        for key, title in kin.WEIGHTINGS:
            self.weightingCombo.addItem(title, key)
        self.weightingCombo.toolTip = "Weights of the frames in the fit (default: the frame duration)"
        self.weightingLabel = qt.QLabel("Weighting:")
        modelForm.addRow(self.weightingLabel, self.weightingCombo)

        # parameters of the fitted models
        self.parameterTable = qt.QTableWidget()
        self.parameterTable.setColumnCount(5)
        self.parameterTable.setHorizontalHeaderLabels(["Parameter", "Initial", "Min", "Max", "Unit"])
        self.parameterTable.verticalHeader().setVisible(False)
        self.parameterTable.toolTip = ("Initial values and bounds of the fit. Untick a parameter to keep it fixed at "
                                       "its initial value (e.g. vB = 0, or the input delay, which is fixed at 0 "
                                       "unless ticked).")
        self.parameterTable.setMinimumHeight(150)
        self.parameterReset = qt.QPushButton("Default parameters")
        self.parameterBox = qt.QWidget()
        parameterLayout = qt.QVBoxLayout(self.parameterBox)
        parameterLayout.setContentsMargins(0, 0, 0, 0)
        parameterLayout.addWidget(self.parameterTable)
        parameterLayout.addWidget(self.parameterReset)
        modelForm.addRow(self.parameterBox)

        # graphical methods
        self.graphicalBox = qt.QWidget()
        graphicalForm = qt.QFormLayout(self.graphicalBox)
        graphicalForm.setContentsMargins(0, 0, 0, 0)
        tStarRow = qt.QHBoxLayout()
        self.tStarSpin = qt.QDoubleSpinBox()
        self.tStarSpin.setRange(0.0, 1000.0)
        self.tStarSpin.decimals = 1
        self.tStarSpin.suffix = " min"
        self.tStarSpin.value = kin.DEFAULT_TSTAR_MIN
        self.tStarSpin.toolTip = "The frames whose middle is at or after t* are used for the line"
        self.autoTStarCheck = qt.QCheckBox("Automatic")
        self.autoTStarCheck.toolTip = ("Earliest t* from which the line fits every point within the maximum error "
                                       "(the voxel-wise maps use the t* found for the target)")
        tStarRow.addWidget(self.tStarSpin, 1)
        tStarRow.addWidget(self.autoTStarCheck)
        graphicalForm.addRow("t*:", tStarRow)
        self.maxErrorSpin = qt.QDoubleSpinBox()
        self.maxErrorSpin.setRange(0.5, 50.0)
        self.maxErrorSpin.decimals = 1
        self.maxErrorSpin.suffix = " %"
        self.maxErrorSpin.value = kin.DEFAULT_MAX_ERROR
        graphicalForm.addRow("Max error (automatic t*):", self.maxErrorSpin)
        self.k2refSpin = qt.QDoubleSpinBox()
        self.k2refSpin.setRange(0.0, 5.0)
        self.k2refSpin.decimals = 4
        self.k2refSpin.singleStep = 0.01
        self.k2refSpin.suffix = " 1/min"
        self.k2refSpin.specialValueText = "not used"
        self.k2refSpin.toolTip = "k2' of the reference region (Logan 1996); 0: the term is left out"
        self.k2refLabel = qt.QLabel("Reference k2':")
        graphicalForm.addRow(self.k2refLabel, self.k2refSpin)
        modelForm.addRow(self.graphicalBox)

        # SUVR window
        self.windowBox = qt.QWidget()
        windowForm = qt.QHBoxLayout(self.windowBox)
        windowForm.setContentsMargins(0, 0, 0, 0)
        self.windowStartSpin = qt.QDoubleSpinBox()
        self.windowEndSpin = qt.QDoubleSpinBox()
        for spin in (self.windowStartSpin, self.windowEndSpin):
            spin.setRange(0.0, 10000.0)
            spin.decimals = 1
            spin.suffix = " min"
        windowForm.addWidget(qt.QLabel("Time window:"))
        windowForm.addWidget(self.windowStartSpin, 1)
        windowForm.addWidget(qt.QLabel("-"))
        windowForm.addWidget(self.windowEndSpin, 1)
        self.windowBox.toolTip = "Time window of the SUVR (min from time zero); default the last 30 min"
        modelForm.addRow(self.windowBox)

        buttons = qt.QHBoxLayout()
        self.previewButton = qt.QPushButton("Preview (target)")
        self.previewButton.toolTip = "Fit the target segment only and show its fitted curve"
        self.fitAllButton = qt.QPushButton("Fit all segments")
        self.fitAllButton.toolTip = "Fit every segment (but the reference) into the results table"
        buttons.addWidget(self.previewButton)
        buttons.addWidget(self.fitAllButton)
        modelForm.addRow(buttons)
        exportRow = qt.QHBoxLayout()
        self.tableCheck = qt.QCheckBox("Results table in the scene")
        self.tableCheck.checked = True
        self.tableCheck.toolTip = "Fit all segments also makes (or updates) a table node with the results"
        self.exportButton = qt.QPushButton("Export CSV...")
        self.exportButton.toolTip = "Save the results as CSV, and the curves next to it (<name>_TACs.csv)"
        exportRow.addWidget(self.tableCheck)
        exportRow.addStretch(1)
        exportRow.addWidget(self.exportButton)
        modelForm.addRow(exportRow)
        self.resultLabel = qt.QLabel()
        self.resultLabel.wordWrap = True
        modelForm.addRow(self.resultLabel)
        self.layout.addWidget(modelBox)

        # --- voxel-wise
        mapBox = ctk.ctkCollapsibleButton()
        mapBox.text = "Voxel-wise calculation"
        mapBox.collapsed = False
        mapForm = qt.QFormLayout(mapBox)
        self.mapThresholdSpin = qt.QDoubleSpinBox()
        self.mapThresholdSpin.setRange(0.0, 100.0)
        self.mapThresholdSpin.decimals = 1
        self.mapThresholdSpin.suffix = " %"
        self.mapThresholdSpin.value = DEFAULT_MAP_THRESHOLD
        self.mapThresholdSpin.toolTip = ("Voxels whose mean activity is below this % of the 99.9th percentile are "
                                         "left at 0")
        mapForm.addRow("Background threshold:", self.mapThresholdSpin)
        self.insideCheck = qt.QCheckBox("Only inside the segments")
        mapForm.addRow(self.insideCheck)
        self.mapButton = qt.QPushButton("Compute parametric maps")
        mapForm.addRow(self.mapButton)
        self.mapLabel = qt.QLabel()
        self.mapLabel.wordWrap = True
        mapForm.addRow(self.mapLabel)
        self.layout.addWidget(mapBox)

        # --- view
        viewBox = ctk.ctkCollapsibleButton()
        viewBox.text = "View"
        viewBox.collapsed = True
        view = qt.QFormLayout(viewBox)
        self.orientationCombo = qt.QComboBox()
        for name in dynamicviews.ORIENTATIONS:
            self.orientationCombo.addItem(name)
        self.orientationCombo.setCurrentIndex(self.orientationCombo.findText(DEFAULT_ORIENTATION))
        view.addRow("Orientation:", self.orientationCombo)
        self.frameColorCombo = qt.QComboBox()
        self.summedColorCombo = qt.QComboBox()
        for combo, default in ((self.frameColorCombo, FRAME_COLORMAP), (self.summedColorCombo, SUMMED_COLORMAP)):
            for label in colormaps.LABELS:
                combo.addItem(label)
            combo.setCurrentIndex(max(0, combo.findText(default)))
        view.addRow("Frame colors:", self.frameColorCombo)
        view.addRow("Summed image colors:", self.summedColorCombo)
        self.frameWindowCheck = qt.QCheckBox("Each frame its own window (0 - 99.9th percentile)")
        view.addRow(self.frameWindowCheck)
        self.maxViewsSpin = qt.QSpinBox()
        self.maxViewsSpin.setRange(4, 100)
        self.maxViewsSpin.value = 64
        view.addRow("Frame views (max):", self.maxViewsSpin)
        closeRow = qt.QHBoxLayout()
        self.openButton = qt.QPushButton("Open layout")
        self.closeButton = qt.QPushButton("Close layout")
        closeRow.addWidget(self.openButton)
        closeRow.addWidget(self.closeButton)
        view.addRow(closeRow)
        self.layout.addWidget(viewBox)
        self.layout.addStretch(1)

        # --- connections
        self.sequenceSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onSequenceChanged)
        self.segmentationSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onSegmentationChanged)
        self.targetCombo.connect("currentIndexChanged(int)", self.onTargetChanged)
        self.referenceCombo.connect("currentIndexChanged(int)", self.onReferenceChanged)
        self.inputCombo.connect("currentIndexChanged(int)", self.onSettingsChanged)
        self.plasmaPath.connect("currentPathChanged(QString)", self.onSettingsChanged)
        self.plasmaUnitCombo.connect("currentIndexChanged(int)", self.onSettingsChanged)
        self.timeZeroCombo.connect("currentIndexChanged(int)", self.onTimeZeroChanged)
        self.injectionSpin.connect("valueChanged(double)", self.onTimeZeroChanged)
        self.modelCombo.connect("currentIndexChanged(int)", self.onModelChanged)
        self.weightingCombo.connect("currentIndexChanged(int)", self.onSettingsChanged)
        self.parameterTable.connect("itemChanged(QTableWidgetItem*)", self.onParameterEdited)
        self.parameterReset.connect("clicked()", self.onParameterReset)
        self.tStarSpin.connect("valueChanged(double)", self.onSettingsChanged)
        self.autoTStarCheck.connect("toggled(bool)", self.onSettingsChanged)
        self.maxErrorSpin.connect("valueChanged(double)", self.onSettingsChanged)
        self.k2refSpin.connect("valueChanged(double)", self.onSettingsChanged)
        self.windowStartSpin.connect("valueChanged(double)", self.onSettingsChanged)
        self.windowEndSpin.connect("valueChanged(double)", self.onSettingsChanged)
        self.previewButton.connect("clicked()", self.onPreview)
        self.fitAllButton.connect("clicked()", self.onFitAll)
        self.exportButton.connect("clicked()", self.onExport)
        self.mapButton.connect("clicked()", self.onMaps)
        self.orientationCombo.connect("currentIndexChanged(int)", self.onOrientationChanged)
        self.frameColorCombo.connect("currentIndexChanged(int)", self.onColorsChanged)
        self.summedColorCombo.connect("currentIndexChanged(int)", self.onColorsChanged)
        self.frameWindowCheck.connect("toggled(bool)", self.onFrameWindowToggled)
        self.openButton.connect("clicked()", self.onOpen)
        self.closeButton.connect("clicked()", self.onClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.NodeAboutToBeRemovedEvent, self.onNodeRemoved)
        self.onModelChanged()
        self.onTimeZeroChanged()
        self._updateButtons()

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

    # --- state ---------------------------------------------------------------------------------
    def modelKey(self):
        return self.modelCombo.currentData or kin.MODEL_ORDER[0]

    def targetId(self):
        return self.targetCombo.currentData

    def referenceId(self):
        return self.referenceCombo.currentData

    def usesPlasmaFile(self):
        return kin.model(self.modelKey()).inputKind == "plasma" and self.inputCombo.currentData == INPUT_FILE

    def options(self):
        key = self.modelKey()
        opts = kin.defaultOptions()
        opts["weighting"] = self.weightingCombo.currentData
        opts["parameters"] = self._parametersFromTable(key)
        opts["tStar"] = self.tStarSpin.value
        opts["autoTStar"] = self.autoTStarCheck.checked
        opts["maxError"] = self.maxErrorSpin.value
        opts["k2ref"] = self.k2refSpin.value or None
        if self.windowEndSpin.value > self.windowStartSpin.value:
            opts["window"] = (self.windowStartSpin.value, self.windowEndSpin.value)
        return opts

    def plasma(self):
        """(times min, values) of the plasma file, or None (blood pool segment). Raises on a bad file."""
        if not self.usesPlasmaFile():
            return None
        path = self.plasmaPath.currentPath
        if not path:
            raise ValueError("Pick the plasma curve file")
        return self.logic.readPlasmaCsv(path, self.plasmaUnitCombo.currentData)

    # --- inputs -----------------------------------------------------------------------------------
    def selectInputs(self):
        if self.sequenceSelector.currentNode() is None:
            nodes = [n for n in slicer.util.getNodesByClass("vtkMRMLSequenceNode")
                     if (n.GetAttribute(PREFIX + "Kind") == "dynamic" or n.GetAttribute(PREFIX + "FrameDurationS"))
                     and self.logic.isDynamicSequence(n)]
            if nodes:
                self.sequenceSelector.setCurrentNode(nodes[-1])
        if self.segmentationSelector.currentNode() is None:
            nodes = slicer.util.getNodesByClass("vtkMRMLSegmentationNode")
            if nodes:
                self.segmentationSelector.setCurrentNode(nodes[-1])

    def onSequenceChanged(self, node=None):
        self._invalidate(curves=True)
        sequence = self.sequenceSelector.currentNode()
        valid = sequence is not None and self.logic.isDynamicSequence(sequence)
        if valid:
            intervals = self.logic.intervals(sequence)
            injection = self.logic.injectionTimeS(intervals)
            try:
                self.timeZeroCombo.model().item(1).setEnabled(injection is not None)
            except Exception:
                pass
            if injection is None and self.timeZeroCombo.currentData == TIME_INJECTION:
                self.timeZeroCombo.setCurrentIndex(0)
            self._setDefaultWindow()
            self.onOpen()
        else:
            self.onClose()
        self._update()

    def onSegmentationChanged(self, node=None):
        if self.observedSegmentation is not None:
            for event, callback in self._segmentEvents():
                self.removeObserver(self.observedSegmentation.GetSegmentation(), event, callback)
            self.observedSegmentation = None
        segmentation = self.segmentationSelector.currentNode()
        if segmentation is not None:
            self.observedSegmentation = segmentation
            for event, callback in self._segmentEvents():
                self.addObserver(segmentation.GetSegmentation(), event, callback)
        self._fillSegmentCombos()
        self._invalidate(curves=True)
        if self.session is not None:
            self.session.setSegmentation(segmentation, (self.targetId(), self.referenceId()))
        self._update()

    def _segmentEvents(self):
        """(event, callback): segment added / removed and painted (the source representation changed) need new
        curves; renamed / recolored (SegmentModified) only new names and colors."""
        out, seen = [], set()
        for name in ("SegmentAdded", "SegmentRemoved", "SourceRepresentationModified", "MasterRepresentationModified",
                     "SegmentModified"):
            event = getattr(slicer.vtkSegmentation, name, None)
            if event is None or event in seen:
                continue
            seen.add(event)
            out.append((event, self._onSegmentRenamed if name == "SegmentModified" else self._onSegmentEdited))
        return out

    def _onSegmentEdited(self, caller=None, event=None):
        if not self._computing:
            self._segmentsChanged = True
            self.segmentTimer.start()

    def _onSegmentRenamed(self, caller=None, event=None):
        if not self._computing:
            self.segmentTimer.start()

    def onSegmentsEdited(self):
        """Segments added, removed or painted: the combos and the curves are made again; renamed or recolored: the
        combos, the plot and the table."""
        self._fillSegmentCombos()
        if self._segmentsChanged or self.analysis is None:
            self._invalidate(curves=True)
        else:
            infos = {i: (n, c) for i, n, c in self.logic.segmentInfo(self.analysis.segmentation)}
            for segmentId, segment in self.analysis.segments.items():
                if segmentId in infos:
                    segment["name"], segment["color"] = infos[segmentId]
        self._segmentsChanged = False
        if self.session is not None:
            self.session.showSegments((self.targetId(), self.referenceId()))
        self._update()

    def _fillSegmentCombos(self):
        segmentation = self.segmentationSelector.currentNode()
        target, reference = self.targetId(), self.referenceId()
        self._updating = True
        try:
            infos = self.logic.segmentInfo(segmentation) if segmentation is not None else []
            for combo in (self.targetCombo, self.referenceCombo):
                combo.clear()
                for segmentId, name, color in infos:
                    pixmap = qt.QPixmap(12, 12)
                    pixmap.fill(qt.QColor(color))
                    combo.addItem(qt.QIcon(pixmap), name, segmentId)
            ids = [i for i, _, _ in infos]
            names = {i: n.lower() for i, n, _ in infos}
            if target in ids:
                self.targetCombo.setCurrentIndex(ids.index(target))
            if reference in ids:
                self.referenceCombo.setCurrentIndex(ids.index(reference))
            elif len(ids) > 1:
                self.referenceCombo.setCurrentIndex(self._guessReference(ids, names))
            if self.targetCombo.currentData == self.referenceCombo.currentData and len(ids) > 1:
                self.targetCombo.setCurrentIndex(next(i for i, s in enumerate(ids)
                                                      if s != self.referenceCombo.currentData))
        finally:
            self._updating = False

    def _guessReference(self, ids, names):
        """The segment most likely to be the reference: by its name, for the kind of model."""
        plasma = kin.model(self.modelKey()).inputKind == "plasma"
        words = (("blood", "aort", "ventric", "lv", "carotid", "sinus", "atri", "heart")
                 if plasma else ("cerebell", "reference", "ref", "occipital", "pons", "white"))
        for i, segmentId in enumerate(ids):
            if any(word in names[segmentId] for word in words):
                return i
        return len(ids) - 1

    def onTargetChanged(self, index=None):
        if self._updating:
            return
        if self.session is not None:
            self.session.showSegments((self.targetId(), self.referenceId()))
        self._refreshPlot()
        self._refreshTable()

    def onReferenceChanged(self, index=None):
        if self._updating:
            return
        self._invalidate()
        if self.session is not None:
            self.session.showSegments((self.targetId(), self.referenceId()))
        self._update()

    def onTimeZeroChanged(self, *args):
        self.injectionSpin.enabled = self.timeZeroCombo.currentData == TIME_MANUAL
        if self.analysis is not None:
            try:
                self.analysis = self.logic.withTimeZero(self.analysis, self.timeZeroCombo.currentData,
                                                        self.injectionSpin.value)
            except ValueError as error:
                self.infoLabel.text = str(error)
        self._setDefaultWindow()
        self._invalidate()
        self._update()

    def onSettingsChanged(self, *args):
        self._invalidate()
        self._update()

    def onModelChanged(self, *args):
        key = self.modelKey()
        spec = kin.model(key)
        self.modelDescription.text = spec.description
        plasma = spec.inputKind == "plasma"
        self.referenceLabel.text = "Blood pool segment:" if plasma else "Reference segment:"
        self.referenceCombo.toolTip = (
            "Segment of the blood pool (image-derived input, e.g. the left ventricle, the aorta or a sinus); with "
            "a plasma file it is the whole blood of vB" if plasma else
            "Region without specific binding (e.g. the cerebellum for many receptor tracers)")
        self.inputLabel.visible = self.inputCombo.visible = plasma
        self.plasmaFileLabel.visible = self.plasmaRow.visible = plasma and self.inputCombo.currentData == INPUT_FILE
        fitted = spec.method in ("compartment", "srtm")
        self.parameterBox.visible = fitted
        self.weightingLabel.visible = self.weightingCombo.visible = spec.method != "ratio"
        self.graphicalBox.visible = spec.method == "graphical"
        self.k2refLabel.visible = self.k2refSpin.visible = key == "LoganRef"
        self.windowBox.visible = spec.method == "ratio"
        if fitted:
            self._fillParameterTable(key)
        segmentation = self.segmentationSelector.currentNode()
        if segmentation is not None and not self._updating:
            infos = self.logic.segmentInfo(segmentation)
            ids = [i for i, _, _ in infos]
            if len(ids) > 1:
                self._updating = True
                try:
                    guess = self._guessReference(ids, {i: n.lower() for i, n, _ in infos})
                    if ids[guess] != self.targetId():
                        self.referenceCombo.setCurrentIndex(guess)
                finally:
                    self._updating = False
                if self.session is not None:
                    self.session.showSegments((self.targetId(), self.referenceId()))
        voxelwise = key in kin.VOXEL_MAPS
        self.mapButton.toolTip = ("Maps of " + ", ".join(kin.VOXEL_MAPS[key]) if voxelwise else
                                  f"{spec.title} has no voxel-wise version; use 1TC, Patlak, Logan, SRTM or SUVR")
        self._invalidate()
        self._update()

    def _setDefaultWindow(self):
        if self.analysis is None:
            sequence = self.sequenceSelector.currentNode()
            if sequence is None or not self.logic.isDynamicSequence(sequence):
                return
            try:
                zero = self.logic.timeZeroS(self.logic.intervals(sequence), self.timeZeroCombo.currentData,
                                            self.injectionSpin.value)
                frames = kin.Frames.fromIntervals(self.logic.intervals(sequence), zero)
            except ValueError:
                return
        else:
            frames = self.analysis.frames
        start, end = kin.defaultWindow(frames)
        for spin, value in ((self.windowStartSpin, start), (self.windowEndSpin, end)):
            spin.blockSignals(True)
            spin.value = round(value, 1)
            spin.blockSignals(False)

    # --- parameter table ------------------------------------------------------------------------
    def _fillParameterTable(self, key):
        params = self.parameterRows.get(key) or kin.model(key).defaultParameters()
        self.parameterRows[key] = params
        self._updating = True
        try:
            self.parameterTable.setRowCount(len(params))
            for r, p in enumerate(params):
                name = qt.QTableWidgetItem(p.name)
                name.setFlags(qt.Qt.ItemIsUserCheckable | qt.Qt.ItemIsEnabled)
                name.setCheckState(qt.Qt.Checked if p.fitted else qt.Qt.Unchecked)
                name.setToolTip(p.description + (" (ticked: fitted, else fixed at the initial value)"))
                self.parameterTable.setItem(r, 0, name)
                for c, value in enumerate((p.initial, p.lower, p.upper), start=1):
                    self.parameterTable.setItem(r, c, qt.QTableWidgetItem(f"{value:g}"))
                unit = qt.QTableWidgetItem(p.unit)
                unit.setFlags(qt.Qt.ItemIsEnabled)
                self.parameterTable.setItem(r, 4, unit)
            self.parameterTable.resizeColumnsToContents()
        finally:
            self._updating = False

    def _parametersFromTable(self, key):
        params = self.parameterRows.get(key)
        return [p.copy() for p in params] if params else None

    def onParameterEdited(self, item):
        if self._updating:
            return
        key = self.modelKey()
        params = self.parameterRows.get(key)
        if not params or item.row() >= len(params):
            return
        p = params[item.row()]
        if item.column() == 0:
            p.fitted = item.checkState() == qt.Qt.Checked
        elif item.column() in (1, 2, 3):
            try:
                value = float(item.text())
            except ValueError:
                self._fillParameterTable(key)
                return
            setattr(p, ("initial", "lower", "upper")[item.column() - 1], value)
        self._invalidate()
        self._update()

    def onParameterReset(self):
        key = self.modelKey()
        self.parameterRows[key] = kin.model(key).defaultParameters()
        self._fillParameterTable(key)
        self._invalidate()
        self._update()

    # --- curves -----------------------------------------------------------------------------------
    def _invalidate(self, curves=False):
        if curves:
            self.analysis = None
        self.results, self.resultsModel = {}, None
        self.resultLabel.text = ""
        if curves and self.session is not None:
            self.session.plot.setGraphical(None)

    def ensureAnalysis(self):
        """TACs of every segment (read once per series / segmentation / segment edit)."""
        sequence = self.sequenceSelector.currentNode()
        segmentation = self.segmentationSelector.currentNode()
        if sequence is None or segmentation is None or not self.logic.isDynamicSequence(sequence):
            return None
        if self.analysis is None:
            count = sequence.GetNumberOfDataNodes()
            dialog = None
            if count * max(1, len(self.logic.segmentIds(segmentation))) > 200:
                dialog = slicer.util.createProgressDialog(windowTitle="Dynamic PET Modeling",
                                                          labelText="Time-activity curves of the segments...",
                                                          maximum=count)

            def progress(index, total):
                if dialog is not None:
                    dialog.setValue(index)
                    slicer.app.processEvents()
                    return not dialog.wasCanceled
                return True

            qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)
            self._computing = True
            try:
                self.analysis = self.logic.prepare(sequence, segmentation, self.timeZeroCombo.currentData,
                                                   self.injectionSpin.value, progress=progress)
            except Exception as error:
                logging.exception("Dynamic PET Modeling: time-activity curves")
                self.infoLabel.text = f"Could not read the curves: {error}"
                self.analysis = None
            finally:
                self._computing = False
                qt.QApplication.restoreOverrideCursor()
                if dialog is not None:
                    dialog.close()
        return self.analysis

    def _update(self):
        self._updateInfo()
        self._refreshPlot()
        self._refreshTable()
        self._updateButtons()

    def _updateInfo(self):
        sequence = self.sequenceSelector.currentNode()
        if sequence is None or not self.logic.isDynamicSequence(sequence):
            self.infoLabel.text = "Pick a dynamic PET sequence." if sequence is None else \
                "Not a dynamic sequence of volumes with at least two frames."
            return
        intervals = self.logic.intervals(sequence)
        units = self.logic.units(sequence) or "unknown units"
        injection = self.logic.injectionTimeS(intervals)
        text = f"{len(intervals)} frames, {units}, {dynamicviews.formatTime(intervals[-1]['end'] - intervals[0]['start'])}."
        if injection is not None:
            text += f" Injection {dynamicviews.formatTime(intervals[0]['start'] - injection)} before the first frame."
        else:
            text += " No injection time in the series."
        if self.segmentationSelector.currentNode() is None:
            text += " Pick a segmentation."
        elif self.targetCombo.count == 0:
            text += " The segmentation has no segment."
        self.infoLabel.text = text

    def _updateButtons(self):
        ready = (self.sequenceSelector.currentNode() is not None and self.segmentationSelector.currentNode()
                 is not None and self.targetCombo.count > 0 and self.referenceCombo.count > 0)
        plasmaFile = self.usesPlasmaFile()
        self.plasmaFileLabel.visible = self.plasmaRow.visible = (
            kin.model(self.modelKey()).inputKind == "plasma" and self.inputCombo.currentData == INPUT_FILE)
        self.previewButton.enabled = ready and (self.targetId() != self.referenceId() or plasmaFile)
        self.fitAllButton.enabled = ready
        self.exportButton.enabled = bool(self.results)
        self.mapButton.enabled = ready and self.modelKey() in kin.VOXEL_MAPS
        self.closeButton.enabled = self.session is not None
        self.openButton.enabled = self.sequenceSelector.currentNode() is not None
        self.tStarSpin.enabled = not self.autoTStarCheck.checked
        self.maxErrorSpin.enabled = self.autoTStarCheck.checked

    def _refreshPlot(self):
        if self.session is None or self.session.plot is None:
            return
        plot = self.session.plot
        analysis = self.analysis if self.segmentationSelector.currentNode() is not None else None
        if analysis is None and self.segmentationSelector.currentNode() is not None:
            analysis = self.ensureAnalysis()
        if analysis is None:
            plot.setCurves([], message="Pick a segmentation to see the time-activity curves")
            plot.setMarkers()
            return
        frames = analysis.frames
        mids = np.where(frames.valid, frames.mid, np.nan)
        target, reference = self.targetId(), self.referenceId()
        plasmaFile = self.usesPlasmaFile()
        curves = []
        for segmentId, segment in analysis.segments.items():
            role = "target" if segmentId == target else ("reference" if segmentId == reference else "")
            if segmentId == reference and kin.model(self.modelKey()).inputKind == "plasma":
                role = "blood"
            curve = {"name": segment["name"], "color": segment["color"], "times": mids,
                     "values": np.where(frames.valid, segment["values"], np.nan), "emphasis": bool(role),
                     "role": role}
            result = self.results.get(segmentId)
            if result is not None and not result.get("error"):
                if result.get("curve") is not None:
                    curve["fitTimes"], curve["fitValues"] = result["curve"]
                elif result.get("fitted") is not None:
                    curve["fitTimes"], curve["fitValues"] = mids, result["fitted"]
            curves.append(curve)
        if plasmaFile:
            try:
                times, values = self.plasma()
                curves.append({"name": "Plasma (file)", "color": "#d02020", "times": times, "values": values,
                               "emphasis": True, "role": "input"})
            except Exception:
                pass
        plot.setCurves(curves, analysis.units)
        spec = kin.model(self.modelKey())
        targetResult = self.results.get(target)
        tStar = None
        if spec.method == "graphical":
            tStar = (targetResult["values"].get("t*") if targetResult and not targetResult.get("error")
                     else (None if self.autoTStarCheck.checked else self.tStarSpin.value))
        window = self.options().get("window") if spec.method == "ratio" else None
        plot.setMarkers(tStar=tStar, window=window)
        graphical = targetResult.get("graphical") if targetResult and not targetResult.get("error") else None
        if graphical:
            graphical = dict(graphical, name=analysis.name(target), color=analysis.segments[target]["color"])
        plot.setGraphical(graphical)
        self.session.graphicalRadio.enabled = bool(graphical)
        if graphical:
            plot.showGraphical(self.session.graphicalRadio.checked)
        else:
            self.session.tacRadio.checked = True

    def _refreshTable(self):
        if self.session is None or self.session.table is None:
            return
        analysis = self.analysis
        key = self.modelKey()
        columns = kin.resultColumns(key)
        header = ["Segment", "Voxels"] + [kin.columnTitle(c) for c in columns] + ["Notes"]
        rows = []
        if analysis is not None:
            reference = self.referenceId()
            plasma = kin.model(key).inputKind == "plasma"
            for segmentId, segment in analysis.segments.items():
                result = self.results.get(segmentId)
                role = ""
                if segmentId == reference and not self.usesPlasmaFile():
                    role = "blood pool (input)" if plasma else "reference"
                elif segmentId == reference:
                    role = "whole blood (vB)"
                if result is None:
                    values = [""] * len(columns)
                    notes = role
                elif result.get("error"):
                    values = [""] * len(columns)
                    notes = result["error"]
                else:
                    values = [kin.formatNumber(v) for v in kin.resultRow(result)]
                    notes = "; ".join(result.get("warnings") or [])
                rows.append((segmentId, [segment["name"], str(segment.get("voxels") or "")] + values + [notes],
                             segment["color"], role))
        self.session.setTable(header, rows, boldRow=self.targetId())

    # --- fits --------------------------------------------------------------------------------------
    def _guard(self, action, title):
        try:
            return action()
        except Exception as error:
            logging.exception(f"Dynamic PET Modeling: {title}")
            slicer.util.errorDisplay(f"{title}:\n{error}")
            return None

    def onPreview(self):
        analysis = self.ensureAnalysis()
        if analysis is None:
            return
        key = self.modelKey()
        target, reference = self.targetId(), self.referenceId()

        def run():
            qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)
            try:
                return self.logic.fitSegment(analysis, key, target, reference, self.options(), self.plasma())
            finally:
                qt.QApplication.restoreOverrideCursor()

        result = self._guard(run, "Could not fit the target")
        if result is None:
            return
        if self.resultsModel != key:
            self.results = {}
        self.results[target] = result
        self.resultsModel = key
        self.resultLabel.text = self._summary(analysis.name(target), result)
        self._refreshPlot()
        self._refreshTable()
        self._updateButtons()

    @staticmethod
    def _summary(name, result):
        values = ", ".join(f"{k} {kin.formatNumber(v)}" for k, v in result["values"].items()
                           if v is not None and k != "delay")
        metrics = result.get("metrics") or {}
        fit = ", ".join(f"{k} {kin.formatNumber(metrics.get(k))}" for k in ("R2", "NRMSE") if metrics.get(k)
                        is not None)
        text = f"{name}: {values}" + (f" ({fit})" if fit else "")
        if result.get("warnings"):
            text += " - " + "; ".join(result["warnings"])
        return text

    def onFitAll(self):
        analysis = self.ensureAnalysis()
        if analysis is None:
            return
        key = self.modelKey()
        reference = self.referenceId()
        count = len(analysis.segments)
        dialog = slicer.util.createProgressDialog(windowTitle="Dynamic PET Modeling", labelText="Fitting...",
                                                  maximum=max(1, count))

        def progress(index, total, name):
            dialog.maximum = total
            dialog.setValue(index)
            dialog.setLabelText(f"Fitted {name}")
            slicer.app.processEvents()
            return not dialog.wasCanceled

        def run():
            qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)
            try:
                return self.logic.fitSegments(analysis, key, reference, self.options(), self.plasma(),
                                              progress=progress)
            finally:
                qt.QApplication.restoreOverrideCursor()
                dialog.close()

        results = self._guard(run, "Could not fit the segments")
        if results is None:
            return
        self.results, self.resultsModel = results, key
        failed = [analysis.name(s) for s, r in results.items() if r.get("error")]
        self.resultLabel.text = f"{len(results) - len(failed)} segments fitted with {kin.model(key).title}."
        if failed:
            self.resultLabel.text += f" Not fitted: {', '.join(failed)}."
        if self.targetId() in results and not results[self.targetId()].get("error"):
            self.resultLabel.text += "\n" + self._summary(analysis.name(self.targetId()), results[self.targetId()])
        if self.tableCheck.checked:
            name = f"{analysis.sequence.GetName()} {key} results"
            existing = slicer.mrmlScene.GetFirstNodeByName(name)
            table = existing if existing is not None and existing.IsA("vtkMRMLTableNode") else None
            self.logic.resultsTable(analysis, results, key, name, table)
        self._refreshPlot()
        self._refreshTable()
        self._updateButtons()

    def onExport(self):
        if not self.results or self.analysis is None:
            return
        key = self.resultsModel or self.modelKey()
        default = os.path.join(qt.QDir.homePath(), f"{self.analysis.sequence.GetName()} {key}.csv")
        path = qt.QFileDialog.getSaveFileName(slicer.util.mainWindow(), "Export the results as CSV", default,
                                              "CSV files (*.csv)")
        if not path:
            return
        if not path.lower().endswith(".csv"):
            path += ".csv"
        written = self._guard(lambda: self.logic.exportCsv(path, self.analysis, self.results, key), "Export failed")
        if written:
            self.resultLabel.text = "Saved " + " and ".join(os.path.basename(p) for p in written) + "."

    def onMaps(self):
        analysis = self.ensureAnalysis()
        if analysis is None:
            return
        key = self.modelKey()
        options = self.options()
        target = self.results.get(self.targetId())
        if kin.model(key).method == "graphical" and self.autoTStarCheck.checked:
            if target is None or target.get("error") or self.resultsModel != key:
                self.onPreview()
                target = self.results.get(self.targetId())
            if target is not None and target["values"].get("t*") is not None:
                options["tStar"] = target["values"]["t*"]
        dialog = slicer.util.createProgressDialog(windowTitle="Dynamic PET Modeling",
                                                  labelText="Computing the parametric maps...", maximum=1)

        def progress(index, total):
            dialog.maximum = total
            dialog.setValue(index)
            slicer.app.processEvents()
            return not dialog.wasCanceled

        def run():
            qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)
            try:
                return self.logic.parametricMaps(analysis, key, self.referenceId(), options, self.plasma(),
                                                 self.mapThresholdSpin.value, self.insideCheck.checked, progress)
            finally:
                qt.QApplication.restoreOverrideCursor()
                dialog.close()

        maps = self._guard(run, "Could not compute the parametric maps")
        if not maps:
            return
        names = ", ".join(node.GetName() for node in maps.values())
        tStar = f" (t* {options['tStar']:.1f} min)" if kin.model(key).method == "graphical" else ""
        self.mapLabel.text = f"Made {names}{tStar}. The first one is shown over the summed image."
        if self.session is not None:
            self.session.setMapVolume(next(iter(maps.values())))

    # --- layout --------------------------------------------------------------------------------
    def onOpen(self):
        sequence = self.sequenceSelector.currentNode()
        if sequence is None or not self.logic.isDynamicSequence(sequence):
            return
        self._closeOtherModules()
        self._closeSession()
        try:
            self.session = ModelingSession(sequence, self.maxViewsSpin.value, self.orientationCombo.currentText,
                                           self.frameColorCombo.currentText, self.summedColorCombo.currentText,
                                           self.frameWindowCheck.checked)
            self.session.onRowClicked = self._selectTarget
            self.session.open()
            self.session.setSegmentation(self.segmentationSelector.currentNode(),
                                         (self.targetId(), self.referenceId()))
        except Exception as error:
            logging.exception("Could not open Dynamic PET Modeling")
            slicer.util.errorDisplay(f"Could not open the modelling layout:\n{error}")
            self._closeSession()
        self._update()

    def _selectTarget(self, segmentId):
        index = self.targetCombo.findData(segmentId)
        if index >= 0:
            self.targetCombo.setCurrentIndex(index)

    def _closeSession(self):
        if self.session is not None:
            self.session.close()
            self.session = None

    def onClose(self):
        self._closeSession()
        if hasattr(self, "closeButton"):
            self._updateButtons()

    @staticmethod
    def _closeOtherModules():
        """The other PET4D modules take the main layout too: close them first."""
        for name in ("DynamicPETEditorWidget", "DynamicPETMotionCorrectionWidget", "DynamicPETRegistrationWidget"):
            try:
                widget = getattr(slicer.modules, name, None)
                if widget is not None and getattr(widget, "session", None) is not None:
                    widget.onClose()
            except Exception:
                logging.warning(f"Could not close {name}", exc_info=True)

    def onOrientationChanged(self, *args):
        if self.session is not None:
            self.session.setOrientation(self.orientationCombo.currentText)

    def onColorsChanged(self, *args):
        if self.session is not None:
            self.session.setColors(self.frameColorCombo.currentText, self.summedColorCombo.currentText)

    def onFrameWindowToggled(self, checked):
        if self.session is not None:
            self.session.setFrameWindows(checked)

    # --- scene ---------------------------------------------------------------------------------
    def enter(self):
        self.selectInputs()
        if self.session is None and self.sequenceSelector.currentNode() is not None:
            self.onOpen()

    def exit(self):
        if self.session is not None:
            self.onClose()
            slicer.app.layoutManager().setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutFourUpView)

    def cleanup(self):
        self.onClose()
        self.removeObservers()
        ScriptedLoadableModuleWidget.cleanup(self)

    def onSceneClose(self, caller, event):
        self.onClose()
        self._invalidate(curves=True)

    @vtk.calldata_type(vtk.VTK_OBJECT)
    def onNodeRemoved(self, caller, event, node):
        if self.session is not None and node in (self.session.sequence, self.segmentationSelector.currentNode()):
            qt.QTimer.singleShot(0, self.onClose)


# ---------------------------------------------------------------------------
# Test: a synthetic series with a target, a reference and a blood pool
# ---------------------------------------------------------------------------
class DynamicPETModelingTest(ScriptedLoadableModuleTest):
    DURATIONS = [10] * 6 + [30] * 4 + [60] * 5 + [120] * 5 + [300] * 8      # 58 min

    def setUp(self):
        slicer.mrmlScene.Clear(0)

    def runTest(self):
        self.setUp()
        self.test_Modeling()

    def makeSeries(self):
        """Three cubes: target (SRTM R1 1.2, BPnd 2), reference (1TC K1 0.1, k2 0.15), blood (the input)."""
        ends = np.cumsum(self.DURATIONS).astype(float)
        starts = ends - np.array(self.DURATIONS)
        frames = kin.Frames(starts, ends)
        grid = frames.grid()
        t = grid
        blood = (851.1 * t - 42.69) * np.exp(-4.134 * t) + 21.88 * np.exp(-0.1191 * t) + 20.81 * np.exp(-0.01043 * t)
        convolver = kin._Convolver(grid, blood)
        reference = 0.1 * convolver.exp(0.15)
        target = 0.12 * convolver.exp(0.06)
        curves = {"blood": frames.frameMeans(grid, blood), "reference": frames.frameMeans(grid, reference),
                  "target": frames.frameMeans(grid, target)}
        shape = (12, 12, 30)
        labels = np.zeros(shape, dtype=np.int16)
        labels[3:9, 3:9, 2:8] = 1
        labels[3:9, 3:9, 12:18] = 2
        labels[3:9, 3:9, 22:28] = 3
        sequence = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSequenceNode", "Synthetic dynamic Bq/ml")
        sequence.SetIndexName("Time")
        sequence.SetIndexUnit("s")
        sequence.SetAttribute(PREFIX + "Kind", "dynamic")
        sequence.SetAttribute(PREFIX + "Units", "Bq/ml")
        for f in range(len(frames)):
            array = np.zeros(shape, dtype=np.float32)
            array[labels == 1] = curves["target"][f]
            array[labels == 2] = curves["reference"][f]
            array[labels == 3] = curves["blood"][f]
            node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", f"F{f + 1}")
            slicer.util.updateVolumeFromArray(node, array)
            node.SetSpacing(2.0, 2.0, 2.0)
            node.SetAttribute(PREFIX + "Frame", str(f + 1))
            node.SetAttribute(PREFIX + "StartFromStudyStartS", f"{starts[f]:g}")
            node.SetAttribute(PREFIX + "DurationS", f"{self.DURATIONS[f]:g}")
            node.SetAttribute(PREFIX + "StartFromInjectionS", f"{starts[f] + 30.0:g}")
            sequence.SetDataNodeAtValue(node, f"{starts[f]:g}")
            slicer.mrmlScene.RemoveNode(node)
        labelNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode", "labels")
        slicer.util.updateVolumeFromArray(labelNode, labels)
        labelNode.SetSpacing(2.0, 2.0, 2.0)
        segmentation = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", "Regions")
        slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(labelNode, segmentation)
        names = ["Target", "Cerebellum", "Blood"]
        ids = DynamicPETModelingLogic.segmentIds(segmentation)
        for segmentId, name in zip(ids, names):
            segmentation.GetSegmentation().GetSegment(segmentId).SetName(name)
        return sequence, segmentation, dict(zip(names, ids)), curves

    def test_Modeling(self):
        import tempfile
        sequence, segmentation, ids, curves = self.makeSeries()
        logic = DynamicPETModelingLogic()
        analysis = logic.prepare(sequence, segmentation)
        np.testing.assert_allclose(analysis.curve(ids["Target"]), curves["target"], rtol=1e-5)
        self.assertEqual(analysis.segments[ids["Target"]]["voxels"], 216)
        results = logic.fitSegments(analysis, "SRTM", referenceId=ids["Cerebellum"],
                                    segmentIds=[ids["Target"]])
        self.assertAlmostEqual(results[ids["Target"]]["values"]["BPnd"], 2.0, delta=0.1)
        result = logic.fitSegment(analysis, "1TC", ids["Cerebellum"], ids["Blood"])
        self.assertAlmostEqual(result["values"]["VT"], 0.1 / 0.15, delta=0.04)
        injected = logic.withTimeZero(analysis, TIME_INJECTION)
        self.assertAlmostEqual(injected.timeZeroS, -30.0)
        table = logic.resultsTable(analysis, results, "SRTM")
        self.assertEqual(table.GetNumberOfRows(), 1)
        folder = tempfile.mkdtemp()
        written = logic.exportCsv(os.path.join(folder, "srtm.csv"), analysis, results, "SRTM")
        self.assertTrue(all(os.path.exists(p) for p in written))
        maps = logic.parametricMaps(analysis, "SRTM", ids["Cerebellum"], thresholdPercent=1.0)
        bp = slicer.util.arrayFromVolume(maps["BPnd"])
        self.assertAlmostEqual(float(np.median(bp[3:9, 3:9, 2:8])), 2.0, delta=0.2)
        self.delayDisplay("Dynamic PET Modeling test passed")
