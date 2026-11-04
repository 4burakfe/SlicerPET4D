"""
DICOM Gated / Dynamic PET plugin for 3D Slicer

Loads PET series (Modality PT) from the DICOM browser in SUVbw or Bq/ml:
    gated     ECG gated PET (Series Type GATED\\IMAGE or Number of Time Slots > 1, e.g. GE Discovery cardiac
              gated PET): a sequence of gates plus their mean as the static image
    dynamic   dynamic PET (Series Type DYNAMIC\\IMAGE or Number of Time Slices > 1): a sequence of time frames
              plus a table of the frame timing
    static    one volume (offered with a low confidence, so Slicer's PET SUV plugin is used for plain static
              PET when it is installed; pick this loadable in the DICOM browser's Advanced view to use it)
Slicer's PET SUV and MultiVolume plugins stack the images of gated and dynamic series into one volume or fail
to split them; this plugin orders them by Image Index (0054,1330), else Trigger Time (gated) or Frame
Reference Time / acquisition time (dynamic), else repeated slice positions, and applies every image's own
rescale slope. The DICOM work is done in DICOMGatedDynamicPETLib/petseries.py (no Slicer dependency).

Loadables (one series gives SUV and Bq/ml, the DICOM browser selects the first). The SUV method (SUVbw,
SUVlbm, SUVbsa) and which units come first are set in the Gated / Dynamic PET Settings module; by default:
    gated     "<series> gated SUVbw" (1.05), "<series> gated Bq/ml" (1.04)
    dynamic   "<series> dynamic Bq/ml" (1.05), "<series> dynamic SUVbw" (1.04)   activity first, for kinetics
    static    "<series> SUVbw" (0.6), "<series> Bq/ml" (0.59)

Results, placed in the series' subject hierarchy study:
    gated     "<series> mean SUVbw": the static image, mean of the gates (each gate is a full concentration
              image), paired with the sequence (MPI.GatedSequence) so Tarvos lists one study; and the sequence
              "<series> gated SUVbw" (index "Gate") with a browser and a "(current gate)" proxy volume
    dynamic   the sequence "<series> dynamic Bq/ml" (index "Time", s from the scan start to the frame middle)
              with a browser and a "(current frame)" proxy volume, the table "<series> frame timing" and
              "<series> summed Bq/ml": the frames summed (weighted by duration, or by counts as a static scan
              would see them with the SummedWeighting setting; a weighted mean, so still a concentration)
    static    the volume "<series> SUVbw"
Every volume carries DICOM.Modality = PT, DICOM.instanceUIDs and attributes with the prefix of its kind
(StaticPET., GatedPET., DynamicPET.): units and scale, SUV inputs, radiopharmaceutical and radionuclide,
injection and scan times, decay correction, corrections applied, scanner and reconstruction; gates also
carry the ECG gating fields (read by Tarvos Gated QC), trigger time and frame duration; time frames carry
their start (clock, from injection, from study start = start of the first frame, from scan start), duration
(s and ms), middle, end
and decay factor. The sequences carry the per-frame timing as JSON lists (e.g. DynamicPET.FrameDurationS).
"""

import json
import os
import sys
import logging
import traceback

import numpy as np
import qt
import vtk
import slicer
from slicer.ScriptedLoadableModule import ScriptedLoadableModule, ScriptedLoadableModuleWidget, ScriptedLoadableModuleTest
from DICOMLib import DICOMPlugin, DICOMLoadable

_MODULE_DIR = os.path.dirname(os.path.abspath(globals().get("__file__", "") or "."))
if globals().get("__file__") and _MODULE_DIR not in sys.path:
    sys.path.insert(0, _MODULE_DIR)

from DICOMGatedDynamicPETLib import petseries  # noqa: E402
from DICOMGatedDynamicPETLib import settings as petsettings  # noqa: E402

UNIT_CODES = {petseries.UNITS_ACTIVITY: ("Bq/ml", "UCUM", "Becquerels/milliliter"),
              petseries.UNITS_SUV: ("{SUVbw}g/ml", "UCUM", "Standardized Uptake Value body weight"),
              petseries.UNITS_SUL: ("{SUVlbm}g/ml", "UCUM", "Standardized Uptake Value lean body mass"),
              petseries.UNITS_SUVBSA: ("{SUVbsa}cm2/ml", "UCUM", "Standardized Uptake Value body surface area")}
# (first loadable, second loadable) confidence per kind; above 1.0 beats MultiVolumeImporter
CONFIDENCE = {petseries.GATED: (1.05, 1.04), petseries.DYNAMIC: (1.05, 1.04), petseries.STATIC: (0.6, 0.59)}
KIND_WORD = {petseries.GATED: "gated", petseries.DYNAMIC: "dynamic", petseries.STATIC: ""}


# ---------------------------------------------------------------------------
# Module registration (hidden module, only used by the DICOM browser)
# ---------------------------------------------------------------------------
class DICOMGatedDynamicPET(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        parent.title = "DICOM Gated / Dynamic PET"
        parent.categories = ["Developer Tools.DICOM Plugins"]
        parent.contributors = ["Burak Demir"]
        parent.helpText = ("Loads gated, dynamic and static PET series from the DICOM browser in SUVbw or Bq/ml: "
                           "gates and time frames as sequences, with tracer, decay correction and frame timing "
                           "stored on the volumes.")
        parent.acknowledgementText = "SUV calculation follows the QIBA vendor-neutral pseudo-code."
        parent.hidden = True

        try:
            slicer.modules.dicomPlugins
        except AttributeError:
            slicer.modules.dicomPlugins = {}
        slicer.modules.dicomPlugins["DICOMGatedDynamicPETPlugin"] = DICOMGatedDynamicPETPluginClass


# ---------------------------------------------------------------------------
# DICOM plugin: examine + load
# ---------------------------------------------------------------------------
class DICOMGatedDynamicPETPluginClass(DICOMPlugin):
    def __init__(self):
        super().__init__()
        self.loadType = "Gated / Dynamic PET"
        self.tags["modality"] = "0008,0060"
        self.tags["seriesDescription"] = "0008,103e"
        self.tags["seriesType"] = "0054,1000"
        self.tags["timeSlots"] = "0054,0071"
        self.tags["timeSlices"] = "0054,0101"

    def examineForImport(self, fileLists):
        return self.examine(fileLists)

    def examine(self, fileLists):
        loadables = []
        for files in fileLists:
            if not files:
                continue
            try:
                loadables += self.examineFiles(list(files))
            except Exception:
                logging.warning("Gated / dynamic PET plugin could not examine a series:\n" + traceback.format_exc())
        return loadables

    def examineFiles(self, files):
        db = slicer.dicomDatabase
        first = files[0]
        kind = petseries.seriesKind(db.fileValue(first, self.tags["modality"]), db.fileValue(first, self.tags["seriesType"]),
                                    db.fileValue(first, self.tags["timeSlots"]), db.fileValue(first, self.tags["timeSlices"]))
        if kind is None:
            return []
        import pydicom
        header = pydicom.dcmread(first, stop_before_pixels=True)
        description = (db.fileValue(first, self.tags["seriesDescription"]) or "PET").strip() or "PET"
        frames = petseries.expectedFrames(header, kind)
        noun = {petseries.GATED: "gates", petseries.DYNAMIC: "time frames"}.get(kind)
        countText = f"{frames} {noun}, " if noun and frames > 1 else ""
        title = {petseries.GATED: "Gated PET", petseries.DYNAMIC: "Dynamic PET", petseries.STATIC: "PET"}[kind]
        suv = petseries.suvInfo([header])
        method = petsettings.suvMethod()
        suvScale, _, _, suvProblems = petseries.suvScales(suv, method)
        suvNote = ""
        if suvScale is None and method != "bw" and suv["suvScale"] is not None:
            suvNote = (f"SUV by {petseries.SUV_METHOD_TITLES[method]} not available ({', '.join(suvProblems)}), "
                       "SUVbw offered instead")
            method, suvScale, suvProblems = "bw", suv["suvScale"], []
        suvUnits = petseries.SUV_METHODS[method]
        available = {suvUnits: suvScale is not None, petseries.UNITS_ACTIVITY: suv["activityScale"] is not None}
        order = (suvUnits, petseries.UNITS_ACTIVITY) if petsettings.preferredUnits(kind) == petsettings.SUV \
            else (petseries.UNITS_ACTIVITY, suvUnits)

        loadables = []
        confidences = iter(CONFIDENCE[kind])
        for units in order:
            if not available[units]:
                continue
            tooltip = f"{title} ({countText}{len(files)} images) in {units}"
            if units == petseries.UNITS_ACTIVITY and not available[suvUnits] and suvProblems:
                tooltip += "\nSUV not available: " + ", ".join(suvProblems)
            if units == suvUnits and suvNote:
                tooltip += "\n" + suvNote
            loadables.append(self._loadable(files, loadableName(description, kind, units), kind, units,
                                            next(confidences), tooltip))
        if loadables:
            for loadable in loadables[1:]:
                loadable.selected = False
            warnings = []
            missing = petseries.missingImagesWarning(header, kind, len(files))
            if missing:
                warnings.append(missing)
            if suvNote:
                warnings.append(suvNote)
            if suvProblems and not available[suvUnits]:
                warnings.append("SUV not available: " + ", ".join(suvProblems))
            if kind == petseries.DYNAMIC and not petseries.isDecayCorrected(header):
                warnings.append("Frames are not decay corrected (Decay Correction NONE); loading asks whether to "
                                "decay correct them to the first frame start")
            if warnings:
                for loadable in loadables:
                    loadable.warning = "\n".join(warnings)
        return loadables

    def _loadable(self, files, name, kind, units, confidence, tooltip):
        loadable = DICOMLoadable()
        loadable.files = files
        loadable.name = name
        loadable.tooltip = tooltip
        loadable.selected = True
        loadable.confidence = confidence
        _KIND_FOR_LOADABLE[name + "|" + files[0]] = (kind, units)
        return loadable

    def load(self, loadable):
        kind, units = _KIND_FOR_LOADABLE.get(loadable.name + "|" + loadable.files[0], (None, None))
        try:
            return self.loadSeries(list(loadable.files), loadable.name, units, kind, loadable)
        except Exception as error:
            logging.error("PET loading failed:\n" + traceback.format_exc())
            slicer.util.errorDisplay(f"Could not load the PET series:\n{error}")
            return None

    # --- loading -------------------------------------------------------------------------------
    def loadSeries(self, files, name, units=None, kind=None, loadable=None, summed=None, decayCorrect=None,
                   weighting=None):
        """Load a PET series in units (Bq/ml, SUVbw, SUVlbm or SUVbsa; from the name when None); returns the
        node to show (static volume, gated mean or dynamic summed volume, else the sequence proxy). summed:
        make the static image of a gated / dynamic series (None: the setting). decayCorrect: decay correct the
        frames of a dynamic series that is not decay corrected (Decay Correction NONE) to the first frame
        start (None: ask). weighting: how the frames of a dynamic series are summed, petseries.SUM_DURATION or
        SUM_COUNTS (None: the setting)."""
        import pydicom
        progress = slicer.util.createProgressDialog(labelText="Reading PET images...", maximum=len(files) + 1)
        datasets = []
        try:
            for i, path in enumerate(files):
                datasets.append(pydicom.dcmread(path))
                if i % 20 == 0:
                    progress.setValue(i)
                    slicer.app.processEvents()
        finally:
            progress.close()

        kind = kind or petseries.datasetKind(datasets[0]) or petseries.STATIC
        if units is None:
            units = next((u for u in petseries.SUV_METHODS.values() if u in name), petseries.UNITS_ACTIVITY)
        if summed is None:
            summed = kind == petseries.STATIC or petsettings.summedVolume(kind)
        layout = petseries.arrangeFrames(datasets, kind)
        frames, slots, method = layout["frames"], layout["slots"], layout["method"]
        frameStarts = petseries.frameStartTimes(frames, datasets) if kind == petseries.DYNAMIC else None
        decayReference, decayFactors, decayNote = None, None, None
        if kind == petseries.DYNAMIC:
            layout["warnings"] += petseries.frameStartWarnings(frames, layout["numbers"], datasets)
            if not petseries.isDecayCorrected(datasets[0]):
                decayReference, factors, problem = petseries.decayCorrectionFactors(frames, datasets)
                if problem:
                    layout["warnings"].append(f"Frames are not decay corrected and cannot be corrected ({problem})")
                    decayNote = f"none (not possible: {problem})"
                elif decayCorrect if decayCorrect is not None else _askDecayCorrection(name, decayReference):
                    decayFactors = layout["decayFactors"] = factors
                    decayNote = ("frames decay corrected by the loader to the first frame start "
                                 f"({decayReference.isoformat()})")
                else:
                    decayNote = "none (declined at load)"
        suv = petseries.suvInfo(datasets, frameStarts, decayReference if decayFactors else None)
        suvExtra = {}
        if petseries.isSuvUnits(units):
            suvScale, frameScales, suvExtra, problems = petseries.suvScales(suv, petseries.suvMethodOf(units))
            scales = frameScales or [suvScale] * len(frames)
        else:
            scales, problems = [suv["activityScale"]] * len(frames), suv["problems"]
        if scales[0] is None:
            raise ValueError(f"{units} not available: " + ", ".join(problems or ["unknown units"]))
        if decayFactors:
            scales = [scale * factor for scale, factor in zip(scales, decayFactors)]

        origin, spacing, direction = petseries.gridGeometry(slots[0], layout["grid"])
        ijkToRas = vtk.vtkMatrix4x4()
        for r, row in enumerate(petseries.lpsToRasMatrix(origin, spacing, direction)):
            for c, value in enumerate(row):
                ijkToRas.SetElement(r, c, value)

        common = {"Kind": kind, "Units": units, "FrameOrdering": method, "Frames": str(len(frames))}
        if layout["warnings"]:
            common["LoadWarnings"] = " | ".join(layout["warnings"])
        if decayNote:
            common["LoaderDecayCorrection"] = decayNote
        if decayFactors or (suv["frameSuvScales"] and petseries.isSuvUnits(units)):
            common["Scale"] = "per frame"
        else:
            common["Scale"] = f"{scales[0]:.8g}"
        common.update(petseries.acquisitionInfo(datasets))
        common.update(petseries.suvAttributes(suv))
        common.update(suvExtra)
        if kind == petseries.GATED:
            common.update({k: str(v) for k, v in petseries.gatingInfo(frames).items()})
        prefix = petseries.ATTRIBUTE_PREFIXES[kind]

        if kind == petseries.STATIC:
            node = self._volume(name, petseries.frameArray(slots[0], scales[0]), ijkToRas, units, prefix, common,
                                frames[0])
            if loadable is not None:
                self._placeInSubjectHierarchy(loadable, node, ())
            logging.info(f"PET: {len(frames[0])} slices loaded as '{node.GetName()}' in {units} (scale {scales[0]:.6g})")
        elif kind == petseries.GATED:
            node = self._loadGated(name, layout, scales, ijkToRas, units, prefix, common, loadable, summed)
        else:
            weighting = weighting or petsettings.summedWeighting()
            decayCorrected = petseries.isDecayCorrected(datasets[0]) or bool(decayFactors)
            node = self._loadDynamic(name, layout, scales, ijkToRas, units, prefix, common, loadable, summed,
                                     weighting, decayCorrected)
        _showLater(node)
        if layout["warnings"]:
            _warnUser(name, layout["warnings"])
        return node

    def _volume(self, name, array, ijkToRas, units, prefix, attributes, frame, addToScene=True):
        node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", slicer.mrmlScene.GenerateUniqueName(name))
        slicer.util.updateVolumeFromArray(node, array)
        node.SetIJKToRASMatrix(ijkToRas)
        node.SetAttribute("DICOM.Modality", "PT")
        node.SetAttribute("DICOM.instanceUIDs", " ".join(str(ds.SOPInstanceUID) for ds in frame))
        for key, value in attributes.items():
            node.SetAttribute(prefix + key, str(value))
        setVoxelUnits(node, units)
        if addToScene:
            _setPetDisplay(node, 0.0, float(np.percentile(array, 99.9)) if array.size else 1.0)
        return node

    def _sequence(self, name, nodes, indexValues, indexName, indexUnit, attributes, prefix, upper):
        """Sequence of the frame nodes (removed from the scene) with a browser and a proxy; returns
        (sequence, browser, proxy)."""
        baseName = slicer.mrmlScene.GenerateUniqueName(name)
        sequenceNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSequenceNode", baseName)
        sequenceNode.SetIndexName(indexName)
        sequenceNode.SetIndexUnit(indexUnit)
        sequenceNode.SetIndexType(sequenceNode.NumericIndex)
        sequenceNode.SetAttribute("DICOM.Modality", "PT")
        for key, value in attributes.items():
            sequenceNode.SetAttribute(prefix + key, str(value))
        for node, value in zip(nodes, indexValues):
            sequenceNode.SetDataNodeAtValue(node, value)
            slicer.mrmlScene.RemoveNode(node)

        browserNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSequenceBrowserNode", f"{baseName} browser")
        slicer.modules.sequences.logic().AddSynchronizedNode(sequenceNode, None, browserNode)
        if hasattr(browserNode, "SetAndObservePrimarySequenceNodeID"):
            browserNode.SetAndObservePrimarySequenceNodeID(sequenceNode.GetID())
        else:
            browserNode.SetAndObserveMasterSequenceNodeID(sequenceNode.GetID())
        browserNode.SetPlaybackLooped(True)
        proxyNode = browserNode.GetProxyNode(sequenceNode)
        proxyNode.SetName(f"{baseName} (current {indexName.lower() if indexName == 'Gate' else 'frame'})")
        proxyNode.CreateDefaultDisplayNodes()
        _setPetDisplay(proxyNode, 0.0, upper)   # one window for all frames, so the cine does not flicker
        return sequenceNode, browserNode, proxyNode

    def _loadGated(self, name, layout, scales, ijkToRas, units, prefix, common, loadable, summed=True):
        gates, method, numbers = layout["frames"], layout["method"], layout["numbers"]
        nodes, upper, summer = [], 0.0, petseries.SliceWiseMean()
        gateName = name if " gated " in f" {name} " else f"{name} gated"
        for i, (gate, slots, g) in enumerate(zip(gates, layout["slots"], numbers)):
            array = petseries.frameArray(slots, scales[i])
            upper = max(upper, float(np.percentile(array, 99.9)))
            summer.add(array, slots)
            node = self._volume(f"{gateName} G{g}", array, ijkToRas, units, prefix, common, gate, addToScene=False)
            node.SetAttribute(prefix + "Gate", str(g))
            trigger, duration = petseries.gateTiming(gate)
            if trigger is not None:
                node.SetAttribute(prefix + "TriggerTimeMs", str(trigger))
            if duration is not None:
                node.SetAttribute(prefix + "FrameDurationMs", str(duration))
            nodes.append(node)
        sequenceNode, browserNode, proxyNode = self._sequence(
            gateName, nodes, [str(g) for g in numbers], "Gate", "", {}, prefix, upper)
        browserNode.SetPlaybackRateFps(len(gates))

        if not summed:
            if loadable is not None:
                self._placeInSubjectHierarchy(loadable, sequenceNode, (browserNode, proxyNode))
            logging.info(f"Gated PET: {len(gates)} gates x {len(layout['grid'])} slices loaded as "
                         f"'{sequenceNode.GetName()}' in {units} (ordered by {method}, scale {scales[0]:.6g})")
            return proxyNode

        # Static image of the acquisition: mean of the gates (each gate is a full concentration image); a
        # slice missing in some gates is the mean of the gates that have it
        meanName = name.replace(" gated ", " mean ", 1) if " gated " in name else f"{name} mean"
        mean = summer.mean()
        meanNode = self._volume(meanName, mean.astype(np.float32), ijkToRas, units, prefix, common, gates[0])
        meanNode.SetAttribute(prefix + "StaticFrom", f"mean of {len(gates)} gates")
        meanNode.SetAttribute("MPI.SummedFromGates", str(len(gates)))
        meanNode.SetAttribute("MPI.AveragedGates", "1")
        meanNode.SetAttribute("MPI.GatedSequence", sequenceNode.GetID())

        if loadable is not None:
            self._placeInSubjectHierarchy(loadable, meanNode, (sequenceNode, browserNode, proxyNode))
        logging.info(f"Gated PET: {len(gates)} gates x {len(layout['grid'])} slices loaded as '{sequenceNode.GetName()}' "
                     f"in {units} (ordered by {method}, scale {scales[0]:.6g}), mean of the gates as '{meanNode.GetName()}'")
        return meanNode

    def _loadDynamic(self, name, layout, scales, ijkToRas, units, prefix, common, loadable, summed=True,
                     weighting=petseries.SUM_DURATION, decayCorrected=True):
        frames, method = layout["frames"], layout["method"]
        timing = petseries.frameTiming(frames, numbers=layout["numbers"])
        for row, factor in zip(timing, layout.get("decayFactors") or []):
            row["LoaderDecayFactor"] = round(factor, 6)
        durations = [row["DurationS"] for row in timing]
        try:
            halfLife = float(common.get("RadionuclideHalfLifeS"))
        except (TypeError, ValueError):
            halfLife = None
        weights, weightingText = petseries.summationWeights(
            durations, [row["StartFromStudyStartS"] for row in timing], halfLife, weighting, decayCorrected)
        nodes, upper, summer = [], 0.0, petseries.SliceWiseMean()
        for f, (slots, frame, row) in enumerate(zip(layout["slots"], frames, timing)):
            array = petseries.frameArray(slots, scales[f])
            upper = max(upper, float(np.percentile(array, 99.9)))
            if summed:
                # summed acquisition = weighted mean of the concentrations (by duration or by counts), slice by
                # slice over the frames that have the slice
                summer.add(array, slots, weights[f] if weights else 1.0)
            node = self._volume(f"{name} F{row['Frame']}", array, ijkToRas, units, prefix, common, frame,
                                addToScene=False)
            node.SetAttribute(prefix + "Frame", str(row["Frame"]))
            if common.get("Scale") == "per frame":
                node.SetAttribute(prefix + "FrameScale", f"{scales[f]:.8g}")
            if layout.get("decayFactors"):
                node.SetAttribute(prefix + "LoaderDecayFactor", f"{layout['decayFactors'][f]:.8g}")
            for key, value in row.items():
                if key != "Frame" and value is not None:
                    node.SetAttribute(prefix + key, str(value))
            nodes.append(node)
        indexValues = frameIndexValues(timing)
        sequenceNode, browserNode, proxyNode = self._sequence(
            name, nodes, indexValues, "Time", "s", petseries.timingLists(timing), prefix, upper)
        sequenceNode.SetAttribute(prefix + "SumWeighting", weighting)   # followed when the frames are summed again
        sequenceNode.SetAttribute(prefix + "IndexMeaning", "seconds from the scan start to the frame middle"
                                  if timing[0]["MidFromScanStartS"] is not None else "frame number")
        if timing[0]["StartDateTime"]:
            # study start of the frame timing: the start of the first PET frame (not the DICOM Study Time)
            sequenceNode.SetAttribute(prefix + "StudyStartDateTime", timing[0]["StartDateTime"])
            sequenceNode.SetAttribute(prefix + "StudyStartDefinition", "start of the first frame")
        tableNode = frameTimingTable(f"{name} frame timing", timing)
        tableNode.SetAttribute(prefix + "SequenceNodeID", sequenceNode.GetID())
        sequenceNode.SetAttribute(prefix + "FrameTimingTableID", tableNode.GetID())

        summedNode = None
        if summed:
            summedName = name.replace(" dynamic ", " summed ", 1) if " dynamic " in name else f"{name} summed"
            array = summer.mean()
            summedNode = self._volume(summedName, array, ijkToRas, units, prefix, common, frames[0])
            summedNode.SetAttribute(prefix + "StaticFrom", f"{len(frames)} frames summed, {weightingText}")
            summedNode.SetAttribute(prefix + "SummedFrames", str(len(frames)))
            summedNode.SetAttribute(prefix + "SumWeighting", weighting)
            if weights:
                summedNode.SetAttribute(prefix + "DurationS", f"{sum(durations):.6g}")
            first, last = timing[0], timing[-1]
            for key, value in (("StartFromInjectionS", first["StartFromInjectionS"]),
                               ("EndFromInjectionS", last["EndFromInjectionS"]),
                               ("StartDateTime", first["StartDateTime"])):
                if value is not None:
                    summedNode.SetAttribute(prefix + key, str(value))
            summedNode.SetAttribute(prefix + "DynamicSequence", sequenceNode.GetID())
            sequenceNode.SetAttribute(prefix + "SummedVolumeID", summedNode.GetID())

        if loadable is not None:
            mainNode, others = (summedNode, (sequenceNode, browserNode, proxyNode, tableNode)) if summedNode \
                else (sequenceNode, (browserNode, proxyNode, tableNode))
            self._placeInSubjectHierarchy(loadable, mainNode, others)
        logging.info(f"Dynamic PET: {len(frames)} frames x {len(layout['grid'])} slices loaded as "
                     f"'{sequenceNode.GetName()}' in {units} (ordered by {method})"
                     + (f", summed as '{summedNode.GetName()}'" if summedNode else ""))
        if summedNode is not None:
            return summedNode
        return proxyNode

    def _placeInSubjectHierarchy(self, loadable, mainNode, others):
        try:
            self.addSeriesInSubjectHierarchy(loadable, mainNode)
            shNode = slicer.vtkMRMLSubjectHierarchyNode.GetSubjectHierarchyNode(slicer.mrmlScene)
            parent = shNode.GetItemParent(shNode.GetItemByDataNode(mainNode))
            if parent:
                for node in others:
                    item = shNode.GetItemByDataNode(node) or shNode.CreateItem(parent, node)
                    shNode.SetItemParent(item, parent)
        except Exception:
            logging.warning("Could not place the PET series in the subject hierarchy:\n" + traceback.format_exc())


class DICOMGatedDynamicPETWidget(ScriptedLoadableModuleWidget):
    """No interface: the plugin works inside the DICOM browser."""


# Loadable name + first file -> (kind, units) (Qt-wrapped loadables can lose Python attributes)
_KIND_FOR_LOADABLE = {}


def loadableName(description, kind, units):
    word = KIND_WORD[kind]
    return f"{description} {word} {units}" if word else f"{description} {units}"


def frameIndexValues(timing):
    """Sequence index of each time frame: seconds from the scan start to the frame middle (frame numbers when
    the timing is unknown, or when two frames would get the same value)."""
    values = [row["MidFromScanStartS"] for row in timing]
    if any(v is None for v in values) or len({round(v, 3) for v in values}) != len(values):
        return [str(row["Frame"]) for row in timing]
    return [f"{v:.3f}".rstrip("0").rstrip(".") for v in values]


TEXT_TIMING_COLUMNS = ("StartDateTime", "StartSource")


def frameTimingTable(name, timing):
    """Table node with one row per time frame and the columns of petseries.FRAME_TIMING_COLUMNS."""
    tableNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTableNode", slicer.mrmlScene.GenerateUniqueName(name))
    table = tableNode.GetTable()
    for key, description in petseries.FRAME_TIMING_COLUMNS:
        column = vtk.vtkStringArray() if key in TEXT_TIMING_COLUMNS else vtk.vtkDoubleArray()
        column.SetName(key)
        table.AddColumn(column)
    table.SetNumberOfRows(len(timing))
    for r, row in enumerate(timing):
        for c, (key, _) in enumerate(petseries.FRAME_TIMING_COLUMNS):
            value = row.get(key)
            if key in TEXT_TIMING_COLUMNS:
                table.GetColumn(c).SetValue(r, value or "")
            else:
                table.GetColumn(c).SetValue(r, float("nan") if value is None else float(value))
    for key, description in petseries.FRAME_TIMING_COLUMNS:
        tableNode.SetColumnDescription(key, description)
    tableNode.Modified()
    return tableNode


def setVoxelUnits(volumeNode, units):
    try:
        entry = slicer.vtkCodedEntry()
        entry.SetValueSchemeMeaning(*UNIT_CODES[units])
        volumeNode.SetVoxelValueUnits(entry)
    except Exception:
        logging.warning("Could not set voxel value units:\n" + traceback.format_exc())


def _setPetDisplay(volumeNode, lower, upper):
    volumeNode.CreateDefaultDisplayNodes()
    displayNode = volumeNode.GetDisplayNode()
    if not displayNode:
        return
    colorNode = slicer.mrmlScene.GetFirstNodeByName("PET-DICOM")
    if colorNode:
        displayNode.SetAndObserveColorNodeID(colorNode.GetID())
    displayNode.AutoWindowLevelOff()
    displayNode.SetWindowLevelMinMax(lower, upper if upper > lower else lower + 1.0)
    displayNode.SetInterpolate(True)


def _askDecayCorrection(name, reference):
    """Ask whether to decay correct a dynamic series that is not decay corrected; yes without asking in tests."""
    if slicer.app.testingEnabled():
        return True
    answer = qt.QMessageBox.question(
        slicer.util.mainWindow(), "Dynamic PET not decay corrected",
        f"The frames of '{name}' are not decay corrected (Decay Correction NONE).\n\n"
        f"Decay correct them to the start of the first frame ({reference.strftime('%H:%M:%S')})? Frame to frame "
        "values (time activity curves, kinetic modelling, the summed volume) need decay corrected frames.\n\n"
        "Yes: each frame is multiplied by the decay from the first frame start and during the frame.\n"
        "No: the frames are loaded as stored.",
        qt.QMessageBox.Yes | qt.QMessageBox.No, qt.QMessageBox.Yes)
    return answer == qt.QMessageBox.Yes


def _warnUser(name, warnings):
    """Log the load warnings and show them after loading (not while testing)."""
    text = f"'{name}' was loaded with problems:\n- " + "\n- ".join(warnings)
    logging.warning(text)
    if not slicer.app.testingEnabled():
        qt.QTimer.singleShot(0, lambda: slicer.util.warningDisplay(text, windowTitle="PET series incomplete"))


def _showLater(volumeNode):
    volumeNodeID = volumeNode.GetID()
    qt.QTimer.singleShot(0, lambda: _show(volumeNodeID))


def _show(volumeNodeID):
    volumeNode = slicer.mrmlScene.GetNodeByID(volumeNodeID)
    if volumeNode is None:
        return
    try:
        slicer.util.setSliceViewerLayers(background=volumeNode, fit=True)
    except Exception:
        logging.warning("Could not show the PET volume:\n" + traceback.format_exc())


# ---------------------------------------------------------------------------
# Test: synthetic static, gated and dynamic PET written to a temporary folder and loaded through the plugin
# ---------------------------------------------------------------------------
class DICOMGatedDynamicPETTest(ScriptedLoadableModuleTest):
    def setUp(self):
        slicer.mrmlScene.Clear(0)

    def runTest(self):
        self.setUp()
        self.test_LoadStatic()
        self.setUp()
        self.test_LoadGated()
        self.setUp()
        self.test_LoadDynamic()
        self.setUp()
        self.test_LoadDynamicNotDecayCorrected()
        self.setUp()
        self.test_LoadGatedIncomplete()
        self.setUp()
        self.test_LoadWithoutSummedVolumes()
        self.setUp()
        self.test_LoadStaticLeanBodyMass()

    @staticmethod
    def _write(kind, **options):
        import tempfile
        return petseries.writeSyntheticSeries(tempfile.mkdtemp(prefix="petseries"), kind=kind, **options)

    def test_LoadStatic(self):
        files, expected = self._write(petseries.STATIC)
        node = DICOMGatedDynamicPETPluginClass().loadSeries(files, "Synthetic SUVbw", petseries.UNITS_SUV)
        np.testing.assert_allclose(slicer.util.arrayFromVolume(node), expected[0], rtol=1e-4)
        self.assertEqual(node.GetAttribute("StaticPET.Radiopharmaceutical"), "Fluorodeoxyglucose")
        self.assertEqual(node.GetAttribute("StaticPET.DecayCorrection"), "START")
        self.delayDisplay("Static PET loaded")

    def test_LoadGated(self):
        files, expected = self._write(petseries.GATED)
        mean = DICOMGatedDynamicPETPluginClass().loadSeries(files, "Synthetic gated SUVbw", petseries.UNITS_SUV,
                                                            summed=True)
        self.assertEqual(mean.GetName(), "Synthetic mean SUVbw")
        sequence = slicer.mrmlScene.GetNodeByID(mean.GetAttribute("MPI.GatedSequence"))
        self.assertEqual(sequence.GetNumberOfDataNodes(), 4)
        for g in range(4):
            node = sequence.GetNthDataNode(g)
            self.assertEqual(node.GetAttribute("GatedPET.Gate"), str(g + 1))
            np.testing.assert_allclose(slicer.util.arrayFromVolume(node), expected[g], rtol=1e-4)
        np.testing.assert_allclose(slicer.util.arrayFromVolume(mean), np.mean(expected, axis=0), rtol=1e-4)
        self.assertIsNone(mean.GetAttribute("GatedPET.Gate"))
        self.assertEqual(mean.GetAttribute("GatedPET.Radiopharmaceutical"), "Fluorodeoxyglucose")
        self.assertAlmostEqual(sequence.GetNthDataNode(0).GetSpacing()[2], 3.27, places=3)
        self.delayDisplay("Gated PET loaded")

    def test_LoadDynamic(self):
        durations = [10.0, 10.0, 30.0, 60.0]
        files, expected = self._write(petseries.DYNAMIC, frames=4, frameDurationsS=durations)
        summed = DICOMGatedDynamicPETPluginClass().loadSeries(files, "Synthetic dynamic Bq/ml",
                                                              petseries.UNITS_ACTIVITY, summed=True)
        self.assertEqual(summed.GetName(), "Synthetic summed Bq/ml")
        sequence = slicer.mrmlScene.GetNodeByID(summed.GetAttribute("DynamicPET.DynamicSequence"))
        weightedMean = sum(1000.0 * (i + 1) * d for i, d in enumerate(durations)) / sum(durations)
        np.testing.assert_allclose(slicer.util.arrayFromVolume(summed), weightedMean, rtol=1e-4)
        self.assertEqual(summed.GetAttribute("DynamicPET.DurationS"), "110")
        self.assertEqual(sequence.GetNumberOfDataNodes(), 4)
        self.assertEqual([sequence.GetNthIndexValue(i) for i in range(4)], ["5", "15", "35", "80"])
        third = sequence.GetNthDataNode(2)
        self.assertEqual(third.GetAttribute("DynamicPET.DurationS"), "30")
        self.assertEqual(third.GetAttribute("DynamicPET.DurationMs"), "30000")
        self.assertEqual(third.GetAttribute("DynamicPET.StartFromInjectionS"), "20")
        self.assertEqual(third.GetAttribute("DynamicPET.StartFromStudyStartS"), "20")   # from the first frame
        np.testing.assert_allclose(slicer.util.arrayFromVolume(third), np.full_like(expected[2], 3000.0), rtol=1e-4)
        self.assertEqual(json.loads(sequence.GetAttribute("DynamicPET.FrameDurationS")), durations)
        table = slicer.mrmlScene.GetNodeByID(sequence.GetAttribute("DynamicPET.FrameTimingTableID"))
        self.assertEqual(table.GetNumberOfRows(), 4)
        self.assertEqual(sequence.GetAttribute("DynamicPET.SumWeighting"), petseries.SUM_DURATION)
        # weighted by counts: as one static acquisition of the 110 s, the early frames weigh more
        counted = DICOMGatedDynamicPETPluginClass().loadSeries(files, "Synthetic dynamic Bq/ml",
                                                               petseries.UNITS_ACTIVITY, summed=True,
                                                               weighting=petseries.SUM_COUNTS)
        halfLife = float(counted.GetAttribute("DynamicPET.RadionuclideHalfLifeS"))
        weights, _ = petseries.summationWeights(durations, [0.0, 10.0, 20.0, 50.0], halfLife, petseries.SUM_COUNTS)
        countMean = sum(1000.0 * (i + 1) * w for i, w in enumerate(weights)) / sum(weights)
        np.testing.assert_allclose(slicer.util.arrayFromVolume(counted), countMean, rtol=1e-4)
        self.assertEqual(counted.GetAttribute("DynamicPET.SumWeighting"), petseries.SUM_COUNTS)
        self.assertIn("count weighted", counted.GetAttribute("DynamicPET.StaticFrom"))
        self.delayDisplay("Dynamic PET loaded")

    def test_LoadDynamicNotDecayCorrected(self):
        durations = [60.0, 600.0, 1200.0]
        files, _ = self._write(petseries.DYNAMIC, frames=3, frameDurationsS=durations, decayCorrection="NONE")
        plugin = DICOMGatedDynamicPETPluginClass()
        summed = plugin.loadSeries(files, "Decay dynamic Bq/ml", petseries.UNITS_ACTIVITY, decayCorrect=True)
        sequence = slicer.mrmlScene.GetNodeByID(summed.GetAttribute("DynamicPET.DynamicSequence"))
        halfLife = 6586.2
        rate = np.log(2.0) / halfLife
        starts = [0.0, 60.0, 660.0]
        factors = [2.0 ** (t / halfLife) * rate * d / -np.expm1(-rate * d) for t, d in zip(starts, durations)]
        for f in range(3):
            node = sequence.GetNthDataNode(f)
            np.testing.assert_allclose(slicer.util.arrayFromVolume(node), 1000.0 * (f + 1) * factors[f], rtol=1e-4)
            self.assertAlmostEqual(float(node.GetAttribute("DynamicPET.LoaderDecayFactor")), factors[f], places=6)
        self.assertIn("decay corrected by the loader", summed.GetAttribute("DynamicPET.LoaderDecayCorrection"))
        # SUV of the corrected frames: one scale, the dose at the first frame start (= injection here)
        summed = plugin.loadSeries(files, "Decay dynamic SUVbw", petseries.UNITS_SUV, decayCorrect=True)
        sequence = slicer.mrmlScene.GetNodeByID(summed.GetAttribute("DynamicPET.DynamicSequence"))
        np.testing.assert_allclose(slicer.util.arrayFromVolume(sequence.GetNthDataNode(2)),
                                   3000.0 * factors[2] * 70.0 * 1000.0 / 370e6, rtol=1e-4)
        # declined: the frames stay as stored
        summed = plugin.loadSeries(files, "Raw dynamic Bq/ml", petseries.UNITS_ACTIVITY, decayCorrect=False)
        sequence = slicer.mrmlScene.GetNodeByID(summed.GetAttribute("DynamicPET.DynamicSequence"))
        np.testing.assert_allclose(slicer.util.arrayFromVolume(sequence.GetNthDataNode(2)), 3000.0, rtol=1e-4)
        self.assertIn("declined", summed.GetAttribute("DynamicPET.LoaderDecayCorrection"))
        self.delayDisplay("Not decay corrected dynamic PET loaded")

    def test_LoadGatedIncomplete(self):
        import pydicom
        files, expected = self._write(petseries.GATED)
        kept = []
        for path in files:
            ds = pydicom.dcmread(path, stop_before_pixels=True)
            gate, sliceIndex = divmod(int(ds.ImageIndex) - 1, int(ds.NumberOfSlices))
            if gate == 2 or (gate == 1 and sliceIndex == 3):
                continue   # gate 3 and slice 4 of gate 2 are missing
            kept.append(path)
        mean = DICOMGatedDynamicPETPluginClass().loadSeries(kept, "Incomplete gated SUVbw", petseries.UNITS_SUV,
                                                            summed=True)
        sequence = slicer.mrmlScene.GetNodeByID(mean.GetAttribute("MPI.GatedSequence"))
        self.assertEqual(sequence.GetNumberOfDataNodes(), 3)
        self.assertEqual([sequence.GetNthIndexValue(i) for i in range(3)], ["1", "2", "4"])
        second = slicer.util.arrayFromVolume(sequence.GetNthDataNode(1))
        self.assertEqual(second.shape[0], 6)
        # the hole is interpolated from slices 3 and 5 of that gate
        np.testing.assert_allclose(second[3], 0.5 * (expected[1][2] + expected[1][4]), rtol=1e-4)
        meanArray = slicer.util.arrayFromVolume(mean)
        np.testing.assert_allclose(meanArray[3], np.mean([expected[0][3], expected[3][3]], axis=0), rtol=1e-4)
        np.testing.assert_allclose(meanArray[0], np.mean([expected[0][0], expected[1][0], expected[3][0]], axis=0),
                                   rtol=1e-4)
        self.assertIn("gate 3", mean.GetAttribute("GatedPET.LoadWarnings"))
        self.assertIn("interpolated", mean.GetAttribute("GatedPET.LoadWarnings"))
        self.delayDisplay("Incomplete gated PET loaded with warnings")

    def test_LoadWithoutSummedVolumes(self):
        files, _ = self._write(petseries.GATED)
        proxy = DICOMGatedDynamicPETPluginClass().loadSeries(files, "Synthetic gated SUVbw", petseries.UNITS_SUV,
                                                             summed=False)
        self.assertTrue(slicer.modules.sequences.logic().GetFirstBrowserNodeForProxyNode(proxy))
        self.assertEqual(slicer.mrmlScene.GetNodesByClass("vtkMRMLScalarVolumeNode").GetNumberOfItems(), 1)
        files, _ = self._write(petseries.DYNAMIC, frames=3)
        proxy = DICOMGatedDynamicPETPluginClass().loadSeries(files, "Synthetic dynamic Bq/ml",
                                                             petseries.UNITS_ACTIVITY, summed=False)
        self.assertTrue(slicer.modules.sequences.logic().GetFirstBrowserNodeForProxyNode(proxy))
        self.assertIsNone(proxy.GetAttribute("DynamicPET.SummedFrames"))
        self.delayDisplay("Gated and dynamic PET loaded without summed volumes")

    def test_LoadStaticLeanBodyMass(self):
        files, expected = self._write(petseries.STATIC)   # 70 kg, 170 cm, sex O: no lean body mass
        with self.assertRaises(ValueError):
            DICOMGatedDynamicPETPluginClass().loadSeries(files, "Synthetic SUVlbm", petseries.UNITS_SUL)
        node = DICOMGatedDynamicPETPluginClass().loadSeries(files, "Synthetic SUVbsa", petseries.UNITS_SUVBSA)
        bsa = petseries.bodySurfaceAreaM2(70.0, 170.0)
        np.testing.assert_allclose(slicer.util.arrayFromVolume(node), expected[0] * bsa * 1e4 / 70000.0, rtol=1e-4)
        self.assertEqual(node.GetAttribute("StaticPET.Units"), "SUVbsa")
        self.delayDisplay("Static PET loaded in SUVbsa")
