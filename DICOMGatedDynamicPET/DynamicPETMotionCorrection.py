"""
Dynamic PET Motion Correction (3D Slicer module)

Registers every frame of a dynamic PET sequence (as loaded by the DICOM Gated / Dynamic PET plugin) to a reference
with Slicer's BRAINSFit (rigid, affine or B-spline, optionally a second pass), and resamples it with BRAINSResample
on its own grid.

Layout, like Review/Edit:
    upper 2/3   the reference in a large view on the left (inverted grey) and a near-square grid of small views on
                the right, one per frame: the frame in PET-Rainbow2 (0.25 opacity) over the reference in inverted grey. All views in
                one orientation (coronal by default), linked. After registration the views show the corrected
                frames (or the original ones, to compare).
    lower 1/3   the frame timeline; after a rigid registration two rows above the frame blocks give each frame's
                shift (mm along R, A, S) and angles (degrees about R, A, S).
The reference is a frame of the series or, optionally, another volume (the summed image). Frames with less than a
percentage of the highest frame activity are not registered (kept as they are), and a registration that moves any
point of the middle of the volume further than the motion limit along R, A or S is rejected (the frame is kept).
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
from DICOMGatedDynamicPETLib import motioncorrection as mc  # noqa: E402
from DICOMGatedDynamicPETLib.motionplot import MotionPlot  # noqa: E402
from DICOMGatedDynamicPETLib.timeline import FrameTimeline  # noqa: E402
from DynamicPETEditor import guardViewVolumes  # noqa: E402

LAYOUT_ID = 7802
MAIN_VIEW = "DynMCMain"
FRAME_VIEW_PREFIX = "DynMCF"
EMPTY_VIEW_PREFIX = "DynMCEmpty"
VIEW_GROUP = 79
PREFIX = "DynamicPET."
MC_PREFIX = "DynamicPET.MotionCorrection."
CORRECTED_SUFFIX = " motion corrected"
HELPER_ATTRIBUTE = "DynamicPETMotionCorrection.Helper"
DEFAULT_ORIENTATION = "Coronal"
FRAME_COLORMAP = "PET-Rainbow2"
REFERENCE_COLORMAP = "Inverted grey"
DEFAULT_OPACITY = 0.25
THRESHOLD_FRACTION = 0.03     # frame voxels below 3 % of the frame's window are transparent over the reference
AUTO_PERCENTILE = 99.9


class DynamicPETMotionCorrection(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        parent.title = "Dynamic PET Motion Correction"
        parent.categories = ["Nuclear Medicine"]
        parent.dependencies = []
        parent.contributors = ["Burak Demir"]
        parent.helpText = (
            "Frame-by-frame motion correction of a dynamic PET series with BRAINSFit (rigid, affine or B-spline, "
            "optionally two passes). Pick the dynamic sequence and the reference frame (or a summed image): every "
            "frame is shown in color over the reference in inverted grey. Register, review the corrected frames "
            "and the detected shift / angles on the timeline (rigid), then save the corrected series.")
        parent.acknowledgementText = ("Part of the DICOM Gated / Dynamic PET extension. Registration by BRAINSFit "
                                      "(Johnson HJ et al., BRAINSFit: Mutual Information Rigid Registrations of "
                                      "Whole-Brain 3D Images, Using the Insight Toolkit, Insight Journal 2007).")


# ---------------------------------------------------------------------------
# Logic
# ---------------------------------------------------------------------------
class DynamicPETMotionCorrectionLogic(ScriptedLoadableModuleLogic):
    """Registration of the frames of a dynamic sequence to a reference. Settings (dict):
        method               "Rigid", "Affine" or "BSpline"
        twoPass, secondMethod    a second pass ("Rigid", default, "Affine" or "BSpline") to the motion corrected sum of
                                 the first, starting from it
        skipLowActivity, lowActivityPercent    frames below that % of the highest frame activity are kept
        limitMotion, maxMotionMm    a registration moving more than that along R, A or S is rejected
        samplingPercentage, bsplineGrid    BRAINSFit details
    """

    @staticmethod
    def defaultSettings():
        return {"method": mc.DEFAULT_METHOD, "twoPass": False, "secondMethod": mc.DEFAULT_SECOND_METHOD,
                "skipLowActivity": True, "lowActivityPercent": mc.DEFAULT_LOW_ACTIVITY_PERCENT,
                "limitMotion": True, "maxMotionMm": mc.DEFAULT_MAX_MOTION_MM,
                "samplingPercentage": mc.DEFAULT_SAMPLING, "bsplineGrid": mc.DEFAULT_BSPLINE_GRID}

    @staticmethod
    def frameArray(sequenceNode, index):
        return slicer.util.arrayFromVolume(sequenceNode.GetNthDataNode(index))

    @classmethod
    def frameActivities(cls, sequenceNode):
        return [mc.frameActivity(cls.frameArray(sequenceNode, i)) for i in range(sequenceNode.GetNumberOfDataNodes())]

    @staticmethod
    def helperVolume(name, source=None):
        """A hidden volume that is not saved with the scene; shares the voxels and geometry of source."""
        node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", name)
        node.SetHideFromEditors(True)
        node.SetSaveWithScene(False)
        node.SetAttribute(HELPER_ATTRIBUTE, "1")
        if source is not None:
            DynamicPETMotionCorrectionLogic.shareVoxels(node, source)
        return node

    @staticmethod
    def shareVoxels(node, source):
        node.SetAndObserveImageData(source.GetImageData())
        matrix = vtk.vtkMatrix4x4()
        source.GetIJKToRASMatrix(matrix)
        node.SetIJKToRASMatrix(matrix)

    @staticmethod
    def _transformNode(transformType, name):
        className = "vtkMRMLBSplineTransformNode" if transformType == "BSpline" else "vtkMRMLLinearTransformNode"
        node = slicer.mrmlScene.AddNewNodeByClass(className, name)
        node.SetHideFromEditors(True)
        node.SetSaveWithScene(False)
        return node

    @staticmethod
    def _runCli(module, parameters):
        cliNode = slicer.cli.runSync(module, None, parameters, update_display=False)
        try:
            if cliNode.GetStatus() & cliNode.ErrorsMask:
                raise RuntimeError(cliNode.GetErrorText() or "CLI failed")
        finally:
            slicer.mrmlScene.RemoveNode(cliNode)

    def registerPass(self, fixed, moving, transformType, settings, initialTransform=None):
        """One BRAINSFit run; returns the output transform node (moving -> fixed, as BRAINSFit gives it)."""
        output = self._transformNode(transformType, f"MC {transformType}")
        parameters = {
            "fixedVolume": fixed.GetID(),
            "movingVolume": moving.GetID(),
            "transformType": transformType,
            "initializeTransformMode": "Off",
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
        if initialTransform is not None:
            parameters["initialTransform"] = initialTransform.GetID()
        try:
            self._runCli(slicer.modules.brainsfit, parameters)
        except Exception:
            slicer.mrmlScene.RemoveNode(output)
            raise
        return output

    def resample(self, moving, transformNode, output):
        """moving resampled through the transform on its own grid, into output (float)."""
        self._runCli(slicer.modules.brainsresample, {
            "inputVolume": moving.GetID(), "referenceVolume": moving.GetID(), "outputVolume": output.GetID(),
            "pixelType": "float", "warpTransform": transformNode.GetID(), "interpolationMode": "Linear",
            "defaultValue": 0.0})

    @staticmethod
    def motionMatrix(transformNode):
        """4x4 numpy motion (reference -> frame) of a linear transform node from BRAINSFit."""
        matrix = vtk.vtkMatrix4x4()
        transformNode.GetMatrixTransformFromParent(matrix)
        return slicer.util.arrayFromVTKMatrix(matrix)

    @staticmethod
    def displacement(transformNode, points):
        """Largest motion (dR, dA, dS) of the points (reference space) under the transform."""
        transform = transformNode.GetTransformFromParent()
        moved = [transform.TransformPoint([float(v) for v in p]) for p in points]
        return mc.largestDisplacement(points, moved)

    def registerFrames(self, sequenceNode, referenceVolume, referenceFrame, settings, progress=None):
        """Register every frame of the sequence to referenceVolume. referenceFrame is the index of the frame the
        reference is (kept as it is), or None (a summed image). progress(index, total, text) returns False to stop.

        With two passes the first pass registers the frames to referenceVolume; the summed image of the frames it
        corrected and of the reference frame (weighted as the series' summed image: by duration or by counts) is
        then the target of the second pass, which starts from the first pass's transform (a frame the first pass
        could not correct starts from no motion). A frame the second pass fails on or moves over the limit keeps its
        first pass result. Low activity frames are left out of both.

        Returns a result per frame: {"status", "shift", "angles", "message", "image" (vtkImageData of the
        corrected frame, or None when the frame is kept)}."""
        count = sequenceNode.GetNumberOfDataNodes()
        types = mc.transformTypes(settings["method"], settings["twoPass"], settings["secondMethod"])
        twoPass = len(types) > 1
        rowsPass = mc.rowsPass(settings["method"], settings["twoPass"], settings["secondMethod"])
        rigidRows = mc.showsMotionRows(settings["method"])
        activities = self.frameActivities(sequenceNode)
        low = mc.lowActivityFrames(activities, settings["lowActivityPercent"]) if settings["skipLowActivity"] \
            else [False] * count
        bounds = [0.0] * 6
        referenceVolume.GetRASBounds(bounds)
        points = mc.limitSamplePoints(bounds)
        center = [(bounds[2 * a] + bounds[2 * a + 1]) / 2.0 for a in range(3)]
        limit = settings["maxMotionMm"] if settings["limitMotion"] else None
        total = count * len(types)

        results = [{"status": mc.NOT_RUN, "shift": None, "angles": None, "message": "", "image": None}
                   for _ in range(count)]
        weights = None
        if twoPass:
            weighting = self.summationWeighting(sequenceNode, self.summedVolumeOf(sequenceNode))
            weights, _ = self.summationWeights(sequenceNode, weighting)
        summed = {"total": None, "weight": 0.0}

        def addToSum(index, array):
            weight = weights[index] if weights else 1.0
            part = np.asarray(array, dtype=np.float64) * weight
            summed["total"] = part if summed["total"] is None else summed["total"] + part
            summed["weight"] += weight

        moving = self.helperVolume("MC moving frame")
        resampled = self.helperVolume("MC resampled frame")
        target = None
        firstPass = {}          # frame index -> first pass transform, the start of the second pass
        stopped = False
        try:
            # --- first pass: to the reference
            for i in range(count):
                result = results[i]
                if progress is not None and progress(i, total, f"Frame {i + 1} of {count}"
                                                     + (" (first pass)" if twoPass else "")) is False:
                    stopped = True
                    break
                if i == referenceFrame:
                    result["status"] = mc.REFERENCE
                    if rigidRows:
                        result["shift"], result["angles"] = (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
                    if twoPass:
                        addToSum(i, self.frameArray(sequenceNode, i))
                    continue
                if low[i]:
                    result["status"] = mc.LOW_ACTIVITY
                    result["message"] = (f"activity {100.0 * activities[i] / max(activities):.2f} % of the highest "
                                         f"frame (< {settings['lowActivityPercent']:g} %), not registered")
                    continue
                self.shareVoxels(moving, sequenceNode.GetNthDataNode(i))
                transform = None
                try:
                    transform = self.registerPass(referenceVolume, moving, types[0], settings)
                    if self._correctFrame(result, transform, moving, resampled, points, center, limit, rigidRows) \
                            and twoPass:
                        addToSum(i, slicer.util.arrayFromVolume(resampled))
                        firstPass[i] = transform
                        transform = None
                except Exception as error:
                    logging.exception(f"Motion correction of frame {i + 1} failed")
                    result.update(status=mc.FAILED, message=str(error), shift=None, angles=None, image=None)
                finally:
                    if transform is not None:
                        slicer.mrmlScene.RemoveNode(transform)

            # --- second pass: to the summed image of the frames the first pass corrected
            if twoPass and not stopped and summed["total"] is not None and summed["weight"] > 0:
                target = self.helperVolume("MC first pass sum")
                slicer.util.updateVolumeFromArray(target, (summed["total"] / summed["weight"]).astype(np.float32))
                matrix = vtk.vtkMatrix4x4()
                sequenceNode.GetNthDataNode(0).GetIJKToRASMatrix(matrix)
                target.SetIJKToRASMatrix(matrix)
                summed["total"] = None
                for i in range(count):
                    result = results[i]
                    if progress is not None and progress(count + i, total,
                                                         f"Frame {i + 1} of {count} (second pass)") is False:
                        break
                    if result["status"] in (mc.REFERENCE, mc.LOW_ACTIVITY, mc.NOT_RUN):
                        continue
                    self.shareVoxels(moving, sequenceNode.GetNthDataNode(i))
                    self._secondPass(result, target, moving, resampled, types[1], settings, firstPass.get(i),
                                     points, center, limit, rigidRows and rowsPass == 1)
        finally:
            for node in [moving, resampled, target] + list(firstPass.values()):
                if node is not None:
                    slicer.mrmlScene.RemoveNode(node)
        logging.info(f"Dynamic PET motion correction of '{sequenceNode.GetName()}' "
                     f"({mc.methodText(settings['method'], settings['twoPass'], settings['secondMethod'])}): "
                     + mc.summaryText([r["status"] for r in results]))
        return results

    def _correctFrame(self, result, transform, moving, resampled, points, center, limit, rows):
        """Check the transform against the motion limit and resample the frame through it into result (rows: also
        the rigid motion). False when the frame is over the limit (kept as it was)."""
        moved = self.displacement(transform, points)
        if rows:
            result["shift"], result["angles"] = mc.rigidMotion(self.motionMatrix(transform), center)
        if mc.exceedsLimit(moved, limit):
            result.update(status=mc.OVER_LIMIT, image=None,
                          message=f"moves up to {mc.vectorText(moved)} mm along R, A, S (limit {limit:g} mm), "
                                  "kept as it was")
            return False
        self.resample(moving, transform, resampled)
        image = vtk.vtkImageData()
        image.DeepCopy(resampled.GetImageData())
        result.update(status=mc.REGISTERED, image=image, message="")
        return True

    def _secondPass(self, result, target, moving, resampled, transformType, settings, initial, points, center, limit,
                    rows):
        """Second pass of one frame to the first pass sum, into result; a first pass result is kept when the second
        pass fails or goes over the limit."""
        firstOk = result["status"] == mc.REGISTERED
        second = {"status": mc.NOT_RUN, "shift": None, "angles": None, "message": "", "image": None}
        transform = None
        try:
            transform = self.registerPass(target, moving, transformType, settings, initial)
            if self._correctFrame(second, transform, moving, resampled, points, center, limit, rows):
                if rows:
                    result["shift"], result["angles"] = second["shift"], second["angles"]
                elif not firstOk:
                    result["shift"] = result["angles"] = None    # first pass motion was rejected
                result.update(status=mc.REGISTERED, image=second["image"], message="")
            elif firstOk:
                result["message"] = f"second pass {second['message'].replace(', kept as it was', '')}; first pass kept"
            else:
                result.update(status=mc.OVER_LIMIT, image=None, message=second["message"])
                if rows:
                    result["shift"], result["angles"] = second["shift"], second["angles"]
        except Exception as error:
            logging.exception("Second pass of motion correction failed")
            if firstOk:
                result["message"] = f"second pass failed ({error}); first pass kept"
            else:
                result.update(status=mc.FAILED, message=str(error), image=None)
        finally:
            if transform is not None:
                slicer.mrmlScene.RemoveNode(transform)

    # --- harden / revert --------------------------------------------------------------------------
    @staticmethod
    def summedVolumeOf(sequenceNode):
        nodeID = sequenceNode.GetAttribute(PREFIX + "SummedVolumeID")
        return slicer.mrmlScene.GetNodeByID(nodeID) if nodeID else None

    @staticmethod
    def correctedName(name):
        return name if name.endswith(CORRECTED_SUFFIX) else name + CORRECTED_SUFFIX

    def harden(self, sequenceNode, results, settings, referenceText, resum=True, summedSource=None):
        """A new sequence '<name> motion corrected' with every frame of the series: the corrected image of each
        registered frame and an unchanged copy of the others (reference, low activity, over the limit, failed, not
        run), with the motion on every frame; its browser; a table of the per-frame motion; with resum, a new summed
        volume of the corrected frames, as the loader makes it (duration-weighted mean). summedSource is the summed
        image it replaces (default: the sequence's own). The original sequence and summed image are left as they
        are (removeOriginals() removes them). Returns (undo record for revert(), new sequence, table, new summed)."""
        method = mc.methodText(settings["method"], settings["twoPass"], settings["secondMethod"])
        scene = slicer.mrmlScene
        name = scene.GenerateUniqueName(self.correctedName(sequenceNode.GetName()))
        corrected = scene.AddNewNodeByClass("vtkMRMLSequenceNode", name)
        corrected.SetIndexName(sequenceNode.GetIndexName())
        corrected.SetIndexUnit(sequenceNode.GetIndexUnit())
        corrected.SetIndexType(sequenceNode.GetIndexType())
        for key in sequenceNode.GetAttributeNames() or ():
            if not key.startswith(MC_PREFIX) and key != PREFIX + "SummedVolumeID":
                corrected.SetAttribute(key, sequenceNode.GetAttribute(key))
        for i in range(sequenceNode.GetNumberOfDataNodes()):
            source = sequenceNode.GetNthDataNode(i)
            result = results[i] if i < len(results) else {"status": mc.NOT_RUN, "image": None, "shift": None}
            frame = self._frameCopy(source, result["image"])
            frame.SetAttribute(MC_PREFIX + "Status", result["status"])
            if result.get("shift") is not None:
                frame.SetAttribute(MC_PREFIX + "ShiftRASMm", mc.vectorText(result["shift"]))
                frame.SetAttribute(MC_PREFIX + "AnglesRASDeg", mc.vectorText(result["angles"]))
            corrected.SetDataNodeAtValue(frame, sequenceNode.GetNthIndexValue(i))
        corrected.SetAttribute(MC_PREFIX + "Method", method)
        corrected.SetAttribute(MC_PREFIX + "Reference", referenceText)
        corrected.SetAttribute(MC_PREFIX + "OriginalSequence", sequenceNode.GetName())
        browser, proxy = self._browserOf(corrected, sequenceNode)
        table = self.motionTable(f"{name} motion", results)
        table.SetAttribute(MC_PREFIX + "SequenceNodeID", corrected.GetID())
        corrected.SetAttribute(MC_PREFIX + "TableID", table.GetID())

        source = summedSource or self.summedVolumeOf(sequenceNode)
        summedNode = self.resum(corrected, source) if resum else None
        if summedNode is not None:
            corrected.SetAttribute(PREFIX + "SummedVolumeID", summedNode.GetID())
        elif source is not None:
            corrected.SetAttribute(PREFIX + "SummedVolumeID", source.GetID())
        created = [n for n in (browser, proxy, corrected, table, summedNode) if n is not None]
        self._placeNextTo(sequenceNode, created)
        undo = {"original": sequenceNode, "originalSummed": source, "created": created}
        changed = sum(1 for r in results if r["image"] is not None)
        logging.info(f"Dynamic PET motion correction: '{name}' made from '{sequenceNode.GetName()}' ({method}, "
                     f"{changed} of {corrected.GetNumberOfDataNodes()} frames corrected)")
        return undo, corrected, table, summedNode

    @staticmethod
    def _frameCopy(source, image=None):
        """A frame node (not in the scene) with the name, attributes and geometry of source and the given image
        (default: a copy of the image of source)."""
        node = getattr(slicer, source.GetClassName())()
        node.SetName(source.GetName())
        for key in source.GetAttributeNames() or ():
            node.SetAttribute(key, source.GetAttribute(key))
        matrix = vtk.vtkMatrix4x4()
        source.GetIJKToRASMatrix(matrix)
        node.SetIJKToRASMatrix(matrix)
        try:
            node.SetVoxelValueUnits(source.GetVoxelValueUnits())
        except Exception:
            pass
        if image is None:
            image = vtk.vtkImageData()
            image.DeepCopy(source.GetImageData())
        node.SetAndObserveImageData(image)
        return node

    def _browserOf(self, sequenceNode, like=None):
        """Browser and current frame volume of a new sequence, played and shown as the browser of like."""
        scene, logic = slicer.mrmlScene, slicer.modules.sequences.logic()
        name = sequenceNode.GetName()
        browser = scene.AddNewNodeByClass("vtkMRMLSequenceBrowserNode", scene.GenerateUniqueName(f"{name} browser"))
        logic.AddSynchronizedNode(sequenceNode, None, browser)
        if hasattr(browser, "SetAndObservePrimarySequenceNodeID"):
            browser.SetAndObservePrimarySequenceNodeID(sequenceNode.GetID())
        else:
            browser.SetAndObserveMasterSequenceNodeID(sequenceNode.GetID())
        browser.SetPlaybackLooped(True)
        old = logic.GetFirstBrowserNodeForSequenceNode(like) if like is not None else None
        oldProxy = old.GetProxyNode(like) if old is not None else None
        if old is not None:
            browser.SetPlaybackRateFps(old.GetPlaybackRateFps())
            browser.SetSelectedItemNumber(old.GetSelectedItemNumber())
        proxy = browser.GetProxyNode(sequenceNode)
        if proxy is None:
            return browser, None
        proxy.SetName(f"{name} (current frame)")
        proxy.CreateDefaultDisplayNodes()
        oldDisplay = oldProxy.GetDisplayNode() if oldProxy is not None else None
        if oldDisplay is not None and proxy.GetDisplayNode() is not None:
            proxy.GetDisplayNode().CopyContent(oldDisplay)
        else:
            upper = max([float(np.percentile(self.frameArray(sequenceNode, i)[::2, ::2, ::2], 99.9))
                         for i in range(sequenceNode.GetNumberOfDataNodes())] or [1.0])
            _setPetDisplay(proxy, upper)
        return browser, proxy

    def removeOriginals(self, sequenceNode, summedSource=None, removeSummed=True):
        """Remove the original series from the scene after harden(): the sequence, its browsers and their current
        frame volumes, and (removeSummed) its summed image: the sequence's own and summedSource when that was
        summed from the sequence (a summed image picked from elsewhere stays). Returns the names removed."""
        scene = slicer.mrmlScene
        summed = []
        if removeSummed:
            own = self.summedVolumeOf(sequenceNode)
            for node in (own, summedSource):
                if node is not None and node not in summed and (
                        node is own or node.GetAttribute(PREFIX + "DynamicSequence") == sequenceNode.GetID()):
                    summed.append(node)
        removed = []
        try:
            logic = slicer.modules.sequences.logic()
            browsers = vtk.vtkCollection()
            logic.GetBrowserNodesForSequenceNode(sequenceNode, browsers)
            for b in [browsers.GetItemAsObject(i) for i in range(browsers.GetNumberOfItems())]:
                proxy = b.GetProxyNode(sequenceNode)
                primary = b.GetPrimarySequenceNode() if hasattr(b, "GetPrimarySequenceNode") else \
                    b.GetMasterSequenceNode()
                if primary is sequenceNode:
                    removed.append(b.GetName())
                    scene.RemoveNode(b)
                else:
                    b.RemoveSynchronizedSequenceNode(sequenceNode.GetID())
                if proxy is not None and scene.IsNodePresent(proxy):
                    removed.append(proxy.GetName())
                    scene.RemoveNode(proxy)
        except Exception:
            logging.warning("Could not remove the browser of the original sequence", exc_info=True)
        removed.append(sequenceNode.GetName())
        scene.RemoveNode(sequenceNode)
        for node in summed:
            if scene.IsNodePresent(node):
                removed.append(node.GetName())
                scene.RemoveNode(node)
        logging.info("Dynamic PET motion correction: removed the originals " + ", ".join(f"'{n}'" for n in removed))
        return removed

    @staticmethod
    def summationWeighting(sequenceNode, source=None):
        """How the summed image of the sequence is made, from the tags the loader wrote (DynamicPET.SumWeighting
        of the summed image, else of the sequence): petseries.SUM_DURATION (default) or SUM_COUNTS."""
        from DICOMGatedDynamicPETLib import petseries
        for node in (source, sequenceNode):
            value = node.GetAttribute(PREFIX + "SumWeighting") if node is not None else None
            if value in petseries.SUM_WEIGHTINGS:
                return value
        return petseries.SUM_DURATION

    @staticmethod
    def _frameNumbers(sequenceNode, key):
        values = []
        for i in range(sequenceNode.GetNumberOfDataNodes()):
            try:
                values.append(float(sequenceNode.GetNthDataNode(i).GetAttribute(PREFIX + key)))
            except (TypeError, ValueError):
                values.append(None)
        return values

    def summationWeights(self, sequenceNode, weighting):
        """(weights, description) of the frames of the sequence, as the loader sums them (petseries)."""
        from DICOMGatedDynamicPETLib import petseries
        durations = self._frameNumbers(sequenceNode, "DurationS")
        starts = self._frameNumbers(sequenceNode, "StartFromStudyStartS")
        first = sequenceNode.GetNthDataNode(0) if sequenceNode.GetNumberOfDataNodes() else None
        halfLife, decayCorrected = None, True
        if first is not None:
            try:
                halfLife = float(first.GetAttribute(PREFIX + "RadionuclideHalfLifeS"))
            except (TypeError, ValueError):
                pass
            loaderCorrection = first.GetAttribute(PREFIX + "LoaderDecayCorrection") or ""
            decayCorrected = (first.GetAttribute(PREFIX + "DecayCorrection") or "").upper() != "NONE" or \
                loaderCorrection.startswith("frames decay corrected")
        return petseries.summationWeights(durations, starts, halfLife, weighting, decayCorrected)

    def resum(self, sequenceNode, source=None):
        """New summed volume of the frames of the sequence, as the loader makes it (petseries): weighted by
        duration or by counts, as the summed image it replaces (or the sequence) is tagged; with the attributes and
        display of that summed image."""
        from DICOMGatedDynamicPETLib import petseries
        count = sequenceNode.GetNumberOfDataNodes()
        arrays = [self.frameArray(sequenceNode, i) for i in range(count)]
        weighting = self.summationWeighting(sequenceNode, source)
        weights, weightingText = self.summationWeights(sequenceNode, weighting)
        scene = slicer.mrmlScene
        if source is not None:
            name = self.correctedName(source.GetName())
        else:
            base = sequenceNode.GetName()
            if base.endswith(CORRECTED_SUFFIX):
                base = base[:-len(CORRECTED_SUFFIX)]
            name = self.correctedName(base.replace(" dynamic ", " summed ", 1) if " dynamic " in base
                                      else f"{base} summed")
        summed = scene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", scene.GenerateUniqueName(name))
        array = petseries.weightedMean(arrays, weights).astype(np.float32)
        slicer.util.updateVolumeFromArray(summed, array)
        first = sequenceNode.GetNthDataNode(0)
        matrix = vtk.vtkMatrix4x4()
        first.GetIJKToRASMatrix(matrix)
        summed.SetIJKToRASMatrix(matrix)
        template = source if source is not None else first
        for key in template.GetAttributeNames() or ():
            summed.SetAttribute(key, template.GetAttribute(key))
        try:
            summed.SetVoxelValueUnits(template.GetVoxelValueUnits())
        except Exception:
            pass
        summed.SetAttribute(PREFIX + "StaticFrom", f"{count} motion corrected frames summed, {weightingText}")
        summed.SetAttribute(PREFIX + "SumWeighting", weighting)
        summed.SetAttribute(PREFIX + "SummedFrames", str(count))
        summed.SetAttribute(PREFIX + "DynamicSequence", sequenceNode.GetID())
        summed.SetAttribute(MC_PREFIX + "Method", sequenceNode.GetAttribute(MC_PREFIX + "Method") or "")
        summed.CreateDefaultDisplayNodes()
        sourceDisplay = source.GetDisplayNode() if source is not None else None
        if sourceDisplay is not None:
            summed.GetDisplayNode().CopyContent(sourceDisplay)
        else:
            _setPetDisplay(summed, float(np.percentile(array, AUTO_PERCENTILE)) if array.size else 1.0)
        return summed

    def revert(self, undo):
        """Undo harden(): remove the new sequence, its browser, the table and the new summed image (the original
        series was left as it was)."""
        for node in undo["created"]:
            if slicer.mrmlScene.IsNodePresent(node):
                slicer.mrmlScene.RemoveNode(node)

    @staticmethod
    def refreshProxies(sequenceNode):
        """The browsers of the sequence show the frames as they are now."""
        try:
            logic = slicer.modules.sequences.logic()
            browsers = vtk.vtkCollection()
            logic.GetBrowserNodesForSequenceNode(sequenceNode, browsers)
            for b in range(browsers.GetNumberOfItems()):
                logic.UpdateProxyNodesFromSequences(browsers.GetItemAsObject(b))
        except Exception:
            logging.warning("Could not refresh the sequence browser", exc_info=True)

    @staticmethod
    def motionTable(name, results):
        table = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTableNode", slicer.mrmlScene.GenerateUniqueName(name))
        columns = [("Frame", vtk.vtkIntArray), ("Status", vtk.vtkStringArray)]
        columns += [(title, vtk.vtkDoubleArray) for title in
                    ("ShiftRMm", "ShiftAMm", "ShiftSMm", "AngleRDeg", "AngleADeg", "AngleSDeg")]
        columns.append(("Note", vtk.vtkStringArray))
        arrays = []
        for title, arrayClass in columns:
            array = arrayClass()
            array.SetName(title)
            arrays.append(array)
        for i, result in enumerate(results):
            arrays[0].InsertNextValue(i + 1)
            arrays[1].InsertNextValue(result["status"])
            values = list(result["shift"] or (float("nan"),) * 3) + list(result["angles"] or (float("nan"),) * 3)
            for array, value in zip(arrays[2:8], values):
                array.InsertNextValue(float(value))
            arrays[8].InsertNextValue(result.get("message") or "")
        for array in arrays:
            table.AddColumn(array)
        return table

    @staticmethod
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
            logging.warning("Could not place the motion corrected series in the subject hierarchy", exc_info=True)


def _setPetDisplay(volumeNode, upper):
    volumeNode.CreateDefaultDisplayNodes()
    display = volumeNode.GetDisplayNode()
    if display is None:
        return
    colorNode = slicer.mrmlScene.GetFirstNodeByName("PET-DICOM")
    if colorNode:
        display.SetAndObserveColorNodeID(colorNode.GetID())
    display.AutoWindowLevelOff()
    display.SetWindowLevelMinMax(0.0, upper if upper > 0 else 1.0)
    display.SetInterpolate(True)


def _percentile(array):
    sample = array.ravel()
    sample = sample[::4] if sample.size > 400000 else sample
    sample = sample[np.isfinite(sample)]
    return float(np.percentile(sample, AUTO_PERCENTILE)) if sample.size else 1.0


# ---------------------------------------------------------------------------
# Session: layout, views and timeline dock
# ---------------------------------------------------------------------------
class MotionCorrectionSession:
    """The motion correction layout of one sequence: the reference in the large view, every frame (PET-Rainbow2) over
    the reference (inverted grey) in the small views, and the timeline dock."""

    def __init__(self, sequenceNode, maxViews=64, orientation=DEFAULT_ORIENTATION, linked=True):
        self.sequenceNode = sequenceNode
        self.maxViews = maxViews
        self.orientation = orientation
        self.linked = linked
        self.intervals = []
        self.viewFrames = []
        self.frameVolumes = []
        self.referenceVolume = None        # the volume shown in the large view and under every frame
        self.referenceHelper = None        # hidden volume of the reference frame (None for a summed image)
        self.referenceText = ""
        self.overlays = {}
        self.observers = []                # (node, tag): view guards
        self.guarded = False
        self.dock = None
        self.timeline = None
        self.previousLayout = None
        self.currentIndex = 0
        self.opacity = DEFAULT_OPACITY
        self.frameWindows = False          # each frame its own window
        self.frameColor = FRAME_COLORMAP
        self.referenceColor = REFERENCE_COLORMAP
        self.results = None
        self.hardened = False              # the corrections are in the sequence itself
        self.plots = []                    # shift (mm) and angle (deg) plots, rigid only
        self.showCorrected = True
        self.originalUppers = []
        self.frameMaximum = 1.0            # highest voxel of the frames (range of the frame window)
        self.frameWindow = None            # (lower, upper) of all frames; None: 0 - highest 99.9th percentile
        self.referenceWindow = None        # (lower, upper) of the reference; None: 0 - its 99.9th percentile
        self.referenceMaximum = 1.0
        self.correctedUppers = []

    # --- open / close ---------------------------------------------------------------------------
    def open(self):
        from DynamicPETEditor import DynamicPETEditorLogic
        logic = DynamicPETEditorLogic()
        self.intervals = logic.frameIntervals(self.sequenceNode)
        self.viewFrames = logic.shownFrames(len(self.intervals), self.maxViews)
        arrays = [DynamicPETMotionCorrectionLogic.frameArray(self.sequenceNode, i)
                  for i in range(self.sequenceNode.GetNumberOfDataNodes())]
        self.originalUppers = [_percentile(array) for array in arrays]
        self.frameMaximum = max([float(np.nanmax(array)) for array in arrays if array.size] or [1.0])
        self._makeFrameVolumes()
        self._setLayout()
        self._makeDock()
        self._setupViews()

    def close(self):
        for node, tag in self.observers:
            node.RemoveObserver(tag)
        self.observers = []
        self.guarded = False
        for sliceView, renderer, actor in self.overlays.values():
            renderer.RemoveActor2D(actor)
            sliceView.scheduleRender()
        self.overlays = {}
        layoutManager = slicer.app.layoutManager()
        if layoutManager.layout == LAYOUT_ID:
            self._link(False)
        if self.dock is not None:
            slicer.util.mainWindow().removeDockWidget(self.dock)
            self.dock.deleteLater()
            self.dock = None
            self.timeline = None
            self.plots = []
        if self.previousLayout is not None and layoutManager.layout == LAYOUT_ID:
            layoutManager.setLayout(self.previousLayout)
        for node in self.frameVolumes + [self.referenceHelper]:
            if node is not None and slicer.mrmlScene.IsNodePresent(node):
                slicer.mrmlScene.RemoveNode(node)
        self.frameVolumes = []
        self.referenceHelper = None

    # --- volumes ---------------------------------------------------------------------------------
    def _makeFrameVolumes(self):
        frameColor = colormaps.colorNodeForLabel(self.frameColor).GetID()
        for index in self.viewFrames:
            volume = DynamicPETMotionCorrectionLogic.helperVolume(
                f"{self.sequenceNode.GetName()} MC view F{index + 1}", self.sequenceNode.GetNthDataNode(index))
            volume.CreateDefaultDisplayNodes()
            display = volume.GetDisplayNode()
            display.SetAndObserveColorNodeID(frameColor)
            display.SetInterpolate(True)
            self.frameVolumes.append(volume)

    def setReference(self, frameIndex=None, volume=None):
        """The reference: a frame of the series (frameIndex) or another volume (the summed image). It is shown (and
        registered to) through a hidden helper volume sharing its voxels, in inverted grey, so the user's own
        volume keeps its display."""
        oldHelper, self.referenceHelper = self.referenceHelper, None
        if volume is None:
            frameIndex = min(max(0, frameIndex or 0), len(self.intervals) - 1)
            source = self.sequenceNode.GetNthDataNode(frameIndex)
            self.referenceText = f"frame {dynamicviews.frameLabel(self.intervals[frameIndex])}"
        else:
            source = volume
            self.referenceText = volume.GetName()
        helper = self.referenceHelper = DynamicPETMotionCorrectionLogic.helperVolume(
            f"{self.sequenceNode.GetName()} MC reference", source)
        helper.CreateDefaultDisplayNodes()
        display = helper.GetDisplayNode()
        display.SetAndObserveColorNodeID(colormaps.colorNodeForLabel(self.referenceColor).GetID())
        display.AutoWindowLevelOff()
        array = slicer.util.arrayFromVolume(helper)
        self.referenceMaximum = float(np.nanmax(array)) if array.size else 1.0
        self.referenceWindow = (0.0, _percentile(array))
        display.SetWindowLevelMinMax(*self.referenceWindow)
        display.SetInterpolate(True)
        self.referenceVolume = helper
        self._assignVolumes()
        if oldHelper is not None:       # removed after the views show the new one
            slicer.mrmlScene.RemoveNode(oldHelper)

    # --- layout and views ----------------------------------------------------------------------
    def _setLayout(self):
        layoutManager = slicer.app.layoutManager()
        layoutNode = layoutManager.layoutLogic().GetLayoutNode()
        xml = dynamicviews.layoutXml(len(self.viewFrames), orientation=self.orientation, mainView=MAIN_VIEW,
                                     framePrefix=FRAME_VIEW_PREFIX, emptyPrefix=EMPTY_VIEW_PREFIX,
                                     viewGroup=VIEW_GROUP, mainLabel="Ref")
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

    def setLinked(self, linked):
        self.linked = linked
        self._link(linked)

    def _setupViews(self):
        main = self._widget(MAIN_VIEW)
        self._addOverlay(MAIN_VIEW, main, 15)
        for v, (widget, _) in enumerate(self._frameWidgets()):
            if widget is None:
                continue
            widget.sliceController().setVisible(False)
            self._addOverlay(dynamicviews.frameViewName(v, FRAME_VIEW_PREFIX), widget, 10)
        for e in range(dynamicviews.emptyViewCount(len(self.frameVolumes))):
            widget = self._widget(dynamicviews.emptyViewName(e, EMPTY_VIEW_PREFIX))
            if widget is None:
                continue
            widget.sliceController().setVisible(False)
            self.observers.append(guardViewVolumes(widget, lambda: (None, None, None)))

    def _guardVolumes(self):
        """The reference and frame views keep their volumes when Slicer shows a newly loaded volume everywhere."""
        if self.guarded or self.referenceVolume is None:
            return
        self.guarded = True
        main = self._widget(MAIN_VIEW)
        if main is not None:
            self.observers.append(guardViewVolumes(main, lambda: (self.referenceVolume.GetID(), None, None)))
        for widget, volume in self._frameWidgets():
            if widget is not None:
                self.observers.append(guardViewVolumes(
                    widget, lambda volume=volume: (self.referenceVolume.GetID(), volume.GetID(), None)))

    def _assignVolumes(self):
        """Reference as background of every view, the frames as foreground of the small views. The views are
        unlinked meanwhile, as a linked view passes its volumes on to the others."""
        if self.referenceVolume is None:
            return
        self._link(False)
        try:
            main = self._widget(MAIN_VIEW)
            if main is not None:
                composite = main.mrmlSliceCompositeNode()
                composite.SetBackgroundVolumeID(self.referenceVolume.GetID())
                composite.SetForegroundVolumeID(None)
                composite.SetLabelVolumeID(None)
            for widget, volume in self._frameWidgets():
                if widget is None:
                    continue
                composite = widget.mrmlSliceCompositeNode()
                composite.SetBackgroundVolumeID(self.referenceVolume.GetID())
                composite.SetForegroundVolumeID(volume.GetID())
                composite.SetForegroundOpacity(self.opacity)
                composite.SetLabelVolumeID(None)
            self.setOrientation(self.orientation, link=False)
        finally:
            self._link(self.linked)
        self._applyFrameWindows()
        self._updateOverlays()
        self._guardVolumes()

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
                self._link(self.linked)

    def setOpacity(self, opacity):
        self.opacity = opacity
        self._link(False)
        try:
            for widget, _ in self._frameWidgets():
                if widget is not None:
                    widget.mrmlSliceCompositeNode().SetForegroundOpacity(opacity)
        finally:
            self._link(self.linked)

    def setFrameColor(self, label):
        self.frameColor = label
        colorID = colormaps.colorNodeForLabel(label).GetID()
        for volume in self.frameVolumes:
            volume.GetDisplayNode().SetAndObserveColorNodeID(colorID)

    def setReferenceColor(self, label):
        self.referenceColor = label
        if self.referenceHelper is not None:
            self.referenceHelper.GetDisplayNode().SetAndObserveColorNodeID(colormaps.colorNodeForLabel(label).GetID())

    def setReferenceWindow(self, lower, upper):
        if upper <= lower:
            return
        self.referenceWindow = (float(lower), float(upper))
        if self.referenceHelper is not None:
            display = self.referenceHelper.GetDisplayNode()
            display.AutoWindowLevelOff()
            display.SetWindowLevelMinMax(lower, upper)

    def setFrameWindow(self, lower, upper):
        """One window for all frames (used while each frame its own window is off)."""
        if upper <= lower:
            return
        self.frameWindow = (float(lower), float(upper))
        self._applyFrameWindows()

    def autoFrameWindow(self):
        return 0.0, max(self.originalUppers or [1.0])

    def setFrameWindows(self, perFrame):
        self.frameWindows = perFrame
        self._applyFrameWindows()

    def _applyFrameWindows(self):
        uppers = self.correctedUppers if (self.results and (self.showCorrected or self.hardened)
                                          and self.correctedUppers) else self.originalUppers
        common = self.frameWindow or (0.0, max(uppers or [1.0]))
        for volume, index in zip(self.frameVolumes, self.viewFrames):
            display = volume.GetDisplayNode()
            lower, upper = (0.0, uppers[index]) if self.frameWindows else common
            upper = upper if upper > lower else lower + 1.0
            display.AutoWindowLevelOff()
            display.SetWindowLevelMinMax(lower, upper)
            display.SetApplyThreshold(True)       # below the window (or 3 % of it): transparent over the reference
            display.SetLowerThreshold(max(lower, THRESHOLD_FRACTION * upper))
            display.SetUpperThreshold(max(abs(upper), self.frameMaximum, 1.0) * 1e6)

    # --- results ---------------------------------------------------------------------------------
    def setResults(self, results, motionRows, hardened=False):
        """Registration results (DynamicPETMotionCorrectionLogic.registerFrames), or None to clear them. hardened:
        the corrected frames are in the sequence itself (the views show the sequence, the motion stays shown)."""
        self.results = results
        self.hardened = bool(results) and hardened
        self.correctedUppers = []
        if results:
            from vtk.util import numpy_support
            for i, result in enumerate(results):
                if result["image"] is None:
                    self.correctedUppers.append(self.originalUppers[i])
                    continue
                values = numpy_support.vtk_to_numpy(result["image"].GetPointData().GetScalars())
                self.correctedUppers.append(_percentile(values))
        entries = [{k: r[k] for k in ("status", "shift", "angles", "message")} for r in results] \
            if results and motionRows else []
        if self.timeline is not None:
            self.timeline.setMotion(entries)
        for plot in self.plots:
            plot.setMotion(entries)
            plot.setVisible(bool(entries))
        self.setShowCorrected(self.showCorrected)
        self._updateToolbar()

    def setShowCorrected(self, corrected):
        self.showCorrected = corrected
        for volume, index in zip(self.frameVolumes, self.viewFrames):
            result = self.results[index] if (self.results and not self.hardened) else None
            image = result["image"] if (corrected and result and result["image"] is not None) else None
            volume.SetAndObserveImageData(image if image is not None
                                          else self.sequenceNode.GetNthDataNode(index).GetImageData())
        self._applyFrameWindows()
        self._updateOverlays()

    # --- overlays --------------------------------------------------------------------------------
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
            if overlay is None:
                continue
            sliceView, _, actor = overlay
            text = dynamicviews.frameLabel(self.intervals[index])
            if self.results:
                status = self.results[index]["status"]
                if status != mc.REGISTERED:
                    text += f"\n{status}"
                elif not self.showCorrected and not self.hardened:
                    text += "\noriginal"
            actor.SetInput(text)
            highlight = index == self.currentIndex
            actor.GetTextProperty().SetColor(*((1.0, 0.85, 0.2) if highlight else (1.0, 1.0, 1.0)))
            actor.GetTextProperty().SetBold(highlight)
            sliceView.scheduleRender()
        overlay = self.overlays.get(MAIN_VIEW)
        if overlay is not None:
            sliceView, _, actor = overlay
            actor.SetInput(f"Reference: {self.referenceText}")
            sliceView.scheduleRender()

    # --- timeline dock ---------------------------------------------------------------------------
    def _makeDock(self):
        mainWindow = slicer.util.mainWindow()
        self.dock = qt.QDockWidget("Dynamic PET Motion Correction timeline")
        self.dock.setObjectName("DynamicPETMotionCorrectionTimeline")
        self.dock.setFeatures(qt.QDockWidget.DockWidgetMovable | qt.QDockWidget.DockWidgetFloatable
                              | qt.QDockWidget.DockWidgetClosable)
        container = qt.QWidget()
        layout = qt.QVBoxLayout(container)
        layout.setContentsMargins(6, 4, 6, 4)
        toolbar = qt.QHBoxLayout()
        previousButton = qt.QToolButton()
        previousButton.text = "<"
        previousButton.toolTip = "Previous frame"
        nextButton = qt.QToolButton()
        nextButton.text = ">"
        nextButton.toolTip = "Next frame"
        self.frameLabel = qt.QLabel()
        self.frameLabel.setMinimumWidth(320)
        self.summaryLabel = qt.QLabel()
        showAllButton = qt.QPushButton("Show all")
        showAllButton.toolTip = ("Show the whole series (mouse wheel: zoom in time, Shift + wheel: pan, Ctrl + wheel "
                                 "over the motion plot: zoom its values, double click: reset)")
        for widget in (previousButton, nextButton, self.frameLabel):
            toolbar.addWidget(widget)
        toolbar.addStretch(1)
        toolbar.addWidget(self.summaryLabel)
        toolbar.addWidget(showAllButton)
        layout.addLayout(toolbar)
        self.plots = [MotionPlot("shift"), MotionPlot("angles")]
        for plot in self.plots:
            plot.setFrames(self.intervals)
            plot.onCurrentFrameChanged = self.setCurrentIndex
            plot.onViewRangeChanged = self._onViewRangeChanged
            plot.setVisible(False)       # shown after a rigid registration
        self.timeline = FrameTimeline()
        self.timeline.setFrames(self.intervals)
        self.timeline.onCurrentFrameChanged = self.setCurrentIndex
        self.timeline.onViewRangeChanged = self._onViewRangeChanged
        splitter = qt.QSplitter(qt.Qt.Vertical)
        for plot in self.plots:
            splitter.addWidget(plot)
        splitter.addWidget(self.timeline)
        splitter.setChildrenCollapsible(False)
        layout.addWidget(splitter, 1)
        self.dock.setWidget(container)
        previousButton.connect("clicked()", lambda: self.setCurrentIndex(max(0, self.currentIndex - 1)))
        nextButton.connect("clicked()", lambda: self.setCurrentIndex(min(len(self.intervals) - 1,
                                                                         self.currentIndex + 1)))
        showAllButton.connect("clicked()", lambda: self._onViewRangeChanged(*self.timeline.fullRange()))
        mainWindow.addDockWidget(qt.Qt.BottomDockWidgetArea, self.dock)
        try:
            mainWindow.resizeDocks([self.dock], [max(170, int(mainWindow.height / 3))], qt.Qt.Vertical)
        except Exception:
            self.dock.setMinimumHeight(max(170, int(mainWindow.height / 3)))
        self._updateToolbar()

    def _onViewRangeChanged(self, start, end):
        """The time window changed in the plot or the timeline: both show it."""
        for widget in [self.timeline] + self.plots:
            if widget is not None:
                widget.setViewRange(start, end)

    def setCurrentIndex(self, index):
        if 0 <= index < len(self.intervals):
            self.currentIndex = index
            if self.timeline is not None:
                self.timeline.setCurrentIndex(index)
            for plot in self.plots:
                plot.setCurrentIndex(index)
            self._updateOverlays()
            self._updateToolbar()

    def _updateToolbar(self):
        if self.dock is None or not (0 <= self.currentIndex < len(self.intervals)):
            return
        frame = self.intervals[self.currentIndex]
        text = (f"Frame {frame['number']} ({self.currentIndex + 1} of {len(self.intervals)}):  "
                f"{dynamicviews.frameTimes(self.intervals, self.currentIndex)}")
        if self.results:
            result = self.results[self.currentIndex]
            text += f"  |  {result['status']}"
            if result["shift"] is not None:
                text += (f", shift {mc.vectorText(result['shift'])} mm, angles {mc.vectorText(result['angles'])} "
                         "deg (R, A, S)")
        self.frameLabel.text = text
        self.summaryLabel.text = mc.summaryText([r["status"] for r in self.results]) if self.results else \
            "Not registered yet"


# ---------------------------------------------------------------------------
# Module panel
# ---------------------------------------------------------------------------
class DynamicPETMotionCorrectionWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    def __init__(self, parent=None):
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)
        self.session = None
        self.logic = DynamicPETMotionCorrectionLogic()
        self.activities = []
        self.results = None
        self.resultsSettings = None
        self.undo = None              # harden() record, for Revert
        self.hardened = False         # the results are hardened into a motion corrected series
        self._updating = False

        # --- input and reference
        inputBox = qt.QGroupBox("Dynamic series and reference")
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
        self.referenceCombo = qt.QComboBox()
        self.referenceCombo.toolTip = ("Frame every other frame is registered to (the frame with the highest activity "
                                       "is picked for you). It is shown in the large view.")
        form.addRow("Reference frame:", self.referenceCombo)
        self.summedSelector = slicer.qMRMLNodeComboBox()
        self.summedSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.summedSelector.noneEnabled = True
        self.summedSelector.noneDisplay = "None (use the reference frame)"
        self.summedSelector.addEnabled = False
        self.summedSelector.removeEnabled = False
        self.summedSelector.renameEnabled = False
        self.summedSelector.setMRMLScene(slicer.mrmlScene)
        self.summedSelector.setCurrentNode(None)
        self.summedSelector.toolTip = ("Optional: register to this volume (e.g. the summed image of the series) "
                                       "instead of a frame. All frames are then registered.")
        form.addRow("Summed image (optional):", self.summedSelector)
        self.resumCheck = qt.QCheckBox("Resum the summed image")
        self.resumCheck.checked = True
        self.resumCheck.toolTip = ("On Harden transforms, make a new summed image of the corrected frames "
                                   "next to the old one, as the DICOM loader makes it: weighted by duration, or by "
                                   "counts when the series was loaded with count weighting (its tags say which)")
        form.addRow(self.resumCheck)
        self.infoLabel = qt.QLabel()
        self.infoLabel.wordWrap = True
        form.addRow(self.infoLabel)
        self.layout.addWidget(inputBox)

        # --- view
        viewBox = qt.QGroupBox("View")
        view = qt.QFormLayout(viewBox)
        self.orientationCombo = qt.QComboBox()
        for name in dynamicviews.ORIENTATIONS:
            self.orientationCombo.addItem(name)
        self.orientationCombo.setCurrentIndex(self.orientationCombo.findText(DEFAULT_ORIENTATION))
        self.orientationCombo.toolTip = "Orientation of all views"
        view.addRow("Orientation:", self.orientationCombo)
        self.opacitySlider = ctk.ctkSliderWidget()
        self.opacitySlider.minimum, self.opacitySlider.maximum = 0.0, 1.0
        self.opacitySlider.singleStep, self.opacitySlider.decimals = 0.05, 2
        self.opacitySlider.value = DEFAULT_OPACITY
        self.opacitySlider.toolTip = "Opacity of the frames over the reference (inverted grey)"
        view.addRow("Frame opacity:", self.opacitySlider)
        self.frameWindowCheck = qt.QCheckBox("Each frame its own window (0 - 99.9th percentile)")
        self.frameWindowCheck.checked = False
        self.frameWindowCheck.toolTip = "Off: one window for all frames, so early low-count frames look faint"
        view.addRow(self.frameWindowCheck)
        self.frameWindowRange = ctk.ctkRangeWidget()
        self.frameWindowRange.decimals = 2
        self.frameWindowRange.toolTip = ("Display range of all frames (when each frame its own window is off); frame "
                                         "voxels below it are transparent over the reference")
        view.addRow("Frame window:", self.frameWindowRange)
        self.referenceWindowRange = ctk.ctkRangeWidget()
        self.referenceWindowRange.decimals = 2
        self.referenceWindowRange.toolTip = "Display range of the reference (large view and under every frame)"
        view.addRow("Reference window:", self.referenceWindowRange)
        self.autoWindowButton = qt.QPushButton("Auto windows")
        self.autoWindowButton.toolTip = "Both windows back to 0 - 99.9th percentile"
        view.addRow(self.autoWindowButton)
        self.frameColorCombo = qt.QComboBox()
        self.referenceColorCombo = qt.QComboBox()
        for combo, default in ((self.frameColorCombo, FRAME_COLORMAP), (self.referenceColorCombo, REFERENCE_COLORMAP)):
            for label in colormaps.LABELS:
                combo.addItem(label)
            combo.setCurrentIndex(max(0, combo.findText(default)))
        self.frameColorCombo.toolTip = "Color map of the frames (small views)"
        self.referenceColorCombo.toolTip = "Color map of the reference (large view, and under every frame)"
        view.addRow("Frame colors:", self.frameColorCombo)
        view.addRow("Reference colors:", self.referenceColorCombo)
        self.linkCheck = qt.QCheckBox("Link views (slice, zoom, pan)")
        self.linkCheck.checked = True
        view.addRow(self.linkCheck)
        self.maxViewsSpin = qt.QSpinBox()
        self.maxViewsSpin.setRange(4, 100)
        self.maxViewsSpin.value = 64
        self.maxViewsSpin.toolTip = "Most small frame views; longer series show frames spread evenly over the series"
        view.addRow("Frame views (max):", self.maxViewsSpin)
        showRow = qt.QHBoxLayout()
        self.showOriginalRadio = qt.QRadioButton("Original")
        self.showCorrectedRadio = qt.QRadioButton("Motion corrected")
        self.showCorrectedRadio.checked = True
        showRow.addWidget(self.showOriginalRadio)
        showRow.addWidget(self.showCorrectedRadio)
        showRow.addStretch(1)
        self.showWidget = qt.QWidget()
        self.showWidget.setLayout(showRow)
        self.showWidget.enabled = False
        view.addRow("Show frames:", self.showWidget)
        buttons = qt.QHBoxLayout()
        self.openButton = qt.QPushButton("Show layout")
        self.openButton.toolTip = "Switch to the motion correction layout (it opens by itself when a series is picked)"
        self.closeButton = qt.QPushButton("Close layout")
        self.closeButton.enabled = False
        buttons.addWidget(self.openButton)
        buttons.addWidget(self.closeButton)
        view.addRow(buttons)
        self.layout.addWidget(viewBox)

        # --- registration
        regBox = qt.QGroupBox("Registration (BRAINSFit)")
        reg = qt.QFormLayout(regBox)
        self.methodCombo = qt.QComboBox()
        for name in mc.METHODS:
            self.methodCombo.addItem(name)
        self.methodCombo.setCurrentIndex(self.methodCombo.findText(mc.DEFAULT_METHOD))
        self.methodCombo.toolTip = ("Rigid: 3 shifts + 3 angles (whole-body motion; the timeline shows them). "
                                    "Affine: adds scaling and shear. BSpline: local deformation.")
        reg.addRow("Method:", self.methodCombo)
        self.twoPassCheck = qt.QCheckBox("Enable two pass")
        self.twoPassCheck.checked = False
        self.twoPassCheck.toolTip = ("After the first pass, sum the corrected frames (weighted as the series' summed "
                                     "image) and register every frame again to that motion corrected sum, starting "
                                     "from the first pass. The sum is a cleaner target than one frame.")
        self.secondMethodCombo = qt.QComboBox()
        for name in mc.SECOND_PASS_METHODS:
            self.secondMethodCombo.addItem(name)
        self.secondMethodCombo.setCurrentIndex(self.secondMethodCombo.findText(mc.DEFAULT_SECOND_METHOD))
        self.secondMethodCombo.enabled = False
        self.secondMethodCombo.toolTip = ("Method of the second pass, to the motion corrected sum. Rigid: the "
                                          "timeline rows show its (final) motion; Affine / BSpline also correct "
                                          "scaling / local deformation.")
        row = qt.QHBoxLayout()
        row.addWidget(self.twoPassCheck)
        row.addWidget(qt.QLabel("second pass:"))
        row.addWidget(self.secondMethodCombo, 1)
        reg.addRow(row)
        self.lowActivityCheck = qt.QCheckBox("Do not register low activity frames, below")
        self.lowActivityCheck.checked = True
        self.lowActivitySpin = qt.QDoubleSpinBox()
        self.lowActivitySpin.setRange(0.01, 50.0)
        self.lowActivitySpin.setDecimals(2)
        self.lowActivitySpin.setSingleStep(0.5)
        self.lowActivitySpin.setSuffix(" %")
        self.lowActivitySpin.value = mc.DEFAULT_LOW_ACTIVITY_PERCENT
        self.lowActivitySpin.toolTip = ("Frames whose activity (sum of the voxels) is below this percentage of the "
                                        "highest frame activity are kept as they are")
        row = qt.QHBoxLayout()
        row.addWidget(self.lowActivityCheck)
        row.addWidget(self.lowActivitySpin)
        row.addWidget(qt.QLabel("of the max frame"))
        row.addStretch(1)
        reg.addRow(row)
        self.limitCheck = qt.QCheckBox("Limit motion to")
        self.limitCheck.checked = True
        self.limitSpin = qt.QDoubleSpinBox()
        self.limitSpin.setRange(1.0, 1000.0)
        self.limitSpin.setDecimals(0)
        self.limitSpin.setSingleStep(10.0)
        self.limitSpin.setSuffix(" mm")
        self.limitSpin.value = mc.DEFAULT_MAX_MOTION_MM
        self.limitSpin.toolTip = ("A registration that moves the middle of the volume further than this along R, A "
                                  "or S is rejected; the frame is kept as it is")
        row = qt.QHBoxLayout()
        row.addWidget(self.limitCheck)
        row.addWidget(self.limitSpin)
        row.addWidget(qt.QLabel("per axis (R, A, S)"))
        row.addStretch(1)
        reg.addRow(row)
        advanced = ctk.ctkCollapsibleButton()
        advanced.text = "Advanced"
        advanced.collapsed = True
        advancedForm = qt.QFormLayout(advanced)
        self.samplingSpin = qt.QDoubleSpinBox()
        self.samplingSpin.setRange(0.001, 1.0)
        self.samplingSpin.setDecimals(3)
        self.samplingSpin.setSingleStep(0.01)
        self.samplingSpin.value = mc.DEFAULT_SAMPLING
        self.samplingSpin.toolTip = "BRAINSFit sampling percentage (fraction of the voxels used by the metric)"
        advancedForm.addRow("Sampling fraction:", self.samplingSpin)
        self.gridSpin = qt.QSpinBox()
        self.gridSpin.setRange(3, 20)
        self.gridSpin.value = mc.DEFAULT_BSPLINE_GRID
        self.gridSpin.toolTip = "B-spline control grid size along each axis"
        advancedForm.addRow("B-spline grid:", self.gridSpin)
        reg.addRow(advanced)
        self.rowsNote = qt.QLabel()
        self.rowsNote.wordWrap = True
        reg.addRow(self.rowsNote)
        self.registerButton = qt.QPushButton("Register frames")
        self.registerButton.toolTip = "Register every frame to the reference and show the corrected frames"
        reg.addRow(self.registerButton)
        self.removeOriginalsCheck = qt.QCheckBox("Keep only the motion corrected series")
        self.removeOriginalsCheck.checked = True
        self.removeOriginalsCheck.toolTip = (
            "On Harden transforms, remove the original series (sequence, browser and current frame volume) and, "
            "when Resum makes a new summed image, the old summed image, so only the motion corrected ones stay. "
            "The frames that were not corrected (reference, low activity, over the limit, failed) are in the motion "
            "corrected series as they were. This cannot be undone. Off: the originals stay next to the new series.")
        reg.addRow(self.removeOriginalsCheck)
        self.revertButton = qt.QPushButton("Revert changes")
        self.revertButton.toolTip = ("Before hardening: drop the registration. After hardening (originals kept): "
                                     "remove the motion corrected series, its table and summed image, and go back "
                                     "to the original series.")
        self.revertButton.enabled = False
        self.hardenButton = qt.QPushButton("Harden transforms")
        self.hardenButton.toolTip = ("Make the motion corrected series: every frame, corrected or (when it was not "
                                     "registered) as it was, with a table of the per-frame motion and, with Resum, a "
                                     "new summed image")
        self.hardenButton.enabled = False
        row = qt.QHBoxLayout()
        row.addWidget(self.revertButton)
        row.addWidget(self.hardenButton)
        reg.addRow(row)
        self.resultLabel = qt.QLabel("Not registered yet.")
        self.resultLabel.wordWrap = True
        reg.addRow(self.resultLabel)
        self.layout.addWidget(regBox)
        self.layout.addStretch(1)

        self.referenceCombo.connect("currentIndexChanged(int)", self.onReferenceChanged)
        self.summedSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onSummedChanged)
        self.orientationCombo.connect("currentIndexChanged(int)", self.onOrientationChanged)
        self.opacitySlider.connect("valueChanged(double)", self.onOpacityChanged)
        self.frameWindowCheck.connect("toggled(bool)", self.onFrameWindowToggled)
        self.frameColorCombo.connect("currentIndexChanged(int)", self.onFrameColorChanged)
        self.frameWindowRange.connect("valuesChanged(double,double)", self.onFrameWindowChanged)
        self.referenceWindowRange.connect("valuesChanged(double,double)", self.onReferenceWindowChanged)
        self.autoWindowButton.connect("clicked()", self.onAutoWindows)
        self.referenceColorCombo.connect("currentIndexChanged(int)", self.onReferenceColorChanged)
        self.linkCheck.connect("toggled(bool)", self.onLinkToggled)
        self.showCorrectedRadio.connect("toggled(bool)", self.onShowCorrectedToggled)
        self.openButton.connect("clicked()", self.onOpen)
        self.closeButton.connect("clicked()", self.onClose)
        self.methodCombo.connect("currentIndexChanged(int)", self.onMethodChanged)
        self.twoPassCheck.connect("toggled(bool)", self.onMethodChanged)
        self.secondMethodCombo.connect("currentIndexChanged(int)", self.onMethodChanged)
        self.lowActivityCheck.connect("toggled(bool)", self._updateInfo)
        self.lowActivitySpin.connect("valueChanged(double)", self._updateInfo)
        self.limitCheck.connect("toggled(bool)", lambda on: setattr(self.limitSpin, "enabled", on))
        self.registerButton.connect("clicked()", self.onRegister)
        self.revertButton.connect("clicked()", self.onRevert)
        self.hardenButton.connect("clicked()", self.onHarden)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.NodeRemovedEvent, self.onNodeRemoved)
        self.onMethodChanged()
        self.selectDynamicSequence()
        self.onSequenceChanged(self.sequenceSelector.currentNode(), openLayout=False)
        # connected after the first pick, so the layout opens on enter(), not while the panel is built
        self.sequenceSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onSequenceChanged)

    # --- settings ------------------------------------------------------------------------------
    def settings(self):
        return {"method": self.methodCombo.currentText, "twoPass": self.twoPassCheck.checked,
                "secondMethod": self.secondMethodCombo.currentText,
                "skipLowActivity": self.lowActivityCheck.checked, "lowActivityPercent": self.lowActivitySpin.value,
                "limitMotion": self.limitCheck.checked, "maxMotionMm": self.limitSpin.value,
                "samplingPercentage": self.samplingSpin.value, "bsplineGrid": self.gridSpin.value}

    def onMethodChanged(self, *args):
        method = self.methodCombo.currentText
        self.twoPassCheck.enabled = method != "BSpline"
        self.secondMethodCombo.enabled = self.twoPassCheck.checked and method != "BSpline"
        if mc.showsMotionRows(method):
            self.rowsNote.text = ("The timeline shows each frame's shift (mm) and angles (deg)"
                                  + ("." if not self.twoPassCheck.checked or self.secondMethodCombo.currentText ==
                                     "Rigid" else " of the rigid first pass."))
        else:
            self.rowsNote.text = ("Shift / angle rows on the timeline are shown for rigid only: an affine or "
                                  "B-spline transform has no single shift and angle per frame.")

    def referenceFrame(self):
        """Index of the reference frame, or None when a summed image is the reference."""
        if self.summedSelector.currentNode() is not None:
            return None
        return self.referenceCombo.currentIndex if self.referenceCombo.currentIndex >= 0 else 0

    # --- series and reference ------------------------------------------------------------------
    def selectDynamicSequence(self):
        nodes = slicer.util.getNodesByClass("vtkMRMLSequenceNode")
        dynamic = [n for n in nodes if (n.GetAttribute(PREFIX + "FrameDurationS") or n.GetAttribute(PREFIX + "Kind"))
                   and not n.GetAttribute(MC_PREFIX + "Source")]
        if dynamic:
            self.sequenceSelector.setCurrentNode(dynamic[-1])

    def onSequenceChanged(self, node, openLayout=True):
        from DynamicPETEditor import DynamicPETEditorLogic
        self._clearResults()
        self._updating = True
        try:
            self.referenceCombo.clear()
            self.activities = []
            valid = node is not None and DynamicPETEditorLogic.isDynamicSequence(node)
            if valid:
                intervals = DynamicPETEditorLogic.frameIntervals(node)
                self.activities = self.logic.frameActivities(node)
                for interval in intervals:
                    self.referenceCombo.addItem(dynamicviews.frameLabel(interval))
                self.referenceCombo.setCurrentIndex(mc.maxActivityFrame(self.activities))
        finally:
            self._updating = False
        self.registerButton.enabled = valid
        self.openButton.enabled = valid
        self._updateInfo()
        if not valid:
            self.infoLabel.text = "Pick a dynamic PET sequence." if node is None else \
                "Not a sequence of volumes with more than one frame."
            self.onClose()
            return
        if openLayout or self.session is not None:
            self.onOpen()

    def _updateInfo(self, *args):
        node = self.sequenceSelector.currentNode()
        if not self.activities or node is None:
            return
        text = f"{len(self.activities)} frames."
        if self.lowActivityCheck.checked:
            low = sum(mc.lowActivityFrames(self.activities, self.lowActivitySpin.value))
            text += f" {low} below {self.lowActivitySpin.value:g} % of the max frame activity (not registered)."
        self.infoLabel.text = text
        self.lowActivitySpin.enabled = self.lowActivityCheck.checked

    def onReferenceChanged(self, index):
        if self._updating:
            return
        self._clearResults()
        if self.session is not None and self.summedSelector.currentNode() is None and index >= 0:
            self.session.setReference(frameIndex=index)
            self._syncWindowWidgets()

    def onSummedChanged(self, node):
        self.referenceCombo.enabled = node is None
        self._clearResults()
        if self.session is not None:
            if node is None:
                self.session.setReference(frameIndex=self.referenceFrame())
            else:
                self.session.setReference(volume=node)
            self._syncWindowWidgets()

    # --- layout --------------------------------------------------------------------------------
    def onOpen(self):
        node = self.sequenceSelector.currentNode()
        if node is None:
            return
        self._closeReviewEdit()
        results = self.results
        self._closeSession()
        try:
            self.session = MotionCorrectionSession(node, self.maxViewsSpin.value, self.orientationCombo.currentText,
                                                   self.linkCheck.checked)
            self.session.opacity = self.opacitySlider.value
            self.session.frameWindows = self.frameWindowCheck.checked
            self.session.frameColor = self.frameColorCombo.currentText
            self.session.referenceColor = self.referenceColorCombo.currentText
            self.session.showCorrected = self.showCorrectedRadio.checked
            self.session.open()
            summed = self.summedSelector.currentNode()
            if summed is not None:
                self.session.setReference(volume=summed)
            else:
                self.session.setReference(frameIndex=self.referenceFrame())
            if results:
                self.session.setResults(results, mc.showsMotionRows(self.resultsSettings["method"]),
                                        hardened=self.hardened)
        except Exception as error:
            logging.exception("Could not open Dynamic PET Motion Correction")
            slicer.util.errorDisplay(f"Could not open the motion correction layout:\n{error}")
            self._closeSession()
        self.closeButton.enabled = self.session is not None
        self._syncWindowWidgets()

    def _closeSession(self):
        if self.session is not None:
            self.session.close()
            self.session = None

    def onClose(self):
        self._closeSession()
        if hasattr(self, "closeButton"):
            self.closeButton.enabled = False
            self._syncWindowWidgets()

    @staticmethod
    def _closeReviewEdit():
        """Review/Edit and this module both take the main layout and a timeline dock: close Review/Edit first."""
        try:
            widget = getattr(slicer.modules, "DynamicPETEditorWidget", None)
            if widget is not None and getattr(widget, "session", None) is not None:
                widget.onClose()
        except Exception:
            logging.warning("Could not close Dynamic PET Review/Edit", exc_info=True)

    def onOrientationChanged(self, index):
        if self.session is not None:
            self.session.setOrientation(self.orientationCombo.currentText)

    def onOpacityChanged(self, value):
        if self.session is not None:
            self.session.setOpacity(value)

    def onFrameWindowToggled(self, checked):
        self.frameWindowRange.enabled = self.session is not None and not checked
        if self.session is not None:
            self.session.setFrameWindows(checked)

    def _syncWindowWidgets(self, auto=False):
        """The window sliders: their range from the volumes, their values from the session."""
        session = self.session
        for widget in (self.frameWindowRange, self.referenceWindowRange, self.autoWindowButton):
            widget.enabled = session is not None
        if session is None:
            return
        if auto:
            session.frameWindow = None
        frame = session.frameWindow or session.autoFrameWindow()
        for widget, maximum, values in ((self.frameWindowRange, session.frameMaximum, frame),
                                        (self.referenceWindowRange, session.referenceMaximum,
                                         session.referenceWindow or (0.0, 1.0))):
            widget.blockSignals(True)
            try:
                top = max(maximum, values[1], 1e-6)
                widget.singleStep = max(top / 1000.0, 1e-4)
                widget.setRange(min(0.0, values[0]), top)
                widget.setValues(*values)
            finally:
                widget.blockSignals(False)
        self.frameWindowRange.enabled = not self.frameWindowCheck.checked

    def onFrameWindowChanged(self, lower, upper):
        if self.session is not None:
            self.session.setFrameWindow(lower, upper)

    def onReferenceWindowChanged(self, lower, upper):
        if self.session is not None:
            self.session.setReferenceWindow(lower, upper)

    def onAutoWindows(self):
        if self.session is None:
            return
        self.session.frameWindow = None
        self.session.setFrameWindows(self.frameWindowCheck.checked)
        reference = self.summedSelector.currentNode()
        if reference is not None:
            self.session.setReference(volume=reference)
        else:
            self.session.setReference(frameIndex=self.referenceFrame())
        self._syncWindowWidgets(auto=True)

    def onFrameColorChanged(self, index):
        if self.session is not None:
            self.session.setFrameColor(self.frameColorCombo.currentText)

    def onReferenceColorChanged(self, index):
        if self.session is not None:
            self.session.setReferenceColor(self.referenceColorCombo.currentText)

    def onLinkToggled(self, checked):
        if self.session is not None:
            self.session.setLinked(checked)

    def onShowCorrectedToggled(self, corrected):
        if self.session is not None:
            self.session.setShowCorrected(corrected)

    # --- registration --------------------------------------------------------------------------
    def _clearResults(self):
        """Forget the registration (and, after hardening, the chance to revert it: the sequence stays hardened)."""
        self.results = None
        self.resultsSettings = None
        self.undo = None
        self.hardened = False
        if hasattr(self, "hardenButton"):
            self._updateButtons()
            self.resultLabel.text = "Not registered yet."
        if getattr(self, "session", None) is not None:
            self.session.setResults(None, False)

    def _updateButtons(self):
        pending = bool(self.results) and not self.hardened
        self.hardenButton.enabled = pending and any(r["status"] == mc.REGISTERED for r in self.results)
        self.revertButton.enabled = pending or self.undo is not None
        self.showWidget.enabled = pending

    def onRegister(self):
        sequence = self.sequenceSelector.currentNode()
        if sequence is None:
            return
        if self.session is None:
            self.onOpen()
        if self.session is None:
            return
        settings = self.settings()
        referenceFrame = self.referenceFrame()
        reference = self.session.referenceVolume
        count = sequence.GetNumberOfDataNodes()
        dialog = slicer.util.createProgressDialog(windowTitle="Dynamic PET Motion Correction",
                                                  labelText="Registering frames...", maximum=count)

        def progress(index, total, text):
            dialog.maximum = total
            dialog.setValue(index)
            dialog.setLabelText(f"Registering: {text}")
            slicer.app.processEvents()
            return not dialog.wasCanceled

        qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)
        try:
            results = self.logic.registerFrames(sequence, reference, referenceFrame, settings, progress)
        except Exception as error:
            logging.exception("Motion correction failed")
            slicer.util.errorDisplay(f"Motion correction failed:\n{error}")
            return
        finally:
            qt.QApplication.restoreOverrideCursor()
            dialog.close()
        self.results, self.resultsSettings, self.undo, self.hardened = results, settings, None, False
        self.session.setResults(results, mc.showsMotionRows(settings["method"]))
        self._updateButtons()
        method = mc.methodText(settings["method"], settings["twoPass"], settings["secondMethod"])
        self.resultLabel.text = (f"{method} to {self.session.referenceText}: "
                                 f"{mc.summaryText([r['status'] for r in results])}.")
        if any(r["status"] == mc.NOT_RUN for r in results):
            self.resultLabel.text += " Stopped before the last frame."

    def onHarden(self):
        sequence = self.sequenceSelector.currentNode()
        if sequence is None or not self.results or self.hardened:
            return
        results, settings = self.results, self.resultsSettings
        changed = sum(1 for r in results if r["image"] is not None)
        resum = self.resumCheck.checked
        removeOriginals = self.removeOriginalsCheck.checked
        summedSource = self.summedSelector.currentNode() or self.logic.summedVolumeOf(sequence)
        text = (f"This makes the motion corrected series of '{sequence.GetName()}': {changed} corrected frame(s) "
                f"and the other {len(results) - changed} frame(s) as they are (reference, low activity, ...)"
                + (", and a new summed image" if resum else "") + ".\n\n")
        if removeOriginals:
            text += ("The original series" + (" and its summed image" if resum and summedSource is not None else "")
                     + " will be DELETED from the scene. This cannot be undone.")
        else:
            text += ("The original series stays. Revert changes removes the new one until you pick another series "
                     "or reference, or close the scene.")
        text += "\n\nSave the scene to keep the motion corrected series.\n\nAre you sure?"
        if not slicer.util.confirmYesNoDisplay(text, windowTitle="Harden transforms"):
            return
        usedSummed = self.summedSelector.currentNode()
        referenceIndex = self.referenceCombo.currentIndex
        referenceText = self.session.referenceText if self.session else ""
        qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)
        try:
            undo, corrected, table, summed = self.logic.harden(sequence, results, settings, referenceText, resum,
                                                               summedSource)
            # show the new series (the selectors forget the results, they come back below), then drop the old one
            self._selectSeries(corrected, (summed or usedSummed) if usedSummed is not None else None,
                               referenceIndex)
            removed = []
            if removeOriginals:
                removed = self.logic.removeOriginals(sequence, summedSource, removeSummed=summed is not None)
                undo = None
        except Exception as error:
            logging.exception("Could not harden the motion correction")
            slicer.util.errorDisplay(f"Could not harden the motion correction:\n{error}")
            return
        finally:
            qt.QApplication.restoreOverrideCursor()
        self.results, self.resultsSettings, self.undo, self.hardened = results, settings, undo, True
        if self.session is not None:
            self.session.setResults(results, mc.showsMotionRows(settings["method"]), hardened=True)
        self._updateButtons()
        method = mc.methodText(settings["method"], settings["twoPass"], settings["secondMethod"])
        self.resultLabel.text = (f"{method} to {referenceText}: {mc.summaryText([r['status'] for r in results])}."
                                 f"\nHardened into '{corrected.GetName()}' (table '{table.GetName()}'"
                                 + (f", new summed image '{summed.GetName()}'" if summed is not None else "") + ")."
                                 + (f"\nRemoved: {', '.join(removed)}." if removed else ""))

    def _selectSeries(self, sequenceNode, summedNode, referenceIndex):
        """Pick a series (and its summed image as reference, or the frame referenceIndex) in the selectors."""
        self.sequenceSelector.setCurrentNode(sequenceNode)
        if summedNode is not self.summedSelector.currentNode():
            self.summedSelector.setCurrentNode(summedNode)
        if summedNode is None and 0 <= referenceIndex < self.referenceCombo.count:
            self.referenceCombo.setCurrentIndex(referenceIndex)

    def onRevert(self):
        if self.undo is not None:
            undo = self.undo
            try:
                original = undo["original"]
                if slicer.mrmlScene.IsNodePresent(original):
                    usedSummed = self.summedSelector.currentNode() is not None
                    summed = undo["originalSummed"] if usedSummed else None
                    if summed is not None and not slicer.mrmlScene.IsNodePresent(summed):
                        summed = None
                    self._selectSeries(original, summed, self.referenceCombo.currentIndex)
                self.logic.revert(undo)
            except Exception as error:
                logging.exception("Could not revert the motion correction")
                slicer.util.errorDisplay(f"Could not revert the motion correction:\n{error}")
                return
        self._clearResults()
        self.resultLabel.text = "Reverted: the original frames are shown."

    # --- scene ---------------------------------------------------------------------------------
    def enter(self):
        if self.sequenceSelector.currentNode() is None:
            self.selectDynamicSequence()
        if self.session is None and self.sequenceSelector.currentNode() is not None:
            self.onOpen()

    def exit(self):
        """Leaving the module: close the layout and its timeline and go back to the four-up layout. The
        registration is kept: coming back opens the layout with it again."""
        if self.session is not None:
            self.onClose()
            slicer.app.layoutManager().setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutFourUpView)

    def cleanup(self):
        self.onClose()
        self.removeObservers()
        ScriptedLoadableModuleWidget.cleanup(self)

    def onSceneClose(self, caller, event):
        self.onClose()
        self.results = None
        self.undo = None
        self.hardened = False

    @vtk.calldata_type(vtk.VTK_OBJECT)
    def onNodeRemoved(self, caller, event, node):
        if self.session is not None and node in (self.session.sequenceNode, self.summedSelector.currentNode()):
            qt.QTimer.singleShot(0, self.onClose)


# ---------------------------------------------------------------------------
# Test: a synthetic dynamic series with one moved frame and one empty frame
# ---------------------------------------------------------------------------
class DynamicPETMotionCorrectionTest(ScriptedLoadableModuleTest):
    SPACING = 4.0
    SHIFT_VOXELS = 2           # frame 4 is moved 8 mm along R

    def setUp(self):
        slicer.mrmlScene.Clear(0)

    def runTest(self):
        self.setUp()
        self.test_Helpers()
        self.setUp()
        self.test_Register()

    def test_Helpers(self):
        angles = (3.0, -2.0, 5.0)
        matrix = np.eye(4)
        matrix[:3, :3] = mc.rotationMatrix(*angles)
        matrix[:3, 3] = (1.0, 2.0, 3.0)
        shift, found = mc.rigidMotion(matrix, (0.0, 0.0, 0.0))
        np.testing.assert_allclose(found, angles, atol=1e-9)
        np.testing.assert_allclose(shift, (1.0, 2.0, 3.0), atol=1e-9)
        self.assertEqual(mc.lowActivityFrames([0.5, 100.0, 50.0]), [True, False, False])

    @classmethod
    def phantom(cls, shape=(40, 48, 48)):
        """Body ellipsoid with two hot spheres (k, j, i)."""
        k, j, i = np.indices(shape, dtype=float)
        center = [(s - 1) / 2.0 for s in shape]
        body = (((k - center[0]) / 16) ** 2 + ((j - center[1]) / 14) ** 2 + ((i - center[2]) / 18) ** 2) <= 1
        image = 1000.0 * body
        for (kc, jc, ic), r, value in (((16, 20, 18), 4, 8000.0), ((24, 26, 30), 5, 5000.0)):
            image += value * (((k - kc) ** 2 + (j - jc) ** 2 + (i - ic) ** 2) <= r * r)
        for _ in range(2):          # a little blur (3-voxel box filter along each axis, numpy only)
            for axis in range(3):
                image = (np.roll(image, -1, axis) + image + np.roll(image, 1, axis)) / 3.0
        return image.astype(np.float32)

    def makeSequence(self):
        base = self.phantom()
        sequence = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSequenceNode", "Synthetic dynamic Bq/ml")
        sequence.SetIndexName("Time")
        sequence.SetIndexUnit("s")
        sequence.SetAttribute(PREFIX + "Kind", "dynamic")
        for f in range(5):
            array = base.copy()
            if f == 1:
                array *= 0.001                                        # "early" frame: below 1 %
            if f == 3:
                array = np.roll(array, self.SHIFT_VOXELS, axis=2)     # moved along i (R in this geometry)
            node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", f"F{f + 1}")
            slicer.util.updateVolumeFromArray(node, array)
            node.SetSpacing(self.SPACING, self.SPACING, self.SPACING)
            node.SetIJKToRASDirections(1, 0, 0, 0, 1, 0, 0, 0, 1)
            node.SetAttribute(PREFIX + "Frame", str(f + 1))
            node.SetAttribute(PREFIX + "StartFromStudyStartS", str(60.0 * f))
            node.SetAttribute(PREFIX + "DurationS", "60")
            node.SetAttribute(PREFIX + "RadionuclideHalfLifeS", "1223.4")   # C-11
            sequence.SetDataNodeAtValue(node, str(60.0 * f + 30.0))
            slicer.mrmlScene.RemoveNode(node)
        return sequence

    def test_Register(self):
        sequence = self.makeSequence()
        logic = DynamicPETMotionCorrectionLogic()
        reference = logic.helperVolume("reference", sequence.GetNthDataNode(0))
        settings = logic.defaultSettings()
        results = logic.registerFrames(sequence, reference, 0, settings)
        statuses = [r["status"] for r in results]
        self.assertEqual(statuses, [mc.REFERENCE, mc.LOW_ACTIVITY, mc.REGISTERED, mc.REGISTERED, mc.REGISTERED])
        shift = results[3]["shift"]
        expected = self.SHIFT_VOXELS * self.SPACING
        self.assertAlmostEqual(shift[0], expected, delta=1.5)
        self.assertLess(max(abs(shift[1]), abs(shift[2])), 1.5)
        self.assertLess(max(abs(a) for a in results[3]["angles"]), 2.0)
        self.assertLess(max(abs(v) for v in results[2]["shift"]), 1.0)
        corrected = slicer.util.arrayFromVolume(reference).astype(float)   # frame 4 moved back onto frame 1
        from vtk.util import numpy_support
        image = results[3]["image"]
        moved = numpy_support.vtk_to_numpy(image.GetPointData().GetScalars()).reshape(image.GetDimensions()[::-1])
        before = np.abs(self.phantom() - np.roll(self.phantom(), self.SHIFT_VOXELS, axis=2)).mean()
        self.assertLess(np.abs(moved - corrected).mean(), 0.5 * before)

        # a motion limit below the shift rejects the moved frame
        settings["maxMotionMm"] = 2.0
        limited = logic.registerFrames(sequence, reference, 0, settings)
        self.assertEqual(limited[3]["status"], mc.OVER_LIMIT)
        self.assertIsNone(limited[3]["image"])

        # two passes: the second to the summed image of the first pass, rigid again, gives the same motion
        twoPass = dict(logic.defaultSettings(), twoPass=True, secondMethod="Rigid")
        steps = []
        second = logic.registerFrames(sequence, reference, 0, twoPass, lambda i, n, text: steps.append((i, n)))
        self.assertEqual([r["status"] for r in second], statuses)
        self.assertEqual(steps[-1], (9, 10))                               # 5 frames, 2 passes
        self.assertAlmostEqual(second[3]["shift"][0], self.SHIFT_VOXELS * self.SPACING, delta=1.5)
        image = second[3]["image"]
        moved = numpy_support.vtk_to_numpy(image.GetPointData().GetScalars()).reshape(image.GetDimensions()[::-1])
        self.assertLess(np.abs(moved - corrected).mean(), 0.5 * before)
        self.assertEqual([n for n in slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode")
                          if n.GetName() == "MC first pass sum"], [])

        # harden makes a motion corrected series with every frame (the low activity one unchanged), the originals
        # stay; revert removes the new series
        original = slicer.util.arrayFromVolume(sequence.GetNthDataNode(3)).copy()
        lowBefore = slicer.util.arrayFromVolume(sequence.GetNthDataNode(1)).copy()
        undo, hardenedSequence, table, summed = logic.harden(sequence, results, logic.defaultSettings(), "frame 1")
        self.assertEqual(hardenedSequence.GetName(), sequence.GetName() + CORRECTED_SUFFIX)
        self.assertEqual(hardenedSequence.GetNumberOfDataNodes(), 5)
        self.assertEqual([hardenedSequence.GetNthIndexValue(i) for i in range(5)],
                         [sequence.GetNthIndexValue(i) for i in range(5)])
        hardened = slicer.util.arrayFromVolume(hardenedSequence.GetNthDataNode(3))
        self.assertLess(np.abs(hardened - corrected).mean(), 0.5 * before)
        np.testing.assert_array_equal(slicer.util.arrayFromVolume(sequence.GetNthDataNode(3)), original)
        np.testing.assert_array_equal(slicer.util.arrayFromVolume(hardenedSequence.GetNthDataNode(1)), lowBefore)
        self.assertEqual(hardenedSequence.GetNthDataNode(3).GetAttribute(MC_PREFIX + "Status"), mc.REGISTERED)
        self.assertEqual(hardenedSequence.GetNthDataNode(1).GetAttribute(MC_PREFIX + "Status"), mc.LOW_ACTIVITY)
        self.assertEqual(hardenedSequence.GetNthDataNode(1).GetAttribute(PREFIX + "DurationS"),
                         sequence.GetNthDataNode(1).GetAttribute(PREFIX + "DurationS"))
        self.assertIsNotNone(slicer.modules.sequences.logic().GetFirstBrowserNodeForSequenceNode(hardenedSequence))
        self.assertEqual(table.GetNumberOfRows(), 5)
        self.assertEqual(hardenedSequence.GetAttribute(PREFIX + "SummedVolumeID"), summed.GetID())
        durations = [60.0] * 5
        expected = sum(slicer.util.arrayFromVolume(hardenedSequence.GetNthDataNode(i)).astype(float) * durations[i]
                       for i in range(5)) / sum(durations)
        np.testing.assert_allclose(slicer.util.arrayFromVolume(summed), expected, rtol=1e-4, atol=1e-3)
        logic.revert(undo)
        for node in [hardenedSequence, summed, table]:
            self.assertFalse(slicer.mrmlScene.IsNodePresent(node))
        self.assertTrue(slicer.mrmlScene.IsNodePresent(sequence))

        # with the originals removed only the motion corrected series and summed image stay
        from DynamicPETEditor import DynamicPETEditorLogic
        oldSummed = logic.resum(sequence)
        oldSummed.SetName("Synthetic summed")
        oldSummed.SetAttribute(PREFIX + "SumWeighting", "counts")        # loaded with count weighting
        sequence.SetAttribute(PREFIX + "SummedVolumeID", oldSummed.GetID())
        browser = DynamicPETEditorLogic.browserFor(sequence)
        undo, kept, keptTable, keptSummed = logic.harden(sequence, results, logic.defaultSettings(), "frame 1")
        logic.removeOriginals(sequence, oldSummed)
        for node in [sequence, oldSummed, browser]:
            self.assertFalse(slicer.mrmlScene.IsNodePresent(node))
        self.assertEqual(kept.GetNumberOfDataNodes(), 5)
        self.assertEqual(keptSummed.GetName(), "Synthetic summed" + CORRECTED_SUFFIX)
        # the new summed image follows the count weighting of the old one
        from DICOMGatedDynamicPETLib import petseries
        weights, _ = petseries.summationWeights([60.0] * 5, [60.0 * f for f in range(5)], 1223.4, "counts")
        self.assertGreater(weights[0] / weights[-1], 1.1)
        countMean = petseries.weightedMean([slicer.util.arrayFromVolume(kept.GetNthDataNode(i)) for i in range(5)],
                                           weights)
        np.testing.assert_allclose(slicer.util.arrayFromVolume(keptSummed), countMean, rtol=1e-4, atol=1e-3)
        self.assertEqual(keptSummed.GetAttribute(PREFIX + "SumWeighting"), "counts")
        self.assertIn("count weighted", keptSummed.GetAttribute(PREFIX + "StaticFrom"))
        np.testing.assert_array_equal(slicer.util.arrayFromVolume(kept.GetNthDataNode(1)), lowBefore)
        sequence = kept

        # the layout opens with the reference in the large view and the frames over it
        session = MotionCorrectionSession(sequence)
        session.open()
        try:
            session.setReference(frameIndex=0)
            session.setResults(results, True)
            layoutManager = slicer.app.layoutManager()
            self.assertEqual(layoutManager.layout, LAYOUT_ID)
            main = layoutManager.sliceWidget(MAIN_VIEW)
            self.assertEqual(main.mrmlSliceCompositeNode().GetBackgroundVolumeID(), session.referenceVolume.GetID())
            self.assertEqual(main.mrmlSliceNode().GetOrientation(), DEFAULT_ORIENTATION)
            view = layoutManager.sliceWidget(dynamicviews.frameViewName(3, FRAME_VIEW_PREFIX))
            self.assertEqual(view.mrmlSliceCompositeNode().GetForegroundVolumeID(), session.frameVolumes[3].GetID())
            self.assertEqual(len(session.timeline.motion), 5)
            self.assertEqual([len(plot.motion) for plot in session.plots], [5, 5])
            session.setFrameColor("Hel")
            self.assertEqual(session.frameVolumes[0].GetDisplayNode().GetColorNodeID(),
                             colormaps.colorNodeForLabel("Hel").GetID())
            # a volume shown everywhere (as when one is loaded) stays out of the editor's views
            slicer.util.setSliceViewerLayers(background=reference, foreground=None)
            empty = layoutManager.sliceWidget(dynamicviews.emptyViewName(0, EMPTY_VIEW_PREFIX))
            self.assertIsNone(empty.mrmlSliceCompositeNode().GetBackgroundVolumeID())
            self.assertEqual(main.mrmlSliceCompositeNode().GetBackgroundVolumeID(), session.referenceVolume.GetID())
            self.assertEqual(view.mrmlSliceCompositeNode().GetForegroundVolumeID(), session.frameVolumes[3].GetID())
            session.setReferenceWindow(10.0, 500.0)
            self.assertEqual(session.referenceVolume.GetDisplayNode().GetWindowLevelMax(), 500.0)
            session.setFrameWindow(0.0, 300.0)
            self.assertEqual(session.frameVolumes[0].GetDisplayNode().GetWindowLevelMax(), 300.0)
            session.setOrientation("Axial")
            self.assertEqual(view.mrmlSliceNode().GetOrientation(), "Axial")
            self.delayDisplay("Dynamic PET Motion Correction open")
        finally:
            session.close()
        helpers = [n for n in slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode")
                   if n.GetAttribute(HELPER_ATTRIBUTE) and n is not reference]
        self.assertEqual(helpers, [])
