"""
Neuro PET Database Manager (3D Slicer module)

Normal databases of brain PET and SPECT in the MNI ICBM 152 2009c space, kept in a database folder picked by the
user (one folder per database, see DICOMGatedDynamicPETLib/normaldb.py for the layout):

    Static tab      the static databases of the folder in a table (name, kind, nuclide, tracer, subjects, reference
                    region, age groups, statistics), with
                        Create new database...      an empty user defined database (the subjects are added later
                                                    from the Neuro workflows, its mean / SD maps computed from them)
                        Add predefined database...  a packaged database (.zip) or mean / SD maps picked one by one
                                                    (master and each age group, with the number of subjects)
                        Export as predefined...     a user defined database as a .zip without its subject images
                        Delete database             the folder removed from disk, after a warning
    Dynamic tab     to come

The reference region mask is copied into every database: made from the CerebrA atlas or the MNI brain mask of the
Neuro atlas folder (the one of the Dynamic Neuro Receptor PET module), or a mask file of the user.
"""

import logging
import os
import sys

import ctk
import numpy as np
import qt
import slicer
import vtk
from slicer.ScriptedLoadableModule import (ScriptedLoadableModule, ScriptedLoadableModuleWidget,
                                           ScriptedLoadableModuleLogic, ScriptedLoadableModuleTest)

_MODULE_DIR = os.path.dirname(os.path.abspath(globals().get("__file__", "") or "."))
if globals().get("__file__") and _MODULE_DIR not in sys.path:
    sys.path.insert(0, _MODULE_DIR)

from DICOMGatedDynamicPETLib import dynamicviews, neuro, normaldb  # noqa: E402

SETTINGS_ROOT = "DICOMGatedDynamicPET/NeuroDatabase/RootFolder"
SETTINGS_ATLAS_FOLDER = "DICOMGatedDynamicPET/Neuro/AtlasFolder"     # shared with Dynamic Neuro Receptor PET
COLUMNS = (("name", "Name"), ("kind", "Type"), ("modality", "Modality"), ("nuclide", "Nuclide"), ("tracer", "Tracer"),
           ("subjects", "Subjects"), ("reference", "Reference region"), ("ageGroups", "Age groups"),
           ("statistics", "Mean / SD"), ("version", "Version"), ("created", "Created"))
DEFAULT_SMOOTHING_MM = 8.0


class NeuroPETDatabaseManager(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        parent.title = "Neuro PET Database Manager"     # PET and SPECT
        parent.categories = ["Nuclear Medicine.PET4D"]
        parent.dependencies = []
        parent.contributors = ["Burak Demir"]
        parent.helpText = (
            "Normal databases of brain PET and SPECT in the MNI ICBM 152 2009c space. Pick the database folder; each "
            "database is a folder in it with its header (database.json), the reference region mask, the subjects (user "
            "defined databases) and the mean / SD maps (master and per age group). <b>Create new database</b> makes "
            "an empty user defined database, <b>Add predefined database</b> one from a package (.zip) or from mean / "
            "SD maps. A user defined database can be exported as a predefined one.")
        parent.acknowledgementText = (
            "Part of the DICOM Gated / Dynamic PET extension. MNI ICBM 152 2009c: Fonov VS et al., NeuroImage 2011;"
            "54:313-327. CerebrA: Manera AL et al., Scientific Data 2020;7:237 (CC0).")


def _button(text, role="action"):
    return dynamicviews.colorButton(qt.QPushButton(text), role)


def _number(text):
    text = str(text or "").strip().replace(",", ".")
    if not text:
        return None
    return float(text)


def _pathEdit(dirs=False, filters="Images (*.nii *.nii.gz *.nrrd *.nhdr *.mha *.mhd)"):
    edit = ctk.ctkPathLineEdit()
    edit.filters = ctk.ctkPathLineEdit.Dirs if dirs else ctk.ctkPathLineEdit.Files
    if not dirs:
        edit.nameFilters = [filters, "All files (*)"]
    edit.settingKey = ""
    return edit


# ---------------------------------------------------------------------------
# Logic
# ---------------------------------------------------------------------------
class NeuroPETDatabaseManagerLogic(ScriptedLoadableModuleLogic):
    """The database folder and the reference masks; the files themselves through normaldb (no Slicer)."""

    @staticmethod
    def rootFolder():
        return str(qt.QSettings().value(SETTINGS_ROOT) or "")

    @staticmethod
    def setRootFolder(folder):
        qt.QSettings().setValue(SETTINGS_ROOT, folder or "")

    @staticmethod
    def atlasFolder():
        return str(qt.QSettings().value(SETTINGS_ATLAS_FOLDER) or "")

    @staticmethod
    def setAtlasFolder(folder):
        qt.QSettings().setValue(SETTINGS_ATLAS_FOLDER, folder or "")

    @staticmethod
    def maskSourceProblem(regionKey, atlasFolder):
        """Why the mask of a reference region cannot be made from the atlas folder, or None."""
        if regionKey == normaldb.REF_CUSTOM:
            return None
        atlas = neuro.findAtlasFiles(atlasFolder)
        if regionKey == normaldb.REF_GLOBAL:
            return None if neuro.BRAIN_MASK in atlas else "The atlas folder has no MNI brain mask " \
                "(mni_icbm152_t1_tal_nlin_sym_09c_mask)"
        if regionKey == normaldb.REF_PONS:
            if neuro.PONS_LABELS not in atlas or neuro.PONS_CSV not in atlas:
                return (f"The atlas folder has no CerebrA with pons ({neuro.PONS_STEM}.seg.nrrd and "
                        f"{neuro.PONS_CSV_NAME})")
            return None
        if neuro.CEREBRA_LABELS not in atlas or neuro.CEREBRA_CSV not in atlas:
            return "The atlas folder has no CerebrA labels (mni_icbm152_CerebrA_tal_nlin_sym_09c and its CSV)"
        return None

    def writeReferenceMask(self, folder, header, atlasFolder, customPath=None):
        """Write the reference region mask of a database into its folder (reference_mask.nii.gz, or the custom
        file's format) and set header["referenceMask"]. Global: the MNI brain mask; cerebellum / brainstem: their
        CerebrA labels (both sides); custom: a copy of the file."""
        import shutil
        key = header.get("referenceRegion")
        if key == normaldb.REF_CUSTOM:
            if not customPath or not os.path.isfile(customPath):
                raise ValueError("Pick the custom reference mask file")
            relative = "reference_mask" + (normaldb.imageExtension(customPath) or ".nii.gz")
            shutil.copyfile(customPath, os.path.join(folder, relative))
            header["referenceMask"] = relative
            return relative
        problem = self.maskSourceProblem(key, atlasFolder)
        if problem:
            raise ValueError(problem)
        atlas = neuro.findAtlasFiles(atlasFolder)
        if key == normaldb.REF_PONS:
            return self._writeSegmentationMask(folder, header, atlas)
        if key == normaldb.REF_GLOBAL:
            source = slicer.util.loadLabelVolume(atlas[neuro.BRAIN_MASK], {"show": False})
            values = None
        else:
            structures = neuro.readCerebraLabels(atlas[neuro.CEREBRA_CSV])
            values = []
            for name in normaldb.referenceStructures(key) or ():
                found = neuro.structureValues(structures, name)
                if not found:
                    raise ValueError(f"CerebrA has no '{name}'")
                values.extend(found)
            source = slicer.util.loadLabelVolume(atlas[neuro.CEREBRA_LABELS], {"show": False})
        target = None
        try:
            array = slicer.util.arrayFromVolume(source)
            mask = (array > 0) if values is None else np.isin(np.rint(array).astype(np.int64), values)
            mask = neuro.erodeMask(mask, float(header.get("referenceErosionMm") or 0),
                                   tuple(float(x) for x in source.GetSpacing())[::-1])
            if not mask.any():
                raise ValueError("The reference mask is empty (eroded too much?)")
            target = slicer.modules.volumes.logic().CreateAndAddLabelVolume(slicer.mrmlScene, source,
                                                                           "reference mask")
            slicer.util.updateVolumeFromArray(target, mask.astype(np.uint8))
            relative = normaldb.REFERENCE_MASK_NAME
            if not slicer.util.saveNode(target, os.path.join(folder, relative)):
                raise ValueError("The reference mask could not be saved")
        finally:
            slicer.mrmlScene.RemoveNode(source)
            if target is not None:
                slicer.mrmlScene.RemoveNode(target)
        header["referenceMask"] = relative
        return relative

    def _writeSegmentationMask(self, folder, header, atlas):
        """Pons mask from the CerebrA with pons (a segmentation file): the segments of the Pons labels of its CSV,
        on the grid of the CerebrA labels (else of the T1 template, else the segmentation's own)."""
        structures = neuro.readCerebraLabels(atlas[neuro.PONS_CSV])
        values = neuro.structureValues(structures, neuro.PONS)
        if not values:
            raise ValueError(f"{neuro.PONS_CSV_NAME} has no '{neuro.PONS}' row")
        wanted = neuro.segmentNamesOfLabels(values, neuro.PONS)
        segmentationNode = slicer.util.loadSegmentation(atlas[neuro.PONS_LABELS], {"show": False})
        referencePath = atlas.get(neuro.CEREBRA_LABELS) or atlas.get(neuro.DEFAULT_TEMPLATE)
        reference = (slicer.util.loadVolume(referencePath, {"show": False, "labelmap": True})
                     if referencePath else None)
        labelNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode", "pons mask")
        try:
            segmentation = segmentationNode.GetSegmentation()
            ids = vtk.vtkStringArray()
            for index in range(segmentation.GetNumberOfSegments()):
                segmentId = segmentation.GetNthSegmentID(index)
                if segmentation.GetSegment(segmentId).GetName().strip().lower() in wanted:
                    ids.InsertNextValue(segmentId)
            if ids.GetNumberOfValues() == 0:
                raise ValueError(f"No pons segment in {os.path.basename(atlas[neuro.PONS_LABELS])} (segments named "
                                 + ", ".join(f"Segment_{v}" for v in values) + " expected)")
            if not slicer.modules.segmentations.logic().ExportSegmentsToLabelmapNode(
                    segmentationNode, ids, labelNode, reference):
                raise ValueError("The pons segments could not be exported")
            mask = neuro.erodeMask(slicer.util.arrayFromVolume(labelNode) > 0,
                                   float(header.get("referenceErosionMm") or 0),
                                   tuple(float(x) for x in labelNode.GetSpacing())[::-1])
            if not mask.any():
                raise ValueError("The pons mask is empty (eroded too much?)")
            slicer.util.updateVolumeFromArray(labelNode, mask.astype(np.uint8))
            relative = normaldb.REFERENCE_MASK_NAME
            if not slicer.util.saveNode(labelNode, os.path.join(folder, relative)):
                raise ValueError("The reference mask could not be saved")
        finally:
            for node in (segmentationNode, reference, labelNode):
                if node is not None:
                    slicer.mrmlScene.RemoveNode(node)
        header["referenceMask"] = relative
        return relative

    def createUserDefined(self, root, header, atlasFolder, customMask=None):
        """New empty user defined database with its reference mask. Returns the folder."""
        import shutil
        folder = normaldb.createDatabase(root, header)
        try:
            self.writeReferenceMask(folder, header, atlasFolder, customMask)
            normaldb.writeHeader(folder, header)
        except Exception:
            shutil.rmtree(folder, ignore_errors=True)
            raise
        return folder

    def createPredefined(self, root, header, maps, maskPath=None, atlasFolder=None):
        """Predefined database from mean / SD maps; the reference mask is the file given, else made from the atlas
        folder when it can be (else none)."""
        folder = normaldb.createPredefined(root, header, maps, maskPath)
        if not maskPath and atlasFolder and header.get("referenceRegion") != normaldb.REF_CUSTOM \
                and self.maskSourceProblem(header.get("referenceRegion"), atlasFolder) is None:
            saved = normaldb.readHeader(folder)
            try:
                self.writeReferenceMask(folder, saved, atlasFolder)
                normaldb.writeHeader(folder, saved)
            except Exception as e:
                logging.warning(f"Reference mask of {folder} not made: {e}")
        return folder


# ---------------------------------------------------------------------------
# Header form (shared by the dialogs)
# ---------------------------------------------------------------------------
class HeaderForm:
    """Fields of a database header: name, nuclide, tracer, acquisition, segmentation, reference region, smoothing,
    map type (predefined), scanner and description, age groups (with maps and subjects for a predefined one)."""

    def __init__(self, layout, predefined=False, atlasFolder=""):
        self.predefined = predefined
        self.widget = qt.QWidget()
        layout.addWidget(self.widget)
        outer = qt.QVBoxLayout(self.widget)
        outer.setContentsMargins(0, 0, 0, 0)

        box = qt.QGroupBox("Database")
        form = qt.QFormLayout(box)
        self.nameEdit = qt.QLineEdit()
        self.nameEdit.placeholderText = "e.g. FDG normals 2026"
        self.nameEdit.toolTip = "The database folder is named after it."
        form.addRow("Name:", self.nameEdit)
        self.modalityCombo = qt.QComboBox()
        self.modalityCombo.addItems(list(normaldb.MODALITIES))
        self.modalityCombo.toolTip = "PET (SUV / SUVR maps) or SPECT (counts / ratio maps)."
        form.addRow("Modality:", self.modalityCombo)
        self.nuclideCombo = qt.QComboBox()
        self.nuclideCombo.editable = True
        self.nuclideCombo.toolTip = "Nuclides of the modality; another one can be typed."
        form.addRow("Nuclide:", self.nuclideCombo)
        self.tracerCombo = qt.QComboBox()
        self.tracerCombo.editable = True
        self.tracerCombo.toolTip = ("Tracers of the nuclide. The nuclide itself means the tracer is not given; "
                                    "another tracer can be typed.")
        form.addRow("Tracer:", self.tracerCombo)
        acquisition = qt.QHBoxLayout()
        self.startSpin = qt.QDoubleSpinBox()
        self.startSpin.setRange(0, 100000)
        self.startSpin.setDecimals(1)
        self.startSpin.suffix = " min p.i."
        self.startSpin.value = 60
        self.durationSpin = qt.QDoubleSpinBox()
        self.durationSpin.setRange(0, 100000)
        self.durationSpin.setDecimals(1)
        self.durationSpin.suffix = " min"
        self.durationSpin.value = 10
        acquisition.addWidget(qt.QLabel("start"))
        acquisition.addWidget(self.startSpin)
        acquisition.addWidget(qt.QLabel("duration"))
        acquisition.addWidget(self.durationSpin)
        form.addRow("Acquisition:", acquisition)
        spaceLabel = qt.QLabel(f"{normaldb.SPACE}, {normaldb.VOXEL_MM:g} mm")
        spaceLabel.wordWrap = True
        form.addRow("Space:", spaceLabel)
        outer.addWidget(box)

        refBox = qt.QGroupBox("Normalization")
        refForm = qt.QFormLayout(refBox)
        self.segmentationEdit = qt.QLineEdit(normaldb.DEFAULT_SEGMENTATION)
        self.segmentationEdit.toolTip = "Atlas / segmentation of the regions."
        refForm.addRow("Segmentation:", self.segmentationEdit)
        self.referenceCombo = qt.QComboBox()
        for key, title, _ in normaldb.REFERENCE_REGIONS:
            self.referenceCombo.addItem(title, key)
        self.referenceCombo.currentIndex = [k for k, _, _ in normaldb.REFERENCE_REGIONS].index(
            normaldb.DEFAULT_REFERENCE)
        self.referenceCombo.toolTip = ("Region the images are divided by (SUVR). Its mask is copied into the "
                                       "database.")
        refForm.addRow("Reference region:", self.referenceCombo)
        self.maskPath = _pathEdit()
        self.maskPath.toolTip = ("Reference mask in the template space. Custom region: required. Otherwise made from "
                                 "the atlas folder (CerebrA / MNI brain mask)"
                                 + (", or this file when given." if predefined else "."))
        self.maskLabel = qt.QLabel("Mask file:")
        refForm.addRow(self.maskLabel, self.maskPath)
        self.atlasPath = _pathEdit(dirs=True)
        self.atlasPath.currentPath = atlasFolder
        self.atlasPath.toolTip = "Neuro atlas folder (MNI ICBM 152 2009c and CerebrA), shared with the Neuro workflow."
        self.atlasLabel = qt.QLabel("Atlas folder:")
        refForm.addRow(self.atlasLabel, self.atlasPath)
        self.smoothingSpin = qt.QDoubleSpinBox()
        self.smoothingSpin.setRange(0, 30)
        self.smoothingSpin.setDecimals(1)
        self.smoothingSpin.suffix = " mm FWHM"
        self.smoothingSpin.specialValueText = "Off"
        self.smoothingSpin.value = DEFAULT_SMOOTHING_MM
        self.smoothingSpin.toolTip = ("Gaussian smoothing of the maps before the mean / SD" if not predefined
                                      else "Smoothing applied to the maps of the database (for the record).")
        refForm.addRow("Smoothing:", self.smoothingSpin)
        self.erosionSpin = qt.QDoubleSpinBox()
        self.erosionSpin.setRange(0, 10)
        self.erosionSpin.setDecimals(1)
        self.erosionSpin.singleStep = 0.5
        self.erosionSpin.suffix = " mm"
        self.erosionSpin.specialValueText = "Off"
        self.erosionSpin.toolTip = ("Erode the reference mask made from the atlas (not a custom mask file), so "
                                    "spill-over from the neighbouring tissue does not reach it.")
        refForm.addRow("Erode reference:", self.erosionSpin)
        self.regionAtlasCombo = qt.QComboBox()
        for key, title in normaldb.REGION_ATLASES:
            self.regionAtlasCombo.addItem(title, key)
        self.regionAtlasCombo.toolTip = "Regions the patients are reported in (the Neuro workflows use the same)."
        refForm.addRow("Region atlas:", self.regionAtlasCombo)
        self.growSpin = qt.QDoubleSpinBox()
        self.growSpin.setRange(0, 10)
        self.growSpin.setDecimals(1)
        self.growSpin.singleStep = 0.5
        self.growSpin.suffix = " mm"
        self.growSpin.specialValueText = "Off"
        self.growSpin.toolTip = ("Grow the regions into the unlabelled voxels around them (each voxel to the nearest "
                                 "region, inside the brain mask), for the PET / SPECT resolution.")
        refForm.addRow("Grow regions:", self.growSpin)
        self.growWhiteMatterCheck = qt.QCheckBox("Grow into the white matter too")
        self.growWhiteMatterCheck.enabled = False
        self.growSpin.connect("valueChanged(double)", lambda v: setattr(self.growWhiteMatterCheck, "enabled", v > 0))
        refForm.addRow("", self.growWhiteMatterCheck)
        if predefined:
            self.mapTypeCombo = qt.QComboBox()
            self.mapTypeCombo.toolTip = ("What the mean / SD maps hold: PET SUVR (divided by the reference) or SUV, "
                                         "SPECT ratio (counts divided by the reference) or counts.")
            refForm.addRow("Map type:", self.mapTypeCombo)
        outer.addWidget(refBox)

        ageBox = qt.QGroupBox("Age groups" if not predefined else "Mean / SD maps")
        ageLayout = qt.QVBoxLayout(ageBox)
        self.ageCheck = qt.QCheckBox("Stratify by age")
        self.ageCheck.toolTip = ("Separate mean / SD maps per age group (ages in years, both ends inclusive, "
                                 "empty max: open). The master mean / SD of all subjects is always made.")
        ageLayout.addWidget(self.ageCheck)
        headers = ["Group", "Min age", "Max age"]
        if predefined:
            headers += ["Subjects", "Mean map", "SD map"]
        self.ageTable = qt.QTableWidget(0, len(headers))
        self.ageTable.setHorizontalHeaderLabels(headers)
        self.ageTable.verticalHeader().visible = False
        self.ageTable.horizontalHeader().setStretchLastSection(True)
        self.ageTable.minimumHeight = 140
        ageLayout.addWidget(self.ageTable)
        rowButtons = qt.QHBoxLayout()
        self.addGroupButton = qt.QPushButton("Add group")
        self.removeGroupButton = qt.QPushButton("Remove group")
        rowButtons.addWidget(self.addGroupButton)
        rowButtons.addWidget(self.removeGroupButton)
        rowButtons.addStretch(1)
        ageLayout.addLayout(rowButtons)
        outer.addWidget(ageBox)

        textBox = qt.QGroupBox("Notes")
        textForm = qt.QFormLayout(textBox)
        self.scannerEdit = qt.QLineEdit()
        self.scannerEdit.placeholderText = "scanner, reconstruction, ..."
        textForm.addRow("Scanner:", self.scannerEdit)
        self.descriptionEdit = qt.QPlainTextEdit()
        self.descriptionEdit.placeholderText = "Description, source, citation, inclusion criteria, ..."
        self.descriptionEdit.maximumHeight = 80
        textForm.addRow("Description:", self.descriptionEdit)
        outer.addWidget(textBox)

        if predefined:
            self._addRow(normaldb.MASTER_TITLE, master=True)
        self.addGroupButton.connect("clicked()", lambda: self._addRow())
        self.removeGroupButton.connect("clicked()", self._removeRow)
        self.ageCheck.connect("toggled(bool)", lambda on: self._updateAge())
        self.referenceCombo.connect("currentIndexChanged(int)", lambda i: self._updateReference())
        self._updateAge()
        self._updateReference()
        self.modalityCombo.connect("currentIndexChanged(int)", lambda i: self._updateModality())
        self.nuclideCombo.connect("currentTextChanged(QString)", self._updateTracers)
        self._updateModality()

    # --- modality, nuclide, tracer --------------------------------------------------------------------
    def _updateModality(self):
        modality = self.modalityCombo.currentText
        self.nuclideCombo.blockSignals(True)
        self.nuclideCombo.clear()
        self.nuclideCombo.addItems(normaldb.nuclides(modality))
        self.nuclideCombo.currentIndex = 0
        self.nuclideCombo.blockSignals(False)
        self._updateTracers(self.nuclideCombo.currentText)
        if self.predefined:
            self.mapTypeCombo.clear()
            self.mapTypeCombo.addItems(list(normaldb.mapTypes(modality)))

    def _updateTracers(self, nuclide):
        """The tracers of the nuclide: the nuclide itself (tracer not given) first, then the second one (the usual
        tracer) selected."""
        options = normaldb.tracers(nuclide, self.modalityCombo.currentText)
        self.tracerCombo.clear()
        self.tracerCombo.addItems(options)
        self.tracerCombo.currentIndex = 1 if len(options) > 1 else 0

    # --- age table -----------------------------------------------------------------------------------
    def _firstGroupRow(self):
        return 1 if self.predefined else 0

    def _addRow(self, name="", low="", high="", master=False):
        row = self.ageTable.rowCount
        self.ageTable.insertRow(row)
        item = qt.QTableWidgetItem(name)
        if master:
            item.setFlags(qt.Qt.ItemIsEnabled)
        self.ageTable.setItem(row, 0, item)
        for column, text in ((1, low), (2, high)):
            cell = qt.QTableWidgetItem("" if master else str(text))
            if master:
                cell.setFlags(qt.Qt.ItemIsEnabled)
            self.ageTable.setItem(row, column, cell)
        if self.predefined:
            spin = qt.QSpinBox()
            spin.setRange(0, 1000000)
            self.ageTable.setCellWidget(row, 3, spin)
            for column in (4, 5):
                self.ageTable.setCellWidget(row, column, _pathEdit())
        self.ageTable.resizeColumnsToContents()

    def _removeRow(self):
        row = self.ageTable.currentRow()
        if row >= self._firstGroupRow():
            self.ageTable.removeRow(row)

    def _updateAge(self):
        on = self.ageCheck.checked
        self.addGroupButton.enabled = on
        self.removeGroupButton.enabled = on
        if not self.predefined:
            self.ageTable.enabled = on
        else:
            for row in range(self._firstGroupRow(), self.ageTable.rowCount):
                self.ageTable.setRowHidden(row, not on)
        if on and self.ageTable.rowCount == self._firstGroupRow():
            for name, low, high in (("20-39", 20, 39), ("40-59", 40, 59), ("60+", 60, "")):
                self._addRow(name, low, high)

    def _updateReference(self):
        custom = self.referenceKey() == normaldb.REF_CUSTOM
        self.maskPath.visible = custom or self.predefined
        self.maskLabel.visible = custom or self.predefined
        self.maskLabel.text = "Mask file:" if custom else "Mask file (optional):"

    def referenceKey(self):
        return self.referenceCombo.itemData(self.referenceCombo.currentIndex)

    def ageGroups(self):
        groups = []
        for row in range(self._firstGroupRow(), self.ageTable.rowCount):
            def text(column):
                item = self.ageTable.item(row, column)
                return item.text().strip() if item else ""
            try:
                low, high = _number(text(1)), _number(text(2))
            except ValueError:
                low, high = text(1), text(2)        # reported by the validation
            groups.append({"name": text(0), "min": low, "max": high})
        return groups

    def maps(self):
        """{MASTER or age group name: {"mean", "sd", "n"}} of a predefined database."""
        maps = {}
        groups = self.ageGroups() if self.ageCheck.checked else []
        rows = [(0, normaldb.MASTER)] + [(self._firstGroupRow() + i, g["name"]) for i, g in enumerate(groups)]
        for row, name in rows:
            maps[name] = {"n": self.ageTable.cellWidget(row, 3).value,
                          "mean": self.ageTable.cellWidget(row, 4).currentPath,
                          "sd": self.ageTable.cellWidget(row, 5).currentPath}
        return maps

    def header(self):
        stratified = bool(self.ageCheck.checked)
        fields = dict(
            modality=self.modalityCombo.currentText,
            nuclide=self.nuclideCombo.currentText.strip(),
            tracer=self.tracerCombo.currentText.strip(),
            acquisition={"startMinutes": float(self.startSpin.value),
                         "durationMinutes": float(self.durationSpin.value)},
            segmentation=self.segmentationEdit.text.strip(),
            referenceRegion=self.referenceKey(),
            smoothingFwhmMm=float(self.smoothingSpin.value),
            referenceErosionMm=float(self.erosionSpin.value),
            regionAtlas=self.regionAtlasCombo.itemData(self.regionAtlasCombo.currentIndex),
            regionGrowMm=float(self.growSpin.value),
            growIntoWhiteMatter=bool(self.growWhiteMatterCheck.checked) and self.growSpin.value > 0,
            ageStratified=stratified,
            ageGroups=self.ageGroups() if stratified else [],
            scanner=self.scannerEdit.text.strip(),
            description=self.descriptionEdit.toPlainText().strip(),
        )
        if self.predefined:
            fields["mapType"] = self.mapTypeCombo.currentText
        return normaldb.newHeader(self.nameEdit.text.strip(), normaldb.STATIC,
                                  normaldb.PREDEFINED if self.predefined else normaldb.USER_DEFINED, **fields)


def _scrollDialog(title, parent):
    dialog = qt.QDialog(parent)
    dialog.setWindowTitle(title)
    dialog.resize(720, 760)
    outer = qt.QVBoxLayout(dialog)
    scroll = qt.QScrollArea()
    scroll.widgetResizable = True
    content = qt.QWidget()
    layout = qt.QVBoxLayout(content)
    scroll.setWidget(content)
    outer.addWidget(scroll, 1)
    buttons = qt.QDialogButtonBox()
    outer.addWidget(buttons)
    return dialog, layout, buttons


def _error(parent, text):
    slicer.util.errorDisplay(text, parent=parent, windowTitle="Neuro PET Database Manager")


# ---------------------------------------------------------------------------
# Widget
# ---------------------------------------------------------------------------
class NeuroPETDatabaseManagerWidget(ScriptedLoadableModuleWidget):
    """Database folder on top, the Static / Dynamic tabs below: the table of the databases, details of the selected
    one and the database buttons."""

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)
        self.logic = NeuroPETDatabaseManagerLogic()
        self.databases = []         # [(folder, header)] of the static table

        rootBox = qt.QGroupBox("Database folder")
        rootLayout = qt.QVBoxLayout(rootBox)
        self.rootPath = _pathEdit(dirs=True)
        self.rootPath.toolTip = "Folder that holds the databases (one folder per database). Kept in the settings."
        self.rootPath.currentPath = self.logic.rootFolder()
        rootLayout.addWidget(self.rootPath)
        self.rootInfo = qt.QLabel()
        self.rootInfo.wordWrap = True
        rootLayout.addWidget(self.rootInfo)
        self.layout.addWidget(rootBox)

        self.tabs = qt.QTabWidget()
        self.layout.addWidget(self.tabs, 1)
        self.tabs.addTab(self._buildStaticTab(), "Static")
        dynamic = qt.QLabel("Dynamic databases (BPnd and other parametric maps) come next.")
        dynamic.wordWrap = True
        dynamic.alignment = qt.Qt.AlignTop
        self.tabs.addTab(dynamic, "Dynamic")

        self.rootPath.connect("currentPathChanged(QString)", self._onRootChanged)
        self.refresh()

    def _buildStaticTab(self):
        page = qt.QWidget()
        layout = qt.QVBoxLayout(page)
        self.table = qt.QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels([title for _, title in COLUMNS])
        self.table.setSelectionBehavior(qt.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(qt.QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(qt.QAbstractItemView.NoEditTriggers)
        self.table.verticalHeader().visible = False
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.minimumHeight = 220
        layout.addWidget(self.table, 1)

        self.details = qt.QTextBrowser()
        self.details.maximumHeight = 170
        layout.addWidget(self.details)

        grid = qt.QGridLayout()
        self.createButton = _button("Create new database...")
        self.createButton.toolTip = "An empty user defined database; subjects are added from the Neuro workflows."
        self.predefinedButton = _button("Add predefined database...")
        self.predefinedButton.toolTip = "A packaged database (.zip), or mean / SD maps picked one by one."
        self.exportButton = qt.QPushButton("Export as predefined...")
        self.exportButton.toolTip = "The selected user defined database as a .zip without its subject images."
        self.deleteButton = _button("Delete database", "revert")
        self.deleteButton.toolTip = "Remove the selected database folder from disk (asks first)."
        self.openButton = qt.QPushButton("Open folder")
        self.refreshButton = qt.QPushButton("Refresh")
        grid.addWidget(self.createButton, 0, 0)
        grid.addWidget(self.predefinedButton, 0, 1)
        grid.addWidget(self.exportButton, 1, 0)
        grid.addWidget(self.deleteButton, 1, 1)
        grid.addWidget(self.openButton, 2, 0)
        grid.addWidget(self.refreshButton, 2, 1)
        layout.addLayout(grid)

        self.table.connect("itemSelectionChanged()", self._updateSelection)
        self.createButton.connect("clicked()", self.onCreate)
        self.predefinedButton.connect("clicked()", self.onAddPredefined)
        self.exportButton.connect("clicked()", self.onExport)
        self.deleteButton.connect("clicked()", self.onDelete)
        self.openButton.connect("clicked()", self.onOpenFolder)
        self.refreshButton.connect("clicked()", self.refresh)
        return page

    # --- table ---------------------------------------------------------------------------------------
    def root(self):
        return self.rootPath.currentPath

    def _onRootChanged(self, path):
        self.logic.setRootFolder(path)
        self.refresh()

    def refresh(self, select=None):
        root = self.root()
        self.databases, errors = normaldb.listDatabases(root, normaldb.STATIC)
        if not root:
            self.rootInfo.text = "Pick the folder that holds (or will hold) the databases."
        elif not os.path.isdir(root):
            self.rootInfo.text = "The folder does not exist."
        else:
            text = f"{len(self.databases)} static database(s)."
            if errors:
                text += " Unreadable: " + ", ".join(os.path.basename(f) for f, _ in errors)
                for folder, error in errors:
                    logging.warning(f"Database {folder}: {error}")
            self.rootInfo.text = text
        self.table.setRowCount(len(self.databases))
        selectRow = -1
        for row, (folder, header) in enumerate(self.databases):
            summary = normaldb.summary(folder, header)
            for column, (key, _) in enumerate(COLUMNS):
                item = qt.QTableWidgetItem(str(summary[key]))
                if key == "statistics" and summary[key] == "Out of date":
                    item.setForeground(qt.QBrush(qt.QColor("#c8553d")))
                item.setToolTip(folder)
                self.table.setItem(row, column, item)
            if select and os.path.normcase(folder) == os.path.normcase(select):
                selectRow = row
        self.table.resizeColumnsToContents()
        if selectRow >= 0:
            self.table.selectRow(selectRow)
        self._updateSelection()

    def selected(self):
        rows = {index.row() for index in self.table.selectedIndexes()}
        if len(rows) != 1:
            return None, None
        row = rows.pop()
        return self.databases[row] if 0 <= row < len(self.databases) else (None, None)

    def _updateSelection(self):
        folder, header = self.selected()
        rootOk = bool(self.root()) and os.path.isdir(self.root())
        self.createButton.enabled = rootOk
        self.predefinedButton.enabled = rootOk
        self.deleteButton.enabled = folder is not None
        self.openButton.enabled = rootOk
        self.exportButton.enabled = folder is not None and header.get("kind") == normaldb.USER_DEFINED
        self.details.setHtml(self._detailsHtml(folder, header) if folder else "")

    @staticmethod
    def _detailsHtml(folder, header):
        def esc(text):
            return (str(text if text is not None else "").replace("&", "&amp;").replace("<", "&lt;")
                    .replace(">", "&gt;"))
        acquisition = header.get("acquisition") or {}
        rows = [
            ("Folder", folder),
            ("Space", f"{header.get('space', '')}, {header.get('voxelSizeMm', '')} mm"),
            ("Acquisition", f"start {acquisition.get('startMinutes')} min p.i., duration "
                            f"{acquisition.get('durationMinutes')} min"),
            ("Segmentation", header.get("segmentation")),
            ("Reference", normaldb.referenceTitle(header.get("referenceRegion"))
             + (f" (mask {header.get('referenceMask')})" if header.get("referenceMask") else " (no mask)")),
            ("Reference erosion", f"{header.get('referenceErosionMm') or 0:g} mm"),
            ("Regions", f"{normaldb.regionAtlasTitle(header.get('regionAtlas'))}, grown "
                        f"{header.get('regionGrowMm') or 0:g} mm"
                        + (" (into the white matter too)" if header.get("growIntoWhiteMatter") else "")),
            ("Maps", f"{header.get('mapType')}, smoothing {header.get('smoothingFwhmMm') or 0:g} mm FWHM"),
            ("Scanner", header.get("scanner")),
            ("Description", header.get("description")),
        ]
        groups = ((header.get("statistics") or {}).get("groups") or {})
        if groups:
            rows.append(("Mean / SD", "; ".join(f"{g.get('name', k)}: n={g.get('n')}" for k, g in groups.items())))
        ages = header.get("subjectAges")
        if ages and ages.get("n"):
            rows.append(("Ages", f"{ages.get('mean')} ± {ages.get('sd')} ({ages.get('min')}-{ages.get('max')})"))
        rows.append(("Version", f"{header.get('version', 1)}, modified {str(header.get('modified', ''))[:16]}"))
        return "<table>" + "".join(f"<tr><td><b>{esc(k)}</b>&nbsp;</td><td>{esc(v)}</td></tr>" for k, v in rows
                                   if v not in (None, "")) + "</table>"

    # --- buttons -------------------------------------------------------------------------------------
    def onCreate(self):
        dialog, layout, buttons = _scrollDialog("Create new static database", slicer.util.mainWindow())
        info = qt.QLabel("An empty user defined database. Subjects are added later from the Neuro workflow ("
                         "Add patient to database); their maps are copied into the database and its mean / SD maps "
                         "computed.")
        info.wordWrap = True
        layout.addWidget(info)
        form = HeaderForm(layout, predefined=False, atlasFolder=self.logic.atlasFolder())
        layout.addStretch(1)
        create = buttons.addButton("Create", qt.QDialogButtonBox.AcceptRole)
        buttons.addButton(qt.QDialogButtonBox.Cancel)
        buttons.connect("rejected()", dialog.reject)

        def accept():
            header = form.header()
            errors = normaldb.validateHeader(header)
            atlasFolder = form.atlasPath.currentPath
            problem = self.logic.maskSourceProblem(header["referenceRegion"], atlasFolder)
            if problem:
                errors.append(problem + " - pick the Neuro atlas folder or a custom mask.")
            if errors:
                _error(dialog, "\n".join(errors))
                return
            try:
                with slicer.util.tryWithErrorDisplay("Could not create the database", waitCursor=True):
                    folder = self.logic.createUserDefined(self.root(), header, atlasFolder,
                                                          form.maskPath.currentPath)
            except Exception:
                return
            if atlasFolder:
                self.logic.setAtlasFolder(atlasFolder)
            dialog.accept()
            self.refresh(select=folder)

        create.connect("clicked()", accept)
        dialog.exec_()

    def onAddPredefined(self):
        dialog, layout, buttons = _scrollDialog("Add predefined static database", slicer.util.mainWindow())
        sourceBox = qt.QGroupBox("Source")
        sourceLayout = qt.QVBoxLayout(sourceBox)
        packageRadio = qt.QRadioButton("Packaged database (.zip, e.g. exported from another computer)")
        mapsRadio = qt.QRadioButton("Mean and SD maps, picked one by one")
        packageRadio.checked = True
        sourceLayout.addWidget(packageRadio)
        packagePath = _pathEdit(filters="Database package (*.zip)")
        sourceLayout.addWidget(packagePath)
        packageInfo = qt.QLabel()
        packageInfo.wordWrap = True
        sourceLayout.addWidget(packageInfo)
        packageName = qt.QLineEdit()
        packageName.placeholderText = "name (default: the name in the package)"
        sourceLayout.addWidget(packageName)
        sourceLayout.addWidget(mapsRadio)
        layout.addWidget(sourceBox)
        form = HeaderForm(layout, predefined=True, atlasFolder=self.logic.atlasFolder())
        layout.addStretch(1)
        add = buttons.addButton("Add", qt.QDialogButtonBox.AcceptRole)
        buttons.addButton(qt.QDialogButtonBox.Cancel)
        buttons.connect("rejected()", dialog.reject)

        def updateSource():
            package = packageRadio.checked
            packagePath.enabled = package
            packageName.enabled = package
            form.widget.visible = not package

        def readPackage(path):
            if not path or not os.path.isfile(path):
                packageInfo.text = ""
                return
            try:
                header = normaldb.readPackage(path)
            except Exception as e:
                packageInfo.text = f"Not a database package: {e}"
                return
            groups = ((header.get("statistics") or {}).get("groups") or {})
            n = (groups.get(normaldb.MASTER) or {}).get("n")
            packageInfo.text = (f"{header.get('name')}: {header.get('tracer', '')} ({header.get('nuclide', '')}), "
                                f"{n} subjects, reference {normaldb.referenceTitle(header.get('referenceRegion'))}, "
                                f"{len(groups)} mean / SD map pair(s).")

        packageRadio.connect("toggled(bool)", lambda on: updateSource())
        packagePath.connect("currentPathChanged(QString)", readPackage)
        updateSource()

        def accept():
            try:
                with slicer.util.tryWithErrorDisplay("Could not add the database", waitCursor=True):
                    if packageRadio.checked:
                        path = packagePath.currentPath
                        if not path or not os.path.isfile(path):
                            raise ValueError("Pick the database package (.zip)")
                        folder = normaldb.importPackage(self.root(), path, packageName.text.strip() or None)
                    else:
                        header = form.header()
                        if header["referenceRegion"] == normaldb.REF_CUSTOM and not form.maskPath.currentPath:
                            raise ValueError("Pick the custom reference mask file")
                        folder = self.logic.createPredefined(self.root(), header, form.maps(),
                                                             form.maskPath.currentPath or None,
                                                             form.atlasPath.currentPath)
            except Exception:
                return
            dialog.accept()
            self.refresh(select=folder)

        add.connect("clicked()", accept)
        dialog.exec_()

    def onExport(self):
        folder, header = self.selected()
        if not folder:
            return
        path = qt.QFileDialog.getSaveFileName(slicer.util.mainWindow(), "Export as predefined database",
                                              os.path.join(os.path.expanduser("~"),
                                                           normaldb.safeName(header["name"]) + ".zip"),
                                              "Database package (*.zip)")
        if not path:
            return
        if not path.lower().endswith(".zip"):
            path += ".zip"
        try:
            with slicer.util.tryWithErrorDisplay("Could not export the database", waitCursor=True):
                normaldb.exportPredefined(folder, path)
        except Exception:
            return
        slicer.util.infoDisplay(f"Saved {path}", windowTitle="Neuro PET Database Manager")

    def onDelete(self):
        folder, header = self.selected()
        if not folder:
            return
        count = normaldb.subjectCount(folder, header)
        what = (f"{count} subject(s) with their maps" if header.get("kind") == normaldb.USER_DEFINED
                else "the mean / SD maps")
        if not slicer.util.confirmYesNoDisplay(
                f"Delete the database '{header.get('name')}' from disk?\n\n{folder}\n\nThe folder is removed with "
                f"{what}. This cannot be undone.", windowTitle="Delete database"):
            return
        try:
            with slicer.util.tryWithErrorDisplay("Could not delete the database", waitCursor=True):
                normaldb.deleteDatabase(folder, self.root())
        finally:
            self.refresh()

    def onOpenFolder(self):
        folder, _ = self.selected()
        qt.QDesktopServices.openUrl(qt.QUrl.fromLocalFile(folder or self.root()))


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------
class NeuroPETDatabaseManagerTest(ScriptedLoadableModuleTest):
    def setUp(self):
        slicer.mrmlScene.Clear()

    def runTest(self):
        self.setUp()
        self.test_ReferenceMask()

    def test_ReferenceMask(self):
        import tempfile
        atlasFolder, root = tempfile.mkdtemp(), tempfile.mkdtemp()
        with open(os.path.join(atlasFolder, neuro.CSV_NAME), "w") as f:
            f.write("Mindboggle ID,Label Name,RH Label,LH Labels,Notes,Dice Kappa\n"
                    "45,Cerebellum Gray Matter,46,97,,0.83\n16,Brainstem,11,62,,0.65\n")
        array = np.zeros((10, 10, 10), dtype=np.int16)
        array[1:3, 1:3, 1:3] = 46
        array[5:7, 5:7, 5:7] = 97
        array[8, 8, 8] = 11
        source = slicer.util.addVolumeFromArray(array, name="labels", nodeClassName="vtkMRMLLabelMapVolumeNode")
        slicer.util.saveNode(source, os.path.join(atlasFolder, "mni_icbm152_CerebrA_tal_nlin_sym_09c.nrrd"))
        slicer.mrmlScene.RemoveNode(source)
        logic = NeuroPETDatabaseManagerLogic()
        header = normaldb.newHeader("test", nuclide="F-18", tracer="FDG")
        folder = logic.createUserDefined(root, header, atlasFolder)
        saved = normaldb.readHeader(folder)
        self.assertEqual(saved["referenceMask"], normaldb.REFERENCE_MASK_NAME)
        mask = slicer.util.loadLabelVolume(os.path.join(folder, normaldb.REFERENCE_MASK_NAME), {"show": False})
        self.assertEqual(int(np.count_nonzero(slicer.util.arrayFromVolume(mask))), 16)
        normaldb.deleteDatabase(folder, root)
        self.assertFalse(os.path.exists(folder))
        # pons from the CerebrA with pons (a segmentation whose segments are named Segment_<label value>)
        with open(os.path.join(atlasFolder, neuro.PONS_CSV_NAME), "w") as f:
            f.write("Mindboggle ID;Label Name;RH Label;LH Labels;Notes;Dice Kappa\n"
                    "16;Brainstem;11;62;;0,65\n444;Pons;103;104;;0\n")
        ponsArray = np.zeros((10, 10, 10), dtype=np.int16)
        ponsArray[4:6, 4:6, 4:6] = 103
        ponsArray[4:6, 4:6, 6:8] = 104
        ponsArray[0, 0, 0] = 46
        labels = slicer.util.addVolumeFromArray(ponsArray, name="pons", nodeClassName="vtkMRMLLabelMapVolumeNode")
        segmentationNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
        slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(labels, segmentationNode)
        segmentation = segmentationNode.GetSegmentation()
        values = (46, 103, 104)      # imported in the order of the label values
        for index in range(segmentation.GetNumberOfSegments()):
            segmentation.GetNthSegment(index).SetName(f"Segment_{values[index]}")
        slicer.util.saveNode(segmentationNode, os.path.join(atlasFolder, neuro.PONS_STEM + ".seg.nrrd"))
        slicer.mrmlScene.RemoveNode(segmentationNode)
        slicer.mrmlScene.RemoveNode(labels)
        header = normaldb.newHeader("pons", nuclide="F-18", tracer="FDG", referenceRegion=normaldb.REF_PONS)
        self.assertIsNone(logic.maskSourceProblem(normaldb.REF_PONS, atlasFolder))
        folder = logic.createUserDefined(root, header, atlasFolder)
        mask = slicer.util.loadLabelVolume(os.path.join(folder, normaldb.REFERENCE_MASK_NAME), {"show": False})
        self.assertEqual(int(np.count_nonzero(slicer.util.arrayFromVolume(mask))), 16)
        self.delayDisplay("Reference mask test passed")
