"""
Gated / Dynamic PET Settings (3D Slicer module)

Defaults of the DICOM Gated / Dynamic PET plugin, kept in the Slicer application settings
(DICOMGatedDynamicPETLib/settings.py):
    SUV method          body weight (SUVbw), lean body mass (SUVlbm, Janmahasatian) or body surface area
                        (SUVbsa, Du Bois): the SUV loadable offered for every PET series
    preferred units     SUV or Bq/ml, per kind (static, gated, dynamic): the loadable the DICOM browser selects
                        first (default SUV, Bq/ml for dynamic)
    summed volumes      also make the static image of gated (mean of the gates) and dynamic (frames summed)
                        series (default on)
    count weighting     sum the dynamic frames as a static acquisition over the same time would count them
                        (early frames weigh more) instead of by their durations (default off)
Changes apply to series examined in the DICOM browser after the change.
"""

import os
import sys

import qt
from slicer.ScriptedLoadableModule import ScriptedLoadableModule, ScriptedLoadableModuleWidget, ScriptedLoadableModuleTest

_MODULE_DIR = os.path.dirname(os.path.abspath(globals().get("__file__", "") or "."))
if globals().get("__file__") and _MODULE_DIR not in sys.path:
    sys.path.insert(0, _MODULE_DIR)

from DICOMGatedDynamicPETLib import petseries  # noqa: E402
from DICOMGatedDynamicPETLib import settings as petsettings  # noqa: E402

KIND_TITLES = ((petseries.STATIC, "Static PET"), (petseries.GATED, "Gated PET"), (petseries.DYNAMIC, "Dynamic PET"))


class GatedDynamicPETSettings(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        parent.title = "Gated / Dynamic PET Settings"
        parent.categories = ["Nuclear Medicine"]
        parent.dependencies = []
        parent.contributors = ["Burak Demir"]
        parent.helpText = ("Defaults of the DICOM Gated / Dynamic PET loader: the SUV normalisation, which units "
                           "the DICOM browser selects first for static, gated and dynamic PET, whether gated and "
                           "dynamic series also get a summed (static) volume, and how the dynamic frames are summed "
                           "(by duration or by counts). Changes apply to series examined in the DICOM browser after "
                           "the change.")
        parent.acknowledgementText = ("Lean body mass: Janmahasatian et al., Clin Pharmacokinet 2005 (as in PERCIST). "
                                      "Body surface area: Du Bois and Du Bois, 1916.")


class GatedDynamicPETSettingsWidget(ScriptedLoadableModuleWidget):
    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)
        self._updating = False

        # SUV
        suvBox = qt.QGroupBox("SUV")
        suvForm = qt.QFormLayout(suvBox)
        self.suvMethodCombo = qt.QComboBox()
        for method, title in petseries.SUV_METHOD_TITLES.items():
            self.suvMethodCombo.addItem(title[0].upper() + title[1:], method)
        self.suvMethodCombo.toolTip = ("SUV normalisation of the SUV loadable. SUVlbm needs the patient height and sex, "
                                       "SUVbsa the height; without them SUVbw is offered and the DICOM browser says why.")
        suvForm.addRow("SUV method:", self.suvMethodCombo)
        self.layout.addWidget(suvBox)

        # Preferred units
        unitsBox = qt.QGroupBox("Loadable selected first in the DICOM browser")
        unitsForm = qt.QFormLayout(unitsBox)
        self.unitsCombos = {}
        for kind, title in KIND_TITLES:
            combo = qt.QComboBox()
            combo.addItem("SUV", petsettings.SUV)
            combo.addItem("Bq/ml (activity concentration)", petsettings.ACTIVITY)
            combo.toolTip = f"{title}: units of the loadable the DICOM browser selects; the other stays available."
            unitsForm.addRow(f"{title}:", combo)
            self.unitsCombos[kind] = combo
        self.layout.addWidget(unitsBox)

        # Summed volumes
        summedBox = qt.QGroupBox("Summed volumes")
        summedForm = qt.QFormLayout(summedBox)
        self.summedChecks = {}
        texts = {petseries.GATED: ("Gated PET: mean of the gates",
                                   "The static image of the acquisition (each gate is a full concentration image, so "
                                   "the gates are averaged). Tarvos lists a gated PET study through this image."),
                 petseries.DYNAMIC: ("Dynamic PET: frames summed",
                                     "The static image of the whole acquisition: the frames summed by their durations "
                                     "(a duration weighted mean, so the values stay in Bq/ml or SUV).")}
        for kind in (petseries.GATED, petseries.DYNAMIC):
            check = qt.QCheckBox(texts[kind][0])
            check.toolTip = texts[kind][1]
            summedForm.addRow(check)
            self.summedChecks[kind] = check
        self.countWeightCheck = qt.QCheckBox("Dynamic PET: weight the frames by counts (simulate a static scan)")
        self.countWeightCheck.toolTip = (
            "Off: the frames are summed by their durations, the time average of the decay corrected concentration. "
            "On: each frame is weighted by the counts a static acquisition over the same time would get from it "
            "(its duration and the decay since the first frame), so the summed image is what one static scan of "
            "the whole acquisition would show; early frames weigh more (for C-11 over 90 min the last minute "
            "weighs about 5 % of the first). Motion Correction resums with the weighting of the series.")
        summedForm.addRow(self.countWeightCheck)
        self.layout.addWidget(summedBox)

        self.resetButton = qt.QPushButton("Restore defaults")
        self.resetButton.toolTip = ("SUVbw; SUV first for static and gated, Bq/ml first for dynamic; summed volumes "
                                    "on, weighted by duration.")
        self.layout.addWidget(self.resetButton)
        note = qt.QLabel("Changes are saved at once and apply to series examined in the DICOM browser afterwards.")
        note.wordWrap = True
        self.layout.addWidget(note)
        self.layout.addStretch(1)

        self.suvMethodCombo.connect("currentIndexChanged(int)", self.onChanged)
        for combo in self.unitsCombos.values():
            combo.connect("currentIndexChanged(int)", self.onChanged)
        for check in self.summedChecks.values():
            check.connect("toggled(bool)", self.onChanged)
        self.countWeightCheck.connect("toggled(bool)", self.onChanged)
        self.resetButton.connect("clicked()", self.onReset)
        self.updateFromSettings()

    def updateFromSettings(self):
        self._updating = True
        try:
            self.suvMethodCombo.setCurrentIndex(max(0, self.suvMethodCombo.findData(petsettings.suvMethod())))
            for kind, combo in self.unitsCombos.items():
                combo.setCurrentIndex(max(0, combo.findData(petsettings.preferredUnits(kind))))
            for kind, check in self.summedChecks.items():
                check.checked = petsettings.summedVolume(kind)
            self.countWeightCheck.checked = petsettings.summedWeighting() == petseries.SUM_COUNTS
        finally:
            self._updating = False

    def onChanged(self, *_):
        if self._updating:
            return
        petsettings.setValue("SUVMethod", self.suvMethodCombo.currentData)
        for kind, combo in self.unitsCombos.items():
            petsettings.setValue(f"PreferredUnits/{kind}", combo.currentData)
        for kind, check in self.summedChecks.items():
            petsettings.setValue(f"SummedVolume/{kind}", bool(check.checked))
        petsettings.setValue("SummedWeighting", petseries.SUM_COUNTS if self.countWeightCheck.checked
                             else petseries.SUM_DURATION)

    def onReset(self):
        petsettings.resetDefaults()
        self.updateFromSettings()


class GatedDynamicPETSettingsTest(ScriptedLoadableModuleTest):
    def runTest(self):
        self.test_SettingsRoundTrip()

    def test_SettingsRoundTrip(self):
        saved = {key: qt.QSettings().value(f"{petsettings.GROUP}/{key}") for key in petsettings.DEFAULTS}
        try:
            petsettings.resetDefaults()
            self.assertEqual(petsettings.suvMethod(), "bw")
            self.assertEqual(petsettings.preferredUnits(petseries.DYNAMIC), petsettings.ACTIVITY)
            self.assertEqual(petsettings.preferredUnits(petseries.GATED), petsettings.SUV)
            self.assertTrue(petsettings.summedVolume(petseries.GATED))
            self.assertEqual(petsettings.summedWeighting(), petseries.SUM_DURATION)
            petsettings.setValue("SummedWeighting", petseries.SUM_COUNTS)
            self.assertEqual(petsettings.summedWeighting(), petseries.SUM_COUNTS)
            petsettings.setValue("SUVMethod", "lbm")
            petsettings.setValue(f"SummedVolume/{petseries.DYNAMIC}", False)
            self.assertEqual(petsettings.suvMethod(), "lbm")
            self.assertFalse(petsettings.summedVolume(petseries.DYNAMIC))
        finally:
            for key, value in saved.items():
                if value is None:
                    qt.QSettings().remove(f"{petsettings.GROUP}/{key}")
                else:
                    qt.QSettings().setValue(f"{petsettings.GROUP}/{key}", value)
        self.delayDisplay("Settings saved and restored")
