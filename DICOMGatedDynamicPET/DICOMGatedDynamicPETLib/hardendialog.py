"""Harden dialog of Dynamic PET to CT/MR Registration and Motion Correction (Qt): what the frames are resampled
onto (grid, voxel size), with which interpolation, and what to do (e.g. overwrite or save separately)."""

import qt

from . import registration as reg


def hardenDialog(title, text, gridOptions, petSpacing, actions, parent=None):
    """Ask how to harden. gridOptions: [(title, key)] (reg.OUTPUT_CUSTOM gets the voxel size boxes, filled with
    petSpacing (i, j, k), mm); actions: [(button text, key)], the first one the default. Returns None (Cancel) or
    {"action", "grid", "spacing" (custom voxel size or None), "interpolation"}."""
    dialog = qt.QDialog(parent)
    dialog.setWindowTitle(title)
    layout = qt.QVBoxLayout(dialog)
    label = qt.QLabel(text)
    label.wordWrap = True
    layout.addWidget(label)

    gridBox = qt.QGroupBox("Output grid")
    gridLayout = qt.QVBoxLayout(gridBox)
    radios = []
    spins = []
    for index, (optionTitle, key) in enumerate(gridOptions):
        radio = qt.QRadioButton(optionTitle)
        radio.checked = index == 0
        radios.append((radio, key))
        if key == reg.OUTPUT_CUSTOM:
            row = qt.QHBoxLayout()
            row.addWidget(radio)
            for axis, value in zip("ijk", petSpacing):
                spin = qt.QDoubleSpinBox()
                spin.setRange(0.1, 50.0)
                spin.setDecimals(2)
                spin.setSingleStep(0.5)
                spin.setSuffix(" mm")
                spin.value = float(value)
                spin.toolTip = f"Voxel size along {axis}"
                spin.enabled = radio.checked
                spins.append(spin)
                row.addWidget(spin)
            row.addStretch(1)
            gridLayout.addLayout(row)
            radio.connect("toggled(bool)", lambda on: [setattr(s, "enabled", on) for s in spins])
        else:
            gridLayout.addWidget(radio)
    layout.addWidget(gridBox)

    form = qt.QFormLayout()
    interpolationCombo = qt.QComboBox()
    for name in reg.INTERPOLATIONS:
        interpolationCombo.addItem(name)
    interpolationCombo.setCurrentIndex(interpolationCombo.findText(reg.DEFAULT_INTERPOLATION))
    interpolationCombo.toolTip = ("Linear: the usual choice. Nearest neighbour: keeps the voxel values (blocky). "
                                  "Cubic: sharper, may overshoot a little at edges.")
    form.addRow("Resample method:", interpolationCombo)
    layout.addLayout(form)

    buttons = qt.QDialogButtonBox()
    chosen = {"action": None}
    for index, (buttonText, key) in enumerate(actions):
        button = buttons.addButton(buttonText, qt.QDialogButtonBox.AcceptRole)
        if index == 0:
            button.setDefault(True)
        button.connect("clicked()", lambda key=key: (chosen.update(action=key), dialog.accept()))
    buttons.addButton(qt.QDialogButtonBox.Cancel)
    buttons.connect("rejected()", dialog.reject)
    layout.addWidget(buttons)

    try:
        if not dialog.exec_() or chosen["action"] is None:
            return None
        grid = next(key for radio, key in radios if radio.checked)
        return {"action": chosen["action"], "grid": grid,
                "spacing": tuple(float(s.value) for s in spins) if grid == reg.OUTPUT_CUSTOM else None,
                "interpolation": interpolationCombo.currentText}
    finally:
        dialog.deleteLater()
