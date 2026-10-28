"""User settings of the DICOM Gated / Dynamic PET extension, kept in the Slicer application settings under
DICOMGatedDynamicPET/ (edited in the Gated / Dynamic PET Settings module, read by the DICOM plugin).

    SUVMethod               bw | lbm | bsa   SUV normalisation of the SUV loadable (default bw)
    PreferredUnits/<KIND>   SUV | Bq/ml      loadable selected first (default SUV, Bq/ml for DYNAMIC)
    SummedVolume/<KIND>     true | false     also make the static image of a gated / dynamic series
                                             (default true)
"""

import qt

from . import petseries

GROUP = "DICOMGatedDynamicPET"
SUV, ACTIVITY = "SUV", "Bq/ml"
DEFAULTS = {
    "SUVMethod": "bw",
    f"PreferredUnits/{petseries.STATIC}": SUV,
    f"PreferredUnits/{petseries.GATED}": SUV,
    f"PreferredUnits/{petseries.DYNAMIC}": ACTIVITY,
    f"SummedVolume/{petseries.GATED}": True,
    f"SummedVolume/{petseries.DYNAMIC}": True,
}


def _value(key):
    value = qt.QSettings().value(f"{GROUP}/{key}")
    return DEFAULTS[key] if value is None or value == "" else value


def setValue(key, value):
    if key not in DEFAULTS:
        raise KeyError(key)
    qt.QSettings().setValue(f"{GROUP}/{key}", value)


def resetDefaults():
    settings = qt.QSettings()
    for key in DEFAULTS:
        settings.remove(f"{GROUP}/{key}")


def suvMethod():
    method = str(_value("SUVMethod")).lower()
    return method if method in petseries.SUV_METHODS else DEFAULTS["SUVMethod"]


def preferredUnits(kind):
    """SUV or Bq/ml: which loadable of a series of this kind is selected first."""
    value = str(_value(f"PreferredUnits/{kind}"))
    return value if value in (SUV, ACTIVITY) else DEFAULTS[f"PreferredUnits/{kind}"]


def summedVolume(kind):
    """Whether loading a gated / dynamic series also makes its static image."""
    value = _value(f"SummedVolume/{kind}")
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes")
    return bool(value)
