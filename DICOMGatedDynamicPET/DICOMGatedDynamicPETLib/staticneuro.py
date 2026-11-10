"""Static Neuro PET/SPECT helpers without Slicer: what a static brain image is (PET / SPECT, units, nuclide,
tracer, read from the tags the loaders put on the volume) and which normal databases fit it.

Tags read (node attributes, {name: value}):
    PET4D DICOM plugin (static PET)       StaticPET.Kind ("static"), StaticPET.Units ("Bq/ml", "SUVbw", ...),
                                          StaticPET.Radiopharmaceutical, StaticPET.Radionuclide ("^18^Fluorine")
    DICOM SPECT Recon module (SPECT)      SPECTRecon.Units ("counts", "Bq/ml", "SUVbw", "SUL"),
                                          SPECTRecon.Isotope ("Tc-99m"), SPECTRecon.IsotopeDicomText (radionuclide
                                          code meaning, else the radiopharmaceutical text)
"""

import os
import re

from DICOMGatedDynamicPETLib import normaldb

STATIC_PREFIX = "StaticPET."
SPECT_PREFIX = "SPECTRecon."

UNITS_SUV, UNITS_BQML, UNITS_COUNTS, UNITS_UNKNOWN = "SUV", "Bq/ml", "counts", ""

# nuclide: (element name, symbol, mass number) - PET and SPECT nuclides of brain imaging
NUCLIDES = {
    "F-18": ("fluorine", "f", "18"), "C-11": ("carbon", "c", "11"), "O-15": ("oxygen", "o", "15"),
    "N-13": ("nitrogen", "n", "13"), "Ga-68": ("gallium", "ga", "68"), "Cu-64": ("copper", "cu", "64"),
    "Zr-89": ("zirconium", "zr", "89"), "Tc-99m": ("technetium", "tc", "99"), "I-123": ("iodine", "i", "123"),
    "I-131": ("iodine", "i", "131"), "In-111": ("indium", "in", "111"), "Tl-201": ("thallium", "tl", "201"),
}

# tracer name as in normaldb.NUCLIDE_TRACERS: words that name it in DICOM texts (lower case)
TRACER_WORDS = (
    ("FDG", ("fdg", "fluorodeoxyglucose", "fludeoxyglucose", "deoxyglucose")),
    ("FDOPA", ("fdopa", "f-dopa", "fluorodopa", "dopa")),
    ("Florbetapir", ("florbetapir", "amyvid", "av-45", "av45")),
    ("Florbetaben", ("florbetaben", "neuraceq", "bay94-9172", "bay 94-9172")),
    ("Flutemetamol", ("flutemetamol", "vizamyl")),
    ("Flortaucipir", ("flortaucipir", "tauvid", "av-1451", "av1451", "t807")),
    ("MK-6240", ("mk-6240", "mk6240", "florquinitau")),
    ("PI-2620", ("pi-2620", "pi2620")),
    ("Fallypride", ("fallypride",)),
    ("FE-PE2I", ("pe2i",)),
    ("FET", ("fet", "fluoroethyl")),
    ("Flumazenil", ("flumazenil", "fmz")),
    ("UCB-H", ("ucb-h", "ucbh")),
    ("UCB-J", ("ucb-j", "ucbj")),
    ("FMISO", ("fmiso", "misonidazole")),
    ("PiB", ("pib", "pittsburgh")),
    ("Raclopride", ("raclopride",)),
    ("Methionine", ("methionine", "met")),
    ("DASB", ("dasb",)),
    ("PBR28", ("pbr28", "pbr-28")),
    ("PK11195", ("pk11195", "pk-11195")),
    ("Water", ("water", "h2o")),
    ("Ammonia", ("ammonia", "nh3")),
    ("HMPAO", ("hmpao", "exametazime", "ceretec")),
    ("ECD", ("ecd", "bicisate", "neurolite")),
    ("TRODAT-1", ("trodat",)),
    ("Ioflupane (DaTscan)", ("ioflupane", "datscan", "fp-cit", "fpcit")),
    ("IBZM", ("ibzm", "iodobenzamide")),
    ("Iomazenil", ("iomazenil",)),
    ("Thallium chloride", ("thallium chloride", "thallous")),
)


def identifyNuclide(text):
    """Nuclide name ('F-18', 'Tc-99m', ...) from free text such as '^18^Fluorine', 'F18', '99mTc', 'Tc-99m' or
    'Iodine-123'. None when not recognised."""
    if not text:
        return None
    lowered = str(text).lower()
    words = set(re.sub(r"[^a-z]", " ", lowered).split())
    numbers = set(re.findall(r"\d+", lowered))
    for name, (element, symbol, mass) in NUCLIDES.items():
        if mass not in numbers:
            continue
        if element in lowered or symbol in words or (symbol + "m") in words or ("m" + symbol) in words:
            return name
    return None


def identifyTracer(text):
    """Tracer name of normaldb.NUCLIDE_TRACERS from a radiopharmaceutical text ('Fluorodeoxyglucose', 'FDG',
    'Tc99m HMPAO', ...). None when not recognised."""
    if not text:
        return None
    lowered = str(text).lower()
    words = set(re.split(r"[^a-z0-9]+", lowered))
    compact = re.sub(r"[^a-z0-9]", "", lowered)
    for name, names in TRACER_WORDS:
        for word in names:
            joined = re.sub(r"[^a-z0-9]", "", word)
            if " " in word:
                if word in lowered:
                    return name
            elif joined in words or ("-" in word and word in lowered) or (len(joined) > 4 and joined in compact):
                return name
    return None


def unitsKind(units):
    """UNITS_SUV / UNITS_BQML / UNITS_COUNTS / UNITS_UNKNOWN of a units text."""
    text = str(units or "").strip().lower()
    if not text:
        return UNITS_UNKNOWN
    if text.startswith("suv") or text in ("sul", "g/ml"):
        return UNITS_SUV
    if text.replace(" ", "") in ("bq/ml", "bqml", "bqperml", "kbq/ml", "kbqml"):
        return UNITS_BQML
    if text in ("counts", "cnts", "count"):
        return UNITS_COUNTS
    return UNITS_UNKNOWN


def imageInfo(attributes, dicomModality=""):
    """What a static brain image is, from its node attributes ({name: value}) and the DICOM modality ('PT', 'NM',
    ... when known): {"modality": "PET" / "SPECT" / "", "units": text, "unitsKind": UNITS_*, "nuclide": text,
    "tracer": text, "source": text, "static": bool}."""
    attributes = {k: v for k, v in (attributes or {}).items() if v not in (None, "")}
    info = {"modality": "", "units": "", "unitsKind": UNITS_UNKNOWN, "nuclide": "", "tracer": "", "source": "",
            "static": False}
    if attributes.get(SPECT_PREFIX + "Units") is not None or any(k.startswith(SPECT_PREFIX) for k in attributes):
        text = attributes.get(SPECT_PREFIX + "IsotopeDicomText", "")
        info.update(modality=normaldb.SPECT, source="DICOM SPECT reconstruction", static=True,
                    units=attributes.get(SPECT_PREFIX + "Units", UNITS_COUNTS),
                    nuclide=identifyNuclide(attributes.get(SPECT_PREFIX + "Isotope")) or identifyNuclide(text) or "",
                    tracer=identifyTracer(text) or "")
    elif any(k.startswith(STATIC_PREFIX) for k in attributes):
        radiopharmaceutical = attributes.get(STATIC_PREFIX + "Radiopharmaceutical", "")
        info.update(modality=normaldb.PET, source="PET4D DICOM (static PET)",
                    static=attributes.get(STATIC_PREFIX + "Kind", "static") == "static",
                    units=attributes.get(STATIC_PREFIX + "Units", ""),
                    nuclide=(identifyNuclide(attributes.get(STATIC_PREFIX + "Radionuclide"))
                             or identifyNuclide(radiopharmaceutical) or ""),
                    tracer=identifyTracer(radiopharmaceutical) or "")
    else:
        modality = str(dicomModality or "").strip().upper()
        if modality == "PT":
            info.update(modality=normaldb.PET, source="DICOM", static=True)
        elif modality == "NM":
            info.update(modality=normaldb.SPECT, source="DICOM", static=True)
    info["unitsKind"] = unitsKind(info["units"])
    if not info["nuclide"] and info["tracer"]:
        info["nuclide"] = next((n for mod in normaldb.MODALITIES for n, names in normaldb.NUCLIDE_TRACERS[mod]
                                if info["tracer"] in names), "")
    return info


def unitWarnings(info, header=None):
    """Warnings (list of str) about the image's units: a PET is expected in SUV (Bq/ml is fine for ratio maps, the
    scale cancels, not for SUV maps); a SPECT in counts, Bq/ml or SUV."""
    warnings = []
    kind, modality = info.get("unitsKind"), info.get("modality")
    mapType = (header or {}).get("mapType")
    if modality == normaldb.PET:
        if kind == UNITS_BQML:
            text = "The PET is in Bq/ml, not SUV."
            if mapType == normaldb.MAP_SUV:
                text += " The database holds SUV maps: load the PET as SUV to compare it."
            else:
                text += " Ratio (SUVR) comparisons are not affected (the scale cancels); SUV values are not shown."
            warnings.append(text)
        elif kind == UNITS_UNKNOWN:
            warnings.append("The units of the PET are not known (SUV expected).")
    elif modality == normaldb.SPECT:
        if kind == UNITS_UNKNOWN:
            warnings.append("The units of the SPECT are not known (counts, Bq/ml or SUV expected).")
        if mapType == normaldb.MAP_COUNTS and kind != UNITS_COUNTS:
            warnings.append("The database holds count maps but the SPECT is not in counts.")
    elif not modality:
        warnings.append("Not recognised as a PET or SPECT (no PET4D or DICOM SPECT tags): check the modality.")
    return warnings


def _same(a, b):
    return str(a or "").strip().lower() == str(b or "").strip().lower()


def databaseMatch(header, info):
    """(rank, problems) of a normal database for the image: rank 0 when modality, nuclide and tracer fit (a tracer
    that is not known on either side counts as fitting), higher the worse; problems lists the differences."""
    problems = []
    rank = 0
    if info.get("modality") and not _same(header.get("modality", normaldb.PET), info["modality"]):
        problems.append(f"modality {header.get('modality', normaldb.PET)} (image {info['modality']})")
        rank += 4
    nuclide = identifyNuclide(header.get("nuclide")) or header.get("nuclide", "")
    if info.get("nuclide") and nuclide and not _same(nuclide, info["nuclide"]):
        problems.append(f"nuclide {header.get('nuclide')} (image {info['nuclide']})")
        rank += 2
    tracer = header.get("tracer", "")
    tracerKnown = tracer and not _same(tracer, header.get("nuclide"))
    if info.get("tracer") and tracerKnown and not _same(identifyTracer(tracer) or tracer, info["tracer"]):
        problems.append(f"tracer {tracer} (image {info['tracer']})")
        rank += 1
    return rank, problems


def sortDatabases(databases, info):
    """[(folder, header, rank, problems)] of [(folder, header)], the fitting ones first (then by name)."""
    rows = [(folder, header) + databaseMatch(header, info) for folder, header in databases]
    return sorted(rows, key=lambda row: (row[2], str(row[1].get("name", "")).lower()))


def masterMeanPath(folder, header):
    """Absolute path of a database's master mean map (the template of the PET to PET normalization), None when the
    statistics are not there."""
    master = ((header.get("statistics") or {}).get("groups") or {}).get(normaldb.MASTER) or {}
    relative = master.get("mean")
    if not relative:
        return None
    path = os.path.join(folder, relative)
    return path if os.path.isfile(path) else None


def petTemplatePath(folder, header):
    """The template of the PET to PET normalization: the database's frozen template (normalizationTemplate), else its
    master mean map; None when it has neither."""
    return normaldb.templatePath(folder, header) or masterMeanPath(folder, header)


def storedRegions(folder, header):
    """(path, reference segment name or None) of the regions stored with a database's statistics (the segmentation
    the regional statistics and the 3D-SSP were made on); None when it has none, or when the statistics of a user
    defined database are out of date."""
    stats = (header or {}).get("statistics") or {}
    relative = stats.get("regionsSegmentation")
    if not relative or (header.get("kind") != normaldb.PREDEFINED and stats.get("stale")):
        return None
    path = os.path.join(folder, relative)
    return (path, stats.get("regionsReference") or None) if os.path.isfile(path) else None


def referenceMaskPath(folder, header):
    """Absolute path of a database's reference mask copy, None when it has none."""
    relative = header.get("referenceMask")
    if not relative:
        return None
    path = os.path.join(folder, relative)
    return path if os.path.isfile(path) else None


# the Neuro region atlas the reference region of a database is read from when it has no mask copy: the bilateral
# CerebrA (one region of both sides); None: cannot be taken from one atlas region
REFERENCE_ATLAS_REGION = {normaldb.REF_CEREBELLUM_GM: "Cerebellum Gray Matter", normaldb.REF_PONS: "Pons",
                          normaldb.REF_BRAINSTEM: "Brainstem"}


def regionSettings(folder, header):
    """The region settings of a database, as the Regions step applies them: {"regionAtlas", "growMm",
    "growInto", "erodeReferenceMm", "referenceRegion", "referenceTitle", "referenceMaskPath",
    "referenceAtlasRegion", "smoothingFwhmMm", "mapType"}."""
    reference = header.get("referenceRegion") or ""
    return {
        "regionAtlas": header.get("regionAtlas") or normaldb.DEFAULT_REGION_ATLAS,
        "growMm": float(header.get("regionGrowMm") or 0.0),
        "growInto": normaldb.growInto(header),
        "erodeReferenceMm": float(header.get("referenceErosionMm") or 0.0),
        "referenceRegion": reference,
        "referenceTitle": normaldb.referenceTitle(reference),
        "referenceMaskPath": referenceMaskPath(folder, header),
        "referenceAtlasRegion": REFERENCE_ATLAS_REGION.get(reference),
        "smoothingFwhmMm": float(header.get("smoothingFwhmMm") or 0.0),
        "mapType": header.get("mapType", ""),
    }


def settingsText(settings):
    """One line per setting of regionSettings(), for the page."""
    grow = settings["growMm"]
    lines = [f"Regions: {normaldb.regionAtlasTitle(settings['regionAtlas'])}",
             "Grow regions: " + (f"{grow:g} mm" + (f" ({normaldb.GROW_INTO_TEXT[settings['growInto']]})"
                                                    if normaldb.GROW_INTO_TEXT.get(settings["growInto"]) else "")
                                 if grow > 0 else "off"),
             f"Reference: {settings['referenceTitle'] or 'none'}"
             + (" (the database's mask)" if settings["referenceMaskPath"] else "")
             + (f", eroded {settings['erodeReferenceMm']:g} mm" if settings["erodeReferenceMm"] > 0 else ""),
             "Smoothing: " + (f"{settings['smoothingFwhmMm']:g} mm FWHM" if settings["smoothingFwhmMm"] > 0
                              else "off"),
             f"Maps: {settings['mapType'] or '?'}"]
    return lines


# ---------------------------------------------------------------------------
# Import from file (BIDS PET: sub-<id>_ses-<session>_pet.nii.gz with its .json sidecar)
# ---------------------------------------------------------------------------
IMAGE_EXTENSIONS = (".nii.gz", ".nii", ".nrrd", ".nhdr", ".mha", ".mhd")
IMPORTED_ATTRIBUTE = STATIC_PREFIX + "ImportedFrom"


def splitImageName(path):
    """(stem, extension) of an image file name ('sub-1_pet', '.nii.gz')."""
    name = os.path.basename(path)
    for extension in IMAGE_EXTENSIONS:
        if name.lower().endswith(extension):
            return name[:-len(extension)], extension
    stem, extension = os.path.splitext(name)
    return stem, extension


def parseBidsName(path):
    """BIDS entities of a file name: {"sub": "00000016", "ses": "SUV", ..., "suffix": "pet"}; {} when the name is not
    BIDS-like (no 'sub-')."""
    stem = splitImageName(path)[0]
    parts = stem.split("_")
    entities = {}
    for part in parts:
        if "-" in part:
            key, value = part.split("-", 1)
            if key and value:
                entities[key.lower()] = value
        else:
            entities["suffix"] = part.lower()
    return entities if "sub" in entities else {}


def sidecarPath(path):
    """The .json next to an image (same stem), None when there is none."""
    stem, _ = splitImageName(path)
    candidate = os.path.join(os.path.dirname(path), stem + ".json")
    return candidate if os.path.isfile(candidate) else None


def findParticipants(path, levels=4):
    """participants.tsv of the BIDS folder the image lies in (searched up to levels folders up), None when absent."""
    folder = os.path.dirname(os.path.abspath(path))
    for _ in range(levels + 1):
        candidate = os.path.join(folder, "participants.tsv")
        if os.path.isfile(candidate):
            return candidate
        parent = os.path.dirname(folder)
        if parent == folder:
            break
        folder = parent
    return None


def readParticipant(tsvPath, subject):
    """The participants.tsv row of a subject ('sub-<id>' or '<id>') as {column: value}, {} when not found."""
    import csv
    if not tsvPath or not subject:
        return {}
    wanted = {str(subject).lower(), ("sub-" + str(subject)).lower()}
    with open(tsvPath, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if str(row.get("participant_id", "")).strip().lower() in wanted:
                return {k: (v or "").strip() for k, v in row.items() if k}
    return {}


def _bidsUnits(entities, sidecar):
    """Units of an imported image: SUV when the file name says so (a session, acquisition or reconstruction
    'SUV...', as in the cnpet SUV set, whose sidecars still say Bq/mL), else the sidecar's Units."""
    for key in ("ses", "acq", "rec", "desc"):
        if str(entities.get(key, "")).lower().startswith("suv"):
            return "SUVbw", "file name"
    units = str(sidecar.get("Units", "") or "")
    return ({"bq/ml": "Bq/ml", "kbq/ml": "kBq/ml"}.get(units.lower(), units), "sidecar") if units else ("", "")


def importAttributes(path, sidecar=None, participant=None):
    """Node attributes ({name: text}) of an image imported from a file, so it reads as a static PET (or a SPECT,
    when the BIDS suffix is 'spect' / the sidecar says so): units, nuclide, tracer, timing, age and sex."""
    entities = parseBidsName(path)
    sidecar = sidecar or {}
    participant = participant or {}
    spect = entities.get("suffix") in ("spect", "nm") or str(sidecar.get("Modality", "")).upper() in ("NM", "SPECT")
    units, unitsSource = _bidsUnits(entities, sidecar)
    prefix = SPECT_PREFIX if spect else STATIC_PREFIX
    out = {IMPORTED_ATTRIBUTE: os.path.abspath(path)}
    if spect:
        out[SPECT_PREFIX + "Units"] = units or UNITS_COUNTS
        nuclide = identifyNuclide(sidecar.get("TracerRadionuclide"))
        if nuclide:
            out[SPECT_PREFIX + "Isotope"] = nuclide
        if sidecar.get("TracerName"):
            out[SPECT_PREFIX + "IsotopeDicomText"] = str(sidecar["TracerName"])
    else:
        out[STATIC_PREFIX + "Kind"] = "static"
        if units:
            out[STATIC_PREFIX + "Units"] = units
        if sidecar.get("TracerName"):
            out[STATIC_PREFIX + "Radiopharmaceutical"] = str(sidecar["TracerName"])
        if sidecar.get("TracerRadionuclide"):
            out[STATIC_PREFIX + "Radionuclide"] = str(sidecar["TracerRadionuclide"])
    if unitsSource:
        out[prefix + "UnitsSource"] = unitsSource
    starts, durations = sidecar.get("FrameTimesStart") or [], sidecar.get("FrameDuration") or []
    if isinstance(starts, list) and len(starts) == 1 and isinstance(durations, list) and len(durations) == 1:
        out[prefix + "StartFromInjectionS"] = f"{float(starts[0]):g}"
        out[prefix + "DurationS"] = f"{float(durations[0]):g}"
    elif isinstance(starts, list) and len(starts) > 1:
        out[prefix + "Frames"] = str(len(starts))
    for key, attribute in (("Manufacturer", "Manufacturer"), ("ManufacturersModelName", "ScannerModel"),
                           ("ReconMethodName", "ReconstructionMethod")):
        if sidecar.get(key) not in (None, "", "n/a"):
            out[prefix + attribute] = str(sidecar[key])
    if entities.get("sub"):
        out[prefix + "SubjectID"] = entities["sub"]
    if entities.get("ses"):
        out[prefix + "Session"] = entities["ses"]
    age = participant.get("age") or participant.get("Age")
    if age not in (None, "", "n/a"):
        out[prefix + "PatientAge"] = age
    sex = participant.get("sex") or participant.get("Sex") or participant.get("gender")
    if sex not in (None, "", "n/a"):
        out[prefix + "PatientSex"] = sex[:1].upper()
    return out


def importName(path):
    """Node name of an imported image: the file name without its extension."""
    return splitImageName(path)[0]


# ---------------------------------------------------------------------------
# Age and sex of the patient (read from the tags, editable on the Data page)
# ---------------------------------------------------------------------------
SEXES = (("", "Unknown"), ("F", "Female"), ("M", "Male"), ("O", "Other"))


def parseAge(text):
    """Age in years from a DICOM Patient's Age ('052Y', '006M', '003W', '010D') or a number; None when not given."""
    text = str(text or "").strip().upper()
    if not text or text == "N/A":
        return None
    match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*([YMWD]?)", text)
    if not match:
        return None
    value, unit = float(match.group(1)), match.group(2) or "Y"
    return round(value / {"Y": 1.0, "M": 12.0, "W": 52.1775, "D": 365.25}[unit], 2)


def parseSex(text):
    """'F' / 'M' / 'O' / '' from a DICOM Patient's Sex or a word."""
    text = str(text or "").strip().upper()
    if text.startswith(("F", "K", "W")):        # female, kadın, woman
        return "F"
    if text.startswith(("M", "E")):             # male, erkek
        return "M"
    if text.startswith("O"):
        return "O"
    return ""


def taggedAgeSex(attributes):
    """(age in years or None, sex '' / 'F' / 'M' / 'O') from the loaders' tags of a volume."""
    age = sex = None
    for prefix in (STATIC_PREFIX, SPECT_PREFIX):
        if age is None:
            age = parseAge(attributes.get(prefix + "PatientAge"))
        if not sex:
            sex = parseSex(attributes.get(prefix + "PatientSex"))
    return age, sex or ""


# ---------------------------------------------------------------------------
# Name and ID of the patient (read from the DICOM tags, editable on the Data page)
# ---------------------------------------------------------------------------
def formatPatientName(text):
    """A DICOM Person Name ('DOE^JOHN^^^' or 'Doe^John=...') as plain text ('DOE JOHN'); '' when not given."""
    text = str(text or "").split("=")[0]            # the alphabetic representation only
    parts = [p.strip() for p in text.split("^") if p.strip()]
    return " ".join(parts)


def taggedSubjectId(attributes):
    """The patient's ID from the loaders' tags of a volume: the BIDS subject of an imported file ('sub-16'), else
    ''."""
    for prefix in (STATIC_PREFIX, SPECT_PREFIX):
        subject = str((attributes or {}).get(prefix + "SubjectID") or "").strip()
        if subject:
            return subject if subject.startswith("sub-") else "sub-" + subject
    return ""


def patientText(name, patientId):
    """'NAME (ID)', 'NAME', 'ID' or '' for the workflow toolbar."""
    name, patientId = str(name or "").strip(), str(patientId or "").strip()
    if name and patientId:
        return f"{name} ({patientId})"
    return name or patientId


# ---------------------------------------------------------------------------
# A patient added to a user defined database
# ---------------------------------------------------------------------------
def suggestedSubjectId(attributes, name=""):
    """Subject ID offered when a patient is added to a database: the BIDS subject of an imported file, else the
    volume's name (as a file name)."""
    for prefix in (STATIC_PREFIX, SPECT_PREFIX):
        subject = (attributes or {}).get(prefix + "SubjectID")
        if subject:
            return "sub-" + str(subject) if not str(subject).startswith("sub-") else str(subject)
    return normaldb.safeName(name) or "subject"


def ratioMap(values, mask):
    """(values / mean of values in the mask, that mean) of a patient's map on the database grid; ValueError when the
    mask misses the image or the mean is not positive."""
    import numpy as np
    values = np.asarray(values, dtype=np.float32)
    mask = np.asarray(mask) > 0
    if values.shape != mask.shape:
        raise ValueError("The image and the reference mask are not on the same grid.")
    inside = values[mask & np.isfinite(values)]
    if inside.size == 0:
        raise ValueError("The reference mask holds no image voxels.")
    mean = float(inside.mean())
    if not mean > 0:
        raise ValueError(f"The mean in the reference region is {mean:g}: is the image normalized to the MNI space?")
    return (values / mean).astype(np.float32), mean


# ---------------------------------------------------------------------------
# Batch: many normal subjects (BIDS PET files) into a database
# ---------------------------------------------------------------------------
def findPetFiles(root):
    """BIDS PET images under root (sub-<id>[_ses-<session>]..._pet.nii[.gz]), sorted by path."""
    found = []
    for folder, dirs, files in os.walk(root or ""):
        dirs[:] = sorted(d for d in dirs if not d.startswith("."))
        for name in files:
            stem, extension = splitImageName(name)
            if extension.lower() in (".nii", ".nii.gz") and parseBidsName(name).get("suffix") == "pet":
                found.append(os.path.join(folder, name))
    return sorted(found)


def bidsSession(path):
    return parseBidsName(path).get("ses", "")


def bidsSubjectId(path):
    """Database subject ID of a BIDS file ('sub-<id>'), "" when the name is not BIDS."""
    subject = parseBidsName(path).get("sub")
    return f"sub-{subject}" if subject else ""


def batchNormalization(norm, defaultTemplate, defaultEngine, engineDefaults, petTemplate=False):
    """The normalization a batch runs with: the database's settings (normaldb.NORM_KEYS) with the module defaults where
    they are not specified. engineDefaults: {engine: (linear, deformable)}. The PET/SPECT to MR template method, or
    the PET to PET one when petTemplate (the database has its template); ValueError for the others (no CT/MR)."""
    norm = dict(normaldb.emptyNormalization(), **(norm or {}))
    method = norm.get("method") or normaldb.NORM_PET
    if method == normaldb.NORM_PET_TEMPLATE and not petTemplate:
        raise ValueError("The database's normalization is PET to PET but it has no template (mean map) yet.")
    if method not in (normaldb.NORM_PET, normaldb.NORM_PET_TEMPLATE):
        title = dict(normaldb.NORM_METHODS).get(method, method)
        raise ValueError(f"The database's normalization is '{title}': the batch registers the PET / SPECT to a "
                         "template only (no CT/MR).")
    engine = norm.get("engine") or defaultEngine
    linear, deformable = engineDefaults[engine]
    template = (norm.get("template") or defaultTemplate) if method == normaldb.NORM_PET else ""
    return {"method": method, "template": template, "engine": engine,
            "linear": norm.get("linear") or linear,
            "deformable": deformable if norm.get("deformable") is None else bool(norm["deformable"]),
            "brainMask": True if norm.get("brainMask") is None else bool(norm["brainMask"])}


def rasArray(array, ijkToRas):
    """A (k, j, i) volume array (as Slicer gives it) as an [x, y, z] array whose indices grow to the right, anterior
    and superior (axis-aligned grids; oblique ones are taken to their nearest axes)."""
    import numpy as np
    out = np.asarray(array).transpose(2, 1, 0)         # [i, j, k]
    directions = np.asarray(ijkToRas, dtype=float)[:3, :3]
    order = [int(np.argmax(np.abs(directions[axis, :]))) for axis in range(3)]   # RAS axis -> ijk axis
    if sorted(order) != [0, 1, 2]:
        raise ValueError("The image grid is too oblique for the QC picture.")
    out = out.transpose(order)
    for axis in range(3):
        if directions[axis, order[axis]] < 0:
            out = np.flip(out, axis)
    return out


HOT = ((0.0, (0, 0, 0)), (0.35, (230, 0, 0)), (0.7, (255, 210, 0)), (1.0, (255, 255, 255)))


def _hot(values):
    import numpy as np
    stops = np.array([s for s, _ in HOT])
    colors = np.array([c for _, c in HOT], dtype=float)
    return np.stack([np.interp(values, stops, colors[:, c]) for c in range(3)], axis=-1)


def qcMontage(anatomy, pet, scale=2, opacity=0.5):
    """RGB (uint8) picture of the PET over the anatomy (both [x, y, z] arrays of rasArray() on the same grid): axial,
    coronal and sagittal slices through the PET's centre of mass side by side, as Slicer shows them (patient's right
    on the left, anterior up / left)."""
    import numpy as np
    anatomy = np.nan_to_num(np.asarray(anatomy, dtype=np.float32))
    pet = np.nan_to_num(np.asarray(pet, dtype=np.float32))
    if anatomy.shape != pet.shape:
        raise ValueError("The PET and the anatomy are not on the same grid.")
    low, high = np.percentile(anatomy, (1, 99)) if anatomy.size else (0.0, 1.0)
    grey = np.clip((anatomy - low) / max(high - low, 1e-6), 0, 1)
    positive = pet[pet > 0]
    top = float(np.percentile(positive, 99.5)) if positive.size else 1.0
    level = np.clip(pet / max(top, 1e-6), 0, 1)
    weights = np.where(pet > 0.25 * top, pet, 0)
    if weights.sum() > 0:
        centre = [int(round(float((weights.sum(axis=tuple(a for a in range(3) if a != axis)) *
                                   np.arange(pet.shape[axis])).sum() / weights.sum()))) for axis in range(3)]
    else:
        centre = [n // 2 for n in pet.shape]
    x, y, z = centre
    # axial (anterior up, patient's right on the left), coronal (superior up), sagittal (anterior on the left)
    pictures = []
    for index in (np.s_[:, :, z], np.s_[:, y, :], np.s_[x, :, :]):
        g, v = grey[index].T[::-1, ::-1], level[index].T[::-1, ::-1]     # up = +, right side of the screen = -
        rgb = np.repeat(g[..., None] * 255.0, 3, axis=-1)
        alpha = np.where(v > 0.2, opacity, 0.0)[..., None]
        rgb = (1 - alpha) * rgb + alpha * _hot(v)
        pictures.append(np.kron(rgb, np.ones((scale, scale, 1))) if scale > 1 else rgb)
    height = max(p.shape[0] for p in pictures)
    padded = [np.pad(p, ((0, height - p.shape[0]), (0, 4 * scale), (0, 0))) for p in pictures]
    return np.clip(np.concatenate(padded, axis=1), 0, 255).astype(np.uint8)
