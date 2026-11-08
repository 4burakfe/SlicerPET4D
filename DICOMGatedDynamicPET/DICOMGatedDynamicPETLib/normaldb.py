"""Neuro PET / SPECT normal databases on disk (no Slicer imports; numpy only): the folder layout, the header
(database.json), the subjects table (subjects.csv), age groups, the database list of a root folder, deletion,
predefined databases made from mean / SD maps, and the packaged (.zip) form of a predefined database.

A root folder (picked in the Neuro PET Database Manager) holds one folder per database, named after it:

    <root>/<database name>/
        database.json           header: name, mode (static / dynamic), kind (predefined / user defined), nuclide,
                                tracer, space, acquisition, segmentation, reference region, smoothing, age groups,
                                free text, statistics (subjects per group, mean / SD map files, when computed)
        reference_mask.nii.gz   copy of the reference region mask in the template space
        subjects.csv            user defined only: one row per subject (ID, age, scan date, added, map files)
        subjects/               user defined only: the maps of each subject (PET: SUV and SUVR, SPECT: counts and
                                ratio, so the reference region can be changed later)
        stats/                  mean and SD maps: master_mean / master_sd (all subjects) and <group>_mean /
                                <group>_sd for each age group

Predefined databases have no subject images: they hold only the mean / SD maps and the number of subjects that the
user gives. A user defined database starts empty; subjects are added from the Neuro workflows and its statistics are
computed from them. A user defined database can be exported as a predefined one (.zip, without subjects/).
"""

import csv
import datetime
import io
import json
import os
import re
import shutil
import zipfile

import numpy as np

FORMAT_VERSION = 1
HEADER_NAME = "database.json"
SUBJECTS_CSV = "subjects.csv"
SUBJECTS_DIR = "subjects"
STATS_DIR = "stats"
REFERENCE_MASK_NAME = "reference_mask.nii.gz"

STATIC, DYNAMIC = "static", "dynamic"
MODES = ((STATIC, "Static"), (DYNAMIC, "Dynamic"))
PREDEFINED, USER_DEFINED = "predefined", "user"
KINDS = ((PREDEFINED, "Predefined"), (USER_DEFINED, "User defined"))

SPACE = "MNI ICBM 152 2009c nonlinear symmetric"
VOXEL_MM = 1.0
DEFAULT_SEGMENTATION = "CerebrA (Manera et al. 2020)"

PET, SPECT = "PET", "SPECT"
MODALITIES = (PET, SPECT)
# What the maps hold: PET SUVR (SUV / reference) or SUV; SPECT counts / reference (ratio) or counts
MAP_SUVR, MAP_SUV, MAP_RATIO, MAP_COUNTS = "SUVR", "SUV", "Ratio", "Counts"
MAP_TYPES_BY_MODALITY = {PET: (MAP_SUVR, MAP_SUV), SPECT: (MAP_RATIO, MAP_COUNTS)}
MAP_TYPES = (MAP_SUVR, MAP_SUV, MAP_RATIO, MAP_COUNTS)

# Reference regions of the static normalization: key, title, CerebrA structures (None: not from CerebrA)
REF_GLOBAL, REF_CEREBELLUM, REF_CEREBELLUM_GM, REF_PONS, REF_BRAINSTEM, REF_CUSTOM = (
    "global", "cerebellum", "cerebellumGM", "pons", "brainstem", "custom")
REFERENCE_REGIONS = (
    (REF_GLOBAL, "Global (whole brain)", None),
    (REF_CEREBELLUM, "Whole cerebellum (grey + white matter)", ("Cerebellum Gray Matter", "Cerebellum White Matter")),
    (REF_CEREBELLUM_GM, "Cerebellar grey matter", ("Cerebellum Gray Matter",)),
    # the pons of the CerebrA with pons (..._wpons.seg.nrrd in the atlas folder); CerebrA itself has only the
    # brainstem, which reaches a little further down
    (REF_PONS, "Pons", ("Pons",)),
    (REF_BRAINSTEM, "Brainstem (CerebrA)", ("Brainstem",)),
    (REF_CUSTOM, "Custom mask file", None),
)
DEFAULT_REFERENCE = REF_CEREBELLUM_GM

# Nuclides of each modality with their brain tracers (the order of the lists). The tracer list of a nuclide starts
# with the nuclide itself (tracer not given); any other nuclide or tracer can be typed.
NUCLIDE_TRACERS = {
    PET: (
        ("F-18", ("FDG", "FDOPA", "Florbetapir", "Florbetaben", "Flutemetamol", "Flortaucipir", "MK-6240", "PI-2620",
                  "Fallypride", "FE-PE2I", "FET", "Flumazenil", "UCB-H", "FMISO")),
        ("C-11", ("PiB", "Raclopride", "UCB-J", "Flumazenil", "Methionine", "DASB", "PBR28", "PK11195")),
        ("O-15", ("Water",)),
        ("N-13", ("Ammonia",)),
        ("Ga-68", ("DOTATATE", "PSMA-11")),
        ("Cu-64", ()),
        ("Zr-89", ()),
    ),
    SPECT: (
        ("Tc-99m", ("HMPAO", "ECD", "TRODAT-1")),
        ("I-123", ("Ioflupane (DaTscan)", "IBZM", "Iomazenil")),
        ("Tl-201", ("Thallium chloride",)),
    ),
}


def nuclides(modality=PET):
    """Nuclides offered for a modality."""
    return [n for n, _ in NUCLIDE_TRACERS.get(modality, ())]


def tracers(nuclide, modality=None):
    """Tracers offered for a nuclide: the nuclide itself first (tracer not given), then its tracers (of the modality,
    else of any modality). A nuclide not in the lists offers only itself."""
    nuclide = str(nuclide or "").strip()
    if not nuclide:
        return []
    out = [nuclide]
    for mod in ([modality] if modality else MODALITIES):
        for name, names in NUCLIDE_TRACERS.get(mod, ()):
            if name.lower() == nuclide.lower():
                out.extend(t for t in names if t not in out)
    return out


def mapTypes(modality=PET):
    return MAP_TYPES_BY_MODALITY.get(modality, MAP_TYPES_BY_MODALITY[PET])

# Regions of the workflows (the Neuro modules' region atlases, same keys) and how they are drawn; stored in the
# header so the subjects added and the patients compared use the same regions
REGION_ATLASES = (("lateral", "CerebrA, left and right separate"), ("bilateral", "CerebrA bilateral"),
                  ("grouped", "CerebrA broad regions, left and right"),
                  ("groupedBilateral", "CerebrA broad regions, sides merged"))
DEFAULT_REGION_ATLAS = "lateral"

MASTER = "master"
MASTER_TITLE = "Master (all subjects)"
IMAGE_EXTENSIONS = (".nii.gz", ".nii", ".nrrd", ".nhdr", ".mha", ".mhd")
SUBJECT_COLUMNS = ("id", "age", "sex", "scanDate", "added", "suvFile", "suvrFile", "referenceMean", "reference",
                   "normalization", "source")

# Normalization of the subjects to the MNI space, stored categorically (the Neuro workflows offer it as the default and
# warn when a patient is normalized otherwise; "" / None: not specified, anything goes)
NORM_PET, NORM_ANATOMY, NORM_PET_TEMPLATE = "pet", "anatomy", "petTemplate"      # the Neuro modules' method keys
NORM_METHODS = (("", "Not specified"), (NORM_PET, "PET/SPECT to MR template"),
                (NORM_ANATOMY, "CT/MR to MR template, same transform to the PET/SPECT"),
                (NORM_PET_TEMPLATE, "PET/SPECT to PET template (the database's mean)"))
NORM_TEMPLATES = (("", "Not specified"), ("T1", "T1"), ("GM", "Grey matter probability"),
                  ("pseudoPET", "Synthetic PET (GM + 0.3 WM)"), ("T2", "T2"), ("PD", "PD"))
NORM_ENGINES = (("", "Not specified"), ("brainsfit", "BRAINSFit"), ("iterative", "Iterative multi-resolution"))
NORM_LINEAR = (("", "Not specified"), ("Rigid", "Rigid"), ("ScaleSkewVersor3D", "Rigid + scale + skew"),
               ("Affine", "Affine"))
NORM_KEYS = ("method", "template", "engine", "linear", "deformable", "brainMask")


# ---------------------------------------------------------------------------
# Names
# ---------------------------------------------------------------------------
def safeName(text):
    """A folder / file name made from text: letters, digits, space, '-', '_', '.' and '+' kept, the rest '_'; no
    leading / trailing dots or spaces. Empty for an empty name."""
    name = re.sub(r"[^\w \-.+]", "_", str(text or "").strip(), flags=re.UNICODE)
    return name.strip(" .")


def imageExtension(path):
    """Extension of an image file (.nii.gz counts as one), lower case; '' when not an image extension."""
    lower = str(path).lower()
    for ext in IMAGE_EXTENSIONS:
        if lower.endswith(ext):
            return ext
    return ""


# Where grown regions may go (white matter: probability >= 0.5 in the MNI WM map)
GROW_NO_WM, GROW_WITH_WM, GROW_ONLY_WM = "noWhiteMatter", "withWhiteMatter", "onlyWhiteMatter"
GROW_INTO = ((GROW_NO_WM, "CSF and cortex edge (not white matter)"), (GROW_WITH_WM, "White matter too"),
             (GROW_ONLY_WM, "White matter only"))
GROW_INTO_TEXT = {GROW_NO_WM: "", GROW_WITH_WM: "into the white matter too", GROW_ONLY_WM: "into the white matter only"}


def growInto(header):
    """GROW_* of a header ("growInto"; older headers: the "growIntoWhiteMatter" flag)."""
    value = header.get("growInto")
    if value in GROW_INTO_TEXT:
        return value
    return GROW_WITH_WM if header.get("growIntoWhiteMatter") else GROW_NO_WM


def regionAtlasTitle(key):
    return dict(REGION_ATLASES).get(key, key or "")


def referenceTitle(key):
    return next((title for k, title, _ in REFERENCE_REGIONS if k == key), key or "")


def referenceStructures(key):
    return next((structures for k, _, structures in REFERENCE_REGIONS if k == key), None)


def now():
    return datetime.datetime.now().replace(microsecond=0).isoformat()


# ---------------------------------------------------------------------------
# Age groups
# ---------------------------------------------------------------------------
def groupKey(name):
    """File name key of an age group ('40-59' -> 'age_40-59')."""
    return "age_" + (safeName(name).replace(" ", "_") or "group")


def _age(value):
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    return float(value)


def validateAgeGroups(groups):
    """Errors (list of str) of age groups [{"name", "min", "max"}]: ages in years, min and max inclusive, max None
    for an open upper end. Names must be unique and the ranges must not overlap."""
    errors = []
    if not groups:
        return ["Define at least one age group"]
    seen = set()
    ranges = []
    for index, group in enumerate(groups):
        name = str(group.get("name", "")).strip()
        label = name or f"group {index + 1}"
        if not name:
            errors.append(f"Age group {index + 1} has no name")
        elif groupKey(name).lower() in seen:
            errors.append(f"Age group name '{name}' is used twice")
        seen.add(groupKey(name).lower())
        try:
            low, high = _age(group.get("min")), _age(group.get("max"))
        except (TypeError, ValueError):
            errors.append(f"Age group '{label}': the ages are not numbers")
            continue
        low = 0.0 if low is None else low
        if low < 0 or (high is not None and high < low):
            errors.append(f"Age group '{label}': the age range is not valid")
            continue
        ranges.append((low, float("inf") if high is None else high, label))
    ranges.sort()
    for (_, high, a), (low, _, b) in zip(ranges, ranges[1:]):
        if low <= high:
            errors.append(f"Age groups '{a}' and '{b}' overlap")
    return errors


def ageGroupOf(age, groups):
    """Name of the age group an age falls in (min <= age <= max), or None."""
    if age is None:
        return None
    for group in groups or ():
        low, high = _age(group.get("min")), _age(group.get("max"))
        if (low is None or age >= low) and (high is None or age <= high):
            return group["name"]
    return None


def ageGroupText(group):
    low, high = _age(group.get("min")), _age(group.get("max"))
    if high is None:
        return f"{group['name']} ({low or 0:g}+)"
    return f"{group['name']} ({low or 0:g}-{high:g})"


def ageSummary(ages):
    """{"n", "mean", "sd", "min", "max"} of a list of ages (None ignored); no identifying data."""
    values = np.array([float(a) for a in ages if a is not None and str(a).strip() != ""], dtype=float)
    if values.size == 0:
        return {"n": 0}
    return {"n": int(values.size), "mean": round(float(values.mean()), 1),
            "sd": round(float(values.std(ddof=1)), 1) if values.size > 1 else 0.0,
            "min": round(float(values.min()), 1), "max": round(float(values.max()), 1)}


# ---------------------------------------------------------------------------
# Normalization settings
# ---------------------------------------------------------------------------
def emptyNormalization():
    return {"method": "", "template": "", "engine": "", "linear": "", "deformable": None, "brainMask": None}


def isSpecified(normalization):
    return any(v not in ("", None) for v in (normalization or {}).values())


def _yesNo(value, yes, no):
    return yes if value else no


def normalizationText(normalization):
    """One line of the specified normalization settings ('Not specified' when none)."""
    n = dict(emptyNormalization(), **(normalization or {}))
    parts = []
    if n["method"]:
        method = dict(NORM_METHODS).get(n["method"], n["method"])
        if n["method"] == NORM_PET and n["template"]:
            method += f" ({dict(NORM_TEMPLATES).get(n['template'], n['template'])})"
        parts.append(method)
    elif n["template"]:
        parts.append(f"template {dict(NORM_TEMPLATES).get(n['template'], n['template'])}")
    if n["engine"]:
        parts.append(dict(NORM_ENGINES).get(n["engine"], n["engine"]))
    if n["linear"]:
        parts.append(dict(NORM_LINEAR).get(n["linear"], n["linear"]))
    if n["deformable"] is not None:
        parts.append(_yesNo(n["deformable"], "then deformable", "no deformable"))
    if n["brainMask"] is not None:
        parts.append(_yesNo(n["brainMask"], "skull-stripped template", "whole template"))
    return ", ".join(parts) if parts else "Not specified"


def normalizationDifferences(expected, used):
    """How used (the settings a patient was normalized with) differs from expected (a database's): ["engine
    BRAINSFit (database: Iterative multi-resolution)", ...], only for the settings the database specifies. The
    template counts only for the PET/SPECT to MR template method."""
    expected = dict(emptyNormalization(), **(expected or {}))
    used = dict(emptyNormalization(), **(used or {}))
    titles = {"method": dict(NORM_METHODS), "template": dict(NORM_TEMPLATES), "engine": dict(NORM_ENGINES),
              "linear": dict(NORM_LINEAR)}
    out = []
    for key in NORM_KEYS:
        want, got = expected[key], used[key]
        if want in ("", None):
            continue
        if key == "template" and (expected["method"] or used["method"]) not in ("", NORM_PET):
            continue
        if want != got:
            if key in titles:
                text = (f"{key} {titles[key].get(got, got) if got not in ('', None) else 'not known'} (database: "
                        f"{titles[key].get(want, want)})")
            else:
                text = (f"{'deformable stage' if key == 'deformable' else 'skull-stripped template'} "
                        f"{'not known' if got is None else _yesNo(got, 'on', 'off')} (database: "
                        f"{_yesNo(want, 'on', 'off')})")
            out.append(text)
    return out


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
def newHeader(name, mode=STATIC, kind=USER_DEFINED, **fields):
    """Header of a new database with the defaults; fields override them (see the keys below)."""
    header = {
        "formatVersion": FORMAT_VERSION,
        "version": 1,           # database version: +1 at every new computation of the statistics
        "name": str(name).strip(),
        "mode": mode,
        "kind": kind,
        "modality": PET,
        "created": now(),
        "modified": now(),
        "nuclide": "",
        "tracer": "",
        "space": SPACE,
        "voxelSizeMm": VOXEL_MM,
        "segmentation": DEFAULT_SEGMENTATION,
        "acquisition": {"startMinutes": None, "durationMinutes": None},
        "referenceRegion": DEFAULT_REFERENCE if mode == STATIC else "",
        "referenceMask": "",
        "mapType": MAP_SUVR,
        "smoothingFwhmMm": 0.0,
        "referenceErosionMm": 0.0,
        "regionAtlas": DEFAULT_REGION_ATLAS,
        "regionGrowMm": 0.0,
        "growInto": GROW_NO_WM,
        "normalization": emptyNormalization(),
        "ageStratified": False,
        "ageGroups": [],
        "scanner": "",
        "description": "",
        "statistics": {"computed": None, "stale": False, "groups": {}},
        # a mean map frozen as the template of the PET to PET normalization: {"file", "made", "fromVersion", "n"}
        "normalizationTemplate": {},
    }
    for key, value in fields.items():
        if key not in header:
            raise KeyError(key)
        header[key] = value
    if "mapType" not in fields:
        header["mapType"] = mapTypes(header["modality"])[0]     # SUVR (PET) / Ratio (SPECT)
    return header


def validateHeader(header):
    """Errors (list of str) of a header before the database is made."""
    errors = []
    if not safeName(header.get("name")):
        errors.append("Give the database a name")
    if header.get("mode") not in dict(MODES):
        errors.append("Unknown mode")
    if header.get("kind") not in dict(KINDS):
        errors.append("Unknown database kind")
    if header.get("modality") not in MODALITIES:
        errors.append("Pick the modality (PET or SPECT)")
    elif header.get("mapType") not in mapTypes(header.get("modality")):
        errors.append(f"The map type of a {header.get('modality')} database is "
                      + " or ".join(mapTypes(header.get("modality"))))
    if not str(header.get("tracer", "")).strip():
        errors.append("Give the tracer")
    if not str(header.get("nuclide", "")).strip():
        errors.append("Give the nuclide")
    if header.get("mode") == STATIC and header.get("referenceRegion") not in dict(
            (k, t) for k, t, _ in REFERENCE_REGIONS):
        errors.append("Pick the reference region of the normalization")
    for key, title in (("startMinutes", "start"), ("durationMinutes", "duration")):
        value = (header.get("acquisition") or {}).get(key)
        if value is not None and float(value) < 0:
            errors.append(f"The acquisition {title} cannot be negative")
    if float(header.get("smoothingFwhmMm") or 0) < 0:
        errors.append("The smoothing cannot be negative")
    if float(header.get("referenceErosionMm") or 0) < 0:
        errors.append("The reference erosion cannot be negative")
    if float(header.get("regionGrowMm") or 0) < 0:
        errors.append("The region growth cannot be negative")
    if header.get("regionAtlas") not in dict(REGION_ATLASES):
        errors.append("Unknown region atlas")
    if header.get("growInto") is not None and header.get("growInto") not in GROW_INTO_TEXT:
        errors.append("Unknown region growth target (growInto)")
    if (header.get("normalization") or {}).get("method", "") not in dict(NORM_METHODS):
        errors.append("Unknown normalization method")
    if header.get("ageStratified"):
        errors.extend(validateAgeGroups(header.get("ageGroups")))
    return errors


def headerPath(folder):
    return os.path.join(folder, HEADER_NAME)


def readHeader(folder):
    with open(headerPath(folder), encoding="utf-8") as f:
        header = json.load(f)
    if not isinstance(header, dict) or "name" not in header:
        raise ValueError(f"{headerPath(folder)} is not a database header")
    header["growInto"] = growInto(header)
    header.pop("growIntoWhiteMatter", None)
    defaults = newHeader(header.get("name", ""), header.get("mode", STATIC), header.get("kind", USER_DEFINED))
    for key, value in defaults.items():
        header.setdefault(key, value)
    header["normalization"] = dict(emptyNormalization(), **(header.get("normalization") or {}))
    return header


def writeHeader(folder, header, touch=True):
    """Write database.json (through a temporary file, so a crash never leaves half a header)."""
    if touch:
        header["modified"] = now()
    path = headerPath(folder)
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as f:
        json.dump(header, f, indent=2, ensure_ascii=False)
        f.write("\n")
    os.replace(temporary, path)


def isDatabase(folder):
    return os.path.isfile(headerPath(folder))


# ---------------------------------------------------------------------------
# Subjects (user defined)
# ---------------------------------------------------------------------------
def readSubjects(folder):
    path = os.path.join(folder, SUBJECTS_CSV)
    if not os.path.isfile(path):
        return []
    with open(path, newline="", encoding="utf-8-sig") as f:
        return [dict(row) for row in csv.DictReader(f)]


def writeSubjects(folder, subjects):
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=SUBJECT_COLUMNS, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    for subject in subjects:
        writer.writerow({k: "" if subject.get(k) is None else subject.get(k) for k in SUBJECT_COLUMNS})
    with open(os.path.join(folder, SUBJECTS_CSV), "w", encoding="utf-8", newline="") as f:
        f.write(out.getvalue())


def subjectFileNames(subjectId, extension=".nii.gz"):
    """(SUV file, SUVR file) of a subject, relative to the database folder."""
    name = safeName(subjectId) or "subject"
    return f"{SUBJECTS_DIR}/{name}_SUV{extension}", f"{SUBJECTS_DIR}/{name}_SUVR{extension}"


def hasSubject(folder, subjectId):
    wanted = str(subjectId).strip().lower()
    return any(str(s.get("id", "")).strip().lower() == wanted for s in readSubjects(folder))


def addSubject(folder, subject, normalization=None, replace=False):
    """Add a subject row (SUBJECT_COLUMNS; its files already written into subjects/) to a user defined database:
    the statistics become out of date, and a database whose normalization is not specified takes the subject's.
    A subject ID already there: ValueError unless replace. Returns the header written."""
    header = readHeader(folder)
    if header.get("kind") != USER_DEFINED:
        raise ValueError("Subjects can be added to user defined databases only.")
    subjectId = str(subject.get("id", "")).strip()
    if not subjectId:
        raise ValueError("The subject needs an ID.")
    subjects = readSubjects(folder)
    same = [i for i, s in enumerate(subjects) if str(s.get("id", "")).strip().lower() == subjectId.lower()]
    if same and not replace:
        raise ValueError(f"Subject '{subjectId}' is already in the database.")
    row = dict(subject, id=subjectId)
    row.setdefault("added", now())
    if same:
        subjects[same[0]] = row
    else:
        subjects.append(row)
    os.makedirs(os.path.join(folder, SUBJECTS_DIR), exist_ok=True)
    writeSubjects(folder, subjects)
    stats = header.setdefault("statistics", {"computed": None, "stale": False, "groups": {}})
    if stats.get("computed"):
        stats["stale"] = True
    if normalization and not isSpecified(header.get("normalization")):
        header["normalization"] = dict(emptyNormalization(), **normalization)
    writeHeader(folder, header)
    return header


def removeSubjects(folder, subjectIds):
    """Remove subjects (rows of subjects.csv and their files in subjects/) from a user defined database; computed
    statistics become out of date. Returns the IDs removed."""
    header = readHeader(folder)
    if header.get("kind") != USER_DEFINED:
        raise ValueError("Subjects can be removed from user defined databases only.")
    wanted = {str(i).strip().lower() for i in subjectIds}
    subjects = readSubjects(folder)
    keep, removed = [], []
    for subject in subjects:
        if str(subject.get("id", "")).strip().lower() in wanted:
            removed.append(subject)
        else:
            keep.append(subject)
    if not removed:
        return []
    subjectsRoot = os.path.realpath(os.path.join(folder, SUBJECTS_DIR))
    for subject in removed:
        for key in ("suvFile", "suvrFile"):
            relative = subject.get(key)
            path = os.path.realpath(os.path.join(folder, relative)) if relative else ""
            if path and path.startswith(subjectsRoot + os.sep) and os.path.isfile(path):
                os.remove(path)
    writeSubjects(folder, keep)
    stats = header.setdefault("statistics", {"computed": None, "stale": False, "groups": {}})
    if stats.get("computed"):
        stats["stale"] = True
    writeHeader(folder, header)
    return [s.get("id") for s in removed]


def subjectCount(folder, header):
    """Subjects of a database: the rows of subjects.csv (user defined), the number given for the master maps
    (predefined)."""
    if header.get("kind") == USER_DEFINED:
        return len(readSubjects(folder))
    master = ((header.get("statistics") or {}).get("groups") or {}).get(MASTER) or {}
    try:
        return int(master.get("n") or 0)
    except (TypeError, ValueError):
        return 0


# ---------------------------------------------------------------------------
# Databases of a root folder
# ---------------------------------------------------------------------------
def databaseFolder(root, name):
    return os.path.join(root, safeName(name))


def listDatabases(root, mode=None):
    """([(folder, header)], [(folder, error)]) of the databases in root (sorted by name), only those of mode when
    given. A folder without database.json is not a database; one with a broken header is reported in the errors."""
    found, errors = [], []
    if not root or not os.path.isdir(root):
        return found, errors
    for entry in sorted(os.listdir(root), key=str.lower):
        folder = os.path.join(root, entry)
        if not os.path.isdir(folder) or not isDatabase(folder):
            continue
        try:
            header = readHeader(folder)
        except (OSError, ValueError) as e:
            errors.append((folder, str(e)))
            continue
        if mode is None or header.get("mode") == mode:
            found.append((folder, header))
    return found, errors


def createDatabase(root, header):
    """Make the folder of a new database in root (named after it) with its header; subjects.csv and subjects/ for
    a user defined one. Returns the folder. Raises ValueError for a header with errors or a name already used."""
    errors = validateHeader(header)
    if errors:
        raise ValueError("\n".join(errors))
    if not root or not os.path.isdir(root):
        raise ValueError("Pick the database folder first")
    folder = databaseFolder(root, header["name"])
    if os.path.exists(folder):
        raise ValueError(f"A database or folder named '{os.path.basename(folder)}' already exists in {root}")
    os.makedirs(os.path.join(folder, STATS_DIR))
    if header["kind"] == USER_DEFINED:
        os.makedirs(os.path.join(folder, SUBJECTS_DIR))
        writeSubjects(folder, [])
    writeHeader(folder, header)
    return folder


# Settings that can be edited after a database is made; those of a predefined database's maps stay as they are
EDITABLE_KEYS = ("name", "modality", "nuclide", "tracer", "acquisition", "segmentation", "referenceRegion",
                 "referenceErosionMm", "smoothingFwhmMm", "mapType", "regionAtlas", "regionGrowMm", "growInto",
                 "normalization", "ageStratified", "ageGroups", "scanner", "description")
PREDEFINED_FIXED_KEYS = ("referenceRegion", "referenceErosionMm", "smoothingFwhmMm", "mapType", "ageStratified",
                         "ageGroups")
STATISTICS_KEYS = ("referenceRegion", "referenceErosionMm", "smoothingFwhmMm", "ageStratified", "ageGroups")


def editedHeader(old, new):
    """(header, changes) of a database whose settings are edited: the editable keys from new (not the fixed ones of a
    predefined database), the rest (kind, mode, statistics, reference mask, version, ...) kept. changes: {"name",
    "reference" (region or erosion: the mask and the subjects' SUVR maps are to be made again), "statistics" (the
    mean / SD maps out of date), "normalization", "keys" [changed keys]}. Raises ValueError for a header with errors."""
    header = json.loads(json.dumps(old))
    fixed = PREDEFINED_FIXED_KEYS if old.get("kind") == PREDEFINED else ()
    changed = []
    for key in EDITABLE_KEYS:
        if key in fixed or key not in new:
            continue
        if json.dumps(new[key], sort_keys=True) != json.dumps(old.get(key), sort_keys=True):
            header[key] = json.loads(json.dumps(new[key]))
            changed.append(key)
    header["name"] = str(header.get("name", "")).strip()
    if not header["ageStratified"]:
        header["ageGroups"] = []
    errors = validateHeader(header)
    if errors:
        raise ValueError("\n".join(errors))
    changes = {"name": "name" in changed and safeName(header["name"]) != safeName(old.get("name", "")),
               "reference": any(k in changed for k in ("referenceRegion", "referenceErosionMm")),
               "statistics": any(k in changed for k in STATISTICS_KEYS),
               "normalization": "normalization" in changed, "keys": changed}
    stats = header.setdefault("statistics", {"computed": None, "stale": False, "groups": {}})
    if changes["statistics"] and stats.get("computed"):
        stats["stale"] = True
    return header, changes


def renameDatabase(folder, root, name):
    """Move a database folder to the folder of its new name in root. Returns the new folder."""
    target = databaseFolder(root, name)
    if os.path.normcase(os.path.realpath(target)) == os.path.normcase(os.path.realpath(folder)):
        return folder
    if os.path.exists(target):
        raise ValueError(f"A database or folder named '{os.path.basename(target)}' already exists in {root}")
    if not _inside(folder, root):
        raise ValueError(f"{folder} is not in the database folder {root}")
    os.rename(folder, target)
    return target


def _inside(path, root):
    path, root = os.path.realpath(path), os.path.realpath(root)
    return os.path.commonpath([path, root]) == root and path != root


def deleteDatabase(folder, root):
    """Delete a database folder from disk. Only a database (with database.json) directly inside root."""
    if not isDatabase(folder):
        raise ValueError(f"{folder} is not a database")
    if not _inside(folder, root) or os.path.dirname(os.path.realpath(folder)) != os.path.realpath(root):
        raise ValueError(f"{folder} is not in the database folder {root}")
    shutil.rmtree(folder)


def summary(folder, header):
    """Table row of a database: {"name", "kind", "modality", "mode", "nuclide", "tracer", "subjects", "reference",
    "ageGroups", "created", "statistics", "version"}."""
    stats = header.get("statistics") or {}
    if header.get("kind") == PREDEFINED:
        state = "Predefined maps"
    elif not stats.get("computed"):
        state = "Not computed"
    elif stats.get("stale"):
        state = "Out of date"
    else:
        state = "Up to date"
    groups = header.get("ageGroups") or []
    return {
        "name": header.get("name", os.path.basename(folder)),
        "version": header.get("version", 1),
        "kind": dict(KINDS).get(header.get("kind"), header.get("kind", "")),
        "modality": header.get("modality", PET),
        "mode": dict(MODES).get(header.get("mode"), header.get("mode", "")),
        "nuclide": header.get("nuclide", ""),
        "tracer": header.get("tracer", ""),
        "subjects": subjectCount(folder, header),
        "reference": referenceTitle(header.get("referenceRegion")),
        "ageGroups": ", ".join(ageGroupText(g) for g in groups) if header.get("ageStratified") and groups else "No",
        "created": str(header.get("created", ""))[:10],
        "statistics": state,
    }


TEMPLATE_NAME = "template.nii.gz"


def subjectMapKey(header):
    """Column of subjects.csv whose maps the mean / SD are made of: the ratio maps (SUVR / Ratio) or the SUV /
    count maps, by the database's map type."""
    return "suvrFile" if header.get("mapType") in (MAP_SUVR, MAP_RATIO) else "suvFile"


def statisticsGroups(header, subjects):
    """{key: {"name", "ids"}} of the mean / SD maps to make: the master (all subjects) and, when age stratified, each
    age group (subjects without an age go into the master only)."""
    groups = {MASTER: {"name": MASTER_TITLE, "ids": [s["id"] for s in subjects]}}
    if header.get("ageStratified"):
        for group in header.get("ageGroups") or []:
            groups[groupKey(group["name"])] = {"name": group["name"], "ids": []}
        for subject in subjects:
            try:
                age = float(subject.get("age")) if str(subject.get("age") or "").strip() else None
            except ValueError:
                age = None
            name = ageGroupOf(age, header.get("ageGroups") or [])
            if name is not None:
                groups[groupKey(name)]["ids"].append(subject["id"])
    return groups


def gaussianSigmas(fwhmMm, spacing):
    """Gaussian sigma in voxels of each axis for a FWHM in mm (spacing of the same axes)."""
    return tuple(float(fwhmMm) / (2.0 * np.sqrt(2.0 * np.log(2.0))) / float(s) for s in spacing)


def smoothMap(array, fwhmMm, spacing):
    """Gaussian smoothing of a map (FWHM in mm; spacing in mm of the array's axes); NaN taken as 0."""
    array = np.nan_to_num(np.asarray(array, dtype=np.float32))
    if not fwhmMm or fwhmMm <= 0:
        return array
    from scipy import ndimage
    return ndimage.gaussian_filter(array, gaussianSigmas(fwhmMm, spacing), mode="constant").astype(np.float32)


class RunningStatistics:
    """Mean and SD (n - 1) of maps added one at a time, without keeping them all in memory."""

    def __init__(self):
        self.n = 0
        self.total = None
        self.squares = None

    def add(self, array):
        array = np.asarray(array, dtype=np.float64)
        if self.total is None:
            self.total, self.squares = np.zeros_like(array), np.zeros_like(array)
        elif array.shape != self.total.shape:
            raise ValueError(f"A map of shape {array.shape} does not fit the others {self.total.shape}.")
        self.total += array
        self.squares += array * array
        self.n += 1

    def mean(self):
        return (self.total / self.n).astype(np.float32)

    def sd(self):
        if self.n < 2:
            raise ValueError("The SD needs at least two subjects.")
        variance = (self.squares - self.total * self.total / self.n) / (self.n - 1)
        return np.sqrt(np.clip(variance, 0, None)).astype(np.float32)


def templatePath(folder, header):
    """The frozen normalization template of a database (normalizationTemplate), None when there is none."""
    relative = (header.get("normalizationTemplate") or {}).get("file")
    path = os.path.join(folder, relative) if relative else ""
    return path if path and os.path.isfile(path) else None


def freezeTemplate(folder):
    """Copy the master mean map of a database as its normalization template (template.<ext>) and make PET to PET its
    normalization method: the subjects and the patients are then normalized to this fixed map, which does not
    change when the statistics are computed again (freeze again for the next iteration). Returns the header."""
    header = readHeader(folder)
    master = ((header.get("statistics") or {}).get("groups") or {}).get(MASTER) or {}
    source = os.path.join(folder, master["mean"]) if master.get("mean") else ""
    if not source or not os.path.isfile(source):
        raise ValueError("The database has no master mean map: compute the mean / SD maps first.")
    relative = "template" + (imageExtension(source) or ".nii.gz")
    shutil.copyfile(source, os.path.join(folder, relative))
    previous = header.get("normalizationTemplate") or {}
    header["normalizationTemplate"] = {"file": relative, "made": now(), "fromVersion": header.get("version", 1),
                                       "n": master.get("n"), "smoothingFwhmMm": header.get("smoothingFwhmMm"),
                                       "iteration": int(previous.get("iteration") or 0) + 1}
    header["normalization"] = dict(emptyNormalization(), **(header.get("normalization") or {}))
    header["normalization"].update(method=NORM_PET_TEMPLATE, template="")
    writeHeader(folder, header)
    return header


def statsFileName(key, which, extension=".nii.gz"):
    """stats/ file of a group: key MASTER or groupKey(name), which 'mean' or 'sd'."""
    return f"{STATS_DIR}/{key}_{which}{extension}"


# ---------------------------------------------------------------------------
# Predefined databases
# ---------------------------------------------------------------------------
def createPredefined(root, header, maps, maskPath=None):
    """Make a predefined database from mean / SD maps. maps: {MASTER or age group name: {"mean": path, "sd": path,
    "n": subjects}}; the master maps are required, and one entry per age group when the header is age stratified.
    The files are copied into stats/ (their format kept) and the mask, when given, as the reference mask. Returns
    the folder."""
    header = dict(header, kind=PREDEFINED)
    errors = validateHeader(header)
    groups = [MASTER] + ([g["name"] for g in header.get("ageGroups") or []] if header.get("ageStratified") else [])
    for name in groups:
        entry = maps.get(name) or {}
        title = MASTER_TITLE if name == MASTER else f"age group '{name}'"
        for which in ("mean", "sd"):
            path = entry.get(which)
            if not path or not os.path.isfile(path):
                errors.append(f"Pick the {which.upper() if which == 'sd' else which} map of the {title}")
            elif not imageExtension(path):
                errors.append(f"The {which} map of the {title} is not a NIfTI / NRRD / MetaImage file")
        try:
            if int(entry.get("n") or 0) < 1:
                errors.append(f"Give the number of subjects of the {title}")
        except (TypeError, ValueError):
            errors.append(f"The number of subjects of the {title} is not a whole number")
    if maskPath and not (os.path.isfile(maskPath) and imageExtension(maskPath)):
        errors.append("The reference mask is not an image file")
    if errors:
        raise ValueError("\n".join(errors))
    folder = createDatabase(root, header)
    try:
        statGroups = {}
        for name in groups:
            entry = maps[name]
            key = MASTER if name == MASTER else groupKey(name)
            files = {}
            for which in ("mean", "sd"):
                relative = statsFileName(key, which, imageExtension(entry[which]))
                shutil.copyfile(entry[which], os.path.join(folder, relative))
                files[which] = relative
            statGroups[key] = {"name": name, "n": int(entry["n"]), **files}
        if maskPath:
            relative = "reference_mask" + imageExtension(maskPath)
            shutil.copyfile(maskPath, os.path.join(folder, relative))
            header["referenceMask"] = relative
        header["statistics"] = {"computed": now(), "stale": False, "groups": statGroups}
        header["version"] = 1
        writeHeader(folder, header)
    except Exception:
        shutil.rmtree(folder, ignore_errors=True)
        raise
    return folder


def exportPredefined(folder, zipPath):
    """Write a user defined database as a predefined one: a .zip with database.json (kind predefined, the subjects
    reduced to an age summary), the reference mask and stats/; the subject maps stay out. Needs computed
    statistics."""
    header = readHeader(folder)
    if header.get("kind") != USER_DEFINED:
        raise ValueError("Only a user defined database can be exported as a predefined one")
    stats = header.get("statistics") or {}
    if not stats.get("computed") or not stats.get("groups"):
        raise ValueError("Compute the mean and SD maps of the database first")
    subjects = readSubjects(folder)
    exported = dict(header, kind=PREDEFINED, exported=now(), exportedFrom=header.get("name"),
                    subjectAges=ageSummary(s.get("age") for s in subjects))
    exported["statistics"] = dict(stats, stale=False)
    base = safeName(header["name"])
    files = [header.get("referenceMask")] if header.get("referenceMask") else []
    if templatePath(folder, header):
        files.append(header["normalizationTemplate"]["file"])
    elif header.get("normalizationTemplate"):
        exported["normalizationTemplate"] = {}
    for group in stats["groups"].values():
        files.extend(group.get(w) for w in ("mean", "sd") if group.get(w))
    with zipfile.ZipFile(zipPath, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(f"{base}/{HEADER_NAME}", json.dumps(exported, indent=2, ensure_ascii=False) + "\n")
        for relative in files:
            path = os.path.join(folder, relative)
            if not os.path.isfile(path):
                raise ValueError(f"{relative} of the database is missing")
            archive.write(path, f"{base}/{relative}")
    return zipPath


def _zipHeader(archive):
    """(prefix, header) of the database.json in a packaged database: at the top or in one top folder."""
    names = [n for n in archive.namelist() if n.rstrip("/").split("/")[-1] == HEADER_NAME]
    names = [n for n in names if n.count("/") <= 1]
    if len(names) != 1:
        raise ValueError("The file is not a packaged database (one database.json expected)")
    prefix = names[0][:-len(HEADER_NAME)]
    header = json.loads(archive.read(names[0]).decode("utf-8"))
    if not isinstance(header, dict) or "name" not in header:
        raise ValueError("The database.json of the package is not a database header")
    return prefix, header


def readPackage(zipPath):
    """Header of a packaged (.zip) database, without extracting it."""
    with zipfile.ZipFile(zipPath) as archive:
        return _zipHeader(archive)[1]


def importPackage(root, zipPath, name=None):
    """Extract a packaged database into root as a predefined database (named name, else the name in its header).
    Paths that would leave the database folder are refused. Returns the folder."""
    if not root or not os.path.isdir(root):
        raise ValueError("Pick the database folder first")
    with zipfile.ZipFile(zipPath) as archive:
        prefix, header = _zipHeader(archive)
        header["kind"] = PREDEFINED
        if name:
            header["name"] = str(name).strip()
        folder = databaseFolder(root, header["name"])
        if not safeName(header["name"]):
            raise ValueError("The package has no database name")
        if os.path.exists(folder):
            raise ValueError(f"A database or folder named '{os.path.basename(folder)}' already exists in {root}")
        members = []
        for info in archive.infolist():
            if not info.filename.startswith(prefix) or info.is_dir():
                continue
            relative = info.filename[len(prefix):]
            if not relative or relative == HEADER_NAME:
                continue
            if relative.startswith(SUBJECTS_DIR + "/") or relative == SUBJECTS_CSV:
                continue        # a predefined database has no subjects
            target = os.path.join(folder, *relative.split("/"))
            if relative.startswith("/") or "\\" in relative or ".." in relative.split("/") or ":" in relative \
                    or not _inside(target, folder):
                raise ValueError(f"The package has an unsafe path: {info.filename}")
            members.append((info, target))
        os.makedirs(os.path.join(folder, STATS_DIR))
        try:
            for info, target in members:
                os.makedirs(os.path.dirname(target), exist_ok=True)
                with archive.open(info) as source, open(target, "wb") as out:
                    shutil.copyfileobj(source, out)
            header.setdefault("imported", now())
            writeHeader(folder, header, touch=False)
        except Exception:
            shutil.rmtree(folder, ignore_errors=True)
            raise
    return folder


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def meanAndSd(arrays):
    """Voxel-wise mean and SD (sample, ddof 1; 0 for one subject) of same-shape arrays, as float32."""
    stack = None
    count = 0
    total = None
    squares = None
    for array in arrays:
        values = np.asarray(array, dtype=np.float64)
        if total is None:
            total, squares = np.zeros_like(values), np.zeros_like(values)
            stack = values.shape
        elif values.shape != stack:
            raise ValueError(f"Map shapes differ: {values.shape} and {stack}")
        total += values
        squares += values * values
        count += 1
    if count == 0:
        raise ValueError("No maps")
    mean = total / count
    if count > 1:
        variance = np.maximum(squares - count * mean * mean, 0.0) / (count - 1)
    else:
        variance = np.zeros_like(mean)
    return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)
