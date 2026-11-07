"""Pure helpers of the Dynamic Neuro Receptor PET workflow (numpy only, no Slicer imports): the MNI ICBM 152 2009c
template files in the atlas folder, the MR contrast (T1 / T2 / PD) of the patient's anatomy, the template that fits
it, and the CerebrA labels (names, left / right, colours, merged structures).

Atlases (not shipped with the extension; the user points to a folder that holds them, searched recursively):
    MNI ICBM 152 nonlinear symmetric 2009c (Fonov et al. 2009, 2011):
        mni_icbm152_<t1|t2|pd|gm|wm|csf>_tal_nlin_sym_09c.nii, mni_icbm152_t1_tal_nlin_sym_09c_mask.nii (brain)
    CerebrA (Manera et al. 2020):   mni_icbm152_CerebrA_tal_nlin_sym_09c.nii and CerebrA_LabelDetails.csv
    CerebrA with pons (optional, made by the user in Slicer): mni_icbm152_CerebrA_tal_nlin_sym_09c_wpons.seg.nrrd, a
        segmentation whose segments are named Segment_<label value> (the pons in a second layer, inside the
        brainstem), and mni_icbm152_CerebrA_tal_nlin_sym_09c_wpons.csv (CerebrA_LabelDetails with a Pons row)
    CerebrA bilateral (made by the workflow on first use, next to the CerebrA files): left and right merged,
        mni_icbm152_CerebrA_bilateral_tal_nlin_sym_09c.nii and CerebrA_bilateral_LabelDetails.csv
"""

import colorsys
import csv
import io
import os
import re

import numpy as np

# ---------------------------------------------------------------------------
# Atlas files
# ---------------------------------------------------------------------------
TEMPLATE_SUFFIX = "_tal_nlin_sym_09c"
TEMPLATES = (   # key, title, file name stem
    ("T1", "T1 (default)", "mni_icbm152_t1" + TEMPLATE_SUFFIX),
    ("T2", "T2", "mni_icbm152_t2" + TEMPLATE_SUFFIX),
    ("PD", "PD", "mni_icbm152_pd" + TEMPLATE_SUFFIX),
    ("GM", "Grey matter probability", "mni_icbm152_gm" + TEMPLATE_SUFFIX),
    ("WM", "White matter probability", "mni_icbm152_wm" + TEMPLATE_SUFFIX),
    ("CSF", "CSF probability", "mni_icbm152_csf" + TEMPLATE_SUFFIX),
)
DEFAULT_TEMPLATE = "T1"
BRAIN_MASK = "mask"
CEREBRA_LABELS = "cerebra"
CEREBRA_CSV = "cerebraCsv"
BILATERAL_LABELS = "cerebraBilateral"
BILATERAL_CSV = "cerebraBilateralCsv"
BILATERAL_STEM = "mni_icbm152_CerebrA_bilateral" + TEMPLATE_SUFFIX
PONS_LABELS = "cerebraPons"
PONS_CSV = "cerebraPonsCsv"
PONS_STEM = "mni_icbm152_CerebrA" + TEMPLATE_SUFFIX + "_wpons"
PONS_CSV_NAME = PONS_STEM + ".csv"
PONS = "Pons"
# Structures of the CerebrA with pons that overlap others (the pons lies inside the brainstem): kept apart from the
# label volume, in overlay label maps made on first use next to the segmentation file
OVERLAY_STRUCTURES = (PONS,)
PONS_OVERLAY = "cerebraPonsOverlay"
PONS_BILATERAL_OVERLAY = "cerebraPonsBilateralOverlay"
PONS_OVERLAY_STEM = PONS_STEM + "_overlay"
PONS_BILATERAL_OVERLAY_STEM = BILATERAL_STEM + "_wpons_overlay"
EXTRA_FILES = (
    (BRAIN_MASK, "mni_icbm152_t1" + TEMPLATE_SUFFIX + "_mask"),
    (CEREBRA_LABELS, "mni_icbm152_CerebrA" + TEMPLATE_SUFFIX),
    (BILATERAL_LABELS, BILATERAL_STEM),
    (PONS_LABELS, PONS_STEM),
    (PONS_OVERLAY, PONS_OVERLAY_STEM),
    (PONS_BILATERAL_OVERLAY, PONS_BILATERAL_OVERLAY_STEM),
)
CSV_NAME = "CerebrA_LabelDetails.csv"
BILATERAL_CSV_NAME = "CerebrA_bilateral_LabelDetails.csv"
GENERATED = (BILATERAL_LABELS, BILATERAL_CSV, PONS_OVERLAY, PONS_BILATERAL_OVERLAY)  # made by the workflow
OPTIONAL = (PONS_LABELS, PONS_CSV)                # the CerebrA with pons, never reported missing
IMAGE_EXTENSIONS = (".seg.nrrd", ".nii", ".nii.gz", ".nrrd", ".mnc")
MAX_SEARCH_DEPTH = 4
REQUIRED = (DEFAULT_TEMPLATE,)       # without these the normalization cannot run


def _stem(fileName):
    lower = fileName.lower()
    for ext in IMAGE_EXTENSIONS:
        if lower.endswith(ext):
            return fileName[:-len(ext)]
    return None


def findAtlasFiles(folder, maxDepth=MAX_SEARCH_DEPTH):
    """{key: path} of the atlas files found in folder and its subfolders (up to maxDepth levels): the templates
    (TEMPLATES keys), the brain mask (BRAIN_MASK), the CerebrA labels (CEREBRA_LABELS) and their CSV (CEREBRA_CSV).
    File names are matched without regard to case; the first one found (shallowest) wins."""
    wanted = {stem.lower(): key for key, _, stem in TEMPLATES}
    wanted.update({stem.lower(): key for key, stem in EXTRA_FILES})
    found = {}
    if not folder or not os.path.isdir(folder):
        return found
    root = os.path.abspath(folder)
    baseDepth = root.rstrip(os.sep).count(os.sep)
    for directory, subdirs, files in os.walk(root):
        depth = directory.rstrip(os.sep).count(os.sep) - baseDepth
        if depth >= maxDepth:
            subdirs[:] = []
        subdirs.sort()
        for fileName in sorted(files):
            csvKeys = {CSV_NAME.lower(): CEREBRA_CSV, BILATERAL_CSV_NAME.lower(): BILATERAL_CSV,
                       PONS_CSV_NAME.lower(): PONS_CSV}
            if fileName.lower() in csvKeys:
                key = csvKeys[fileName.lower()]
                found.setdefault(key, os.path.join(directory, fileName))
                continue
            stem = _stem(fileName)
            key = wanted.get(stem.lower()) if stem else None
            if key is not None:
                found.setdefault(key, os.path.join(directory, fileName))
    return found


def missingFiles(found):
    """Names of the files the workflow looks for that are not in found (findAtlasFiles)."""
    names = {key: stem + ".nii" for key, _, stem in TEMPLATES}
    names.update({key: stem + ".nii" for key, stem in EXTRA_FILES})
    names[CEREBRA_CSV] = CSV_NAME
    return [names[k] for k in names if k not in found and k not in GENERATED and k not in OPTIONAL]


def bilateralPaths(found):
    """(labels path, CSV path) of the bilateral CerebrA: the files found, else where they are written (next to the
    CerebrA labels), or (None, None) without CerebrA."""
    if BILATERAL_LABELS in found and BILATERAL_CSV in found:
        return found[BILATERAL_LABELS], found[BILATERAL_CSV]
    source = found.get(CEREBRA_LABELS)
    if not source:
        return None, None
    folder = os.path.dirname(source)
    return os.path.join(folder, BILATERAL_STEM + ".nii"), os.path.join(folder, BILATERAL_CSV_NAME)


def hasPons(found):
    """Whether the atlas folder holds the CerebrA with pons (segmentation and CSV)."""
    return PONS_LABELS in found and PONS_CSV in found


def labelsCsv(found):
    """CSV of the CerebrA structures: the one of the CerebrA with pons when it is there (all CerebrA rows and the
    pons), else CerebrA_LabelDetails.csv; None without either."""
    if hasPons(found):
        return found[PONS_CSV]
    return found.get(CEREBRA_CSV)


def ponsOverlayPaths(found):
    """(left / right overlay, bilateral overlay) of the CerebrA with pons: the files found, else where they are
    written (next to the segmentation file), or (None, None) without it."""
    if not hasPons(found):
        return None, None
    folder = os.path.dirname(found[PONS_LABELS])
    return (found.get(PONS_OVERLAY) or os.path.join(folder, PONS_OVERLAY_STEM + ".nii"),
            found.get(PONS_BILATERAL_OVERLAY) or os.path.join(folder, PONS_BILATERAL_OVERLAY_STEM + ".nii"))


def overlayValues(structures, mergeSides=False):
    """{label value: name} of the overlapping structures (OVERLAY_STRUCTURES) in the CerebrA values (sides) or the
    bilateral numbering (labelTable with mergeSides)."""
    entries, _ = labelTable(structures, mergeSides)
    wanted = {n.lower() for n in OVERLAY_STRUCTURES}
    out = {}
    for value, name, _ in entries:
        base = name.lower()
        if not mergeSides and base[-2:] in (" r", " l"):
            base = base[:-2]
        if base in wanted:
            out[value] = name
    return out


def templateTitle(key):
    return next((title for k, title, _ in TEMPLATES if k == key), key)


# ---------------------------------------------------------------------------
# Contrast of the patient's anatomy
# ---------------------------------------------------------------------------
# words in the series description / protocol / sequence name, checked in this order (word boundaries, any case)
CONTRAST_WORDS = (
    ("T2", ("flair", "t2", "t2w", "tse_t2", "space_t2", "cube_t2", "stir", "dark_fluid", "darkfluid")),
    ("PD", ("pd", "pdw", "proton density")),
    ("T1", ("t1", "t1w", "mprage", "mp-rage", "mp2rage", "spgr", "fspgr", "bravo", "tfe", "t1_tfe", "vibe", "flash",
            "ir-fspgr", "irfspgr", "3d_t1", "t1_mpr", "mpr")),
)
# echo / repetition time rules (ms) when the names say nothing
T2_MIN_TE, T2_MIN_TR = 60.0, 1500.0
T1_MAX_TE, T1_MAX_TR = 30.0, 1000.0
PD_MIN_TR = 1500.0


def _hasWord(text, word):
    return re.search(r"(?<![a-z0-9])" + re.escape(word) + r"(?![a-z0-9])", text) is not None


def _number(value):
    try:
        text = str(value).split("\\")[0].strip()
        return float(text) if text else None
    except (TypeError, ValueError):
        return None


def classifyAnatomy(modality=None, texts=(), repetitionTime=None, echoTime=None, inversionTime=None):
    """(contrast, reason) of the patient's CT/MR: contrast "CT", "T1", "T2", "PD" or None (unknown). texts: series
    description, protocol name, sequence name, volume name...; times in ms (DICOM strings are fine)."""
    modality = (modality or "").strip().upper()
    if modality == "CT":
        return "CT", "modality CT"
    joined = " ".join(str(t) for t in texts if t).lower()
    if _hasWord(joined, "t1") and _hasWord(joined, "flair"):     # T1 FLAIR is T1 weighted
        return "T1", "'T1' in the series description / protocol"
    for contrast, words in CONTRAST_WORDS:
        for word in words:
            if _hasWord(joined, word):
                return contrast, f"'{word.upper()}' in the series description / protocol"
    tr, te, ti = _number(repetitionTime), _number(echoTime), _number(inversionTime)
    if te is not None and tr is not None:
        if te >= T2_MIN_TE and tr >= T2_MIN_TR:
            return "T2", f"long TE ({te:g} ms) and TR ({tr:g} ms)"
        if te <= T1_MAX_TE and (tr <= T1_MAX_TR or (ti is not None and ti > 0)):
            detail = f" and TR ({tr:g} ms)" if tr <= T1_MAX_TR else f" with inversion ({ti:g} ms)"
            return "T1", f"short TE ({te:g} ms)" + detail
        if te <= T1_MAX_TE and tr >= PD_MIN_TR:
            return "PD", f"short TE ({te:g} ms) and long TR ({tr:g} ms)"
    if modality == "MR":
        return None, "MR, contrast not recognised"
    if _hasWord(joined, "ct"):
        return "CT", "'CT' in the name"
    return None, "not recognised"


def templateForContrast(contrast):
    """Template key that fits a contrast: T2 and PD their own template, T1, CT and unknown the T1."""
    return contrast if contrast in ("T2", "PD") else DEFAULT_TEMPLATE


# ---------------------------------------------------------------------------
# CerebrA labels
# ---------------------------------------------------------------------------
RIGHT, LEFT, BOTH = "R", "L", "R+L"
REFERENCE_STRUCTURE = "Cerebellum Gray Matter"    # default reference region of the reference tissue models


def readCerebraLabels(source):
    """Structures of CerebrA_LabelDetails.csv (path or text): [{"id" (Mindboggle ID), "name", "right" (label value),
    "left" (label value), "notes"}], in the file's order. Columns: Mindboggle ID, Label Name, RH Label, LH Labels,
    Notes, Dice Kappa (header names matched loosely)."""
    if isinstance(source, str) and ("\n" in source or "," in source) and not os.path.isfile(source):
        text = source
    else:
        with open(source, newline="", encoding="utf-8-sig") as f:
            text = f.read()
    text = text.lstrip("﻿")
    firstLine = text.split("\n", 1)[0]
    delimiter = ";" if firstLine.count(";") > firstLine.count(",") else ","     # Excel saves with ';' in some locales
    reader = csv.reader(io.StringIO(text), delimiter=delimiter)
    rows = [r for r in reader if any(c.strip() for c in r)]
    if not rows:
        raise ValueError("The CerebrA label file is empty")
    header = [c.strip().lower() for c in rows[0]]

    def column(*words):
        for i, name in enumerate(header):
            if all(w in name for w in words):
                return i
        raise ValueError(f"The CerebrA label file has no '{' '.join(words)}' column")

    nameCol, rightCol, leftCol = column("name"), column("rh"), column("lh")
    try:
        idCol = column("id")
    except ValueError:
        idCol = None
    try:
        notesCol = column("note")
    except ValueError:
        notesCol = None
    out = []
    for row in rows[1:]:
        def cell(i):
            return row[i].strip() if i is not None and i < len(row) else ""
        name = cell(nameCol)
        right, left = _number(cell(rightCol)), _number(cell(leftCol))
        if not name or (right is None and left is None):
            continue
        out.append({"id": cell(idCol), "name": name, "right": int(right) if right is not None else None,
                    "left": int(left) if left is not None else None, "notes": cell(notesCol)})
    if not out:
        raise ValueError("No labels in the CerebrA label file")
    return out


def structureColor(index, count, side=RIGHT):
    """(r, g, b) 0-1 of a structure: hues spread by the golden ratio, the left side a little darker."""
    hue = (index * 0.618033988749895) % 1.0
    value = 0.92 if side != LEFT else 0.70
    saturation = 0.65 if index % 2 == 0 else 0.85
    return colorsys.hsv_to_rgb(hue, saturation, value)


def labelTable(structures, mergeSides=False):
    """Segments to make from the CerebrA labels. Returns (entries, mapping): entries [(new label value, name,
    (r, g, b))], mapping {CerebrA label value: new label value}. Separate sides: one segment per side, 'Putamen R' /
    'Putamen L' (the CerebrA values are kept); merged: one segment per structure, numbered 1, 2, ..."""
    entries, mapping = [], {}
    count = len(structures)
    if mergeSides:
        for index, s in enumerate(structures):
            value = index + 1
            entries.append((value, s["name"], structureColor(index, count, BOTH)))
            for side in ("right", "left"):
                if s[side] is not None:
                    mapping[s[side]] = value
        return entries, mapping
    for index, s in enumerate(structures):
        for side, suffix in (("right", RIGHT), ("left", LEFT)):
            if s[side] is not None:
                value = s[side]
                entries.append((value, f"{s['name']} {suffix}", structureColor(index, count, suffix)))
                mapping[value] = value
    entries.sort(key=lambda e: e[0])
    return entries, mapping


BILATERAL_HEADER = ("Label", "Label Name", "RH Label", "LH Label")


def bilateralCsvText(structures):
    """CSV of the bilateral CerebrA: one row per structure, its new label value (1, 2, ...) and the CerebrA values of
    the sides it merges."""
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(BILATERAL_HEADER)
    for index, s in enumerate(structures):
        writer.writerow((index + 1, s["name"], "" if s["right"] is None else s["right"],
                         "" if s["left"] is None else s["left"]))
    return out.getvalue()


def readBilateralLabels(source):
    """Segments of CerebrA_bilateral_LabelDetails.csv (path or text): [(label value, name, (r, g, b))]."""
    if isinstance(source, str) and "\n" in source and not os.path.isfile(source):
        text = source
    else:
        with open(source, newline="", encoding="utf-8-sig") as f:
            text = f.read()
    rows = [r for r in csv.reader(io.StringIO(text.lstrip("\ufeff"))) if any(c.strip() for c in r)]
    if not rows or rows[0][0].strip().lower() != "label":
        raise ValueError("Not a bilateral CerebrA label file")
    entries = []
    body = rows[1:]
    for index, row in enumerate(body):
        value = _number(row[0]) if row else None
        if value is None or len(row) < 2 or not row[1].strip():
            continue
        entries.append((int(value), row[1].strip(), structureColor(index, len(body), BOTH)))
    if not entries:
        raise ValueError("No labels in the bilateral CerebrA label file")
    return entries


def remapLabels(array, mapping):
    """Label array with its values replaced through mapping (values not in it become 0)."""
    array = np.asarray(array)
    values = np.rint(array).astype(np.int64) if array.dtype.kind == "f" else array.astype(np.int64)
    top = max(int(values.max(initial=0)), max(mapping, default=0)) + 1
    lookup = np.zeros(top, dtype=np.int32)
    for source, target in mapping.items():
        if 0 <= source < top:
            lookup[source] = target
    values = np.clip(values, 0, top - 1)
    return lookup[values]


def structureValues(structures, name):
    """CerebrA label values (right, left) of the structure called name (any case), or ()."""
    for s in structures:
        if s["name"].strip().lower() == name.strip().lower():
            return tuple(v for v in (s["right"], s["left"]) if v is not None)
    return ()


def segmentNamesOfLabels(values, name=None):
    """Segment names (lower case) that stand for these CerebrA label values in a segmentation file: 'Segment_<value>'
    (Slicer's names of a label map imported as segments), and name / 'name R' / 'name L' when given."""
    names = {f"segment_{int(v)}" for v in values}
    if name:
        base = name.strip().lower()
        names.update({base, f"{base} r", f"{base} l"})
    return names
