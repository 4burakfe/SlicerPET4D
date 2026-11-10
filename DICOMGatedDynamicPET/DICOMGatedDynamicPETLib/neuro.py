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
    # made on first use from the grey and white matter maps (pseudoPetArray), saved next to them
    ("pseudoPET", "Synthetic PET (GM + 0.3 WM, smoothed)", "mni_icbm152_pseudopet" + TEMPLATE_SUFFIX),
)
PSEUDO_PET = "pseudoPET"
PSEUDO_PET_WM_WEIGHT = 0.3      # FDG: white matter uptake about a third of the grey matter's
PSEUDO_PET_FWHM_MM = 6.0        # smoothed to a PET resolution
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
GROUPS_CSV = "cerebraGroups"
GROUPS_CSV_NAME = "CerebrA_groups.csv"
GROUPED_LABELS, GROUPED_CSV = "cerebraGrouped", "cerebraGroupedCsv"
GROUPED_BILATERAL_LABELS, GROUPED_BILATERAL_CSV = "cerebraGroupedBilateral", "cerebraGroupedBilateralCsv"
GROUPED_OVERLAY, GROUPED_BILATERAL_OVERLAY = "cerebraGroupedOverlay", "cerebraGroupedBilateralOverlay"
GROUPED_STEM = "mni_icbm152_CerebrA_grouped" + TEMPLATE_SUFFIX
GROUPED_BILATERAL_STEM = "mni_icbm152_CerebrA_grouped_bilateral" + TEMPLATE_SUFFIX
GROUPED_CSV_NAME = "CerebrA_grouped_LabelDetails.csv"
CT_TEMPLATE = "ctTemplate"      # a CT template in the MNI space: a patient's CT is registered to it (CT to CT)
CT_TEMPLATE_GM = "ctTemplateGm" # the grey matter map in the CT template's space (its MNI is not quite the 2009c)
CT_CORRECTION_SUFFIX = "_to_icbm152_2009c.json"     # the CT template to 2009c matrix, cached next to the CT template


def ctCorrectionPath(found):
    """Where the CT template to ICBM 152 2009c matrix is kept (next to the CT template), None without one."""
    path = found.get(CT_TEMPLATE)
    if not path:
        return None
    stem = _stem(os.path.basename(path)) or os.path.basename(path)
    return os.path.join(os.path.dirname(path), stem + CT_CORRECTION_SUFFIX)


def ctCorrectedPath(found):
    """Where the CT template moved onto the ICBM 152 2009c is saved (to look at; its geometry carries the
    correction), None without a CT template."""
    path = ctCorrectionPath(found)
    return path[:-len(".json")] + ".nii" if path else None
GROUPED_BILATERAL_CSV_NAME = "CerebrA_grouped_bilateral_LabelDetails.csv"
EXTRA_FILES = (
    (BRAIN_MASK, "mni_icbm152_t1" + TEMPLATE_SUFFIX + "_mask"),
    (CEREBRA_LABELS, "mni_icbm152_CerebrA" + TEMPLATE_SUFFIX),
    (BILATERAL_LABELS, BILATERAL_STEM),
    (PONS_LABELS, PONS_STEM),
    (PONS_OVERLAY, PONS_OVERLAY_STEM),
    (PONS_BILATERAL_OVERLAY, PONS_BILATERAL_OVERLAY_STEM),
    (GROUPED_LABELS, GROUPED_STEM),
    (GROUPED_BILATERAL_LABELS, GROUPED_BILATERAL_STEM),
    (GROUPED_OVERLAY, GROUPED_STEM + "_overlay"),
    (GROUPED_BILATERAL_OVERLAY, GROUPED_BILATERAL_STEM + "_overlay"),
    (CT_TEMPLATE, "scct_unsmooth"),     # optional: the CT template of the SPM Clinical Toolbox (Rorden et al. 2012,
    (CT_TEMPLATE, "scct"),              # BSD-2-Clause), put in the atlas folder by the user
    (CT_TEMPLATE_GM, "scgrey"),         # its grey matter map: puts its MNI space onto the ICBM 152 2009c
)
CSV_NAME = "CerebrA_LabelDetails.csv"
BILATERAL_CSV_NAME = "CerebrA_bilateral_LabelDetails.csv"
GENERATED = (PSEUDO_PET, BILATERAL_LABELS, BILATERAL_CSV, PONS_OVERLAY,     # made by the workflow
             PONS_BILATERAL_OVERLAY,
             GROUPED_LABELS, GROUPED_CSV, GROUPED_BILATERAL_LABELS, GROUPED_BILATERAL_CSV, GROUPED_OVERLAY,
             GROUPED_BILATERAL_OVERLAY)
# never reported missing (the groups: the extension's table)
OPTIONAL = (PONS_LABELS, PONS_CSV, GROUPS_CSV, CT_TEMPLATE, CT_TEMPLATE_GM)
IMAGE_EXTENSIONS = (".seg.nrrd", ".nii", ".nii.gz", ".nrrd", ".mnc")
MAX_SEARCH_DEPTH = 4
REQUIRED = (DEFAULT_TEMPLATE,)       # without these the normalization cannot run


def _stem(fileName):
    lower = fileName.lower()
    for ext in IMAGE_EXTENSIONS:
        if lower.endswith(ext):
            return fileName[:-len(ext)]
    return None


# The atlas files that come with the extension (Resources/Atlas: the MNI ICBM 152 2009c T1, brain mask, grey and white
# matter maps, CerebrA and the CerebrA with pons; their licenses in LICENSE.txt there). The files of the user's atlas
# folder come first (atlasFiles). What the workflow makes from the extension's files (GENERATED) is written into
# generatedFolder(), not into the extension.
BUNDLED_FOLDER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Resources", "Atlas")
GENERATED_FOLDER_VARIABLE = "PET4D_NEURO_ATLAS_CACHE"


def generatedFolder():
    """Where the files made from the extension's atlas files are kept: $PET4D_NEURO_ATLAS_CACHE, else
    ~/.PET4D/NeuroAtlas."""
    return os.environ.get(GENERATED_FOLDER_VARIABLE) or os.path.join(os.path.expanduser("~"), ".PET4D", "NeuroAtlas")


def _sameFolder(a, b):
    return os.path.normcase(os.path.realpath(a)) == os.path.normcase(os.path.realpath(b))


def outputFolder(source):
    """The folder the files made from the atlas file source are written to: next to it; generatedFolder() (made
    when missing) for a file of the extension."""
    folder = os.path.dirname(source)
    if not _sameFolder(folder, BUNDLED_FOLDER):
        return folder
    target = generatedFolder()
    try:
        os.makedirs(target, exist_ok=True)
    except OSError:
        pass        # the workflow then saves in Slicer's temporary folder
    return target


# keys that belong together (made from one another): taken from one place
_CEREBRA_KEYS = (CEREBRA_LABELS, CEREBRA_CSV, BILATERAL_LABELS, BILATERAL_CSV, PONS_LABELS, PONS_CSV, PONS_OVERLAY,
                 PONS_BILATERAL_OVERLAY, GROUPED_LABELS, GROUPED_CSV, GROUPED_BILATERAL_LABELS,
                 GROUPED_BILATERAL_CSV, GROUPED_OVERLAY, GROUPED_BILATERAL_OVERLAY)
_TISSUE_KEYS = ("GM", "WM", PSEUDO_PET)


def atlasFiles(folder=None, bundled=None):
    """{key: path} of the atlas files the workflows use: those of the user's atlas folder (findAtlasFiles) and, for
    what it lacks, the extension's own (bundled: BUNDLED_FOLDER by default) with the files made from them
    (generatedFolder()). The CerebrA files are taken together from the user's folder when it has the CerebrA labels
    (else all from the extension), the grey / white matter maps and the synthetic PET together from the user's folder
    when it has both maps; the other files one by one."""
    bundled = BUNDLED_FOLDER if bundled is None else bundled
    own = findAtlasFiles(bundled, maxDepth=1) if bundled else {}
    if own:
        made = findAtlasFiles(generatedFolder(), maxDepth=1)
        own.update({k: v for k, v in made.items() if k in GENERATED and k not in own})
    user = findAtlasFiles(folder) if folder else {}
    found = dict(own)
    userCerebra = CEREBRA_LABELS in user
    userTissue = "GM" in user and "WM" in user
    for keys, fromUser in ((_CEREBRA_KEYS, userCerebra), (_TISSUE_KEYS, userTissue)):
        if fromUser:
            for key in keys:
                found.pop(key, None)
    for key, path in user.items():
        if (key in _CEREBRA_KEYS and not userCerebra) or (key in _TISSUE_KEYS and not userTissue):
            continue
        found[key] = path
    return found


def bundledKeys(found):
    """The keys of found (atlasFiles) that are the extension's own files (or made from them)."""
    folders = (BUNDLED_FOLDER, generatedFolder())
    return [k for k, p in found.items() if any(_sameFolder(os.path.dirname(p), f) for f in folders)]


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
                       PONS_CSV_NAME.lower(): PONS_CSV, GROUPS_CSV_NAME.lower(): GROUPS_CSV,
                       GROUPED_CSV_NAME.lower(): GROUPED_CSV,
                       GROUPED_BILATERAL_CSV_NAME.lower(): GROUPED_BILATERAL_CSV}
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
    folder = outputFolder(source)
    return os.path.join(folder, BILATERAL_STEM + ".nii"), os.path.join(folder, BILATERAL_CSV_NAME)


def pseudoPetPath(found):
    """Where the synthetic PET template is: the file found, else next to the grey matter map; None without the grey
    and white matter maps."""
    if PSEUDO_PET in found:
        return found[PSEUDO_PET]
    if "GM" not in found or "WM" not in found:
        return None
    stem = next(s for k, _, s in TEMPLATES if k == PSEUDO_PET)
    return os.path.join(outputFolder(found["GM"]), stem + ".nii")


def canMakeTemplate(found, key):
    """Whether the template key can be used: its file is there, or (the synthetic PET) it can be made."""
    return key in found or (key == PSEUDO_PET and pseudoPetPath(found) is not None)


def _probability(array):
    array = np.asarray(array, dtype=np.float32)
    top = float(array.max()) if array.size else 0.0
    return array / top if top > 1.5 else array        # stored 0-100 or 0-255


def pseudoPetArray(gm, wm, spacing=(1.0, 1.0, 1.0), wmWeight=PSEUDO_PET_WM_WEIGHT, fwhmMm=PSEUDO_PET_FWHM_MM):
    """A synthetic FDG-like brain image from the grey and white matter probability maps (same grid; spacing of the
    numpy axes in mm): GM + wmWeight x WM, smoothed with a Gaussian of fwhmMm, scaled to a maximum of 1."""
    gm, wm = _probability(gm), _probability(wm)
    if gm.shape != wm.shape:
        raise ValueError("The grey and white matter maps are not on the same grid.")
    image = gm + float(wmWeight) * wm
    if fwhmMm > 0:
        from scipy import ndimage
        sigma = [fwhmMm / 2.3548 / float(s) for s in spacing]
        image = ndimage.gaussian_filter(image, sigma)
    top = float(image.max()) if image.size else 0.0
    return (image / top if top > 0 else image).astype(np.float32)


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
    folder = outputFolder(found[PONS_LABELS])
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


CT_TEMPLATE_TITLE = "CT template (scct, CT to CT)"


def templateTitle(key):
    if key == CT_TEMPLATE:
        return CT_TEMPLATE_TITLE
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
    """Template key that fits a contrast: T2 and PD their own template, a CT the CT template (CT_TEMPLATE; the T1
    when it is not in the atlas folder), T1 and unknown the T1."""
    if contrast == "CT":
        return CT_TEMPLATE
    return contrast if contrast in ("T2", "PD") else DEFAULT_TEMPLATE


# ---------------------------------------------------------------------------
# A CT for the MR template: the brain only (numpy + scipy)
# ---------------------------------------------------------------------------
# The MR shows no skull (cortical bone has no signal) and the T1's scalp is bright fat, while the CT's bone is its
# brightest and its brain lies in a few HU; on the whole CT range a mutual information histogram holds the brain in
# one bin. So a CT is normalized as its brain only, in a brain window, to the skull-stripped template.
CT_BONE_HU = 200.0              # bone at and above this
CT_BRAIN_HU = (-10.0, 100.0)    # brain, CSF and blood (a haemorrhage too) between these
CT_BRAIN_WINDOW = (0.0, 80.0)   # the brain image clipped to this window
CT_SKULL_CLOSE_MM = 2.0         # the bone grown by this much first (thin bone, sutures, partial volume)
CT_BRAIN_ERODE_MM = 4.0         # the brain eroded by this much to cut it from the orbits and the neck, then grown back


def looksLikeCt(voxels):
    """Whether an image is in Hounsfield units (air at about -1000)."""
    array = np.asarray(voxels)
    return bool(array.size) and float(np.nanmin(array[::2, ::2, ::2])) <= -500.0


def _ellipsoid(radiusMm, spacing):
    """A boolean structuring element of an ellipsoid of radiusMm (numpy axes' spacing in mm)."""
    half = [max(0, int(np.floor(radiusMm / float(s) + 1e-6))) for s in spacing]
    grids = np.ogrid[tuple(slice(-h, h + 1) for h in half)]
    distance = sum((g * float(s) / radiusMm) ** 2 for g, s in zip(grids, spacing))
    return distance <= 1.0 + 1e-6


def _enclosedPerSlice(mask, axis):
    """The voxels enclosed by mask in the slices across one axis (holes of each slice, mask not included)."""
    from scipy import ndimage
    structure = np.zeros((3, 3, 3), dtype=bool)
    index = [slice(None)] * 3
    index[axis] = 1
    structure[tuple(index)] = ndimage.generate_binary_structure(2, 1)
    labels, _ = ndimage.label(~mask, structure=structure)
    border = set()
    for other in (a for a in range(3) if a != axis):
        for end in (0, -1):
            border.update(np.unique(np.take(labels, end, axis=other)).tolist())
    border.discard(0)
    return (labels > 0) & ~np.isin(labels, list(border))


def ctBrainMask(hu, spacing=(1.0, 1.0, 1.0)):
    """The brain of a head CT (Hounsfield units; numpy axes' spacing in mm): soft tissue enclosed by the skull in the
    slices of any of the three orientations, eroded, its largest piece, grown back. Empty when nothing is found."""
    from scipy import ndimage
    hu = np.asarray(hu, dtype=np.float32)
    skull = ndimage.binary_dilation(hu >= CT_BONE_HU, structure=_ellipsoid(CT_SKULL_CLOSE_MM, spacing))
    inside = np.zeros(hu.shape, dtype=bool)
    for axis in range(3):
        inside |= _enclosedPerSlice(skull, axis)
    candidate = inside & (hu >= CT_BRAIN_HU[0]) & (hu <= CT_BRAIN_HU[1])
    ball = _ellipsoid(CT_BRAIN_ERODE_MM, spacing)
    core = ndimage.binary_erosion(candidate, structure=ball)
    labels, count = ndimage.label(core)
    if count == 0:
        return np.zeros(hu.shape, dtype=bool)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    brain = ndimage.binary_dilation(labels == int(np.argmax(sizes)), structure=ball) & candidate
    return ndimage.binary_fill_holes(brain)


# CT intensities stretched for a mutual information histogram (as the Cormack units of the Clinical Toolbox): the soft
# tissue range (CT_STRETCH_HU) gets most of the scale, air and bone a little at either end
CT_STRETCH_HU = (-100.0, 100.0)
CT_STRETCH_POINTS = ((-1000.0, 0.0), (CT_STRETCH_HU[0], 100.0), (CT_STRETCH_HU[1], 1100.0), (1500.0, 1200.0))


def ctStretch(hu):
    """A CT in Hounsfield units mapped (monotonic, piecewise linear) so that -100 ... 100 HU spans 1000 of 1200 units:
    the brain is spread over the histogram instead of a bin or two of the whole HU range."""
    xs, ys = zip(*CT_STRETCH_POINTS)
    return np.interp(np.asarray(hu, dtype=np.float32), xs, ys).astype(np.float32)


def ctBrainImage(hu, spacing=(1.0, 1.0, 1.0)):
    """(image, brain voxel count): the CT's brain in the brain window (CT_BRAIN_WINDOW), 0 outside the brain."""
    mask = ctBrainMask(hu, spacing)
    image = np.clip(np.asarray(hu, dtype=np.float32), *CT_BRAIN_WINDOW)
    image[~mask] = 0.0
    return image, int(mask.sum())


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



# ---------------------------------------------------------------------------
# Grouped CerebrA (broad regions: prefrontal, lateral occipital, ...)
# ---------------------------------------------------------------------------
DEFAULT_GROUPS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Resources", "Atlas",
                                   GROUPS_CSV_NAME)


def groupsCsv(found):
    """Table of the broad regions: CerebrA_groups.csv of the atlas folder when the user put one there, else the
    extension's (Resources/Atlas/CerebrA_groups.csv)."""
    return found.get(GROUPS_CSV) or DEFAULT_GROUPS_PATH


def readGroups(source):
    """Broad regions of a groups table (path or text): [(group name, [CerebrA structure names])]. Columns Group and
    Structures (';' or ',' separated file; the structures of a group separated by ',' in a ';' file, by '|' in a ','
    file); other columns (Notes) are ignored. Raises ValueError for a structure in two groups."""
    if isinstance(source, str) and "\n" in source and not os.path.isfile(source):
        text = source
    else:
        with open(source, newline="", encoding="utf-8-sig") as f:
            text = f.read()
    text = text.lstrip("\ufeff")
    firstLine = text.split("\n", 1)[0]
    delimiter = ";" if firstLine.count(";") >= firstLine.count(",") else ","
    memberSeparator = "," if delimiter == ";" else "|"
    rows = [r for r in csv.reader(io.StringIO(text), delimiter=delimiter) if any(c.strip() for c in r)]
    if not rows:
        raise ValueError("The groups table is empty")
    header = [c.strip().lower() for c in rows[0]]
    if "group" not in header or "structures" not in header:
        raise ValueError("The groups table needs the columns Group and Structures")
    groupCol, membersCol = header.index("group"), header.index("structures")
    groups, seen = [], {}
    for row in rows[1:]:
        name = row[groupCol].strip() if groupCol < len(row) else ""
        members = [m.strip() for m in (row[membersCol] if membersCol < len(row) else "").split(memberSeparator)]
        members = [m for m in members if m]
        if not name or not members:
            continue
        for member in members:
            if member.lower() in seen:
                raise ValueError(f"'{member}' is in the groups '{seen[member.lower()]}' and '{name}'")
            seen[member.lower()] = name
        groups.append((name, members))
    if not groups:
        raise ValueError("No groups in the groups table")
    return groups


def groupedTable(structures, groups, mergeSides=False):
    """Segments of the grouped CerebrA. Returns (entries [(value, name, color)], mapping {CerebrA label value: group
    value}, unknown structure names). Sides separate: 'Prefrontal R' (value 2i+1) and 'Prefrontal L' (2i+2);
    merged: 'Prefrontal' (i+1). CerebrA structures in no group are left out (ventricles, optic chiasm)."""
    byName = {s["name"].strip().lower(): s for s in structures}
    entries, mapping, unknown = [], {}, []
    count = len(groups)
    for index, (name, members) in enumerate(groups):
        found = [byName[m.lower()] for m in members if m.lower() in byName]
        unknown.extend(m for m in members if m.lower() not in byName)
        if not found:
            continue
        if mergeSides:
            value = index + 1
            entries.append((value, name, structureColor(index, count, BOTH)))
            for s in found:
                for side in ("right", "left"):
                    if s[side] is not None:
                        mapping[s[side]] = value
            continue
        for side, suffix, value in (("right", RIGHT, 2 * index + 1), ("left", LEFT, 2 * index + 2)):
            values = [s[side] for s in found if s[side] is not None]
            if values:
                entries.append((value, f"{name} {suffix}", structureColor(index, count, suffix)))
                mapping.update({v: value for v in values})
    return entries, mapping, unknown


GROUPED_HEADER = ("Label", "Label Name", "Structures")


def groupedCsvText(entries, groups):
    """CSV of a grouped CerebrA (read with readBilateralLabels): label value, segment name, its structures."""
    members = {name.lower(): ", ".join(m) for name, m in groups}
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(GROUPED_HEADER)
    for value, name, _ in entries:
        base = name[:-2] if name[-2:] in (" " + RIGHT, " " + LEFT) and name[:-2].lower() in members else name
        writer.writerow((value, name, members.get(base.lower(), "")))
    return out.getvalue()


def groupedPaths(found, mergeSides=False):
    """(labels, CSV, overlay) paths of a grouped CerebrA: the files found, else next to the CerebrA labels; (None,
    None, None) without CerebrA."""
    keys = ((GROUPED_BILATERAL_LABELS, GROUPED_BILATERAL_CSV, GROUPED_BILATERAL_OVERLAY) if mergeSides
            else (GROUPED_LABELS, GROUPED_CSV, GROUPED_OVERLAY))
    stem = GROUPED_BILATERAL_STEM if mergeSides else GROUPED_STEM
    csvName = GROUPED_BILATERAL_CSV_NAME if mergeSides else GROUPED_CSV_NAME
    source = found.get(CEREBRA_LABELS)
    if not source:
        return None, None, None
    folder = outputFolder(source)
    defaults = (os.path.join(folder, stem + ".nii"), os.path.join(folder, csvName),
                os.path.join(folder, stem + "_overlay.nii"))
    return tuple(found.get(k) or d for k, d in zip(keys, defaults))


# ---------------------------------------------------------------------------
# Growing the regions, eroding a reference (numpy + scipy)
# ---------------------------------------------------------------------------
def growLabels(labels, distanceMm, spacing=(1.0, 1.0, 1.0), allowed=None, fixedValues=()):
    """Regions grown by distanceMm into the unlabelled voxels: each such voxel within distanceMm of a region (and in
    allowed, when given) goes to the nearest region, so regions never overlap and the order does not matter. Labelled
    voxels are kept; fixedValues (ventricles, ...) neither grow nor are grown into. spacing: of the array axes (mm).
    """
    from scipy import ndimage
    labels = np.asarray(labels)
    if distanceMm <= 0:
        return labels.copy()
    seeds = labels > 0
    if len(fixedValues):
        seeds &= ~np.isin(labels, list(fixedValues))
    if not seeds.any():
        return labels.copy()
    distance, indices = ndimage.distance_transform_edt(~seeds, sampling=spacing, return_indices=True)
    grow = (labels == 0) & (distance <= distanceMm + 1e-6)
    if allowed is not None:
        grow &= np.asarray(allowed, dtype=bool)
    out = labels.copy()
    nearest = labels[tuple(indices)]
    out[grow] = nearest[grow]
    return out


def erodeMask(mask, distanceMm, spacing=(1.0, 1.0, 1.0)):
    """The mask without its voxels within distanceMm of the outside (mm, any voxel size). Nothing removed for 0."""
    from scipy import ndimage
    mask = np.asarray(mask, dtype=bool)
    if distanceMm <= 0 or not mask.any():
        return mask.copy()
    padded = np.pad(mask, 1)        # the edge of the array counts as outside
    distance = ndimage.distance_transform_edt(padded, sampling=spacing)[1:-1, 1:-1, 1:-1]
    return mask & (distance > distanceMm + 1e-6)


NO_GROW_WORDS = ("ventricle", "optic chiasm")       # CSF and the chiasm: never grown


def fixedValuesOf(entries):
    """Label values of the entries that must not grow (ventricles, optic chiasm)."""
    return [v for v, name, _ in entries if any(w in name.lower() for w in NO_GROW_WORDS)]
