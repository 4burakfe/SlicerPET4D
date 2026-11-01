"""Color maps of Review/Edit: the Tarvos (MPI) extension's tables (colortables.py) plus some of
Slicer's own. The MPI tables are made in the scene as "MPI <label>" color nodes, the same names Tarvos uses,
so both extensions share one node."""

import slicer

from . import colortables

# (combo label, color node name or ID); the first one is the default
COLORMAPS = [
    ("PET-DICOM", "MPI PET-DICOM"),
    ("Hel", "MPI Hel"),
    ("Inferno", "Inferno"),
    ("PET-HotMetalBlue", "PET-HotMetalBlue"),
    ("PET-HotIron", "MPI PET-HotIron"),
    ("Jotun", "MPI Jotun"),
    ("Vanaheim", "MPI Vanaheim"),
    ("Valhalla", "MPI Valhalla"),
    ("Trugold", "MPI Trugold"),
    ("PET-Rainbow2", "PET-Rainbow2"),
    ("Grey", "vtkMRMLColorTableNodeGrey"),
    ("Inverted grey", "vtkMRMLColorTableNodeInvertedGrey"),
    ("PET-DICOM (Slicer)", "PET-DICOM"),
]
LABELS = [label for label, _ in COLORMAPS]


def _builtInNode(key):
    scene = slicer.mrmlScene
    return (scene.GetNodeByID(key) or scene.GetFirstNodeByName(key)
            or scene.GetNodeByID(f"vtkMRMLColorTableNodeFile{key}.txt")
            or scene.GetNodeByID(f"vtkMRMLProceduralColorNodeFile{key}.txt"))


def customColorNode(nodeName):
    """An "MPI <label>" color table in the scene, made when missing."""
    node = slicer.mrmlScene.GetFirstNodeByName(nodeName)
    if node is not None and node.IsA("vtkMRMLColorTableNode"):
        return node
    table = colortables.table(nodeName[4:]) if nodeName.startswith("MPI ") else None
    if table is None:
        return None
    node = slicer.mrmlScene.CreateNodeByClass("vtkMRMLColorTableNode")
    node.UnRegister(None)
    node.SetName(nodeName)
    node.SetTypeToUser()
    node.SetNumberOfColors(256)
    try:
        node.SetNamesInitialised(True)
    except AttributeError:
        pass
    for i, (r, g, b) in enumerate(table):
        node.SetColor(i, f"{i}", float(r), float(g), float(b), 1.0)
    node.SetHideFromEditors(False)
    slicer.mrmlScene.AddNode(node)
    return node


def colorNodeForLabel(label):
    """Color node of a combo label (grey when not available)."""
    for name, key in COLORMAPS:
        if name == label:
            node = customColorNode(key) if key.startswith("MPI ") else None
            node = node or _builtInNode(key)
            if node is not None:
                return node
    return slicer.mrmlScene.GetNodeByID("vtkMRMLColorTableNodeGrey")

