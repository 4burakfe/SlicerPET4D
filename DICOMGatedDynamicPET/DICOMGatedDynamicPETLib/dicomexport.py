"""DICOM export of a dynamic PET series (pydicom and numpy only, no Slicer).

A series of PET Image Storage instances, one per slice and frame, in the study of a template image of the original
series (patient, study, frame of reference, tracer and scanner are copied from it). It is written so that the
DICOM Gated / Dynamic PET loader reads it back with the same values and timing:
    Series Type DYNAMIC\\IMAGE, Number of Time Slices, Number of Slices, Image Index = frame x slices + slice + 1
    Acquisition Date / Time = frame start, Actual Frame Duration, Frame Reference Time (frame middle from the
    series time) and Decay Factor
    Series Date / Time = the decay reference (Decay Correction START, also in Decay Correction DateTime), so the
    loader's SUV reference is the same; NONE when the frames are not decay corrected
    Units BQML (Bq/ml) or GML (SUVbw); every image its own Rescale Slope (int16 pixels)
"""

import datetime
import math
import os

import numpy as np

PET_IMAGE_STORAGE = "1.2.840.10008.5.1.4.1.1.128"
UNITS_DICOM = {"Bq/ml": "BQML", "SUVbw": "GML"}
COPIED_KEYWORDS = (
    "PatientName", "PatientID", "PatientBirthDate", "PatientSex", "PatientAge", "PatientWeight", "PatientSize",
    "StudyInstanceUID", "StudyDate", "StudyTime", "StudyID", "AccessionNumber", "ReferringPhysicianName",
    "StudyDescription", "FrameOfReferenceUID", "PositionReferenceIndicator", "Manufacturer",
    "ManufacturerModelName", "DeviceSerialNumber", "StationName", "InstitutionName", "SoftwareVersions",
    "RadiopharmaceuticalInformationSequence", "CorrectedImage", "AttenuationCorrectionMethod",
    "ScatterCorrectionMethod", "RandomsCorrectionMethod", "ReconstructionMethod", "ConvolutionKernel",
    "PatientPosition", "BodyPartExamined", "SpecificCharacterSet")


class ExportError(ValueError):
    pass


def _dt(value):
    if isinstance(value, datetime.datetime):
        return value
    if not value:
        return None
    try:
        return datetime.datetime.fromisoformat(str(value))
    except ValueError:
        return None


def _number(value):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _date(stamp):
    return stamp.strftime("%Y%m%d")


def _time(stamp):
    return stamp.strftime("%H%M%S.%f")


def geometry(ijkToRas):
    """(row direction, column direction, pixel spacing [between rows, between columns], slice step vector, origin)
    in LPS of an IJK to RAS matrix (array kji: rows j, columns i, slices k)."""
    lps = np.diag([-1.0, -1.0, 1.0, 1.0]) @ np.asarray(ijkToRas, dtype=float)
    axes = lps[:3, :3]
    spacing = np.linalg.norm(axes, axis=0)
    if np.any(spacing <= 0):
        raise ExportError("The image geometry is degenerate.")
    return axes[:, 0] / spacing[0], axes[:, 1] / spacing[1], (spacing[1], spacing[0]), axes[:, 2], lps[:3, 3]


def decayFactor(startS, durationS, halfLifeS):
    """Decay correction of a frame starting startS after the reference, lasting durationS: the DICOM Decay Factor
    2^(start / T1/2) x lambda d / (1 - exp(-lambda d))."""
    if halfLifeS is None or halfLifeS <= 0 or startS is None or not durationS:
        return None
    x = math.log(2.0) / halfLifeS * durationS
    return 2.0 ** (startS / halfLifeS) * x / -math.expm1(-x)


def scaledPixels(image):
    """(int16 pixels, rescale slope) of a float slice: the slope spreads the largest absolute value over 32767."""
    image = np.asarray(image, dtype=np.float64)
    largest = float(np.max(np.abs(image))) if image.size else 0.0
    slope = largest / 32767.0 if largest > 0 else 1.0
    return np.round(image / slope).astype(np.int16), slope


def buildDatasets(frames, ijkToRas, rows, template, units, description, seriesNumber, decayCorrection="START",
                  decayReference=None, halfLifeS=None, seriesInstanceUID=None, derivation=None, frameNotes=None):
    """pydicom datasets of the series. frames: arrays kji (one geometry), rows: per frame {"StartDateTime" (ISO or
    datetime), "DurationS"}; template: a pydicom dataset of the original series; units "Bq/ml" or "SUVbw";
    decayCorrection "START" (to decayReference), "ADMIN" or "NONE"."""
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    if units not in UNITS_DICOM:
        raise ExportError(f"Units {units} cannot be exported (Bq/ml or SUVbw only).")
    if not frames:
        raise ExportError("No frames.")
    shapes = {np.shape(f) for f in frames}
    if len(shapes) != 1 or len(next(iter(shapes))) != 3:
        raise ExportError("The frames are not on one image grid.")
    starts = [_dt(r.get("StartDateTime")) for r in rows]
    durations = [_number(r.get("DurationS")) for r in rows]
    if any(s is None for s in starts) or any(d is None or d <= 0 for d in durations):
        raise ExportError("A frame start or duration is unknown.")
    reference = _dt(decayReference)
    if decayCorrection == "START" and reference is None:
        raise ExportError("The decay correction time is unknown.")
    seriesTime = reference if (decayCorrection == "START" and reference is not None) else min(starts)
    rowDir, colDir, pixelSpacing, sliceStep, origin = geometry(ijkToRas)
    slices = next(iter(shapes))[0]
    seriesUID = seriesInstanceUID or generate_uid()
    now = datetime.datetime.now()
    datasets = []
    for f, (array, start, duration) in enumerate(zip(frames, starts, durations)):
        sinceReference = (start - seriesTime).total_seconds()
        factor = decayFactor(sinceReference, duration, halfLifeS) if decayCorrection == "START" else None
        for k in range(slices):
            meta = FileMetaDataset()
            meta.MediaStorageSOPClassUID = PET_IMAGE_STORAGE
            meta.MediaStorageSOPInstanceUID = generate_uid()
            meta.TransferSyntaxUID = ExplicitVRLittleEndian
            ds = Dataset()
            ds.file_meta = meta
            for keyword in COPIED_KEYWORDS:
                if keyword in template:
                    setattr(ds, keyword, template.data_element(keyword).value)
            ds.SOPClassUID, ds.SOPInstanceUID = PET_IMAGE_STORAGE, meta.MediaStorageSOPInstanceUID
            ds.Modality = "PT"
            ds.SeriesInstanceUID = seriesUID
            ds.SeriesDescription = description
            ds.SeriesNumber = int(seriesNumber)
            ds.SeriesDate, ds.SeriesTime = _date(seriesTime), _time(seriesTime)
            ds.AcquisitionDate, ds.AcquisitionTime = _date(start), _time(start)
            ds.ContentDate, ds.ContentTime = _date(now), _time(now)
            ds.InstanceCreationDate, ds.InstanceCreationTime = _date(now), _time(now)
            ds.ImageType = ["DERIVED", "PRIMARY"]
            ds.DerivationDescription = ((frameNotes[f] if frameNotes and frameNotes[f] else derivation) or
                                        "Dynamic PET (PET4D)")[:1024]
            ds.SeriesType = ["DYNAMIC", "IMAGE"]
            ds.Units = UNITS_DICOM[units]
            ds.CountsSource = "EMISSION"
            ds.DecayCorrection = decayCorrection
            if decayCorrection == "START":
                ds.DecayCorrectionDateTime = seriesTime.strftime("%Y%m%d%H%M%S.%f")
            if factor is not None:
                ds.DecayFactor = f"{factor:.10g}"[:16]
            ds.NumberOfSlices = slices
            ds.NumberOfTimeSlices = len(frames)
            ds.ImageIndex = f * slices + k + 1
            ds.InstanceNumber = f * slices + k + 1
            ds.ActualFrameDuration = int(round(duration * 1000.0))
            ds.FrameReferenceTime = f"{(sinceReference + duration / 2.0) * 1000.0:.6g}"
            position = origin + k * sliceStep
            ds.ImagePositionPatient = [f"{v:.6g}" for v in position]
            ds.ImageOrientationPatient = [f"{v:.8g}" for v in (*rowDir, *colDir)]
            ds.PixelSpacing = [f"{v:.8g}" for v in pixelSpacing]
            ds.SliceThickness = f"{float(np.linalg.norm(sliceStep)):.8g}"
            normal = np.cross(rowDir, colDir)
            ds.SliceLocation = f"{float(np.dot(position, normal)):.6g}"
            pixels, slope = scaledPixels(array[k])
            ds.Rows, ds.Columns = pixels.shape
            ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
            ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 1
            ds.RescaleSlope, ds.RescaleIntercept = f"{slope:.10g}"[:16], "0"
            ds.PixelData = pixels.astype("<i2").tobytes()
            datasets.append(ds)
    return datasets


def writeDatasets(folder, datasets):
    """Write the datasets as files in folder; returns the paths."""
    os.makedirs(folder, exist_ok=True)
    paths = []
    for n, ds in enumerate(datasets):
        path = os.path.join(folder, f"PT{n + 1:05d}.dcm")
        try:
            ds.save_as(path, enforce_file_format=True)          # pydicom 3
        except TypeError:
            ds.is_little_endian, ds.is_implicit_VR = True, False  # pydicom 2
            ds.save_as(path, write_like_original=False)
        paths.append(path)
    return paths
