# DICOM Gated / Dynamic PET (3D Slicer)

A DICOM plugin that loads **ECG gated** and **dynamic** PET series as sequences, in SUV (body weight, lean body mass or body surface area) or Bq/ml, and keeps
the tracer, decay correction and frame timing with every volume. Plain static PET loads too.

> **Intended use.** Research software. It is not a certified medical device and must not be used for
> clinical diagnosis.

## Why

Slicer's PET SUV plugin (PETDICOMExtension) targets static PET, and the MultiVolume importer handles time
series generically. Gated and dynamic PET series are stacks of single-slice images, one per slice and frame.
Those plugins often stack such a series into one volume, fail to split it, or lose the per-image rescale
slope, the per-frame timing and the decay correction. This plugin fills that gap. For static PET it gives way
to the PET SUV plugin when that one is installed (see *Static PET* below).

## What it loads

The series shows up in the DICOM browser with one loadable per unit; the first one is selected.

| Series | Recognised by | Loadables (selected first) | Result |
|---|---|---|---|
| Gated | Series Type `GATED\IMAGE`, or Number of Time Slots (0054,0071) > 1 | *gated SUVbw*, *gated Bq/ml* | **mean of the gates** (static image) + sequence of gates (index *Gate*) |
| Dynamic | Series Type `DYNAMIC\IMAGE`, or Number of Time Slices (0054,0101) > 1 | *dynamic Bq/ml*, *dynamic SUVbw* | sequence of time frames (index *Time*, s) + **frame timing table** |
| Static | any other PT series | *SUVbw*, *Bq/ml* | one volume |

An explicit Series Type wins over the frame counts (a `DYNAMIC` series with Number of Time Slots > 1 stays
dynamic); the counts decide only when the Series Type is missing or unknown.

- Images are split into frames by Image Index (0054,1330), else Trigger Time (gated) or Frame Reference
  Time / acquisition time (dynamic), else repeated slice positions, and sorted along the slice normal.
  Every image keeps its own rescale slope.
- Sequences come with a browser and a *(current gate)* / *(current frame)* volume, one window for all
  frames (no flicker in cine), in the series' patient/study folder.
- Each gate of a gated PET is a full activity image, so the static image is the **mean** of the gates. It
  is paired with the gates (`MPI.GatedSequence`), so the Tarvos (Myocardial Perfusion) extension lists one
  study with its gated sequence.
- Dynamic series load in Bq/ml first, the usual unit for kinetic modelling.

### Incomplete series

Missing frames and slices do not stop the loading; the series loads with what is there and the user is
warned (a message after loading, the Python log, and the `LoadWarnings` attribute on every volume).

- The DICOM browser already shows a warning on the loadable when the series has fewer images than the
  header announces (frames x Number of Slices).
- A **missing gate or time frame** is left out. The others keep their own numbers (gates *1, 2, 4*), so
  gate attributes and frame timing stay right.
- A **missing slice** keeps its place on the slice grid shared by all frames, so the frames keep the same
  geometry. When the frame has slices before and after it, it is linearly interpolated from the nearest
  ones along the slice axis (the same result as hardening Slicer's acquisition geometry regularization
  transform for a scalar volume with a gap). Missing slices at the top or bottom of the stack are left
  empty (0). The mean of the gates and the dynamic summed volume take each slice from the frames that
  really have it, and use the interpolated slices only where no frame has it.
- Duplicate images at one slice position are skipped. With uneven slice spacing the volume is built with
  the median spacing.

## Settings

The **Gated / Dynamic PET Settings** module (Nuclear Medicine category) sets the loader's defaults. Changes are
saved at once and apply to series examined in the DICOM browser afterwards.

| Setting | Choices | Default |
|---|---|---|
| SUV method | body weight (SUVbw), lean body mass (SUVlbm / SUL, Janmahasatian), body surface area (SUVbsa, Du Bois) | SUVbw |
| Loadable selected first, per static / gated / dynamic | SUV or Bq/ml | SUV, SUV, Bq/ml |
| Summed volumes, gated / dynamic | on or off | on, on |

- Every series offers the chosen SUV and Bq/ml; the other one stays in the list, unselected.
- SUVlbm needs the patient height and sex (M / F), SUVbsa the height. Without them the series offers SUVbw,
  and the DICOM browser says why.
- **Gated summed volume:** the mean of the gates (see below). Tarvos lists a gated PET study through this
  image, so keep it on for Tarvos.
- **Dynamic summed volume:** *"<series> summed Bq/ml"*. The frames are summed by their durations, which is
  the duration-weighted mean, so the values stay in Bq/ml or SUV. It links to its sequence
  (`DynamicPET.DynamicSequence`) and carries the total duration and the first start / last end from the
  injection.
- With a summed volume off, the sequence's *(current gate/frame)* volume is shown instead.

## Dynamic PET Editor

The **Dynamic PET Editor** module (Nuclear Medicine) reviews a dynamic series frame by frame. Pick the dynamic
sequence (the most recent one is picked for you) and press *Open editor*:

- **Upper 2/3:** a large axial view of the current frame, and next to it a near-square grid of small axial
  views, one per frame (5 frames: 2 x 3, 20 frames: 4 x 5). The views are linked (Slicer's view link,
  on by default, in their own view group): changing the orientation, slice, zoom or pan of any view, large or
  small, changes all of them.
- **Lower 1/3:** a timeline dock. The ruler has three rows: clock time, time from the study start and time
  from the injection (each when the header tells it). Under it is one block per frame, each as wide as the
  frame's duration. The toolbar line gives the current frame's start - end in all three. Click a block to show that frame. Ctrl / Shift + click selects
  frames, and Left / Right step through them.
- **Play** runs in acquisition time: each frame stays on screen for its duration, scaled by the speed shown
  next to *Slower* / *Faster* (halve / double). The default plays the whole series in 60 s (a 90 min series at
  x90, 1 s = 1:30). It plays the timeline's time window and loops.
- **Time window:** a two-handle slider with spin boxes (seconds from the study start), like a volume's display
  range, zooms the timeline into part of the series. The mouse wheel over the timeline also zooms, around the
  cursor. *Show all* goes back to the whole series.
- **Overlays:** each view shows its frame number and its start - end time from the study start in the top
  left corner. The current frame is highlighted.
- **Window** (module panel): one display range for all frames (min / max slider, *Auto* = 0 to the highest
  99.9th percentile; dragging the window in the large view changes all frames too), or *each frame automatic*
  (0 to its own 99.9th percentile, so late low-count frames stay visible), and the color map: the Tarvos (MPI)
  maps (PET-DICOM, Hel, PET-HotIron, Jotun, Vanaheim, Valhalla, Trugold) and Slicer's Inferno, PET maps and grey.
- **Fusion** (module panel): pick a background volume (CT / MR); every view shows it under its PET frame, with a
  PET opacity slider.
- **Close editor:** goes back to the previous layout and removes the editor's helper volumes.

Series longer than *Frame views (max)* (64 by default) show that many frames, spread over the series. The frame
operations (sum selected, merge, split, reframe, remove) are placeholders for now.

### SUV

SUVbw follows the QIBA vendor-neutral pseudo-code: weight, injected dose decayed to the decay reference.
SUVlbm and SUVbsa use the same decayed dose, with the lean body mass (g) or the body surface area (cm2) in
place of the weight (`SUVMethod`, `LeanBodyMassKg`, `BodySurfaceAreaM2` are stored on the volumes).

- *Decay Correction START*: the dose is decayed to the series time (or GE's scan start when the series time
  is later than the acquisition, else the first acquisition time); one scale for every frame.
- *ADMIN*: no decay.
- *NONE* (frames not decay corrected): a dynamic series is **decay corrected on request**. The DICOM browser
  warns about it, and loading asks whether to correct the frames to the start of the first frame, like a
  START series (time activity curves and kinetic modelling need it). Frame *f* is multiplied by
  2^((start_f - start_1) / T1/2) x λd / (1 - e^(-λd)): the decay from the first frame start plus the decay
  during the frame (what the DICOM Decay Factor of a decay corrected frame holds). SUV then uses one
  scale, the dose decayed to the first frame start. The factor is stored per frame
  (`DynamicPET.LoaderDecayFactor`, timing column `LoaderDecayFactor`) and the choice in
  `DynamicPET.LoaderDecayCorrection`. When the frames are left as stored, Bq/ml stays uncorrected and SUV
  decays the dose to **each frame's own start**, one scale per frame. The starts are the same as in the
  frame timing table; when a frame's start is unknown, SUV is not offered rather than using another
  frame's time.
- Philips counts images use the Philips SUV and activity scale factors; GML images are already in SUVbw.
- When SUV cannot be computed (no weight, dose or times) only Bq/ml is offered, and the reason is in the
  tooltip.

## What is stored on the volumes

Node attributes, with the prefix `StaticPET.`, `GatedPET.` or `DynamicPET.`. Every volume also carries
`DICOM.Modality` = PT and `DICOM.instanceUIDs`.

| Group | Attributes |
|---|---|
| Units | `Units`, `Scale` (or `per frame`), `DICOMUnits`, `Kind`, `Frames`, `FrameOrdering`, `LoadWarnings` (only when incomplete) |
| Tracer | `Radiopharmaceutical`, `Radionuclide`, `RadionuclideHalfLifeS`, `RadionuclidePositronFraction`, `InjectedDoseMBq`, `InjectionTime`, `InjectionToScanStartS` |
| Times | `StudyStart`, `SeriesStart`, `ScanStart`, `ScanStartSource` (ISO date-times) |
| Decay and corrections | `DecayCorrection` (START / ADMIN / NONE), `DecayCorrectionDateTime`, `DecayReference`, `CorrectedImage` (DECY, ATTN, SCAT, DTIM, RAN, NORM), `AttenuationCorrectionMethod`, `ScatterCorrectionMethod`, `RandomsCorrectionMethod` |
| Scanner and reconstruction | `Manufacturer`, `ScannerModel`, `ReconstructionMethod`, `ConvolutionKernel` |
| SUV inputs | `WeightKg`, `InjectedDoseBq`, `DecayedDoseBq`, `HalfLifeS`, `SUVSource`, `SUVProblems`; also `PatientWeightKg`, `PatientHeightCm`, `PatientSex` |
| Gates | `Gate`, `TriggerTimeMs`, `FrameDurationMs`, and the ECG gating fields (`GatingLowRRMs`, `GatingHighRRMs`, `GatingHeartRateBpm`, beats acquired / rejected, ...) named like the DICOM SPECT Recon gating attributes |
| Time frames | `Frame`, `StartDateTime` (clock), `StartFromInjectionS`, `StartFromStudyStartS`, `StartFromScanStartS`, `DurationS`, `DurationMs`, `MidFromInjectionS`, `MidFromScanStartS`, `EndFromInjectionS`, `FrameReferenceTimeMs`, `DecayFactor`, `StartSource`, `LoaderDecayFactor` |

The **study start** of the frame timing is the start of the first PET frame, so frame 1 starts at 0 from the
study start. The DICOM Study Time is often the registration or the CT, not the PET; it stays in `StudyStart`. The
dynamic sequence carries the first frame's start as `DynamicPET.StudyStartDateTime`.

The dynamic sequence node also carries each timing column as a JSON list (`DynamicPET.FrameDurationS`,
`DynamicPET.FrameMidFromInjectionS`, ...). The table *"<series> frame timing"* holds the same values, one
row per frame, for kinetic modelling or export. A frame starts at the earliest acquisition time of its
slices; when a frame has none, it starts where the previous frame ends (previous start + Actual Frame
Duration), and the first frame at the scan start. `StartSource` says which. Frame Reference Time is not used
for the start, since the standard does not say whether it is the frame's start or middle. The loader warns
about frames whose start was estimated, and about frames that start before the previous one ends.

## Static PET

Static PET loadables get a low confidence (0.6). That is above Slicer's plain scalar volume loader and,
as far as we know, below the PET SUV plugin, so the PET SUV plugin is used for static PET when it is
installed. To use this plugin for a static series anyway, pick its loadable in the DICOM browser's
*Advanced* view.

## Installation

Install *DICOMGatedDynamicPET* from the Extensions Manager (Nuclear Medicine category) once it is published,
or add the `DICOMGatedDynamicPET` folder of this repository to *Edit > Application Settings > Modules >
Additional module paths* and restart Slicer. The plugin works inside the DICOM browser; its defaults are in
the *Gated / Dynamic PET Settings* module.

## Developer notes

```
DICOMGatedDynamicPET/
  DICOMGatedDynamicPET.py          DICOM plugin (examine / load), Slicer self-test on synthetic series
  GatedDynamicPETSettings.py       settings module (visible, Nuclear Medicine)
  DynamicPETEditor.py              Dynamic PET Editor module (visible, Nuclear Medicine)
  DICOMGatedDynamicPETLib/
    settings.py                    the settings, kept in the Slicer application settings
    dynamicviews.py, timeline.py   editor layout helpers and the timeline widget
    petseries.py                   PET DICOM: kind, frame ordering, geometry, SUV, tracer, decay, frame timing,
                                   ECG gating, synthetic test series (pydicom + numpy, no Slicer imports)
Testing/                          tests without Slicer (petseries, editor helpers): python -m pytest Testing
```

In Slicer, the self-test runs from the Python console:
`import DICOMGatedDynamicPET; DICOMGatedDynamicPET.DICOMGatedDynamicPETTest().runTest()`.

## License

MIT, see [LICENSE](LICENSE).
