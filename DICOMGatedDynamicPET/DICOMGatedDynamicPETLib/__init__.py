"""Helper package of the DICOM Gated / Dynamic PET plugin (DICOMGatedDynamicPET).

    petseries   PET DICOM series: kind (static / gated / dynamic), frame and slice ordering, geometry, SUV,
                tracer, decay correction, frame timing, ECG gating (pydicom and numpy only, no Slicer imports)
    dynamicviews    Dynamic PET Editor helpers: view grid, layout XML, frame intervals, time labels (no Slicer)
    timeline        Dynamic PET Editor timeline widget (Qt)
    colortables     color tables shared with the Tarvos (MPI) extension (numpy only)
    colormaps       editor color maps as Slicer color nodes ("MPI <label>", shared with Tarvos)
    settings    user settings (SUV method, preferred units, summed volumes) in the Slicer application settings
"""
