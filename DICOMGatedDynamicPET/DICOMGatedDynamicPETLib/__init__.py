"""Helper package of the DICOM Gated / Dynamic PET plugin (DICOMGatedDynamicPET).

    petseries   PET DICOM series: kind (static / gated / dynamic), frame and slice ordering, geometry, SUV,
                tracer, decay correction, frame timing, ECG gating (pydicom and numpy only, no Slicer imports)
    dynamicviews    Review/Edit helpers: view grid, layout XML, frame intervals, time labels (no Slicer)
    timeline        Review/Edit timeline widget (Qt)
    roicurves       Review/Edit spherical ROIs: sphere voxels, time-activity curves, plot axis (numpy only)
    frameedit       Review/Edit frame editing: merge, split, remove, add a series (comparison, decay alignment),
                timing of the saved series (numpy only)
    motioncorrection    Motion Correction helpers: low activity frames, rigid shift / angles, motion limit, timeline
                texts (numpy only)
    motionplot      Motion Correction plot of the rigid shift / angles over time (Qt)
    colortables     color tables shared with the Tarvos (MPI) extension (numpy only)
    colormaps       Review/Edit color maps as Slicer color nodes ("MPI <label>", shared with Tarvos)
    settings    user settings (SUV method, preferred units, summed volumes) in the Slicer application settings
"""
