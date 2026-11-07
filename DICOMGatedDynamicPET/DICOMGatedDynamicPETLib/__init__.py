"""Helper package of the DICOM Gated / Dynamic PET plugin (DICOMGatedDynamicPET).

    petseries   PET DICOM series: kind (static / gated / dynamic), frame and slice ordering, geometry, SUV,
                tracer, decay correction, frame timing, ECG gating (pydicom and numpy only, no Slicer imports)
    dynamicviews    Review/Edit helpers: view grid, layout XML, frame intervals, time labels (no Slicer)
    timeline        Review/Edit timeline widget (Qt)
    roicurves       Review/Edit spherical ROIs: sphere voxels, time-activity curves, plot axis (numpy only)
    frameedit       Review/Edit frame editing: merge, split, remove, add a series (comparison, decay alignment),
                timing of the saved series, undo / redo (numpy only)
    dicomexport     DICOM export of a dynamic series as a new series in the original study (pydicom + numpy)
    motioncorrection    Motion Correction helpers: low activity frames, rigid shift / angles, motion limit, timeline
                texts (numpy only)
    motionplot      Motion Correction plot of the rigid shift / angles over time (Qt)
    registration    Dynamic PET to CT/MR Registration helpers: steps, transform summary, checkerboard, Harden grid,
                layout (numpy only)
    sitkreg     iterative multi-resolution registration (rigid / affine, then B-spline passes), from EasyReg
                (numpy + scipy + SimpleITK)
    resampling  voxels resampled onto another grid through a (B-spline) transform (VTK + numpy)
    kinetics    Dynamic PET Modeling: kinetic models (1TC, 2TC, Patlak, Logan, SRTM, SUVR), fits, metrics, t*,
                voxel-wise maps, CSV (numpy only; usable as a library by other modules)
    tacplot     Dynamic PET Modeling plot of the time-activity curves and the Patlak / Logan plot (Qt)
    neuro       Dynamic Neuro Receptor PET helpers: MNI ICBM 152 2009c atlas files, MR contrast (T1 / T2 / PD),
                CerebrA labels (left / right and bilateral) (numpy only)
    normaldb    Neuro PET Database Manager: normal database folders (database.json, subjects, mean / SD maps), age
                groups, predefined databases and their .zip packages (numpy only)
    hardendialog    Harden dialog (grid, voxel size, resample method, overwrite / save separately) (Qt)
    colortables     color tables shared with the Tarvos (MPI) extension (numpy only)
    colormaps       Review/Edit color maps as Slicer color nodes ("MPI <label>", shared with Tarvos)
    settings    user settings (SUV method, preferred units, summed volumes) in the Slicer application settings
"""
