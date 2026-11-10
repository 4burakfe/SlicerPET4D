cmake_minimum_required(VERSION 3.16.3...3.19.7 FATAL_ERROR)

project(DICOMGatedDynamicPET)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://github.com/4burakfe/SlicerDICOMGatedDynamicPET")
set(EXTENSION_CATEGORY "Nuclear Medicine")
set(EXTENSION_CONTRIBUTORS "Burak Demir")
set(EXTENSION_DESCRIPTION "DICOM plugin for ECG gated and dynamic PET: gates and time frames loaded as sequences in SUVbw or Bq/ml (QIBA SUV, per image rescale), the mean of the gates as the static image, frame timing from injection, study and scan start, tracer and decay correction stored on the volumes. Static PET is supported too. Includes Dynamic PET Review/Edit, frame-by-frame Dynamic PET Motion Correction, Dynamic PET to CT/MR Registration (BRAINSFit) and Dynamic PET Modeling (1TC, 2TC, Patlak, Logan, SRTM, SUVR on segments and voxel-wise). The brain PET / SPECT workflows (Neuro) are in the SlicerNeuroPET extension (Mergen), built on this one.")
set(EXTENSION_ICONURL "https://raw.githubusercontent.com/4burakfe/SlicerDICOMGatedDynamicPET/main/DICOMGatedDynamicPET/Resources/Icons/DICOMGatedDynamicPET.png")
set(EXTENSION_SCREENSHOTURLS "")
set(EXTENSION_DEPENDS "NA") # Specified as a list or "NA" if no dependencies

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(DICOMGatedDynamicPET)

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
