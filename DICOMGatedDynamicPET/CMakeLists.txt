#-----------------------------------------------------------------------------
# DICOM plugin for gated, dynamic and static PET (hidden module, used by the DICOM browser)
set(MODULE_NAME DICOMGatedDynamicPET)

set(MODULE_PYTHON_SCRIPTS
  ${MODULE_NAME}.py
  ${MODULE_NAME}Lib/__init__.py
  ${MODULE_NAME}Lib/petseries.py
  ${MODULE_NAME}Lib/settings.py
  ${MODULE_NAME}Lib/dynamicviews.py
  ${MODULE_NAME}Lib/timeline.py
  ${MODULE_NAME}Lib/roicurves.py
  ${MODULE_NAME}Lib/frameedit.py
  ${MODULE_NAME}Lib/dicomexport.py
  ${MODULE_NAME}Lib/motioncorrection.py
  ${MODULE_NAME}Lib/motionplot.py
  ${MODULE_NAME}Lib/colortables.py
  ${MODULE_NAME}Lib/colormaps.py
  ${MODULE_NAME}Lib/registration.py
  ${MODULE_NAME}Lib/sitkreg.py
  ${MODULE_NAME}Lib/resampling.py
  ${MODULE_NAME}Lib/hardendialog.py
  ${MODULE_NAME}Lib/kinetics.py
  ${MODULE_NAME}Lib/tacplot.py
  ${MODULE_NAME}Lib/neuro.py
  ${MODULE_NAME}Lib/normaldb.py
  )

set(MODULE_PYTHON_RESOURCES
  Resources/Icons/${MODULE_NAME}.png
  )

slicerMacroBuildScriptedModule(
  NAME ${MODULE_NAME}
  SCRIPTS ${MODULE_PYTHON_SCRIPTS}
  RESOURCES ${MODULE_PYTHON_RESOURCES}
  WITH_GENERIC_TESTS
  )

#-----------------------------------------------------------------------------
if(BUILD_TESTING)
  # Register the unittest subclass in the module script as a ctest (also available at runtime)
  slicer_add_python_unittest(SCRIPT ${MODULE_NAME}.py)
endif()

#-----------------------------------------------------------------------------
# Settings module (visible, Nuclear Medicine): defaults of the plugin
set(SETTINGS_MODULE_NAME GatedDynamicPETSettings)

slicerMacroBuildScriptedModule(
  NAME ${SETTINGS_MODULE_NAME}
  SCRIPTS ${SETTINGS_MODULE_NAME}.py
  RESOURCES Resources/Icons/${SETTINGS_MODULE_NAME}.png
  WITH_GENERIC_TESTS
  )

if(BUILD_TESTING)
  slicer_add_python_unittest(SCRIPT ${SETTINGS_MODULE_NAME}.py)
endif()

#-----------------------------------------------------------------------------
# Dynamic PET Review/Edit (visible, Nuclear Medicine, module name DynamicPETEditor): frame-by-frame review layout with a timeline
set(EDITOR_MODULE_NAME DynamicPETEditor)

slicerMacroBuildScriptedModule(
  NAME ${EDITOR_MODULE_NAME}
  SCRIPTS ${EDITOR_MODULE_NAME}.py
  RESOURCES Resources/Icons/${EDITOR_MODULE_NAME}.png
  WITH_GENERIC_TESTS
  )

if(BUILD_TESTING)
  slicer_add_python_unittest(SCRIPT ${EDITOR_MODULE_NAME}.py)
endif()

#-----------------------------------------------------------------------------
# Dynamic PET Motion Correction (visible, Nuclear Medicine): frame-by-frame registration with BRAINSFit
set(MC_MODULE_NAME DynamicPETMotionCorrection)

slicerMacroBuildScriptedModule(
  NAME ${MC_MODULE_NAME}
  SCRIPTS ${MC_MODULE_NAME}.py
  RESOURCES
    Resources/Icons/${MC_MODULE_NAME}.png
    Resources/Images/MotionShift.png
    Resources/Images/MotionAngles.png
  WITH_GENERIC_TESTS
  )

if(BUILD_TESTING)
  slicer_add_python_unittest(SCRIPT ${MC_MODULE_NAME}.py)
endif()

#-----------------------------------------------------------------------------
# Dynamic PET to CT/MR Registration (visible, Nuclear Medicine): the series registered to a CT or MR with BRAINSFit
set(REG_MODULE_NAME DynamicPETRegistration)

slicerMacroBuildScriptedModule(
  NAME ${REG_MODULE_NAME}
  SCRIPTS ${REG_MODULE_NAME}.py
  RESOURCES Resources/Icons/${REG_MODULE_NAME}.png
  WITH_GENERIC_TESTS
  )

if(BUILD_TESTING)
  slicer_add_python_unittest(SCRIPT ${REG_MODULE_NAME}.py)
endif()

#-----------------------------------------------------------------------------
# Dynamic PET Modeling (visible, Nuclear Medicine): kinetic models on the segments of a segmentation
set(KM_MODULE_NAME DynamicPETModeling)

slicerMacroBuildScriptedModule(
  NAME ${KM_MODULE_NAME}
  SCRIPTS ${KM_MODULE_NAME}.py
  RESOURCES Resources/Icons/${KM_MODULE_NAME}.png
  WITH_GENERIC_TESTS
  )

if(BUILD_TESTING)
  slicer_add_python_unittest(SCRIPT ${KM_MODULE_NAME}.py)
endif()

#-----------------------------------------------------------------------------
# Dynamic Neuro Receptor PET (visible, Nuclear Medicine): workflow app (data, motion correction, CT/MR registration,
# normalization to MNI ICBM 152 2009c, CerebrA regions, modeling)
set(NEURO_MODULE_NAME DynamicNeuroPET)

slicerMacroBuildScriptedModule(
  NAME ${NEURO_MODULE_NAME}
  SCRIPTS ${NEURO_MODULE_NAME}.py
  RESOURCES
    Resources/Icons/${NEURO_MODULE_NAME}.png
    Resources/Atlas/CerebrA_groups.csv
  WITH_GENERIC_TESTS
  )

if(BUILD_TESTING)
  slicer_add_python_unittest(SCRIPT ${NEURO_MODULE_NAME}.py)
endif()

#-----------------------------------------------------------------------------
# Neuro PET Database Manager (visible, Nuclear Medicine): normal databases of brain PET (static first; predefined
# from mean / SD maps or a package, user defined filled from the Neuro workflows)
set(NDB_MODULE_NAME NeuroPETDatabaseManager)

slicerMacroBuildScriptedModule(
  NAME ${NDB_MODULE_NAME}
  SCRIPTS ${NDB_MODULE_NAME}.py
  RESOURCES Resources/Icons/${NDB_MODULE_NAME}.png
  WITH_GENERIC_TESTS
  )

if(BUILD_TESTING)
  slicer_add_python_unittest(SCRIPT ${NDB_MODULE_NAME}.py)
endif()
