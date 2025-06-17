cmake_minimum_required(VERSION 3.16.3...3.19.7 FATAL_ERROR)

project(StrokeDetection)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://www.slicer.org/wiki/Documentation/Nightly/Extensions/StrokeDetection")
set(EXTENSION_CONTRIBUTORS "Metin Ilgar Mutlu")
set(EXTENSION_DESCRIPTION "This extension performs stroke detection and segmentation using deep learning models. ")
set(EXTENSION_ICONURL "https://www.example.com/Slicer/Extensions/StrokeDetection.png")
set(EXTENSION_SCREENSHOTURLS "https://www.example.com/Slicer/Extensions/StrokeDetection/Screenshots/1.png")
set(EXTENSION_DEPENDS "NA") # Specified as a list or "NA" if no dependencies

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(StrokeDetectionModule)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
