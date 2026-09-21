cmake_minimum_required(VERSION 3.15)
project(blis_ukr_bench CXX)

# Require C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# --- Configuration Options ---
option(USE_PERF_COUNTERS "Compile and link performance_counters_<backend>.cpp" ON)

set(STRIDE "ROWSTRIDE" CACHE STRING "Memory stride definition")
set_property(CACHE STRIDE PROPERTY STRINGS "ROWSTRIDE" "COLSTRIDE")

set(BLIS_ROOT "" CACHE PATH "Path to custom BLIS installation root")
set(OUTPUT_NAME "bench" CACHE STRING "Name of the compiled output executable")

# --- Source Files ---
set(SOURCES bench.cpp)
if(USE_PERF_COUNTERS)
    set(PC_BACKEND "perf" CACHE STRING "Performance counters backend")
    set_property(CACHE PC_BACKEND PROPERTY STRINGS "perf" "papi" "m1" "simple")
    list(APPEND SOURCES "performance_counters/performance_counters_${PC_BACKEND}.cpp")

    if(PC_BACKEND STREQUAL "papi")
        find_package(PkgConfig)
        pkg_check_modules(PAPI REQUIRED papi)
        set(EXTRA_CFLAGS ${PAPI_CFLAGS_OTHER})
        set(EXTRA_INCLUDES ${PAPI_INCLUDE_DIRS})
        set(EXTRA_LIBS ${PAPI_LIBRARIES})
    endif()
endif()

# --- Executable Target ---
add_executable(blis_ukr_bench ${SOURCES})
set_target_properties(blis_ukr_bench PROPERTIES OUTPUT_NAME ${OUTPUT_NAME})

# Apply Stride Definition
target_compile_definitions(blis_ukr_bench PRIVATE ${STRIDE})

# --- BLIS Auto-Discovery & Linking ---

# 1. Dynamically build search suffixes for uninstalled BLIS build directories
set(BLIS_INC_SUFFIXES "include" "include/blis")
set(BLIS_LIB_SUFFIXES "lib" "lib64")

if(BLIS_ROOT)
    # Autodetect include/<config_name>
    file(GLOB INC_DIRS RELATIVE "${BLIS_ROOT}/include" "${BLIS_ROOT}/include/*")
    foreach(SUBDIR IN LISTS INC_DIRS)
        if(IS_DIRECTORY "${BLIS_ROOT}/include/${SUBDIR}")
            list(APPEND BLIS_INC_SUFFIXES "include/${SUBDIR}")
        endif()
    endforeach()
    
    # Autodetect lib/<config_name>
    file(GLOB LIB_DIRS RELATIVE "${BLIS_ROOT}/lib" "${BLIS_ROOT}/lib/*")
    foreach(SUBDIR IN LISTS LIB_DIRS)
        if(IS_DIRECTORY "${BLIS_ROOT}/lib/${SUBDIR}")
            list(APPEND BLIS_LIB_SUFFIXES "lib/${SUBDIR}")
        endif()
    endforeach()
endif()

# 2. Find headers
find_path(BLIS_INCLUDE_DIR
    NAMES blis.h
    HINTS "${BLIS_ROOT}"
    PATH_SUFFIXES ${BLIS_INC_SUFFIXES}
    REQUIRED
)

# 3. Find library (checks for both standard and multithreaded AOCL names)
find_library(BLIS_LIBRARY
    NAMES blis blis-mt
    HINTS "${BLIS_ROOT}"
    PATH_SUFFIXES ${BLIS_LIB_SUFFIXES}
    REQUIRED
)

target_include_directories(blis_ukr_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/archcomp/include ${BLIS_INCLUDE_DIR} ${EXTRA_INCLUDES})
target_link_libraries(blis_ukr_bench PRIVATE ${BLIS_LIBRARY} ${EXTRA_LIBS})

if(EXTRA_CFLAGS)
    target_compile_options(blis_ukr_bench PRIVATE ${EXTRA_CFLAGS})
endif()

# 4. Auto-detect if the found BLIS is AMD AOCL
file(STRINGS "${BLIS_INCLUDE_DIR}/blis.h" AOCL_MATCH REGEX "AOCL_BLIS_ZEN")
if(AOCL_MATCH)
    message(STATUS "Autodetected AOCL BLIS in ${BLIS_INCLUDE_DIR}")
    target_compile_definitions(blis_ukr_bench PRIVATE AOCL_BLIS)
else()
    message(STATUS "Autodetected standard BLIS in ${BLIS_INCLUDE_DIR}")
endif()

# --- OpenMP ---
find_package(OpenMP REQUIRED)
target_link_libraries(blis_ukr_bench PRIVATE OpenMP::OpenMP_CXX)
