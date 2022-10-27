cmake_minimum_required(VERSION 3.23)
project(
    idealNN
#        CUDA
        DESCRIPTION  "A ideal Neural Network C++ Library"
        HOMEPAGE_URL "github.com/cesare-montresor/IdealNN"
        LANGUAGES    CXX
)

set(CMAKE_CUDA_STANDARD             17)
set(CMAKE_CXX_STANDARD_REQUIRED     ON)
set(CMAKE_CXX_EXTENSIONS            OFF)

# Compile with coverage support
option(TEST_COVERAGE "Enable test coverage" ON)

set(COMPILER_FLAGS
    -Wall -Wextra -Wpedantic -Werror
    $<$<BOOL:${TEST_COVERAGE}>:--coverage>
)

set(LINKER_FLAGS
    $<$<BOOL:${TEST_COVERAGE}>:--coverage>
)

set(INCLUDES
    ${PROJECT_SOURCE_DIR}/third_party
    ${PROJECT_SOURCE_DIR}/test
)

add_executable(
    idealNN
        src/main.cpp
        src/layers/layer.cpp
        src/layers/layer.hpp
        src/sequential.cpp
        src/sequential.hpp
)


set_target_properties(
    idealNN
        PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
)


# ===== Documentation =====
find_package(Doxygen)
if (DOXYGEN_FOUND)
    add_custom_target(doc ALL
        COMMAND ${DOXYGEN_EXECUTABLE} ${PROJECT_SOURCE_DIR}/Doxyfile
        COMMENT "Generating API documentation with Doxygen"
    )
    install(
        FILES       ${CMAKE_BINARY_DIR}/doc
        DESTINATION ${PROJECT_SOURCE_DIR}
        # Skip if documentation hasn't been generated
        OPTIONAL
    )
endif()


# ===== Coverage =====
if (COVERAGE)
    add_custom_target(coverage
        COMMAND gcovr
        --root ${PROJECT_SOURCE_DIR}
        -v
        --html --html-details coverage.html
        COMMENT "Generating code coverage"
    )
endif()