cmake_minimum_required(VERSION 3.16)
project(
    idealNN
        DESCRIPTION  "A ideal Neural Network C++ Library"
        HOMEPAGE_URL "github.com/cesare-montresor/IdealNN"
        LANGUAGES    CXX
)

set(CMAKE_CXX_STANDARD_REQUIRED     ON)
set(CMAKE_CXX_EXTENSIONS            OFF)

file(GLOB SOURCES_TEST ${PROJECT_SOURCE_DIR}/test/*.cpp)

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
)
#    ${PROJECT_SOURCE_DIR}/test



add_library(
    idealNN
        src/Layers/Layer.cpp
        src/Layers/Layer.h
        src/Layers/Sequential.cpp
        src/Layers/Sequential.h
        src/Layers/Dense.cpp src/Layers/Dense.h src/Tensor.cpp src/Tensor.h)


find_package (Eigen3 REQUIRED)
target_link_libraries(idealNN Eigen3::Eigen)


# ==== Testing ====

add_executable(
    idealNN_test
        EXCLUDE_FROM_ALL
        third_party/catch2/catch.hpp
        test/base_test.cpp
)
target_link_libraries(idealNN_test idealNN)

target_sources(idealNN_test PRIVATE ${SOURCES_TEST})
target_include_directories(
    idealNN_test
        PRIVATE src/
        PRIVATE test/
        PRIVATE ${INCLUDES}
)

enable_testing()

add_test(NAME test
        COMMAND ${CMAKE_BINARY_DIR}/idealNN_test )

add_test(NAME valgrind
        COMMAND valgrind --leak-check=full --show-leak-kinds=all --track-fds=yes --track-origins=yes
        ${CMAKE_BINARY_DIR}/idealNN_test)

# ===== Documentation =====
find_package(Doxygen)
if (DOXYGEN_FOUND)
    add_custom_target(doc ALL
        COMMAND ${DOXYGEN_EXECUTABLE} ${PROJECT_SOURCE_DIR}/Doxyfile
        COMMENT "Generating API documentation with Doxygen"
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