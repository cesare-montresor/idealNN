cmake_minimum_required(VERSION 3.16)
project(
    idealNN
        DESCRIPTION  "A ideal Neural Network C++ Library that mimics pytorch API"
        HOMEPAGE_URL "github.com/cesare-montresor/IdealNN"
        LANGUAGES    CXX
)

set(CMAKE_CXX_STANDARD          17)
set(CMAKE_CXX_STANDARD_REQUIRED     ON)
set(CMAKE_CXX_EXTENSIONS            OFF)

file(GLOB_RECURSE SOURCES ${PROJECT_SOURCE_DIR}/src/*)
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




add_library(libidealNN ${SOURCES} )
target_include_directories(libidealNN PUBLIC ${PROJECT_SOURCE_DIR}/src)


find_package (Eigen3 REQUIRED)
target_link_libraries(libidealNN Eigen3::Eigen)


target_compile_options(libidealNN PRIVATE ${COMPILER_FLAGS} )
target_link_options(libidealNN PRIVATE ${LINKER_FLAGS} )

# ==== Testing ====

add_executable(libidealNN_test EXCLUDE_FROM_ALL ${SOURCES_TEST} third_party/catch2/catch.hpp)
target_link_libraries(libidealNN_test libidealNN)


target_include_directories(
    libidealNN_test
        PRIVATE src/
        PRIVATE test/
        PRIVATE ${INCLUDES}
)

enable_testing()

add_test(NAME test
        COMMAND ${CMAKE_BINARY_DIR}/libidealNN_test )

add_test(NAME valgrind
        COMMAND valgrind --leak-check=full --show-leak-kinds=all --track-fds=yes --track-origins=yes
        ${CMAKE_BINARY_DIR}/libidealNN_test)

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
