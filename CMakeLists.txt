cmake_minimum_required(VERSION 3.16)
project(
    IdealNN
        DESCRIPTION  "A educational Neural Network C++ Library that mimics pytorch API"
        HOMEPAGE_URL "github.com/cesare-montresor/IdealNN"
        LANGUAGES    CXX
)

# =============== Options ===============

# Enforce CXX 17 standards
set(CMAKE_CXX_STANDARD          17)
set(CMAKE_CXX_STANDARD_REQUIRED     ON)
# Disable compiler specific extensions.
set(CMAKE_CXX_EXTENSIONS            OFF)


# Produce a static lib
option(STATIC_LIB "Produce a static lib" OFF)
# Compile with coverage support
option(TEST_COVERAGE "Enable test coverage" ON)
# Undefined behavior checking at runtime
option(UB_SANITIZER "Enable undefined behaviour sanitizer" ON)


# set default compiler flags
set(COMPILER_FLAGS
    -Wall -Wextra -Wpedantic -Werror
    $<$<BOOL:${TEST_COVERAGE}>:--coverage>
    $<$<BOOL:${UB_SANITIZER}>:-fsanitize=undefined -fno-omit-frame-pointer>
)

# set default linker flags
set(LINKER_FLAGS
    $<$<BOOL:${TEST_COVERAGE}>:--coverage>
    $<$<BOOL:${UB_SANITIZER}>:-fsanitize=undefined -fno-omit-frame-pointer>
)


## USED FLAGS
# -Wall: This enables all the warnings about constructions that some users consider questionable and that are easy to avoid.
# -Wextra: This enables some extra warning flags that are not enabled by -Wall
# -Wpedantic: Issue all the warnings demanded by strict ISO C and ISO C++
# -Werror: Make all warnings into errors.

## TEST_COVERAGE
# --coverage: This option is used to compile and link code instrumented for coverage analysis. The option is a synonym for -fprofile-arcs -ftest-coverage (when compiling) and -lgcov
# -fno-omit-frame-pointer: Keep the frame pointer in a register for functions that don't need one. (default: -omit-frame-pointer)
## see more: # https://gcc.gnu.org/onlinedocs/gcc/Warning-Options.html

#set(CMAKE_CXX_FLAGS_DEBUG_INIT ${COMPILER_FLAGS})
#set(CMAKE_CXX_FLAGS_RELEASE_INIT "-Wall -Wextra -Wpedantic -Werror")

# List all source files
file(GLOB_RECURSE SOURCES ${PROJECT_SOURCE_DIR}/src/*)
# List all test files
file(GLOB_RECURSE SOURCES_TEST ${PROJECT_SOURCE_DIR}/test/*.cpp)
# List all 3rd party-headers ( catch2 )
set(INCLUDES ${PROJECT_SOURCE_DIR}/third_party )


# =============== Library ===============



# add target library, static
if(STATIC_LIB)
    add_library(idealnn STATIC ${SOURCES} ) #Static
else()
    add_library(idealnn ${SOURCES} )  #Non-static lib
endif()

# setup the /include folder
target_include_directories(idealnn PUBLIC ${PROJECT_SOURCE_DIR}/include) # add .h

# add Eigen3 as requirement
find_package(Eigen3 REQUIRED)
target_link_libraries(idealnn Eigen3::Eigen)

# set compiler and linker flags accordingly
target_compile_options(idealnn PRIVATE ${COMPILER_FLAGS} )
target_link_options(idealnn PRIVATE ${LINKER_FLAGS} )

# ensure presence of bin/ folder for install action
file(MAKE_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
# copy libidealnn.a to main bin/ folder
install(
        FILES       ${CMAKE_BINARY_DIR}/libidealnn.a
        DESTINATION ${PROJECT_SOURCE_DIR}/bin
        OPTIONAL
)

# =============== Testing ===============
#enable testing
enable_testing()
# add executable target for tests
add_executable(idealnn_test EXCLUDE_FROM_ALL ${SOURCES_TEST} third_party/catch2/catch.hpp)
# link test executable with libidealnn
target_link_libraries(idealnn_test idealnn)
# add include directories for src/, tests/ and third_party/
target_include_directories(
        idealnn_test
        PRIVATE src/
        PRIVATE test/
        PRIVATE ${INCLUDES}
)
# set compiler and linker flags for test executable
target_compile_options(idealnn_test PRIVATE ${COMPILER_FLAGS} )
target_link_options(idealnn_test PRIVATE ${LINKER_FLAGS} )


#default test
add_test(NAME test COMMAND ${CMAKE_BINARY_DIR}/idealnn_test )
#valgrind test for memory leaks
add_test(NAME valgrind
    COMMAND valgrind --leak-check=full --show-leak-kinds=all --track-fds=yes --track-origins=yes -s
    ${CMAKE_BINARY_DIR}/idealnn_test
    CONFIGURATIONS valgrind # disabled by default, use ctest -C valgrind
)


# =============== Documentation ===============
find_package(Doxygen)
if (DOXYGEN_FOUND)
    # ensure presence of doc/ folder for install action
    file(MAKE_DIRECTORY ${PROJECT_SOURCE_DIR}/doc)

    #add doc target
    add_custom_target(doc ALL
        COMMAND ${DOXYGEN_EXECUTABLE} ${PROJECT_SOURCE_DIR}/Doxyfile
        COMMENT "Generating API documentation with Doxygen"
    )

    #add install action for doc/ if present
    install(
        DIRECTORY ${CMAKE_BINARY_DIR}/doc
        DESTINATION ${PROJECT_SOURCE_DIR}/
        OPTIONAL
    )

endif()


# ===== Coverage =====
if (TEST_COVERAGE)
    # ensure presence of coverage/ folder for coverage action
    file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/coverage/ )
    # ensure presence of /doc/coverage/ folder for install action
    file(MAKE_DIRECTORY ${PROJECT_SOURCE_DIR}/doc/ )
    file(MAKE_DIRECTORY ${PROJECT_SOURCE_DIR}/doc/coverage/ )

    #add coverage target
    add_custom_target(coverage
        COMMAND ${CMAKE_BINARY_DIR}/idealnn_test || (exit 0)
        COMMAND gcovr
            --root ${PROJECT_SOURCE_DIR}
            --exclude ${PROJECT_SOURCE_DIR}/test/
            --exclude ${PROJECT_SOURCE_DIR}/third_party/
            --exclude ".*.h"
            -v
            --html --html-details coverage/coverage.html
            COMMENT "Generating code coverage"
    )
    add_dependencies(coverage idealnn_test)

    #add install action for doc/coverage/ if present
    install(
            DIRECTORY   ${CMAKE_BINARY_DIR}/coverage
            DESTINATION ${PROJECT_SOURCE_DIR}/doc/
            OPTIONAL
    )
endif()

