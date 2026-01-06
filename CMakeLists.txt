cmake_minimum_required(VERSION 3.16)
project(VolumeRendering VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set the output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/bin)

# Determine architecture
if(CMAKE_SIZEOF_VOID_P EQUAL 8)
    set(ARCH "x64")
else()
    set(ARCH "x86")
endif()

# Find OpenGL
find_package(OpenGL REQUIRED)

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/Source
    ${CMAKE_SOURCE_DIR}/libs
    ${CMAKE_SOURCE_DIR}/libs/GL
    ${CMAKE_SOURCE_DIR}/libs/GLFW
    ${CMAKE_SOURCE_DIR}/libs/glm
    ${CMAKE_SOURCE_DIR}/libs/assimp
    ${CMAKE_SOURCE_DIR}/libs/stb
    ${CMAKE_SOURCE_DIR}/libs/Engine
)

# Collect all source files
file(GLOB_RECURSE SOURCES
    "Source/*.cpp"
    "Source/*.h"
)

# Create executable
add_executable(${PROJECT_NAME} ${SOURCES})

# Link directories based on configuration
target_link_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_SOURCE_DIR}/libs/GL/${ARCH}/Release
    ${CMAKE_SOURCE_DIR}/libs/GLFW/${ARCH}/$<IF:$<CONFIG:Debug>,Debug,Release>
    ${CMAKE_SOURCE_DIR}/libs/assimp/${ARCH}/$<IF:$<CONFIG:Debug>,Debug,Release>
    ${CMAKE_SOURCE_DIR}/libs/Engine/${ARCH}/$<IF:$<CONFIG:Debug>,Debug,Release>
)

# Link libraries (using actual lib names from the libs folder)
target_link_libraries(${PROJECT_NAME} PRIVATE
    OpenGL::GL
    glew32
    glfw3dll
    assimp
    Components
)

# Add preprocessor definitions
target_compile_definitions(${PROJECT_NAME} PRIVATE
    $<$<CONFIG:Debug>:_DEBUG>
)

# Copy DLLs to output directory after build
# Note: assimp debug DLL is named assimpd.dll but release is assimp.dll
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${CMAKE_SOURCE_DIR}/libs/GL/${ARCH}/Release/glew32.dll"
    $<TARGET_FILE_DIR:${PROJECT_NAME}>
)

add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${CMAKE_SOURCE_DIR}/libs/GLFW/${ARCH}/$<IF:$<CONFIG:Debug>,Debug,Release>/glfw3.dll"
    $<TARGET_FILE_DIR:${PROJECT_NAME}>
)

# For assimp, Debug uses assimpd.dll, Release uses assimp.dll
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${CMAKE_SOURCE_DIR}/libs/assimp/${ARCH}/$<IF:$<CONFIG:Debug>,Debug,Release>/$<IF:$<CONFIG:Debug>,assimpd.dll,assimp.dll>"
    $<TARGET_FILE_DIR:${PROJECT_NAME}>
)

add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${CMAKE_SOURCE_DIR}/libs/Engine/${ARCH}/$<IF:$<CONFIG:Debug>,Debug,Release>/Components.dll"
    $<TARGET_FILE_DIR:${PROJECT_NAME}>
)

# Copy Resources folder to output directory
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_SOURCE_DIR}/Resources
    $<TARGET_FILE_DIR:${PROJECT_NAME}>/Resources
)

message(STATUS "Architecture: ${ARCH}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
