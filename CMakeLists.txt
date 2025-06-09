cmake_minimum_required(VERSION 3.10)
project(Volume_Rendering)

set(CMAKE_CXX_STANDARD 17)

include_directories(
        libs
        libs/Engine
        Source
        Resources
)

file(GLOB_RECURSE SOURCES
        Source/*.cpp
)

add_executable(Volume_Rendering ${SOURCES})

target_link_directories(Volume_Rendering PRIVATE
        libs/GLFW/x86/Debug
        libs/GL/x86/Release
        libs/Engine/x86/Debug
)

target_link_libraries(Volume_Rendering
        glfw3dll.lib
        glew32.lib
        Components.lib
        opengl32.lib
)