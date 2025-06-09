cmake_minimum_required(VERSION 3.10)
project(Volume_Rendering)

set(CMAKE_CXX_STANDARD 17)

# Include directories (corect și complet)
include_directories(
        libs
        Source
        Resources
)

# Sursele (toate cpp-urile)
file(GLOB_RECURSE SOURCES
        Source/*.cpp
)

# Adăugăm executabilul
add_executable(Volume_Rendering ${SOURCES})

# Linkare biblioteci precompilate
target_link_directories(Volume_Rendering PRIVATE libs/GLFW/x86/Debug)

target_link_libraries(Volume_Rendering
        glfw3.lib
        opengl32.lib
)

# Dacă ai și o versiune precompilată de Assimp (ex: assimp.lib), adaugă:
# target_link_directories(Volume_Rendering PRIVATE libs/assimp/lib)
# target_link_libraries(Volume_Rendering assimp.lib)
