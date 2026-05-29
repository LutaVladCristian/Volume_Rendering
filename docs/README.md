# Volume Rendering Documentation

This documentation describes the implementation in this repository, a Windows
C++/OpenGL visualization application containing two volume rendering scenes:

- **Ray Casting**: direct volume rendering on the GPU. This is the scene
  selected by `Source/Main.cpp` and therefore the default executable behavior.
- **Marching Cubes**: CPU isosurface extraction followed by standard mesh
  rendering. The scene is compiled into the application but must be selected
  in `Source/Main.cpp` to run.

## Documentation Map

| Document | Purpose |
| --- | --- |
| [BUILD_AND_USAGE.md](BUILD_AND_USAGE.md) | Requirements, build/run workflow, controls, and scene selection |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Application structure, engine loop, input system, and rendering support classes |
| [RENDERING_TECHNIQUES.md](RENDERING_TECHNIQUES.md) | Ray-casting and Marching Cubes algorithm/data-flow walkthroughs |
| [RESOURCES_AND_DATA.md](RESOURCES_AND_DATA.md) | Shaders, textures, models, RAW volumes, and transfer-function inputs |
| [KNOWN_LIMITATIONS.md](KNOWN_LIMITATIONS.md) | Limitations addressed in this revision and remaining risks |

## Scope

The review covers project-authored C++ and GLSL under `Source/`, shared GLSL
under `Resources/Shaders/`, build configuration, scripts, and resource
metadata. The `libs/` directory contains vendored or precompiled dependencies;
its public integration points are described, but third-party library internals
are not project architecture.

## At A Glance

The program starts a 1280 by 720 GLFW window, initializes OpenGL through GLEW,
loads common textures, creates a scene, and runs a per-frame update loop. A
`SimpleScene` supplies a camera, input handlers, common shaders, optional
coordinate-plane rendering, and mesh helpers.

By default, the `RayCasting` scene loads `Resources/Volumes/head256.raw` as an
8-bit three-dimensional texture and `Resources/Volumes/tff.dat` as a
256-entry RGBA transfer-function texture. Rendering uses a cube proxy: one
pass captures intended exit coordinates in an off-screen framebuffer and a
second pass samples through the volume from entry to exit.

The alternative `MarchingCubes` scene defaults to `Bucky.raw`, visits each voxel
cell, evaluates the standard lookup tables at an isovalue, uploads the
generated triangles, and renders normal-derived color.

Either scene and alternate RAW dimensions can be selected from command-line
arguments; see [BUILD_AND_USAGE.md](BUILD_AND_USAGE.md).
