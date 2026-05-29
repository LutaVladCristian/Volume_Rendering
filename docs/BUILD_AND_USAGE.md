# Build And Usage

## Platform And Requirements

The checked-in build system targets Windows and Visual Studio:

- CMake 3.16 or later.
- Visual Studio 2019 or 2022 with the C++ desktop toolchain.
- An OpenGL 4.3-capable driver. Window creation requests an OpenGL 4.3
  core-profile context and the technique shaders declare GLSL 4.30.
- A 64-bit build is the documented and scripted configuration.

Dependencies are bundled in `libs/`:

| Dependency | Use |
| --- | --- |
| GLFW | Window and input/event handling |
| GLEW | OpenGL function loading |
| GLM | Vector, matrix, and quaternion mathematics |
| Assimp | Loading OBJ/model meshes used by shared scene helpers |
| stb_image / stb_image_write | Image texture loading and PNG writing |
| Components library | Precompiled `Camera` and `Transform` implementations |

## Build And Run

The simplest path is the root batch script:

```batch
.\build_and_run.bat
```

For a debug build:

```batch
.\build_and_run.bat debug
```

The script creates `build/`, configures a Visual Studio x64 project, builds
`VolumeRendering.exe`, and runs it from the repository root.

Equivalent commands for a Release build are:

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
.\build\bin\VolumeRendering.exe
```

Run the executable with the repository root as the working directory.
Resource paths and scene-specific shader paths are relative strings beneath
`Resources/`, such as `Resources/Volumes/` and `Resources/Shaders/RayCasting/`.

## Build Output

CMake builds the executable in `build/bin/` and copies:

- `glew32.dll`
- `glfw3.dll`
- `assimp.dll` for Release or `assimpd.dll` for Debug
- `Components.dll`
- the complete `Resources/` directory

Technique-specific shaders are stored under `Resources/Shaders/`, so they are
included by the existing CMake resource-copy step.

## Troubleshooting

The repository currently contains an ignored/generated `build/` directory
whose `CMakeCache.txt` was created from a different absolute checkout path.
Attempting to reuse that cached build from this workspace causes CMake to
reject it as relocated. If this occurs, configure a clean build directory
(or remove and regenerate the existing generated `build/` directory) before
building.

## Selecting A Scene And Dataset

Running without arguments starts ray casting with `head256.raw`:

```powershell
.\build\bin\VolumeRendering.exe
```

The two command-line forms are:

```powershell
.\build\bin\VolumeRendering.exe --ray-casting [raw width height depth [transfer-function]]
.\build\bin\VolumeRendering.exe --marching-cubes [raw width height depth [isovalue]]
```

Examples:

```powershell
.\build\bin\VolumeRendering.exe --marching-cubes
.\build\bin\VolumeRendering.exe --ray-casting Resources\Volumes\brain.raw 200 160 160
.\build\bin\VolumeRendering.exe --marching-cubes Resources\Volumes\Engine.raw 256 256 256 80
```

RAW file byte counts are validated against the supplied dimensions before
rendering. Ray casting also validates that its transfer-function input
contains the required 1024 RGBA bytes.

## Controls

These controls come from `CameraInput` and `SceneInput` and apply to both
scenes unless a scene overrides behavior:

| Input | Behavior |
| --- | --- |
| Hold right mouse button and move mouse | Look around; mouse pointer is captured while held |
| Hold right mouse button + `W` / `S` | Move camera forward / backward |
| Hold right mouse button + `A` / `D` | Move camera left / right |
| Hold right mouse button + `Q` / `E` | Move camera down / up |
| Hold `Shift` during right-button navigation | Double translation delta |
| Hold right mouse button + numpad `*` / `/` | Increase / decrease camera speed |
| Hold right mouse button + numpad `4`, `6`, `8`, `5` | Rotate camera by keyboard |
| `C` without modifiers | Print camera information |
| `F3` | Toggle the helper ground plane state |
| `F5` | Recompile/relink shaders registered in the current scene |
| `Escape` | Close the window |
| `[` / `]` in Ray Casting | Decrease / increase the volume ray step size |

The `MarchingCubes` scene additionally uses:

| Input | Behavior |
| --- | --- |
| `Z` | Increase isovalue by 5 and rebuild the surface |
| `X` | Decrease isovalue by 5 and rebuild the surface |
