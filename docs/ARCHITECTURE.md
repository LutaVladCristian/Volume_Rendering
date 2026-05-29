# Architecture

## Source Layout

```text
Source/
|-- Main.cpp                         Program entry point and scene selection
|-- include/                         Shared OpenGL, GLM, math, utility wrappers
|-- Core/
|   |-- Engine.*                     GLFW/GLEW initialization and shutdown
|   |-- World.*                      Application loop and scene lifecycle
|   |-- Window/                      Window, callbacks, observer-based input
|   |-- GPU/                         Mesh, shader, texture, framebuffer, buffers
|   `-- Managers/                    Resource prefixes and texture cache
|-- Component/
|   |-- SimpleScene.*                Camera/common rendering scene base
|   |-- CameraInput.*                Camera navigation bindings
|   `-- SceneInput.*                 Common scene hotkeys
`-- Laboratoare/
    |-- RayCasting/                  Default direct volume-rendering scene
    `-- MarchingCubes/               Optional isosurface-extraction scene
```

`libs/Engine/Component/` exposes headers for `Camera` and `Transform`, whose
implementations are linked through the bundled `Components` library.

## Startup And Lifetime

`main()` performs the following steps:

1. Sets the random seed and creates `WindowProperties` at 1280 by 720.
2. Calls `Engine::Init`, which initializes GLFW, constructs `WindowObject`,
   initializes GLEW, and loads default textures through `TextureManager`.
3. Parses optional scene/data arguments and constructs `RayCasting` by default
   or `MarchingCubes` when requested.
4. Calls `Init()`, then enters `Run()`.
5. Destroys the active world and its GL resources before `Engine::Exit()`
   clears cached textures, destroys the window, and terminates GLFW.

## Main Loop

`World` is both the scene lifecycle base class and an `InputController`. Its
loop repeats until GLFW reports that the window should close:

```text
PollEvents
  -> ComputeFrameDeltaTime
  -> Dispatch queued window/input events
  -> FrameStart
  -> Update(deltaTime)
  -> FrameEnd
  -> SwapBuffers
```

Derived scenes override `Init`, `FrameStart`, `Update`, and `FrameEnd`.
When paused, input events continue to be dispatched but scene rendering
updates are skipped.

## Window And Input

`WindowObject` wraps GLFW context creation and maintains buffered input state:

- Context request: OpenGL 4.3 core profile.
- Default window properties: visible, resizable, centered, VSync enabled.
- Key and mouse states support both events and per-frame held-button queries.

`WindowCallbacks` forwards GLFW callbacks to the singleton-like window stored
by `Engine`. `InputController` instances subscribe to that window on
construction. Every frame, observers receive resize, movement, button, scroll,
key, and continuous-update events in sequence.

The common observers created by `SimpleScene` are:

- `CameraInput`, which implements first-person camera interaction.
- `SceneInput`, which implements shader reload, ground toggle, and exit keys.

Each scene is also an `InputController`, so a specific technique can add its
own event behavior.

## SimpleScene

`SimpleScene` is the reusable base for both volume techniques. Its constructor:

- Creates and configures a perspective camera and transform.
- Loads a `plane50.obj` ground-plane helper mesh and creates colored axis
  lines.
- Creates four shared shader programs: `Simple`, `Color`, `VertexNormal`, and
  `VertexColor`.
- Enables depth testing.

It exposes render helpers that provide `Model`, `View`, and `Projection`
uniforms, plus screen clearing, coordinate rendering, and shader reloading.
The current two volume scenes use the camera/render helpers, while coordinate
plane drawing is not called in their active frame paths.

## GPU Support Layer

| Type | Responsibility |
| --- | --- |
| `Shader` | Loads GLSL files, compiles/links programs, caches common uniform locations, reloads programs |
| `Mesh` | Holds vertex/index data, loads model files with Assimp, uploads GPU buffers, issues indexed draws |
| `GPUBuffers` | Creates VAOs/VBOs and uploads positions, normals, texture coordinates, or `VertexFormat` buffers |
| `Texture2D` | Loads two-dimensional images, creates texture/FBO attachments, reads or saves texture data |
| `FrameBuffer` | Creates color and depth texture attachments for off-screen passes |
| `SSBO<T>` | Generic shader-storage-buffer wrapper for optional GPU data workflows |
| `ParticleEffect<T>` | Generic point rendering helper backed by an SSBO; unused by the two volume scenes |
| `TextureManager` | Loads and caches common two-dimensional textures |

## Resource Resolution

`RESOURCE_PATH` defines paths relative to the current working directory:

```text
Resources/
|-- Models/
|-- Shaders/
|-- Textures/
`-- Volumes/
```

The technique-specific shader paths are located beneath
`Resources/Shaders/<Technique>/`, allowing CMake's existing `Resources/`
copy step to produce a runnable build output resource tree.
