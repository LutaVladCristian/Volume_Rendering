# Limitations And Resolutions

## Addressed In This Revision

- The ray-casting exit pass now writes interpolated proxy-cube coordinates
  rather than a constant color, so the ray-march pass receives meaningful
  exit points.
- The application requests OpenGL 4.3 core profile to match the GLSL 4.30
  technique shaders.
- Volume upload uses `GL_R8` / `GL_RED` instead of legacy luminance/intensity
  formats.
- Both scenes can be selected from the command line and accept configurable
  RAW input paths and dimensions; ray casting also accepts a transfer-function
  path and exposes runtime step-size adjustment.
- RAW and transfer-function reads are validated before use.
- Technique shaders now reside in `Resources/Shaders/`, so the CMake resource
  copy produces the required runtime files.
- Marching Cubes uses vector-owned triangle storage and clamped
  finite-difference gradients, removing the allocation mismatch and boundary
  indexing hazards.
- Marching Cubes isovalue changes rebuild at most once per key press rather
  than every frame while a key is held.
- Scene objects, GPU textures, framebuffers, cached textures, and uploaded
  mesh buffers now have explicit release paths before the GL context closes.
- `World::Pause()` now prevents scene update/render execution while paused.

## Remaining Constraints

- RAW files still require dimensions supplied externally because the input
  format contains no metadata header.
- Marching Cubes rebuilds the full mesh when the isovalue changes. This is
  correct but can be expensive for large volumes; incremental or GPU
  extraction is outside the current renderer architecture.
- The application has no automated rendering or shader-validation test suite.
  Build verification checks compilation and packaging, while visual output
  still requires launching on an OpenGL 4.3-capable graphics environment.
