# Rendering Techniques

## Ray Casting

`RayCasting` is the default scene created in `Source/Main.cpp`. It implements
GPU direct volume rendering using a proxy cube and a transfer function.

### Initialization

`RayCasting::Init()`:

1. Allocates a framebuffer matching the window resolution with three color
   attachments and a depth attachment.
2. Sets `stepSize` to `0.001`.
3. Positions the camera near the proxy volume.
4. Creates a unit cube mesh whose per-vertex color equals its local position.
   These colors are used as volume texture coordinates.
5. Compiles a back-face program and a ray-casting program.
6. Reads the configured RAW file, defaulting to `head256.raw` at
   `256 x 256 x 225`, validates its size, and uploads it as a single-channel
   `GL_R8` `GL_TEXTURE_3D`.
7. Validates and reads the configured transfer-function file, defaulting to
   `tff.dat`, into a 256-entry RGBA `GL_TEXTURE_1D`.

### Frame Pipeline

Each frame constructs a rotation model matrix for the cube and performs two
passes:

| Pass | Target | Purpose |
| --- | --- | --- |
| Back-face pass | Off-screen `FrameBuffer` | Render front-culled cube faces and store the ray exit position for each pixel |
| Ray-march pass | Default framebuffer | Render cube entry faces, sample an exit position, step through the 3D volume, map intensity through the transfer function, and composite color |

The ray-casting vertex shader passes the cube color as `EntryPoint` and its
clip-space position for mapping to the exit-point texture. The fragment shader:

1. Converts the projected fragment coordinate into a framebuffer lookup.
2. Forms a ray from entry to exit point.
3. Advances by `normalize(direction) * StepSize`, with an upper bound of
   2000 iterations.
4. Fetches scalar intensity from `VolumeTex`.
5. Fetches RGBA from `TransferFunc`.
6. Applies opacity correction and front-to-back compositing.
7. Stops on leaving the volume or reaching full accumulated opacity.

### Resize Behavior

When a resize event is received, `RayCasting::OnWindowResize()` regenerates
the framebuffer attachments for the new width and height.

The `[` and `]` keys adjust `StepSize` between bounded values at runtime,
trading rendering detail for sampling cost.

## Marching Cubes

`MarchingCubes` is compiled but not selected by default. It converts volume
samples into a triangle mesh on the CPU.

### Initialization

`MarchingCubes::Init()`:

1. Positions the camera.
2. Loads and validates a configured RAW volume, defaulting to `Bucky.raw` at
   `32 x 32 x 32`.
3. Initializes the configurable isovalue, defaulting to `50`.
4. Reconstructs the surface into a `Mesh`.
5. Compiles a vertex/fragment shader pair to display interpolated normals as
   color.

### Extraction Pipeline

`reconstructSurface()` iterates across every neighboring sample cube in the
volume:

1. Fill a `GRIDCELL` with eight positions, byte intensities, and bounded
   finite-difference gradient normals.
2. Compute an 8-bit case index by checking whether each sample is below the
   current isovalue.
3. Use `edgeTable` to identify crossed edges.
4. Interpolate intersection positions and normals along those edges.
5. Use `triTable` to emit up to five triangles for the cell.
6. Collect triangles in vector-owned storage and flatten them into position,
   normal, and index arrays.
7. Upload the arrays into a GPU `Mesh`.

In the display pass, the generated coordinates are uniformly scaled by
`5.0 / xsize`, and the fragment shader encodes normalized vertex normals as
RGB output.

### Interactive Isovalue

While this scene is active:

- Pressing `Z` increases `isolevel` by `5`, capped at `255`.
- Pressing `X` decreases `isolevel` by `5`, floored at `0`.
- A changed isovalue immediately rebuilds and uploads the complete triangle
  mesh once per key press.

## Comparison

| Aspect | Ray Casting | Marching Cubes |
| --- | --- | --- |
| Default scene | Yes | No |
| Representation | Samples original voxel texture during rendering | Extracts triangles from voxel values |
| Primary compute location | Fragment shader | CPU preprocessing/reconstruction |
| Appearance control | 1D transfer-function texture and step size | Isovalue and normal visualization |
| Default dataset | `head256.raw` | `Bucky.raw` |
| Interaction implemented | Camera/common scene controls | Camera/common controls plus `Z`/`X` isovalue changes |
