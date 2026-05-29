# Resources And Data

## Resource Directories

| Path | Contents | Used By Current Scenes |
| --- | --- | --- |
| `Resources/Volumes/` | Raw scalar volumes and transfer function | Yes |
| `Resources/Shaders/` | Common mesh/debug/screen GLSL programs | Loaded by `SimpleScene` |
| `Resources/Models/Primitives/` | Basic helper meshes including `plane50.obj` | `SimpleScene` loads the plane |
| `Resources/Models/` | Additional sample model assets | Available through `Mesh`, not used by volume scenes |
| `Resources/Textures/` | Default and sample images | Default textures loaded on engine startup |
| `Resources/Shaders/RayCasting/` | Direct volume rendering shaders | Ray Casting |
| `Resources/Shaders/MarchingCubes/` | Isosurface display shaders | Marching Cubes |

## Volume Data Format

Both rendering techniques read raw files as tightly packed one-byte scalar
samples. There is no header parsing, endian conversion, spacing metadata, or
automatic dimension detection. A dataset must therefore be paired with exact
dimensions supplied through the command-line interface.

Known dimensions from `Resources/Volumes/readme.txt`:

| File | Dimensions | Byte Interpretation |
| --- | ---: | --- |
| `Engine.raw` | `256 x 256 x 256` | unsigned 8-bit scalar |
| `brain.raw` | `200 x 160 x 160` | unsigned 8-bit scalar |
| `vismale.raw` | `128 x 256 x 256` | unsigned 8-bit scalar |
| `chest.raw` | `384 x 384 x 240` | unsigned 8-bit scalar |
| `Bucky.raw` | `32 x 32 x 32` | unsigned 8-bit scalar |

`head256.raw` is the default ray-casting dataset with dimensions
`256 x 256 x 225`.

## Transfer Function

`RayCasting` reads `Resources/Volumes/tff.dat` into a one-dimensional OpenGL
texture:

- Texture extent: 256 texels.
- Texel format: RGBA8.
- Required input bytes for the uploaded region: 1024.
- Sampling: nearest-neighbor.

The ray-marching fragment shader uses each sampled scalar intensity as the
coordinate into this transfer function, producing a color and opacity for
compositing.

## Shader Inventory

### Common Shaders

`SimpleScene` registers these programs:

| Program Name | Vertex Shader | Fragment Shader | Role |
| --- | --- | --- | --- |
| `Simple` | `MVP.Texture.VS.glsl` | `Default.FS.glsl` | Textured mesh rendering |
| `Color` | `MVP.Texture.VS.glsl` | `Color.FS.glsl` | Uniform-color helper lines/plane |
| `VertexNormal` | `MVP.Texture.VS.glsl` | `Normals.FS.glsl` | Normal visualization |
| `VertexColor` | `MVP.Texture.VS.glsl` | `VertexColor.FS.glsl` | Vertex-color visualization |

Additional common screen/model shaders are supplied under `Resources/Shaders`
but are not used by the two volume scene update paths.

### Technique Shaders

| Scene | Shader Files Used |
| --- | --- |
| Ray Casting | `Resources/Shaders/RayCasting/VertexShader_backface.glsl`, `FragmentShader_backface.glsl`, `VertexShader_raycasting.glsl`, `FragmentShader_raycasting.glsl` |
| Marching Cubes | `Resources/Shaders/MarchingCubes/VertexShader.glsl`, `FragmentShader.glsl` |

The scene-specific shader files declare `#version 430`, whereas common
resource shaders declare `#version 330`.

## Texture Startup Set

At engine initialization, `TextureManager` attempts to load:

- `default.png`
- `white.png`
- `black.jpg`
- `noise.png`
- `random.jpg`
- `particle.png`

The first loaded texture also serves as the fallback material texture in
`Mesh::Render()` when a material does not provide a diffuse image.
