# 🧠 Volume Rendering Engine

> *Because flat images are so 2D...*

Implementation documentation: [`docs/README.md`](docs/README.md)

A slick OpenGL-based volume rendering engine featuring **Marching Cubes** and **Ray Casting** techniques. Perfect for visualizing 3D volumetric data like brains, engines, and other cool stuff! 🔥

---

## ✨ Features

| Technique | What it does |
|-----------|--------------|
| 🧊 **Marching Cubes** | Extracts polygon meshes from 3D scalar fields |
| 🔦 **Ray Casting** | Direct volume rendering with transfer functions |

## 📂 Where the Magic Happens

```
Source/Laboratoare/
├── MarchingCubes/   🧊 Isosurface extraction
└── RayCasting/      🔦 Direct volume rendering
```

---

## 🚀 Quick Start

### Option 1: The Easy Way (Recommended)

Just double-click the batch file and watch the magic happen:

```batch
.\build_and_run.bat
```

Want a debug build? No problem:

```batch
.\build_and_run.bat debug
```

### Option 2: The Manual Way (For Control Freaks)

```powershell
# Create and enter build directory
mkdir build
cd build

# Configure with CMake (Visual Studio 2022)
cmake -G "Visual Studio 17 2022" -A x64 ..

# Build it! (Release mode)
cmake --build . --config Release

# Run from project root (important for shader paths!)
cd ..
.\build\bin\VolumeRendering.exe
```

---

## 🛠️ Requirements

- **Visual Studio 2019/2022** with C++ tools
- **CMake 3.16+**
- A graphics card that doesn't hate OpenGL 🎮

---

## 📦 Included Goodies

- 🧠 Brain scan (`brain.raw`)
- 💀 Head CT (`head256.raw`)  
- 🫁 Chest scan (`chest.raw`)
- ⚙️ Engine model (`Engine.raw`)
- 🦌 Bucky the bunny (`Bucky.raw`)

---

## 🎮 Controls

Fire up the engine and explore! Use your mouse and keyboard to navigate the 3D world.

---

<div align="center">

**Made with ☕ and OpenGL**

*Enjoy! :3*

</div>
