# OpenGL Particle System

Real-time particle system in C++ using OpenGL 4.6, GLFW, GLAD, and GLM.
Supports multiple emission patterns and particle types, including smoke, sparks, fireworks, rain, and snow. Particles are rendered as smooth, circular, feathered sprites via custom GLSL shaders.

---

## Project Structure  
├── deps/                 External dependencies (GLFW, GLAD, GLM, etc.)  
├── shaders/              Custom shaders  
│   ├── points.vert       Vertex shader (transform + size)  
│   └── points.frag       Fragment shader (circular sprite with smooth edges)  
├── particle_main.cpp     Simulation + rendering loop  
└── README.md             This file  

---

## Features
- Scalable particle core
- Efficient circular buffer with capacity for 50,000 particles.
- Stable iteration helpers to update and draw only alive particles.
- Emission patterns like Plume, Random, Rising, Falling, Circle.

### Particle types & behaviors

- Smoke: buoyant, expands, fades; wind-responsive drift.
- Sparks: bright, flicker/decay, drag; additive blending.
- Firework: launches with a fuse, emits smoke trail, explodes into a radial spark burst.
- Rain: fast falling streaks that die on ground contact.
- Snow: slow, wind-wobbly flakes that settle and fade on the ground.

### Wind & turbulence

- Time-varying gust system (direction + strength targets with smooth interpolation).
- Low-frequency band-wave turbulence for natural meander.

### Rendering pipeline

- Single VBO of packed DrawParticle { vec4 pos_size, vec4 colour }.
- Type grouping each frame: alpha-blended pass for smoke/snow/rain, additive pass for sparks.
- GLSL shaders produce circular, feathered point sprites with per-particle color/alpha.

### Interactive camera

Controls:
- Right-drag: orbit
- Middle-drag or Shift + Right-drag: pan
- Scroll: zoom

### Emitter placement
- Left-click: place an emitter, with multiple emitters supported.

## Keyboard shortcuts

### Patterns:
  - `1` Plume
  - `2` Random
  - `3` Rising
  - `4` Falling
  - `5` Circle

### Types:
  - `S` Smoke
  - `F` Spark
  - `W` Firework
  - `R` Rain
  - `N` Snow
    
### Global: 
  - `C` Clear all emitters

## Shaders

- Vertex (shaders/points.vert)
  - Applies projection/view transform and writes per-vertex gl_PointSize from pos_size.w. Expects uniform uViewProj.

- Fragment (shaders/points.frag)
  - Computes a circular mask in point coordinates to create round, feathered sprites and tints by the provided colour.

---

## Getting Started

### Prerequisites
- C++17 or later
- OpenGL 4.6 context (the code requests 4.6; most 4.5+ drivers work)
- CMake (recommended) or your preferred build system

### Dependencies:  
- GLFW  
 – window & input  
- GLAD  
 – OpenGL loader  
- GLM  
 – OpenGL math  

## Build & Run
Quick compile (MinGW-w64 / Windows)

From the project root:
```
g++ .\particle_main.cpp .\deps\src\glad.c ^  
  -I .\deps\include ^  
  -I C:\msys64\mingw64\include ^  
  -L C:\msys64\mingw64\lib ^  
  -lglfw3 -lopengl32 -lgdi32 -luser32 -lshell32 -lwinmm -o main.exe  
```
Then run:
```
./main
```
Linux 
```
g++ particle_main.cpp deps/src/glad.c \
  -I deps/include \
  -lglfw -ldl -lX11 -lpthread -lXrandr -lXi -o main
./main
```

macOS (using system frameworks + Homebrew GLFW)
```
clang++ particle_main.cpp deps/src/glad.c \
  -I deps/include \
  -L /opt/homebrew/lib -lglfw \
  -framework OpenGL -framework Cocoa -framework IOKit -framework CoreVideo \
  -std=c++17 -o main
./main
```
