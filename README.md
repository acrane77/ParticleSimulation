# OpenGL Particle System

This project implements a **real-time particle system** in C++ using **OpenGL 4.5**, **GLFW**, **GLAD**, and **GLM**.  
It supports multiple particle emission patterns and particle types, and renders them as smooth, circular, feathered sprites via custom GLSL shaders.

---

## Project Structure
├── deps/ | External dependencies (GLFW, GLAD, GLM, etc.)  
├── shaders/  | Custom shaders  
│ ├── points.vert | Vertex shader (transforms and sizes particles)  
│ └── points.frag | Fragment shader (circular sprite with smooth edges)  
├── particle_main.cpp | Main C++ program (simulation + rendering loop)  
└── README.md | Project documentation  

## Features

- **Particle system architecture**
  - Efficient circular buffer for up to **20,000 particles**.
  - Multiple emission patterns (`Plume`, `Random`, `Rising`, `Falling`, `Circle`).
  - Two particle types with unique behaviors:
    - **Smoke**: fades, expands, and drifts upward.
    - **Sparks**: bright bursts with randomized flicker and decay.

- **Interactive camera controls**
  - **Right Mouse Drag** → Orbit camera.
  - **Middle Mouse Drag / Shift + Right Mouse Drag** → Pan camera.
  - **Scroll Wheel** → Zoom.

- **Emitter placement**
  - **Left Click** to place an emitter at the ground plane.
  - Multiple emitters can be active at once.

- **Keyboard shortcuts**
  - `1`–`5` Selects emission pattern (`Plume`, `Random`, `Rising`, `Falling`, `Circle`).
  - `S` activate Smoke particles.
  - `F` activate Firework burst (sparks).
  - `C` Clear all emitters.

- **Shaders**
  - Vertex shader (`points.vert`): applies projection, view transformations, and particle sizing.
  - Fragment shader (`points.frag`): renders circular, feathered point sprites with per-particle color.

  ---

  ## Getting Started

  ### Prerequisites
- C++17 or later
- OpenGL 4.5+
- CMake (recommended)
- Dependencies:
  - [GLFW](https://www.glfw.org/) – window & input handling
  - [GLAD](https://glad.dav1d.de/) – OpenGL loader
  - [GLM](https://github.com/g-truc/glm) – math library

### Build & Run
In the project terminal run:  
```g++ .\particle_main.cpp .\deps\src\glad.c -I .\deps\include -I C:\msys64\mingw64\include -L C:\msys64\mingw64\lib -lglfw3 -lopengl32 -lgdi32 -luser32 -lshell32 -lwinmm -o main.exe```
Then run the .exe with:  
```./main```
