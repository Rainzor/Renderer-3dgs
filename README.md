# Renderer for 3DGS

The renderer is based on **SIBR** to rendering 3D Gaussian Splatting model files.

**SIBR** is a System for Image-Based Rendering.  It is built around the *sibr-core* in this repo and several *Projects* implementing published research papers.  For more complete documentation, see here: [SIBR Documentation](https://sibr.gitlabpages.inria.fr) 

##### Features:

- Depth Culling by Hierarchical Z-Buffer

## 1. Setup

### Hardware Requirements

- OpenGL 4.5-ready GPU and drivers 
- 4 GB VRAM recommended
- CUDA-ready GPU with Compute Capability 7.0+

### Software Requirements

- Visual Studio or g++, **not Clang** (we used Visual Studio 2019/2022 for Windows)
- CUDA SDK 11+, install *after* Visual Studio
- CMake (recent version)
- 7zip (only on Windows)

## 2. Clone 

- Checkout this repository's master branch:
  
  ```sh
  git clone https://gitlab.inria.fr/sibr/sibr_core.git -b master
  ```

## 3. Compile

### 3.1 GUI

#### Generation of the solution

- Run Cmake-gui once, select the repo root as a source directory, `build/` as the build directory. Configure, select the Visual Studio C++ Win64 compiler
- Select the projects you want to generate among the BUILD elements in the list (you can group Cmake flags by categories to access those faster)
- Generate

#### Compilation

- Open the generated Visual Studio solution (`build/sibr_projects.sln`)
- Build the `ALL_BUILD` target, and then the `INSTALL` target
- The compiled executables will be put in `install/bin`
- If some `*dll` files are not found, you need to copy them to `install/bin`

#### Compilation of the documentation

- Open the generated Visual Studio solution (`build/sibr_projects.sln`)
- Build the `DOCUMENTATION` target
- Run `install/docs/index.html` in a browser


### 3.2 Command

```sh
cd <dir>
cmake -Bbuild .
cmake --build build --target install --config Release -j
```

## 4. Run

```sh
./install/bin/SIBR_gaussianViewer_app.exe -m <path to trained model>
```

<details>
<summary><span style="font-weight: bold;">Primary Command Line Arguments for Real-Time Viewer</span></summary>


  #### --model-path / -m

  Path to trained model.

  #### --iteration

  Specifies which of state to load if multiple are available. Defaults to latest available iteration.

  #### --path / -s

  Argument to override model's path to source dataset.

  #### --rendering-size 

  Takes two space separated numbers to define the resolution at which real-time rendering occurs, ```1200``` width by default. Note that to enforce an aspect that differs from the input images, you need ```--force-aspect-ratio``` too.

  #### --load_images

  Flag to load source dataset images to be displayed in the top view for each camera.

  #### --device

  Index of CUDA device to use for rasterization if multiple are available, ```0``` by default.

  #### --no_interop

  Disables CUDA/GL interop forcibly. Use on systems that may not behave according to spec (e.g., WSL2 with MESA GL 4.5 software rendering).
</details>
<br>

## 5. Reference

```
@misc{sibr2020,
   author       = "Bonopera, Sebastien and Esnault, Jerome and Prakash, Siddhant and Rodriguez, Simon and Thonat, Theo and Benadel, Mehdi and Chaurasia, Gaurav and Philip, Julien and Drettakis, George",
   title        = "sibr: A System for Image Based Rendering",
   year         = "2020",
   url          = "https://gitlab.inria.fr/sibr/sibr_core"
}
```

