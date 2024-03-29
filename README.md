# BA_DEM

# The DEM in two Dimensions - A basic Python Tool
The Discrete Element Method (DEM) is a numerical method that describes a system consisting of individual particles 
tracking the movement and interaction of the particles over time. One of its key features is the explicit 
time integration, which makes it fast and efficient. In the context of DEM, this algorithm is often called the 
Velocity-Verlet-Algorithm (VV), which is a central difference scheme. The interaction between the particles can be 
characterized by applying different contact models. Depending on the application and scale, the 
contact models vary in complexity and include various physical phenomena such as elasticity, plasticity, or cohesion. 
The LS+D (Linear Spring Dashpot Model) and HM+D (Hertz Mindlin Spring Dashpot Model) are popular contact models. 

This python simulation tool was developed as part of a bachelor's thesis. This calculation tool can be operated 
via a graphical user interface (GUI). The core of the simulation program is the DEM_Solver. A simple DEM algorithm is implemented here. 
The central features of this implementation are the explicit time integration with the VV-algorithm and a 
linear spring dashpot model that models the interactions of the particles. The program also supports the interaction 
of particles with rigid body boundaries.

---

<div style="display:flex;justify-content:center;align-items:center;">
  <div style="flex:1;padding-right:20px;">
    <img src="/images/animation_screenshot.png" alt="Image 1" style="width:100%;">
    <p align="center">Figure 1 shows what the animations look like with pygame</p>
  </div>
  <div style="flex:1;padding-left:20px;">
    <img src="/images/gui_screenshot.png" alt="Image 2" style="width:100%;">
    <p align="center">Figure 2 presents the GUI of the programme</p>
  </div>
</div>

---

## Installation
To use the simulation software, the entire repository "BA_DEM" must be downloaded. It must also be ensured that a 
Python 3 version is installed on the device with the [Numpy](https://numpy.org/), [Sympy](https://www.sympy.org/en/index.html), [PyQt](https://doc.qt.io/qtforpython/), [Imageio](https://imageio.readthedocs.io/en/stable/) and [Pygame](https://www.pygame.org/news) packages, which are necessary to run the DEM Tool. 

For the installation of Python, [Anaconda](https://www.anaconda.com/) is recommended. Anaconda is an open source distribution for Python 
that provides a comprehensive collection of libraries and tools. Visit their website for detailed information. 

PyQt, NumPy, SymPy, and Imageio are among other standard packages included in the Anaconda distribution.
Otherwise, packages can be updated or installed with the [anaconda navigator](https://docs.anaconda.com/navigator/index.html) 
if a graphical user interface (GUI) is preferred or [conda](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html) 
as command-line interface (CLI), which can be used via Anaconda Prompt. 

The only package not provided by default in the anaconda environment is pygame. Pygame can be installed via Anaconda Prompt as follows:
```bash
conda install -c cogsci pygame
```

## Usage
### Starting the Program
To start the calculation tool, the Python script "Main" must be executed in a Python environment. 
### Creating an Assembly
The user has different options to create an assembly consisting of particles and rigid boundaries.
1. One can create particles and boundaries manually with the respective properties. For the particles, default values can 
be used, random values can be generated or specific values can be entered. The boundaries can be drawn interactively on the screen with the mouse.
2. Various examples are already stored in the "examples" folder. The particles, boundaries and system properties can be 
transferred to the current assembly via the GUI by importing the corresponding python file. 
The correctness of the input of the desired assembly can be rechecked via the preview. 
### Running the Computation
Here, the system properties such as the coefficient of restitution (COR), the coefficient of friction (mu) or 
the time increment (dt) and the desired simulation time can be selected. In addition, gravity can be switched on or off. 
When the user presses "Run Simulation", the estimated calculation time is determined, and the progress of the calculation is displayed. 
### Rendering the Video
To make the results of the calculation visible to the user, one can create an animation of the particles as a video in this area. 
The storage location, the file name and the frames per second (fps) of the video must be specified. Since this process may 
also take a while, the remaining rendering time and the progress are displayed to the user here. 
### Plotting the Energy 
In order to check the plausibility of the calculation or to view details of the system behaviour, the user has the possibility 
to plot the energies of the individual particles and the entire system over time.
## Contributing
Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

## Contact and further information
If you have suggestions for improvement or need further information feel free to contact the author at any time via email at 
[benedikt.jaist@gmail.com](mailto:benedikt.jaist@gmail.com?subject=DEM%20Simulation%20Tool).

## License
This project is published here on GitHub under the [BSD 4-Clause](https://spdx.org/licenses/BSD-4-Clause.html) licence. 

Please have a look at the "LICENCE.md" file. 
