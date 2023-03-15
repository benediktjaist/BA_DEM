# BA_DEM

# The DEM in two Dimensions - A basic Python Tool
___
The Discrete Element Method (DEM) is a numerical method that describes a system consisting of individual particles 
tracking the movement and interaction of the particles over time. One of its key features is the explicit 
time integration, which makes it fast and efficient. In the context of DEM, this algorithm is often called the 
Velocity-Verlet-Algorithm, which is a central difference scheme. The interaction between the particles can be 
characterized with the application of different contact models. Depending on the application and scale, the 
contact models vary in complexity and include different physical phenomena such as elasticity, plasticity or cohesion. 
Popular contact models are the LS+D (Linear Spring Dashpot Model) and HM+D (Hertz Mindlin Spring Dashpot Model). 

As part of a bachelor's thesis, this python simulation tool was developed. This calculation tool can be operated 
via a GUI. The core of the simulation programme is the DEM_Solver. A simple DEM algorithm is implemented here. 
The central features of this implementation are the explicit time integration with the VV algorithm and a 
linear spring dashpot model that models the interactions of the particles. The programme also supports the interaction 
of particles with rigid body boundaries. 



## Installation
___
To use the simulation software, the entire repository "BA_DEM" must be downloaded. It must also be ensured that a 
Python 3 version is installed on the device with the Numpy, Sympy and PyQt packages used. 

For the installation of Python, [Anaconda](https://www.anaconda.com/) is recommended. Anaconda is an open source distribution for Python 
that provides a comprehensive collection of libraries and tools. Visit their website for detailed information. 

PyQt, NumPy and SymPy are among the standard packages included in the Anaconda distribution.
Otherwise, packages can be updated or installed with the [anaconda navigator](https://docs.anaconda.com/navigator/index.html) 
if a graphical user interface is preferred or [conda](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html) 
as backend solution which can be used via anaconda prompt. 


## Usage
___
### Starting the Program
To start the calculation tool, the Python script "Main" must be executed in a Python environment. 
### Creating an Assembly
The user has different options to create an assembly consisting of particles and rigid boundaries.
1. You can create particles and boundaries manually with the respective properties. For the particles, default values can 
be used, random values can be generated or specific values can be entered. The boundaries can be drawn interactively on the screen with the mouse.
2. Various examples are already stored in the "examples" folder. The particles, boundaries and system properties can be 
transferred to the current assembly via the gui by importing the corresponding python file. 
The correctness of the input of the desired assembly can be checked again via the preview. 
### Running the Computation
Here, the system properties such as the coefficient of restitution (COR), the coefficient of friction (mu) or 
the time increment (dt) and the desired simulation time can be selected. In addition, gravity can be switched on or off. 
As soon as the user presses "Run Simulation", the estimated calculation time is determined and the progress of the calculation is displayed. 
### Rendering the Video
to make the results of the calculation visible to the user, you can create an animation of the particles as a video in this area. 
The storage location, the file name and the frames per second (fps) of the video must be specified. Since this process can 
also take a while, the remaining rendering time and the progress are displayed to the user here. 
### Plotting the Energy 
In order to check the plausibility of the calculation or to view details of the system behaviour, the user has the possibility 
to plot the energies of the individual particles and the entire system over time.
## Contributing
___
Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

## Contact and further information
If you have suggestions for improvement or need further information feel free to contact the author at any time via email at 
[benedikt.jaist@gmail.com](mailto:benedikt.jaist@gmail.com?subject=DEM%20Simulation%20Tool).

## License
___
This project is published here on GitHub under the [BSD 4-Clause](https://spdx.org/licenses/BSD-4-Clause.html) licence. 

Please have a look at the "LICENCE.md" file. 
