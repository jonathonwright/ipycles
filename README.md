iPyCLES project is a Large Eddy Simulation model (PyCLES) coupled with stable water isotope tracer components. Now the isotopic components are still developing, so most parts of iPyCLES are same as PyCLES, but fixed some bugs about python 3 environemnt and local system settings.

## Installation of ipycles in Linux and wsl2: (tested with ubuntu, Debian and Centos):
Important system environments needed to be installed:

`$ sudo apt-get install gcc gfortran-8 csh`

Here the **gfortran version** should be lower than 9 (the lastest version until May, 2021), or some files can't be compiled.

We recommend using [conda](https://docs.conda.io/en/latest/) as the package management system and environment management system for python environment settings. Miniconda can be downloaded using **wget**:

`$ wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh`

then install by:

`$ bash Miniconda3-latest-Linux-x86_64.sh`

**Becareful with the python version**, which can't be higher then 3.8, because cython doesn't support python 3.9 until now (May, 2021).

Install packages needed to compile ipycles:

`$ conda install numpy scipy netcdf4 mpi4py matplotlib cython gcc_linux-64`

Complie ipycles by doing following steps:

1. `$ cd ipycles`

2. `$ python generate_parameters.py`

3. `$ CC=mpicc python setup.py build_ext --inplace`

 More details about installation in defferent platforms and erros can be found : [install.rst](https://github.com/huzizhan/ipycles/blob/master/docs/source/install.rst)
## Run test cases
Run test cases of ipycles follows: [running.rst](https://github.com/huzizhan/ipycles/blob/master/docs/source/running.rst)
## Introduction of [PyCLES](https://github.com/pressel/pycles)
Python Cloud Large Eddy Simulation, or PyCLES (pronounced pickles), is a massively parallel anelastic atmospheric large eddy simulation infrastructure designed to simulate boundary layer clouds and deep convection. PyCLES is written in Python, Cython, and C. It was primarily developed by [Kyle Pressel](http://www.kylepressel.com) and [Colleen Kaul](http://www.colleenkaul.com) as part of the [Climate Dynamics Group](https://climate-dynamics.org/) at both the California Institute of Technology and ETH Zurich. 

The model formulation is describe in detail in: 

Pressel, K. G., C. M. Kaul, T. Schneider, Z. Tan, and S. Mishra, 2015: Large-eddy simulation in an anelastic framework with closed water and entropy balances. Journal of Advances in Modeling Earth Systems, 7, 1425–1456, [doi:10.1002/2015MS000496](http://dx.doi.org/10.1002/2015MS000496). 

PyCLES Related Publications:

Zhang, X., T. Schneider, and C. M. Kaul, 2018: Arctic mixed-phase clouds in large-eddy simulations and a mixed-layer model. Journal of Advances in Modeling Earth Systems, submitted. [PDF](http://climate-dynamics.org/wp-content/uploads/2018/01/isdac-revision.pdf)

Tan, Z., C. M. Kaul, K. G. Pressel, Y. Cohen, T. Schneider, and J. Teixeira, 2018: An extended eddy-diffusivity mass-flux scheme for unified representation of subgrid-scale turbulence and convection. Journal of Advances in Modeling Earth Systems, In Press. [Early Release](http://onlinelibrary.wiley.com/doi/10.1002/2017MS001162/full) 

Pressel, K. G., S. Mishra, T. Schneider, C. M. Kaul, Z. Tan, 2017: Numerics and subgrid-scale modeling in large eddy simulations of stratocumulus clouds. Journal of Advances in Modeling Earth Systems, 9, 1342-1365, [doi:10.1002/2016MS000778](http://dx.doi.org/10.1002/2016MS000778).

Tan, Z., T. Schneider, J. Teixeira, and K. G. Pressel, 2017: Large-eddy simulation of subtropical cloud-topped boundary layers: 2. Cloud response to climate change. Journal of Advances in Modeling Earth Systems, 9, 19-38, [doi:10.1002/2016MS000804](http://dx.doi.org/10.1002/2016MS000804).
 
Schneider, T., J. Teixeira, C. S. Bretherton, F. Brient, K. G. Pressel, C. Schär, and A. P. Siebesma, 2017: Climate goals and computing the future of clouds. Nature Climate Change, 7, 3-5, [doi:10.1038/nclimate3190](http://dx.doi.org/10.1038/nclimate3190).
 
Tan, Z., T. Schneider, J. Teixeira, and K. G. Pressel, 2016: Large-eddy simulation of subtropical cloud-topped boundary layers: 1. A forcing framework with closed surface energy balance. Journal of Advances in Modeling Earth Systems, 8, 1565-1585, [doi:10.1002/2016MS000655](http://dx.doi.org/10.1002/2016MS000655).

Pressel, K. G., C. M. Kaul, T. Schneider, Z. Tan, and S. Mishra, 2015: Large-eddy simulation in an anelastic framework with closed water and entropy balances. Journal of Advances in Modeling Earth Systems, 7, 1425–1456, [doi:10.1002/2015MS000496](http://dx.doi.org/10.1002/2015MS000496).

Ait-Chaalal, F., T. Schneider, B. Meyer, and B. Marston, 2016: Cumulant expansions for atmospheric flows. New Journal of Physics, 18, 025019, [doi:10.1088/1367-2630/18/2/025019](http://dx.doi.org/10.1088/1367-2630/18/2/025019).
