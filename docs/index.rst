.. BayesISOLA documentation master file, created by
   sphinx-quickstart on Wed Apr 21 14:37:45 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

BayesISOLA documentation
=========================

BayesISOLA is an open-source Python module for solution of seismic source inverse problem. It uses the point source approximation and describes the source in terms of centroid moment tensor.

| **Copyright:**	`Jiří Vackář <http://geo.mff.cuni.cz/~vackar/>`_
| **Version:**	developer's snapshot 2022-02-20
| **License:**	GNU Lesser General Public License, Version 3 (http://www.gnu.org/copyleft/lesser.html)

.. toctree::
   :hidden:
   :maxdepth: 3

   BayesISOLA.load_data
   BayesISOLA.grid
   BayesISOLA.process_data
   BayesISOLA.covariance_matrix
   BayesISOLA.resolve_MT
   BayesISOLA.plot
   BayesISOLA.axitra
   BayesISOLA.fileformats
   BayesISOLA.helpers
   BayesISOLA.histogram
   BayesISOLA.inverse_problem
   BayesISOLA.MouseTrap
   BayesISOLA.MT_comps

   
Method
------

The used method is described in the following paper: 
J. Vackář, J. Burjánek, F. Gallovič, J. Zahradník, and J. Clinton (2017). 
Bayesian ISOLA: new tool for automated centroid moment tensor inversion, 
*Geophys. J. Int.*, 210(2), 693–705.
`PDF <http://geo.mff.cuni.cz/~vackar/papers/isola-obspy.pdf>`_

Important note
--------------

The code is still under development. We would be very happy for your feedback.
   
Requirements
------------

Anaconda `package list <_static/package-list.txt>`_ of all necessary Python packages

* `NumPy <http://www.numpy.org>`_: Fundamental package for scientific computing with Python.
* `matplotlib <http://matplotlib.org>`_: Python 2D plotting library which produces publication quality figures in a variety of hardcopy formats and interactive.
* `ObsPy <https://github.com/obspy/obspy/wiki>`_: Python framework for processing seismological data.
* `SciPy <http://www.scipy.org>`_: Python-based ecosystem of open-source software for mathematics, science, and engineering.
* `pyproj <https://github.com/jswhit/pyproj>`_: Python interface to PROJ4 library for cartographic transformations
* `other modules:` math, subprocess, shutil, multiprocessing, re, fractions, warnings, os

Download
--------

* `BayesISOLA on GitHub <https://github.com/vackar/BayesISOLA>`_

Installation
------------

* Download the code `from GitHub <https://github.com/vackar/BayesISOLA>`_
* Install all required packages (recommended via Anaconda, `package list <_static/package-list.txt>`_)
* Compile files `green/gr_xyz.for` and `green/elemse.for` with a Fortran compiler (tested with `ifort` and `gfortran`), the binaries should be at `green/gr_xyz` and `green/elemse`, respectively.
* Run the examples `example_2_SAC.py` (data saved in files) or `example_2_fdsnws.py` (data obtained via fdsnws service)


Examples
--------

All necessary inputs for all examples are included in GitHub repository. Just run ``python example_X.py``. The desired output is linked below each example.

Example 2a: SAC files with covariance matrix
  Find a centroid moment tensor for Sargans (St. Gallen, Switzerland) Dec 12, 2013 earthquake. Network configuration is described in file `network.stn`, event information are in `event.isl`, crustal model in `crustal.dat`, and waveforms are in form of SAC files.
  
  Download: `output directory of example 2a (zip) <_static/example_2_SAC.zip>`_ 
  
Example 2b: fdsnws service with covariance matrix
  The same earthquake and configuration as above, just the waveforms and station responses are obtained using fdsnws service.

  Download: `output directory of example 2b (zip) <_static/example_2_fdsnws.zip>`_ 

  
Module summary
------------------
   
.. currentmodule:: BayesISOLA

.. autosummary::
   
   BayesISOLA.load_data
   BayesISOLA.grid
   BayesISOLA.process_data
   BayesISOLA.covariance_matrix
   BayesISOLA.resolve_MT
   BayesISOLA.plot
   BayesISOLA.axitra
   BayesISOLA.fileformats
   BayesISOLA.helpers
   BayesISOLA.histogram
   BayesISOLA.inverse_problem
   BayesISOLA.MouseTrap
   BayesISOLA.MT_comps

   

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

