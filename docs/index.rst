.. BayesISOLA documentation master file, created by
   sphinx-quickstart on Wed Apr 21 14:37:45 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

BayesISOLA documentation
=========================

BayesISOLA is an open-source module for Python for solution of seismic source inverse problem. It uses the point source approximation and describes the source in terms of centroid moment tensor.

| **Copyright:**	`Jiří Vackář <http://geo.mff.cuni.cz/~vackar/>`_
| **Version:**	developer's snapshot 2021-09-17
| **License:**	GNU Lesser General Public License, Version 3 (http://www.gnu.org/copyleft/lesser.html)

Contents:

.. toctree::
   :maxdepth: 4

   class_isola
   class_isola_toc
   
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

Anaconda `package list <_static/package-list.txt>` of all necessary Python packages

* `NumPy <http://www.numpy.org>`_: Fundamental package for scientific computing with Python.
* `matplotlib <http://matplotlib.org>`_: Python 2D plotting library which produces publication quality figures in a variety of hardcopy formats and interactive.
* `ObsPy <https://github.com/obspy/obspy/wiki>`_: Python framework for processing seismological data.
* `SciPy <http://www.scipy.org>`_: Python-based ecosystem of open-source software for mathematics, science, and engineering.
* `pyproj <https://github.com/jswhit/pyproj>`_: Python interface to PROJ4 library for cartographic transformations
* `other modules:` math, subprocess, shutil, multiprocessing, re, fractions, warnings, os

Download
--------

* `BayesISOLA on GitHub <https://github.com/vackar/BayesISOLA>`

Installation
------------

* Download the code from GitHub
* Compile files `green/gr_xyz.for` and `green/elemse.for` with a Fortran compiler (tested with `ifort`), the binaries should be at `green/gr_xyz` and `green/elemse`, respectively.
* Try to run `src/invert_SAC.py` (Example 1) or `src/invert_SAC_cova.py` (Example 2)


Examples
--------



.. Example 1: SAC files
..   Find a centroid moment tensor for a Corinth Gulf (Greece) Apr 25, 2012 earthquake (`Sokos and Zahradník, SRL, 2013 <http://geo.mff.cuni.cz/~jz/papers/sokos&zahradnik_srl2013.pdf>`_). Network configuration is described in file `network.stn`, event information are in `event.isl`, crustal model in `crustal.dat`, and waveforms are in form of SAC files.
..   
..   Download: `Example 1 directory (zip) <_static/example_1.zip>`_ and `desired output of example 1 (zip) <_static/example_1_output.zip>`_ 
.. 
.. Example 2: SAC files with covariance matrix
..   Find a centroid moment tensor for Sargans (St. Gallen, Switzerland) Dec 12, 2013 earthquake. Network configuration is described in file `network.stn`, event information are in `event.isl`, crustal model in `crustal.dat`, and waveforms are in form of SAC files.
..   
..   Download: `Example 2 directory (zip) <_static/example_2.zip>`_ and `desired output of example 2 (zip) <_static/example_2_output.zip>`_ 


Function summary
------------------
   
.. currentmodule:: BayesISOLA.class_isola

.. autosummary::
   
   ISOLA
   ISOLA.log
   ISOLA.read_crust
   ISOLA.read_event_info
   ISOLA.set_event_info
   ISOLA.read_network_info_DB
   ISOLA.read_network_coordinates
   ISOLA.create_station_index
   ISOLA.write_stations
   ISOLA.add_NEZ
   ISOLA.add_SAC
   ISOLA.load_files
   ISOLA.load_streams_ArcLink
   ISOLA.check_a_station_present
   ISOLA.detect_mouse
   ISOLA.set_frequencies
   ISOLA.count_components
   ISOLA.correct_data
   ISOLA.trim_filter_data
   ISOLA.covariance_matrix
   ISOLA.prefilter_data
   ISOLA.decimate_shift
   ISOLA.set_working_sampling
   ISOLA.min_time
   ISOLA.max_time
   ISOLA.set_time_window
   ISOLA.set_Greens_parameters
   ISOLA.write_Greens_parameters
   ISOLA.verify_Greens_parameters
   ISOLA.verify_Greens_headers
   ISOLA.calculate_or_verify_Green
   ISOLA.set_grid
   ISOLA.set_time_grid
   ISOLA.calculate_Green
   ISOLA.run_inversion
   ISOLA.find_best_grid_point
   ISOLA.print_solution
   ISOLA.print_fault_planes
   ISOLA.plot_MT
   ISOLA.plot_uncertainty
   ISOLA.plot_maps
   ISOLA.plot_slices
   ISOLA.plot_maps_sum
   ISOLA.plot_map_backend
   ISOLA.plot_3D
   ISOLA.plot_seismo
   ISOLA.plot_covariance_function
   ISOLA.plot_noise
   ISOLA.plot_stations
   ISOLA.plot_covariance_matrix
   
   next_power_of_2
   lcmm
   a2mt
   decompose
   my_filter
   decimate
   read_elemse
   attach_ISOLA_paz
   Axitra_wrapper
   invert
..    histogram



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

