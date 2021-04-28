#! /usr/bin/env python3
# -*- coding: utf-8 -*-

class plot:
	"""
    Class for graphic output of MT solution, inverse process, and used data.
    
    :param solution: instance of a MT solution
    :type solution: :class:`~BayesISOLA.resolve_MT`
    """

	from BayesISOLA._plot import plot_stations, plot_covariance_matrix
	from BayesISOLA._plot_solution_summary import plot_MT, plot_uncertainty, plot_MT_uncertainty_centroid
	from BayesISOLA._plot_solution_maps import plot_maps, plot_slices, plot_maps_sum, plot_map_backend, plot_3D
	from BayesISOLA._plot_data import plot_seismo, plot_covariance_function, plot_noise, plot_spectra, plot_seismo_backend_1, plot_seismo_backend_2
	from BayesISOLA._html import html_log

	def __init__(self, solution):
		self.MT   = solution
		self.grid = solution.g
		self.data = solution.d
		self.inp  = solution.d.d
		self.cova = solution.cova
		self.outdir = self.inp.outdir
		self.log = self.inp.log
		self.logtext = self.inp.logtext
		self.movie_writer = 'mencoder' # None for default

