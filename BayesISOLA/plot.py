#! /usr/bin/env python3
# -*- coding: utf-8 -*-

class plot:
	"""
    Class for graphic output of MT solution, inverse process, and used data.
    
    :param solution: instance of a MT solution
    :type solution: :class:`~BayesISOLA.resolve_MT`
    :param maps: plot maps of solutions on the grid using :func:`plot_maps`
    :type maps: bool, optional
    :param slices: plot slices of solutions on the grid using :func:`plot_slices`
    :type slices: bool, optional
    :param maps_sum: plot maps and side-views of the posterior probability density function (using :func:`plot_maps_sum`)
    :type maps_sum: bool, optional
    :param MT: plot the best-fitting moment tensor solution using :func:`plot_MT`
    :type MT: bool, optional
    :param uncertainty: plot several figures of posterior probability density function (histograms and MT beachball), parameter set the number of random samples sampling the PPDF (using :func:`plot_uncertainty`)
    :type uncertainty: integer or `None`, optional
    :param seismo: plot seismograms (both observed and modeled data), y-axis automatically scaled for each station (using :func:`plot_seismo`)
    :type seismo: bool, optional
    :param seismo_sharey: plot seismograms, y-axis the same for all stations (using :func:`plot_seismo`)
    :type seismo_sharey: bool, optional
    :param seismo_cova: plot seismograms filtered using the covariance matrix (using :func:`plot_seismo`)
    :type seismo_cova: bool, optional
    :param noise: plot noise before the event using :func:`plot_noise`
    :type noise: bool, optional
    :param spectra: plot spectra of the signal, filtered signal, and noise (using :func:`plot_spectra`)
    :type spectra: bool, optional
    :param stations: plot map of the stations using :func:`plot_stations`
    :type stations: bool, optional
    :param covariance_matrix: plot the covariance matrix using :func:`plot_covariance_matrix`
    :type covariance_matrix: bool, optional
    :param covariance_function: plot the covariance functions using :func:`plot_covariance_function`
    :type covariance_function: bool, optional
    """

	from BayesISOLA._plot import plot_stations, plot_covariance_matrix
	from BayesISOLA._plot_solution_summary import plot_MT, plot_uncertainty, plot_MT_uncertainty_centroid
	from BayesISOLA._plot_solution_maps import plot_maps, plot_slices, plot_maps_sum, plot_map_backend, plot_3D
	from BayesISOLA._plot_data import plot_seismo, plot_covariance_function, plot_noise, plot_spectra, plot_seismo_backend_1, plot_seismo_backend_2
	from BayesISOLA._html import html_log

	def __init__(self, solution, maps=True, slices=True, maps_sum=True, MT=True, uncertainty=400, seismo=False, seismo_sharey=True, seismo_cova=True, noise=True, spectra=True, stations=True, covariance_matrix=True, covariance_function=False):
		self.MT   = solution
		self.grid = solution.g
		self.data = solution.d
		self.inp  = solution.d.d
		self.cova = solution.cova
		self.outdir = self.inp.outdir
		self.log = self.inp.log
		self.logtext = self.inp.logtext
		self.movie_writer = 'mencoder' # None for default
		self.plots = {'MT': None, 'uncertainty':None, 'stations':None, 'seismo':None, 'seismo_cova':None, 'seismo_sharey':None, 'spectra':None, 'noise':None, 'covariance_function':None, 'covariance_matrix':None, 'maps':None, 'slices':None, 'maps_sum':None}

		if maps:
			self.plot_maps()
		if slices:
			self.plot_slices()
		if maps_sum:
			self.plot_maps_sum()
		if MT:
			self.plot_MT()
		if uncertainty:
			self.plot_uncertainty(n=uncertainty)
		if seismo:
			self.plot_seismo()
		if seismo_sharey:
			self.plot_seismo(outfile='$outdir/seismo_sharey.png', sharey=True)
		if seismo_cova and (len(self.cova.LT) or len(self.cova.LT3)):
			self.plot_seismo(outfile='$outdir/seismo_cova.png', cholesky=True)
		if noise:
			self.plot_noise()
		if spectra:
			self.plot_spectra()
		if stations:
			self.plot_stations()
		if covariance_matrix:
			self.plot_covariance_matrix()
		if covariance_function:
			self.plot_covariance_function()
