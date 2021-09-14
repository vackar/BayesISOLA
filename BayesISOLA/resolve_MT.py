#! /usr/bin/env python3
# -*- coding: utf-8 -*-

class resolve_MT:
	"""
    Class for the solution of the MT inverse problem.
    
    :param data: instance of processed data
    :type data: :class:`~BayesISOLA.process_data`
    :param cova: instance of a covariance matrix
    :type cova: :class:`~BayesISOLA.covariance_matrix`
    :type deviatoric: bool, optional
    :param deviatoric: if ``False``: invert full moment tensor (6 components); if ``True`` invert deviatoric part of the moment tensor (5 components) (default ``False``)
    :type decompose: bool, optional
    :param decompose: performs decomposition of the found moment tensor in each grid point
    :param run_inversion: run :func:`run_inversion`
    :type run_inversion: bool, optional
    :param find_best_grid_point: run :func:`find_best_grid_point`
    :type find_best_grid_point: bool, optional
    :param save_seismo: run :func:`save_seismo`
    :type save_seismo: bool, optional
    :param VR_of_components: run :func:`VR_of_components`
    :type VR_of_components: bool, optional
    :param print_solution: run :func:`print_solution`
    :type print_solution: bool, optional
    :param print_fault_planes: run :func:`print_fault_planes` (only if ``decompose`` is ``True``)
    :type print_fault_planes: bool, optional

    .. rubric:: _`Variables`

    ``centroid`` : Reference to ``grid`` item.
        The best grid point found by the inversion.
    ``mt_decomp`` : list
        Decomposition of the best centroid moment tensor solution calculated by :func:`decompose` or :func:`decompose_mopad`
    ``max_VR`` : tuple (VR, n)
        Variance reduction `VR` from `n` components of a subset of the closest stations
	"""

	from BayesISOLA._inverse import run_inversion, find_best_grid_point
	from BayesISOLA._VR import VR_of_components
	from BayesISOLA._print import print_solution, print_fault_planes
	from BayesISOLA._save import save_seismo

	def __init__(self, data, cova, deviatoric=False, decompose=True, run_inversion=True, find_best_grid_point=True, save_seismo=False, VR_of_components=False, print_solution=True, print_fault_planes=True):
		self.log = data.log
		self.d = data
		self.inp = data.d
		self.cova = cova
		self.g = data.grid
		self.grid = data.grid.grid
		self.threads = data.threads
		self.event = data.d.event
		
		self.deviatoric = deviatoric
		self.decompose = decompose
		self.mt_decomp = []
		self.max_VR = ()
		
		self.log('Inversion of ' + {1:'deviatoric part of', 0:'full'}[self.deviatoric] + ' moment tensor (' + {1:'5', 0:'6'}[self.deviatoric] + ' components)')

		if run_inversion:
			self.run_inversion()
		if find_best_grid_point:
			self.find_best_grid_point()
		if save_seismo:
			self.save_seismo(self.inp.outdir+'/data_observed', self.inp.outdir+'/data_modeles')
		if VR_of_components:
			self.VR_of_components()
		if print_solution:
			self.print_solution()
		if print_fault_planes and decompose:
			self.print_fault_planes()
