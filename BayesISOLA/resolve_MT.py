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

	def __init__(self, data, cova, deviatoric=False, decompose=True):
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


