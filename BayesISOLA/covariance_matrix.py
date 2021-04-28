#! /usr/bin/env python3
# -*- coding: utf-8 -*-

class covariance_matrix:
	"""
    Design covariance matrix(es).

    :param data: instance with processed data
    :type data: :class:`~BayesISOLA.process_data`

    .. rubric:: _`Variables`

    ``Cd_inv`` : list of :class:`~numpy.ndarray`
        Inverse of the data covariance matrix :math:`C_D^{-1}` saved block-by-block. Created by :func:`covariance_matrix`.
    ``Cd`` : list of :class:`~numpy.ndarray`
        Data covariance matrix :math:`C_D^{-1}` saved block-by-block. Optionally created by :func:`covariance_matrix`.
    ``LT`` : list of list of :class:`~numpy.ndarray`
        Cholesky decomposition of the data covariance matrix :math:`C_D^{-1}` saved block-by-block with the blocks corresponding to one component of a station. Created by :func:`covariance_matrix`.
    ``LT3`` : list of :class:`~numpy.ndarray`
        Cholesky decomposition of the data covariance matrix :math:`C_D^{-1}` saved block-by-block with the blocks corresponding to all component of a station. Created by :func:`covariance_matrix`.
    ``Cf`` :  list of 3x3 :class:`~numpy.ndarray` of :class:`~numpy.ndarray`
        List of arrays of the data covariance functions.
    ``Cf_len`` : integer
        Length of covariance functions.    
	"""

	from BayesISOLA._covariance_matrix import covariance_matrix_noise, covariance_matrix_SACF, covariance_matrix_ACF

	def __init__(self, data):
		self.d = data
		self.stations = data.d.stations
		self.log = data.log
		self.Cd_inv = []
		self.Cd = []
		self.LT = []
		self.LT3 = []
		self.Cf = []
		self.Cd_inv_shifts = []
		self.Cd_shifts = []
		self.LT_shifts = []

	def __exit__(self, exc_type, exc_value, traceback):
		self.__del__()
		
	def __del__(self):
		del self.Cd_inv
		del self.Cd
		del self.LT
		del self.LT3
		del self.Cf
		del self.Cd_inv_shifts
		del self.Cd_shifts
		del self.LT_shifts
		
