#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Solves inverse problem in a single grid point for multiple time shifts.

"""

import numpy as np

from obspy import UTCDateTime

from BayesISOLA.fileformats import read_elemse, read_elemse_from_files
from BayesISOLA.helpers import my_filter
from BayesISOLA.MT_comps import decompose, a2mt

def invert(point_id, d_shifts, norm_d, Cd_inv, Cd_inv_shifts, nr, comps, stations, npts_elemse, npts_slice, elemse_start_origin, origin_time, deviatoric=False, decomp=True, invert_displacement=False, elemse_path=None):
	"""
	Solves inverse problem in a single grid point for multiple time shifts.
	
	:param point_id: grid point id, elementary seismograms are readed from 'green/elemse'+point_id+'.dat'
	:type point_id: string	
	:param d_shifts: list of shifted data vectors :math:`d`
	:type d_shifts: list of :class:`~numpy.ndarray`
	:param norm_d: list of norms of vectors :math:`d`
	:type norm_d: list of floats
	:param Cd_inv: inverse of the data covariance matrix :math:`C_D^{-1}` saved block-by-block
	:type Cd_inv: list of :class:`~numpy.ndarray`
	:param Cd_inv_shifts: inverse of the data covariance matrix :math:`C_D^{-1}` saved block-by-block (for all time shifts - ACF)
	:type Cd_inv_shifts: list of :class:`~numpy.ndarray`
	:param nr: number of receivers
	:type nr: integer
	:param comps: number of components used in inversion
	:type comps: integer
	:param stations: ``BayesISOLA.stations`` metadata of inverted stations
	:type stations: list of dictionaries
	:param npts_elemse: number of points of elementary seismograms
	:type npts_elemse: integer
	:param npts_slice: number of points of seismograms used in inversion (npts_slice <= npts_elemse)
	:type npts_slice: integer
	:param elemse_start_origin: time between elementary seismogram start and elementary seismogram origin time
	:type elemse_start_origin: float
	:param origin_time: Event origin time in UTC
	:type origin_time: :class:`~obspy.core.utcdatetime.UTCDateTime`
	:param deviatoric: if ``True``, invert only deviatoric part of moment tensor (5 components), otherwise full moment tensor (6 components)
	:type deviatoric: bool, optional
	:param decomp: if ``True``, decomposes found moment tensor in each grid point
	:type decomp: bool, optional
	:param invert_displacement: calculate L-2 difference between observed and modeled waveforms in displacement (if ``True``), otherwise compare it in velocity (default ``False``)
	:type invert_displacement: bool, optional
	:param elemse_path: path to elementary seismogram file
	:type elemse_path: string, optional
	:returns: Dictionary {'shift': order of `d_shift` item, 'a': coeficients of the elementary seismograms, 'VR': variance reduction, 'CN' condition number, and moment tensor decomposition (keys described at function :func:`decompose`)}
	
	It reads elementary seismograms for specified grid point, filter them and creates matrix :math:`G`.
	Calculates :math:`G^T`, :math:`G^T G`, :math:`(G^T G)^{-1}`, and condition number of :math:`G^T G` (using :func:`~np.linalg.cond`)
	Then reads shifted vectors :math:`d` and for each of them calculates :math:`G^T d` and the solution :math:`(G^T G)^{-1} G^T d`. Calculates variance reduction (VR) of the result.
	
	Finally chooses the time shift where the solution had the best VR and returns its parameters.
	
	Remark: because of parallelisation, this wrapper cannot be part of class :class:`ISOLA`.
	"""

	# params: grid[i]['id'], self.d_shifts, self.Cd_inv, self.nr, self.components, self.stations, self.npts_elemse, self.npts_slice, self.elemse_start_origin, self.deviatoric, self.decompose
	if deviatoric: ne=5
	else: ne=6
	if elemse_path:
		elemse = read_elemse_from_files(nr, elemse_path, stations, origin_time, invert_displacement)
	else:
		elemse = read_elemse(nr, npts_elemse, 'green/elemse'+point_id+'.dat', stations, invert_displacement)
	
	# filtrovat elemse
	for r in range(nr):
		for i in range(ne):
			my_filter(elemse[r][i], stations[r]['fmin'], stations[r]['fmax'])
	
	if npts_slice != npts_elemse:
		dt = elemse[0][0][0].stats.delta
		for st6 in elemse:
			for st in st6:
				#st.trim(UTCDateTime(0)+dt*elemse_start_origin, UTCDateTime(0)+dt*npts_slice+dt*elemse_start_origin+1)
				st.trim(UTCDateTime(0)+elemse_start_origin)
		npts = npts_slice
	else:
		npts = npts_elemse

	# RESIT OBRACENOU ULOHU
	# pro kazdy bod site a cas zdroje
	#   m = (G^T G)^-1 G^T d
	#     pamatovat si m, misfit, kondicni cislo, prip. singularni cisla

	c = 0
	G = np.empty((comps*npts, ne))
	for r in range(nr):
		for comp in range(3):
			if stations[r][{0:'useZ', 1:'useN', 2:'useE'}[comp]]: # this component has flag 'use in inversion'
				weight = stations[r][{0:'weightZ', 1:'weightN', 2:'weightE'}[comp]]
				for i in range(npts):
					for e in range(ne):
						G[c*npts+i, e] = elemse[r][e][comp].data[i] * weight
				c += 1

	res = {}
	sum_c = 0
	for shift in range(len(d_shifts)):
		d_shift = d_shifts[shift]
		# d : vector of data shifted
		#   shift>0 means that elemse start `shift` samples after data zero time

		if Cd_inv_shifts:  # ACF
			Cd_inv = Cd_inv_shifts[shift]
			
		if 'Gt' in vars() and not Cd_inv_shifts: # Gt is the same as at the previous shift
			pass
		elif Cd_inv:
			# evaluate G^T C_D^{-1}
			# G^T C_D^{-1} is in ``GtCd`` saved block-by-block, in ``Gt`` in one ndarray
			idx = 0
			GtCd = []
			for C in Cd_inv:
				size = len(C)
				GtCd.append(np.dot(G[idx:idx+size, : ].T, C))
				idx += size
			Gt = np.concatenate(GtCd, axis=1)
		else:
			Gt = G.transpose()
		
		if not 'det_Ca' in vars() or Cd_inv_shifts: # first cycle or Cd_inv is dependent on shift - must be recalculated
			GtG = np.dot(Gt,G)
			CN = np.sqrt(np.linalg.cond(GtG)) # condition number
			GtGinv = np.linalg.inv(GtG)
			det_Ca = np.linalg.det(GtGinv)

		# Gtd
		Gtd = np.dot(Gt,d_shift)

		# result : coeficients of elementary seismograms
		a = np.dot(GtGinv,Gtd)
		if deviatoric: a = np.append(a, [[0.]], axis=0)
		
		if Cd_inv:
			dGm = d_shift - np.dot(G, a[:ne]) # dGm = d_obs - G m
			idx = 0
			dGmCd_blocks = []
			for C in Cd_inv:
				size = len(C)
				dGmCd_blocks.append(np.dot(dGm[idx:idx+size, : ].T, C))
				idx += size
			dGmCd = np.concatenate(dGmCd_blocks, axis=1)
			misfit = np.dot(dGmCd, dGm)[0,0]
		else:
			synt = np.zeros(comps*npts)
			for i in range(ne):
				synt += G[:,i] * a[i]
			misfit = 0
			for i in range(npts*comps):
				misfit += (d_shift[i,0]-synt[i])**2
		VR = 1 - misfit / norm_d[shift]

		res[shift] = {}
		res[shift]['a'] = a.copy()
		res[shift]['misfit'] = misfit
		res[shift]['VR'] = VR
		res[shift]['CN'] = CN
		res[shift]['GtGinv'] = GtGinv
		res[shift]['det_Ca'] = det_Ca

	shift = max(res, key=lambda s: res[s]['VR']) # best shift

	r = {}
	r['shift'] = shift
	r['a'] = res[shift]['a'].copy()
	r['VR'] = res[shift]['VR']
	r['misfit'] = res[shift]['misfit']
	r['CN'] = res[shift]['CN']
	r['GtGinv'] = res[shift]['GtGinv']
	r['det_Ca'] = res[shift]['det_Ca']
	r['shifts'] = res
	if decomp:
		r.update(decompose(a2mt(r['a']))) # add MT decomposition to dict `r`
	return r
 
