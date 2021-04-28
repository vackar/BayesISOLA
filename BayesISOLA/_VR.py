#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from obspy import UTCDateTime

from BayesISOLA.fileformats import read_elemse
from BayesISOLA.helpers import my_filter

def VR_of_components(self, n=1):
	"""
	Calculates the variance reduction from each component and the variance reduction from a subset of stations.
	
	:param n: minimal number of components used
	:type n: integer, optional
	:return: maximal variance reduction from a subset of stations
	
	Add the variance reduction of each component to ``self.stations`` with keys ``VR_Z``, ``VR_N``, and ``VR_Z``.
	Calculate the variance reduction from a subset of the closest stations (with minimal ``n`` components used) leading to the highest variance reduction and save it to ``self.max_VR``.
	"""
	npts = self.d.npts_slice
	data = self.d.data_shifts[self.centroid['shift_idx']]
	elemse = read_elemse(self.inp.nr, self.d.npts_elemse, 'green/elemse'+self.centroid['id']+'.dat', self.inp.stations, self.d.invert_displacement) # read elemse
	for r in range(self.inp.nr):
		for e in range(6):
			my_filter(elemse[r][e], self.inp.stations[r]['fmin'], self.inp.stations[r]['fmax'])
			elemse[r][e].trim(UTCDateTime(0)+self.d.elemse_start_origin)
	MISFIT = 0
	NORM_D = 0
	COMPS_USED = 0
	max_VR = -99
	self.VRcomp = {}
	for sta in range(self.inp.nr):
		SYNT = {}
		for comp in range(3):
			SYNT[comp] = np.zeros(npts)
			for e in range(6):
				SYNT[comp] += elemse[sta][e][comp].data[0:npts] * self.centroid['a'][e,0]
		comps_used = 0
		for comp in range(3):
			if self.cova.Cd_inv and not self.inp.stations[sta][{0:'useZ', 1:'useN', 2:'useE'}[comp]]:
				self.inp.stations[sta][{0:'VR_Z', 1:'VR_N', 2:'VR_E'}[comp]] = None
				continue
			synt = SYNT[comp]
			d = data[sta][comp][0:npts]
			if self.cova.LT3:
				d    = np.zeros(npts)
				synt = np.zeros(npts)
				x1 = -npts
				for COMP in range(3):
					if not self.inp.stations[sta][{0:'useZ', 1:'useN', 2:'useE'}[COMP]]:
						continue
					x1 += npts; x2 = x1+npts
					y1 = comps_used*npts; y2 = y1+npts
					d    += np.dot(self.cova.LT3[sta][y1:y2, x1:x2], data[sta][COMP].data[0:npts])
					synt += np.dot(self.cova.LT3[sta][y1:y2, x1:x2], SYNT[COMP])
				
			elif self.cova.Cd_inv:
				d    = np.dot(self.cova.LT[sta][comp], d)
				synt = np.dot(self.cova.LT[sta][comp], synt)
				
			else:
				pass
			comps_used += 1
			misfit = np.sum(np.square(d - synt))
			norm_d = np.sum(np.square(d))
			VR = 1 - misfit / norm_d
			self.inp.stations[sta][{0:'VR_Z', 1:'VR_N', 2:'VR_E'}[comp]] = VR
			if self.inp.stations[sta][{0:'useZ', 1:'useN', 2:'useE'}[comp]]:
				MISFIT += misfit
				NORM_D += norm_d
				VR_sum = 1 - MISFIT / NORM_D
				COMPS_USED += 1
				#print sta, comp, VR, VR_sum # DEBUG
		if COMPS_USED >= n:
			if COMPS_USED > 1:
				self.VRcomp[COMPS_USED] = VR_sum
			if VR_sum >= max_VR:
				max_VR = VR_sum
				self.max_VR = (VR_sum, COMPS_USED)
	return max_VR
