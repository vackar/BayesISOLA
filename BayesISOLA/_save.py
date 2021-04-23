#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from obspy import UTCDateTime

from BayesISOLA.fileformats import read_elemse
from BayesISOLA.helpers import my_filter

def save_seismo(self, file_d, file_synt):
	"""
	Saves observed and simulated seismograms into files.
	
	:param file_d: filename for observed seismogram
	:type file_d: string
	:param file_synt: filename for synthetic seismogram
	:type file_synt: string
	
	Uses :func:`numpy.save`.
	"""
	data = self.data_shifts[self.centroid['shift_idx']]
	npts = self.npts_slice
	elemse = read_elemse(self.nr, self.npts_elemse, 'green/elemse'+self.centroid['id']+'.dat', self.stations, self.invert_displacement) # nacist elemse
	for r in range(self.nr):
		for e in range(6):
			my_filter(elemse[r][e], self.stations[r]['fmin'], self.stations[r]['fmax'])
			elemse[r][e].trim(UTCDateTime(0)+self.elemse_start_origin)
	synt = np.zeros((npts, self.nr*3))
	d = np.empty((npts, self.nr*3))
	for r in range(self.nr):
		for comp in range(3):
			for e in range(6):
				synt[:, 3*r+comp] += elemse[r][e][comp].data[0:npts] * self.centroid['a'][e,0]
			d[:, 3*r+comp] = data[r][comp][0:npts]
	np.save(file_d, d)
	np.save(file_synt, synt)
