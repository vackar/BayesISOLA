#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import multiprocessing as mp
import numpy as np
from pyproj import Geod

from BayesISOLA.inverse_problem import invert
 
def run_inversion(self):
	"""
	Runs function :func:`invert` in parallel.
	
	Module :class:`multiprocessing` does not allow running function of the same class in parallel, so the function :func:`invert` cannot be method of class :class:`ISOLA` and this wrapper is needed.
	"""
	grid = self.grid
	todo = []
	for i in range (len(grid)):
		point_id = str(i).zfill(4)
		grid[i]['id'] = point_id
		if not grid[i]['err']:
			todo.append(i)
	
	# create norm_d[shift]
	norm_d = []
	for shift in range(len(self.d_shifts)):
		d_shift = self.d_shifts[shift]
		if self.Cd_inv_shifts:  # ACF
			self.Cd_inv = self.Cd_inv_shifts[shift]
		if self.Cd_inv:
			idx = 0
			dCd_blocks = []
			for C in self.Cd_inv:
				size = len(C)
				dCd_blocks.append(np.dot(d_shift[idx:idx+size, : ].T, C))
				idx += size
			dCd   = np.concatenate(dCd_blocks,   axis=1)
			norm_d.append(np.dot(dCd, d_shift)[0,0])
		else:
			norm_d.append(0)
			for i in range(self.npts_slice*self.components):
				norm_d[-1] += d_shift[i,0]*d_shift[i,0]
	
	if self.threads > 1: # parallel
		pool = mp.Pool(processes=self.threads)
		results = [pool.apply_async(invert, args=(grid[i]['id'], self.d_shifts, norm_d, self.Cd_inv, self.Cd_inv_shifts, self.nr, self.components, self.stations, self.npts_elemse, self.npts_slice, self.elemse_start_origin, self.deviatoric, self.decompose, self.invert_displacement)) for i in todo]
		output = [p.get() for p in results]
	else: # serial
		output = []
		for i in todo:
			res = invert(grid[i]['id'], self.d_shifts, norm_d, self.Cd_inv, self.Cd_inv_shifts, self.nr, self.components, self.stations, self.npts_elemse, self.npts_slice, self.elemse_start_origin, self.deviatoric, self.decompose, self.invert_displacement)
			output.append(res)
	min_misfit = output[0]['misfit']
	for i in todo:
		grid[i].update(output[todo.index(i)])
		grid[i]['shift_idx'] = grid[i]['shift']
		#grid[i]['shift'] = self.shift_min + grid[i]['shift']*self.SHIFT_step/self.max_samprate
		grid[i]['shift'] = self.shifts[grid[i]['shift']]
		min_misfit = min(min_misfit, grid[i]['misfit'])
	self.max_sum_c = self.max_c = self.sum_c = 0
	for i in todo:
		gp = grid[i]
		gp['sum_c'] = 0
		for idx in gp['shifts']:
			GP = gp['shifts'][idx]
			GP['c'] = np.sqrt(gp['det_Ca']) * np.exp(-0.5 * (GP['misfit']-min_misfit))
			gp['sum_c'] += GP['c']
		gp['c'] = gp['shifts'][gp['shift_idx']]['c']
		self.sum_c += gp['sum_c']
		self.max_c = max(self.max_c, gp['c'])
		self.max_sum_c = max(self.max_sum_c, gp['sum_c'])

def find_best_grid_point(self):
	"""
	Set ``self.centroid`` to a grid point with higher variance reduction -- the best solution of the inverse problem.
	"""
	self.centroid = max(self.grid, key=lambda g: g['VR']) # best grid point
	x = self.centroid['x']; y = self.centroid['y']
	az = np.degrees(np.arctan2(y, x))
	dist = np.sqrt(x**2 + y**2)
	g = Geod(ellps='WGS84')
	self.centroid['lon'], self.centroid['lat'], baz = g.fwd(self.event['lon'], self.event['lat'], az, dist)
