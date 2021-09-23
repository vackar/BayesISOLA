#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy as np

def set_grid(self, min_depth=1000):
	"""
	Generates grid ``self.grid`` of points, where the inverse problem will be solved.
	`Rupture length` is estimated as :math:`111 \cdot 10^{M_W}`.
	Horizontal diameter of the grid is determined as ``self.location_unc`` + `rupture_length`.
	Vertical half-size of the grid is ``self.depth_unc`` + `rupture_length`.
	If ``self.circle_shape`` is ``True``, the shape of the grid is cylinder, otherwise it is rectangular box.
	The horizontal grid spacing is defined by ``self.step_x``, the vertical by ``self.step_z``. If it leads to more grid points than ``self.max_points``, the both spacings are increased (both by the same ratio) to approximately fit ``self.max_points``.
	
	:parameter min_depth: minimal grid point depth in meters
	:type min_depth: float, optional
	"""
	step_x = self.step_x; step_z = self.step_z; max_points = self.max_points
	rupture_length = self.data.rupture_length
	if self.grid_radius:
		self.radius = radius = self.grid_radius
	else:
		self.radius = radius = self.location_unc + rupture_length
	if self.grid_min_depth:
		self.depth_min = depth_min = max(min_depth, self.grid_min_depth)
	else:
		self.depth_min = depth_min = max(min_depth, self.data.event['depth'] - self.depth_unc - rupture_length)
	if self.grid_max_depth:
		self.depth_max = depth_max = self.grid_max_depth
	else:
		self.depth_max = depth_max = self.data.event['depth'] + self.depth_unc + rupture_length
	n_points = np.pi*(radius/step_x)**2 * (depth_max-depth_min)/step_z
	if n_points > max_points:
		step_x *= (n_points/max_points)**0.333
		step_z *= (n_points/max_points)**0.333
	n_steps = int(radius/step_x)
	n_steps_z = int((self.depth_unc + rupture_length)/step_z)
	depths = []
	for k in range(-n_steps_z, n_steps_z+1):
		z = self.data.event['depth']+k*step_z
		if z >= depth_min and z <= depth_max:
			depths.append(z)
	self.grid = []
	self.steps_x = []
	for i in range(-n_steps, n_steps+1):
		x = i*step_x
		self.steps_x.append(x)
		for j in range(-n_steps, n_steps+1):
			y = j*step_x
			if math.sqrt(x**2+y**2) > radius and self.circle_shape:
				continue
			for z in depths:
				edge = z==depths[0] or z==depths[-1] or (math.sqrt((abs(x)+step_x)**2+y**2) > radius or math.sqrt((abs(y)+step_x)**2+x**2) > radius) and self.circle_shape or max(abs(i),abs(j))==n_steps
				self.grid.append({'x':x, 'y':y, 'z':z, 'err':0, 'edge':edge})
	self.depths = depths
	self.step_x = step_x; self.step_z = step_z
	self.data.log('\nGrid parameters:\n  number of points: {0:4d}\n  horizontal step: {1:5.0f} m\n  vertical step: {2:5.0f} m\n  grid radius: {3:6.3f} km\n  minimal depth: {4:6.3f} km\n  maximal depth: {5:6.3f} km\nEstimated rupture length: {6:6.3f} km'.format(len(self.grid), step_x, step_z, radius/1e3, depth_min/1e3, depth_max/1e3, rupture_length/1e3))

def set_time_grid(self, fmax, max_samprate):
	"""
	:parameter fmax: Higher range of bandpass filter for data.
	:type fmax: float
	:parameter max_samprate: Maximal sampling rate of the source data, which can be reached by integer decimation from all input samplings.
	:type max_samprate: float
	
	Sets equidistant time grid defined by ``self.shift_min``, ``self.shift_max``, and ``self.shift_step`` (in seconds). The corresponding values ``self.SHIFT_min``, ``self.SHIFT_max``, and ``self.SHIFT_step`` are (rounded) in samples related to the the highest sampling rate common to all stations.
	"""
	if self.grid_max_time:
		self.shift_max  = shift_max  = self.grid_max_time
	else:
		self.shift_max  = shift_max  = self.time_unc + self.data.rupture_length / self.rupture_velocity
	if self.grid_min_time:
		self.shift_min  = shift_min  = self.grid_min_time
	else:
		self.shift_min  = shift_min  = -self.time_unc
	self.shift_step = shift_step = 1./fmax * 0.01
	self.SHIFT_min = int(round(shift_min*max_samprate))
	self.SHIFT_max = int(round(shift_max*max_samprate))
	self.SHIFT_step = max(int(round(shift_step*max_samprate)), 1) # max() to avoid step beeing zero
	self.SHIFT_min = int(round(self.SHIFT_min / self.SHIFT_step)) * self.SHIFT_step # shift the grid to contain zero time shift
	self.data.log('\nGrid-search over time:\n  min = {sn:5.2f} s ({Sn:3d} samples)\n  max = {sx:5.2f} s ({Sx:3d} samples)\n  step = {step:4.2f} s ({STEP:3d} samples)'.format(sn=shift_min, Sn=self.SHIFT_min, sx=shift_max, Sx=self.SHIFT_max, step=shift_step, STEP=self.SHIFT_step))
