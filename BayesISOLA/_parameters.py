#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy as np

from BayesISOLA.helpers import lcmm, next_power_of_2

def set_frequencies(self, fmax, fmin=0., wavelengths=5):
	"""
	Sets frequency range for each station according its distance.
	
	:type fmax: float
	:param fmax: minimal inverted frequency for all stations
	:type fmax: float, optional
	:param fmax: maximal inverted frequency for all stations
	:type wavelengths: float, optional
	:param wavelengths: maximal number of wavelengths between the source and the station; if exceeded, automatically decreases ``fmax``
	
	The maximal frequency for each station is determined according to the following formula:
	
	:math:`\min ( f_{max} = \mathrm{wavelengths} \cdot \mathrm{self.s\_velocity} / r, \; fmax )`,
	
	where `r` is the distance the source and the station.
	"""
	for stn in self.d.stations:
		dist = np.sqrt(stn['dist']**2 + self.d.event['depth']**2)
		stn['fmax'] = min(wavelengths * self.s_velocity / dist, fmax)
		stn['fmin'] = fmin
		self.fmax = max(self.fmax, stn['fmax'])

def set_working_sampling(self, multiple8=False):
	"""
	Determine maximal working sampling as at least 8-multiple of maximal inverted frequency (``self.fmax``). If needed, increases the value to eneables integer decimation factor.
	
	:param multiple8: if ``True``, force the decimation factor to be such multiple, that decimation can be done with factor 8 (more times, if needed) and finaly with factor <= 8. The reason for this is decimation pre-filter unstability for higher decimation factor (now not needed).
	:type multiple8: bool, optional
	"""
	#min_sampling = 4 * self.fmax
	min_sampling = 8 * self.fmax # teoreticky 4*fmax aby fungovala L2 norma????
	SAMPRATE = 1. / lcmm(*self.d.data_deltas) # kazda stanice muze mit jine vzorkovani, bereme nejvetsiho spolecneho delitele (= 1. / nejmensi spolecny nasobek)
	decimate = SAMPRATE / min_sampling
	if multiple8:
		if decimate > 128:
			decimate = int(decimate/64) * 64
		elif decimate > 16:
			decimate = int(decimate/8) * 8
		else:
			decimate = int(decimate)
	else:
		decimate = int(decimate)
	self.max_samprate = SAMPRATE
	# print(min_sampling, SAMPRATE, decimate) # DEBUG
	# print(self.d.data_deltas) # DEBUG
	self.samprate = SAMPRATE / decimate
	self.logtext['samplings'] = samplings_str = ", ".join(["{0:5.1f} Hz".format(1./delta) for delta in  self.d.data_deltas])
	self.log('\nSampling frequencies:\n  Data sampling: {0:s}\n  Common sampling: {3:5.1f}\n  Decimation factor: {1:3d} x\n  Sampling used: {2:5.1f} Hz'.format(samplings_str, decimate, self.samprate, SAMPRATE))

def count_components(self, log=True):
	"""
	Counts number of components, which should be used in inversion (e.g. ``self.d.stations[n]['useZ'] = True`` for `Z` component). This is needed for allocation proper size of matrices used in inversion.
	
	:param log: if true, write into log table of stations and components with information about component usage and weight
	:type log: bool, optional
	"""
	c = 0
	stn = self.d.stations
	for r in range(self.d.nr):
		if stn[r]['useZ']: c += 1
		if stn[r]['useN']: c += 1
		if stn[r]['useE']: c += 1
	self.components = c
	if log:
		out = '\nComponents used in inversion and their weights\nstation     \t   \t Z \t N \t E \tdist\tazimuth\tfmin\tfmax\n            \t   \t   \t   \t   \t(km)    \t(deg)\t(Hz)\t(Hz)\n'
		for r in range(self.d.nr):
			out += '{net:>3s}:{sta:5s} {loc:2s}\t{ch:2s} \t'.format(sta=stn[r]['code'], net=stn[r]['network'], loc=stn[r]['location'], ch=stn[r]['channelcode'])
			for c in range(3):
				if stn[r][self.idx_use[c]]:
					out += '{0:3.1f}\t'.format(stn[r][self.idx_weight[c]])
				else:
					out += '---\t'
			if stn[r]['dist'] > 2000:
				out += '{0:4.0f}    '.format(stn[r]['dist']/1e3)
			elif stn[r]['dist'] > 200:
				out += '{0:6.1f}  '.format(stn[r]['dist']/1e3)
			else:
				out += '{0:8.3f}'.format(stn[r]['dist']/1e3)
			out += '\t{2:3.0f}\t{0:4.2f}\t{1:4.2f}'.format(stn[r]['fmin'], stn[r]['fmax'], stn[r]['az'])
			out += '\n'
		self.logtext['components'] = out
		self.log(out, newline=False)

def min_time(self, distance, mag=0, v=8000):
	"""
	Defines the beginning of inversion time window in seconds from location origin time. Save it into ``self.t_min`` (now save 0 -- FIXED OPTION)
	
	:param distance: station distance in meters
	:type distance: float
	:param mag: magnitude (unused)
	:param v: the first inverted wave-group characteristic velocity in m/s
	:type v: float
	
	Sets ``self.t_min`` as minimal time of interest (in seconds).
	"""
	#t = distance/v		# FIXED OPTION
	##if t<5:
		##t = 0
	#self.t_min = t
	self.t_min = 0		# FIXED OPTION, because Green's functions with beginning in non-zero time are nou implemented yet

def max_time(self, distance, mag=0, v=1000):
	"""
	Defines the end of inversion time window in seconds from location origin time. Calculates it as :math:`\mathrm{distance} / v`.
	Save it into ``self.t_max``.
	
	:param distance: station distance in meters
	:type distance: float
	:param mag: magnitude (unused)
	:param v: the last inverted wave-group characteristic velocity in m/s
	:type v: float
	"""
	t = distance/v		# FIXED OPTION
	self.t_max = t

def set_time_window(self):
	"""
	Determines number of samples for inversion (``self.npts_slice``) and for Green's function calculation (``self.npts_elemse`` and ``self.npts_exp``) from ``self.min_time`` and ``self.max_time``.
	
	:math:`\mathrm{npts\_slice} \le \mathrm{npts\_elemse} = 2^{\mathrm{npts\_exp}} < 2\cdot\mathrm{npts\_slice}`
	"""
	self.min_time(np.sqrt(self.d.stations[0]['dist']**2+self.grid.depth_min**2))
	self.max_time(np.sqrt(self.d.stations[self.d.nr-1]['dist']**2+self.grid.depth_max**2))
	#self.t_min -= 20 # FIXED OPTION
	self.t_min = round(self.t_min * self.samprate) / self.samprate
	if self.t_min > 0:
		self.t_min = 0.
	self.elemse_start_origin = -self.t_min
	self.t_len = self.t_max - self.t_min
	self.npts_slice  =                 int(math.ceil(self.t_max * self.samprate))
	self.npts_elemse = next_power_of_2(int(math.ceil(self.t_len * self.samprate)))
	if self.npts_elemse < 64:		# FIXED OPTION
		self.npts_exp = 6
		self.npts_elemse = 64
	else:
		self.npts_exp = int(math.log(self.npts_elemse, 2))

def set_parameters(self, fmax, fmin=0., wavelengths=5, min_depth=1000, log=True):
	"""
	Sets some technical parameters of the inversion.
	
	Technically, just runs following functions:
		- :func:`set_frequencies`
		- :func:`set_working_sampling`
		- :func:`set_grid`
		- :func:`set_time_grid`
		- :func:`set_time_window`
		- :func:`set_Greens_parameters`
		- :func:`count_components`
	
	The parameters are parameters of the same name of these functions.
	"""
	self.set_frequencies(fmax, fmin, wavelengths)
	self.set_working_sampling()
	self.grid.set_grid() # must be after set_working_sampling
	self.grid.set_time_grid(self.fmax, self.max_samprate)
	self.set_time_window()
	self.set_Greens_parameters()
	self.count_components(log)

def skip_short_records(self, noise=False):
	"""
	Checks whether all records are long enough for the inversion and skips unsuitable ones.
	
	:parameter noise: checks also whether the record is long enough for generating the noise slice for the covariance matrix (if the value is ``True``, choose minimal noise length automatically; if it's numerical, take the value as minimal noise length)
	:type noise: bool or float, optional
	"""
	self.log('\nChecking record length:')
	for st in self.d.data_raw:
		for comp in range(3):
			stats = st[comp].stats
			if stats.starttime > self.d.event['t'] + self.t_min + self.grid.shift_min or stats.endtime < self.d.event['t'] + self.t_max + self.grid.shift_max:
				self.log('  ' + stats.station + ' ' + stats.channel + ': record too short, ignoring component in inversion')
				self.d.stations_index['_'.join([stats.network, stats.station, stats.location, stats.channel[0:2]])]['use'+stats.channel[2]] = False
			if noise:
				if type(noise) in (float,int):
					noise_len = noise
				else:
					noise_len = (self.t_max - self.t_min + self.grid.shift_max + 10)*1.1 - self.grid.shift_min - self.t_min
					#print stats.station, stats.channel, noise_len, '>', self.d.event['t']-stats.starttime # DEBUG
				if stats.starttime > self.d.event['t'] - noise_len:
					self.log('  ' + stats.station + ' ' + stats.channel + ': record too short for noise covariance, ignoring component in inversion')
					self.d.stations_index['_'.join([stats.network, stats.station, stats.location, stats.channel[0:2]])]['use'+stats.channel[2]] = False
