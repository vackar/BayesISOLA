#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
 
from BayesISOLA.helpers import my_filter, next_power_of_2

def correct_data(self):
	"""
	Corrects ``self.data_raw`` for the effect of instrument. Poles and zeros must be contained in trace stats.
	"""
	for st in self.data_raw:
		st.detrend(type='demean')
		st.filter('highpass', freq=0.01) # DEBUG
		for tr in st:
			#tr2 = tr.copy() # DEBUG
			if getattr(tr.stats, 'response', 0):
				tr.remove_response(output="VEL")
			else:
				tr.simulate(paz_remove=tr.stats.paz)
	# 2DO: add prefiltering etc., this is not the best way for the correction
	# 	see http://docs.obspy.org/packages/autogen/obspy.core.trace.Trace.remove_response.html
	self.data_are_corrected = True

def trim_filter_data(self, noise_slice=True, noise_starttime=None, noise_length=None):
	"""
	Filter ``self.data_raw`` using function :func:`prefilter_data`.
	Decimate ``self.data_raw`` to common sampling rate ``self.max_samprate``.
	Optionally, copy a time window for the noise analysis.
	Copy a slice to ``self.data``.
	
	:type starttime: :class:`~obspy.core.utcdatetime.UTCDateTime`
	:param starttime: Specify the start time of trimmed data
	:type length: float
	:param length: Length in seconds of trimmed data.
	:type noise_slice: bool, optional
	:param noise_slice: If set to ``True``, copy a time window of the length ``lenght`` for later noise analysis. Copied noise is in ``self.noise``.
	:type noise_starttime: :class:`~obspy.core.utcdatetime.UTCDateTime`, optional
	:param noise_starttime: Set the starttime of the noise time window. If ``None``, the time window starts in time ``starttime``-``length`` (in other words, it lies just before trimmed data time window).
	:type noise_length: :class:`~obspy.core.utcdatetime.UTCDateTime`, optional
	:param noise_length: Length of the noise time window (in seconds).
	"""

	starttime = self.event['t']+self.shift_min+self.t_min
	length = self.t_max - self.t_min + self.shift_max + 10
	endtime = starttime + length
	if noise_slice:
		if not noise_length:
			noise_length = length*4
		if not noise_starttime:
			noise_starttime = starttime - noise_length
			noise_endtime = starttime
		else:
			noise_endtime = noise_starttime + noise_length
		DECIMATE = int(round(self.max_samprate / self.samprate))

	for st in self.data_raw:
		stats = st[0].stats
		fmax = self.stations_index['_'.join([stats.network, stats.station, stats.location, stats.channel[0:2]])]['fmax']
		self.data.append(st.copy())
	for st in self.data:
		stats = st[0].stats
		fmin = self.stations_index['_'.join([stats.network, stats.station, stats.location, stats.channel[0:2]])]['fmin']
		fmax = self.stations_index['_'.join([stats.network, stats.station, stats.location, stats.channel[0:2]])]['fmax']
		decimate = int(round(st[0].stats.sampling_rate / self.max_samprate))
		if noise_slice:
			self.noise.append(st.slice(noise_starttime, noise_endtime))
			#print self.noise[-1][0].stats.endtime-self.noise[-1][0].stats.starttime, '<', length*1.1 # DEBUG
			if (len(self.noise[-1])!=3 or (self.noise[-1][0].stats.endtime-self.noise[-1][0].stats.starttime < length*1.1)) and self.stations_index['_'.join([stats.network, stats.station, stats.location, stats.channel[0:2]])]['use'+stats.channel[2]]:
				self.log('Noise slice too short to generate covariance matrix (station '+st[0].stats.station+'). Stopping generating noise slices.')
				noise_slice = False
				self.noise = []
			elif len(self.noise[-1]):
				my_filter(self.noise[-1], fmin/2, fmax*2)
				self.noise[-1].decimate(int(decimate*DECIMATE/2), no_filter=True) # noise has 2-times higher sampling than data
		self.prefilter_data(st)
		st.decimate(decimate, no_filter=True)
		st.trim(starttime, endtime)
	# TODO: kontrola, jestli neorezavame mimo puvodni zaznam

def prefilter_data(self, st):
	"""
	Drop frequencies above Green's function computation high limit using :func:`numpy.fft.fft`.
	
	:param st: stream to be filtered
	:type st: :class:`~obspy.core.stream`
	"""
	f = self.freq / self.tl
	for tr in st:
		npts = tr.stats.npts
		NPTS = next_power_of_2(npts)
		TR = np.fft.fft(tr.data,NPTS)
		df = tr.stats.sampling_rate / NPTS
		flim = int(np.ceil(f/df))
		TR[flim:NPTS-flim+1] = 0+0j
		tr_filt = np.fft.ifft(TR)
		tr.data = np.real(tr_filt[0:npts])

def decimate_shift(self):
	"""
	Generate ``self.data_shifts`` where are multiple copies of ``self.data`` (needed for plotting).
	Decimate ``self.data_shifts`` to sampling rate for inversion ``self.samprate``.
	Filter ``self.data_shifts`` by :func:`my_filter`.
	Generate ``self.d_shifts`` where are multiple vectors :math:`d`, each of them shifted according to ``self.SHIFT_min``, ``self.SHIFT_max``, and ``self.SHIFT_step``
	"""
	self.d_shifts = []
	self.data_shifts = []
	self.shifts = []
	starttime = self.event['t']# + self.t_min
	length = self.t_max-self.t_min
	endtime = starttime + length
	decimate = int(round(self.max_samprate / self.samprate))
	for SHIFT in range(self.SHIFT_min, self.SHIFT_max+1, self.SHIFT_step):
		#data = deepcopy(self.data)
		shift = SHIFT / self.max_samprate
		self.shifts.append(shift)
		data = []
		for st in self.data:
			st2 = st.slice(starttime+shift-self.elemse_start_origin, endtime+shift+1) # we add 1 s to be sure, that no index will point outside the range
			st2.trim(starttime+shift-self.elemse_start_origin, endtime+shift+1, pad=True, fill_value=0.) # short records are not inverted, but they should by padded because of plotting
			if self.invert_displacement:
				st2.detrend('linear')
				st2.integrate()
			st2.decimate(decimate, no_filter=True)
			stats = st2[0].stats
			stn = self.stations_index['_'.join([stats.network, stats.station, stats.location, stats.channel[0:2]])]
			fmin = stn['fmin']
			fmax = stn['fmax']
			if stn['accelerograph']:
				st2.integrate()
			my_filter(st2, fmin, fmax)
			st2.trim(starttime+shift, endtime+shift+1) # we add 1 s to be sure, that no index will point outside the range
			data.append(st2)
		self.data_shifts.append(data)
		c = 0
		d_shift = np.empty((self.components*self.npts_slice, 1))
		for r in range(self.nr):
			for comp in range(3):
				if self.stations[r][{0:'useZ', 1:'useN', 2:'useE'}[comp]]: # this component has flag 'use in inversion'
					weight = self.stations[r][{0:'weightZ', 1:'weightN', 2:'weightE'}[comp]]
					for i in range(self.npts_slice):
						try:
							d_shift[c*self.npts_slice+i] = data[r][comp].data[i] * weight
						except:
							self.log('Index out of range while generating shifted data vectors. Waveform file probably too short.', printcopy=True)
							print('values for debugging: ', r, comp, c, self.npts_slice, i, c*self.npts_slice+i, len(d_shift), len(data[r][comp].data), SHIFT)
							raise Exception('Index out of range while generating shifted data vectors. Waveform file probably too short.')
					c += 1
		self.d_shifts.append(d_shift)
