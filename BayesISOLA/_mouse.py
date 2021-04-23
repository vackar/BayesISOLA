#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import os.path

from BayesISOLA.MouseTrap import *

def detect_mouse(self, mouse_len = 2.5*60, mouse_onset = 1*60, fit_t1=-20, fit_t2c=0, fit_t2v=1200, figures=None, figures_mkdir=True):
	"""
	Wrapper for :class:`MouseTrap`
	
	:param mouse_len: synthetic mouse length in second
	:param mouse_onset: the onset of the synthetic mouse is `mouse_onset` seconds after synthetic mouse starttime
	:param fit_t1: mouse fitting starts this amount of seconds after an event origin time (negative value to start fitting before the origin time)
	:param fit_t2c: mouse fitting endtime -- constant term
	:param fit_t2v: mouse fitting endtime -- linear term (see equation below)
	
	Endtime of fitting is :math:`t_2 = \mathrm{fit\_t2c} + \mathrm{dist} / \mathrm{fit\_t2v}` where :math:`\mathrm{dist}` is station epicentral distance.
	"""
	self.log('\nMouse detection:')
	out = ''
	for st0 in self.data_raw:
		st = st0.copy()
		t_start = max(st[0].stats.starttime, st[1].stats.starttime, st[2].stats.starttime)
		t_start_origin = self.event['t'] - t_start
		paz = st[0].stats.paz
		demean(st, t_start_origin)
		ToDisplacement(st)
		#error = PrepareRecord(st, t_start_origin) # demean, integrate, check signal-to-noise ratio
		#if error:
			#print ('    %s' % error)
		# create synthetic m1
		t_len = min(st[0].stats.endtime, st[1].stats.endtime, st[2].stats.endtime) - t_start
		dt = st[0].stats.delta
		m1 = mouse(fit_time_before = 50, fit_time_after = 60)
		m1.create(paz, int(mouse_len/dt), dt, mouse_onset)
		# fit waveform by synthetic m1
		sta = st[0].stats.station
		for comp in range(3):
			stats = st[comp].stats
			dist = self.stations_index['_'.join([stats.network, sta, stats.location, stats.channel[0:2]])]['dist']
			try:
				m1.fit_mouse(st[comp], t_min=t_start_origin+fit_t1, t_max=t_start_origin+fit_t2c+dist/fit_t2v)
			except:
				out += '  ' + sta + ' ' + stats.channel + ': MOUSE detecting problem (record too short?), ignoring component in inversion\n'
				self.stations_index['_'.join([stats.network, sta, stats.location, stats.channel[0:2]])]['use'+stats.channel[2]] = False
			else:
				onset, amp, dummy, dummy, fit = m1.params(degrees=True)
				amp = abs(amp)
				detected=False
				if (amp > 50e-8 and fit > 0.6) or (amp > 10e-8 and fit > 0.8) or (amp > 7e-8 and fit > 0.9) or (amp > 5e-9 and fit > 0.94) or (fit > 0.985): # DEBUGGING: fit > 0.95 in the before-last parentheses?
					out += '  ' + sta + ' ' + stats.channel + ': MOUSE detected, ignoring component in inversion (time of onset: {o:6.1f} s, amplitude: {a:10.2e} m s^-2, fit: {f:7.2f})\n'.format(o=onset-t_start_origin, a=amp, f=fit)
					self.stations_index['_'.join([stats.network, sta, stats.location, stats.channel[0:2]])]['use'+stats.channel[2]] = False
					detected=True
				if figures:
					if not os.path.exists(figures) and figures_mkdir:
						os.mkdir(figures)
					m1.plot(st[comp], outfile=os.path.join(figures, 'mouse_'+('no','YES')[detected]+'_'+sta+str(comp)+'.png'), xmin=t_start_origin-60, xmax=t_start_origin+240, ylabel='raw displacement [counts]', title="{{net:s}}:{{sta:s}} {{ch:s}}, fit: {fit:4.2f}".format(fit=fit))
	self.logtext['mouse'] = out
	self.log(out, newline=False)
