#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import math
#from math import degrees,floor
import numpy as np
from numpy import matrix,array
import shutil
import multiprocessing as mp
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
warnings.filterwarnings('ignore', '.*Conversion of the second argument of issubdtype from.*') # Avoids following warning:
		# /usr/lib/python3.7/site-packages/obspy/signal/detrend.py:31: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
from pyproj import Geod # transformation of geodetic coordinates
import os.path
#from scipy import signal

#import matplotlib as mpl

import obspy
from obspy import UTCDateTime
#import obspy.imaging.scripts.mopad as mopad

#from obspy.signal.cross_correlation import xcorr   # There was some problem with small numbers

#from nearest_correlation import nearcorr # nearest correlation matrix

from MouseTrap import *
from functions import *
from fileformats import *
from inverse_problem import invert
from MT_comps import *
from axitra import Axitra_wrapper


class ISOLA:
	"""
    Class for moment tensor inversion.

    :param location_unc: horizontal uncertainty of the location in meters (default 0)
    :type location_unc: float, optional
    :type depth_unc: float, optional
    :param depth_unc: vertical (depth) uncertainty of the location in meters (default 0)
    :type time_unc: float, optional
    :param time_unc: uncertainty of the origin time in seconds (default 0)
    :type deviatoric: bool, optional
    :param deviatoric: if ``False``: invert full moment tensor (6 components); if ``True`` invert deviatoric part of the moment tensor (5 components) (default ``False``)
    :type step_x: float, optional
    :param step_x: preferred horizontal grid spacing in meter (default 500 m)
    :type step_z: float, optional
    :param step_z: preferred vertical grid spacing in meter (default 500 m)
    :type max_points: integer, optional
    :param max_points: maximal (approximate) number of the grid points (default 100)
    :type grid_radius: float, optional
    :param grid_radius: if non-zero, manually sets radius of the cylinrical grid or half of the edge-length of rectangular grid, in meters (default 0)
    :type grid_min_depth: float, optional
    :param grid_min_depth: if non-zero, manually sets minimal depth of the grid, in meters (default 0)
    :type grid_max_depth: float, optional
    :param grid_max_depth: if non-zero, manually sets maximal depth of the grid, in meters (default 0)
    :type grid_min_time: float, optional
    :param grid_min_time: if non-zero, manually sets minimal time-step of the time grid, in seconds with respect to origin time, negative values for time before epicentral time (default 0)
    :type grid_max_time: float, optional
    :param grid_max_time: if non-zero, manually sets maximal time-step of the time grid, in seconds with respect to origin time (default 0)
    :type threads: integer, optional
    :param threads: number of threads for parallelization (default 2)
    :type invert_displacement: bool, optional
    :param invert_displacement: convert observed and modeled waveforms to displacement prior comparison (if ``True``), otherwise compare it in velocity (default ``False``)
    :type circle_shape: bool, optional
    :param circle_shape: if ``True``, the shape of the grid is cylinder, otherwise it is rectangular box (default ``True``)
    :type use_precalculated_Green: bool, optional
    :param use_precalculated_Green: use Green's functions calculated in the previous run (default ``False``)
    :type rupture_velocity: float, optional
    :param rupture_velocity: rupture propagation velocity in m/s, used for estimating the difference between the origin time and the centroid time (default 1000 m/s)
    :type decompose: bool, optional
    :param decompose: performs decomposition of the found moment tensor in each grid point
    :type s_velocity: float, optional
    :param s_velocity: characteristic S-wave velocity used for calculating number of wave lengths between the source and stations (default 3000 m/s)
    :type logfile: string, optional
    :param logfile: path to the logfile (default '$outdir/log.txt')
    :type outdir: string, optional
    :param outdir: a directory, where the outputs are saved (default 'output')
	
    .. rubric:: _`Variables`
    
    The following description is useful mostly for developers. The parameters from the `Parameters` section are not listed, but they become also variables of the class. Two of them (``step_x`` and ``step_z``) are changed at some point, the others stay intact.

    ``data`` : list of :class:`~obspy.core.stream`
        Prepared data for the inversion. It's filled by function :func:`add_NEZ` or :func:`trim_filter_data`. The list is ordered ascending by epicentral distance of the station.
    ``data_raw`` : list of :class:`~obspy.core.stream`
        Data for the inversion. They are loaded by :func:`add_SAC`, :func:`load_files`, or :func:`load_streams_ArcLink`. Then they are corrected by :func:`correct_data` and trimmed by :func:`trim_filter_data`. The list is ordered ascending by epicentral distance of the station. After processing, the list references to the same streams as ``data``.
    ``data_unfiltered`` : list of :class:`~obspy.core.stream`
        The copy of the ``data`` before it is filtered. Used for plotting results only.
    ``noise`` : list of :class:`~obspy.core.stream`
        Before-event slice of ``data_raw`` for later noise analysis. Created by :func:`trim_filter_data`.
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
    ``fmin`` : float
        Lower range of bandpass filter for data.
    ``fmax`` : float
        Higher range of bandpass filter for data.
    ``data_deltas`` : list of floats
        List of ``stats.delta`` from ``data_raw`` and ``data``.
    ``data_are_corrected`` : bool
        Flag whether the instrument response correction was performed.
    ``event`` : dictionary
        Information about event location and magnitude.
    ``stations`` : list of dictionaries
        Information about stations used in inversion. The list is ordered ascending by epicentral distance.
    ``stations_index`` : dictionary referencing ``station`` items.
        You can also access station information like ``self.stations_index['NETWORK_STATION_LOCATION_CHANNELCODE']['dist']`` insted of ``self.stations[5]['dist']``. `CHANNELCODE` are the first two letters of channel, e.g. `HH`.
    ``nr`` : integer
        Number of stations used, i.e. length of ``stations`` and ``data``.
    ``samprate`` : float
        Sampling rate used in the inversion.
    ``max_samprate`` : float
        Maximal sampling rate of the source data, which can be reached by integer decimation from all input samplings.
    ``t_min`` : float
        Starttime of the inverted time window, in seconds from the origin time.
    ``t_max`` :  float
        Endtime of the inverted time window, in seconds from the origin time.
    ``t_len`` : float
        Length of the inverted time window, in seconds (``t_max``-``t_min``).
    ``npts_elemse`` : integer
        Number of elementary seismogram data points (time series for one component).
    ``npts_slice`` : integer
        Number of data points for one component of one station used in inversion :math:`\mathrm{npts_slice} \le \mathrm{npts_elemse}`.
    ``tl`` : float
        Time window length used in the inversion.
    ``freq`` : integer
        Number of frequencies calculated when creating elementary seismograms.
    ``xl`` : float
        Parameter ``xl`` for `Axitra` code.
    ``npts_exp`` : integer
        :math:`\mathrm{npts_elemse} = 2^\mathrm{npts_exp}`
    ``grid`` : list of dictionaries
        Spatial grid on which is the inverse where is solved.
    ``centroid`` : Reference to ``grid`` item.
        The best grid point found by the inversion.
    ``depths`` : list of floats
        List of grid-points depths.
    ``radius`` : float
        Radius of grid cylinder / half of rectangular box horizontal edge length (depending on grid shape). Value in meters.
    ``depth_min`` : float
        The lowest grid-poind depth (in meters).
    ``depth_max`` : float
        The highest grid-poind depth (in meters).
    ``shift_min``, ``shift_max``, ``shift_step`` : 
        Three variables controling the time grid. The names are self-explaining. Values in second.
    ``SHIFT_min``, ``SHIFT_max``, ``SHIFT_step`` : 
        The same as the previous ones, but values in samples related to ``max_samprate``.
    ``components`` : integer
        Number of components of all stations used in the inversion. Created by :func:`count_components`.
    ``data_shifts`` : list of lists of :class:`~obspy.core.stream`
        Shifted and trimmed ``data`` ready.
    ``d_shifts`` : list of :class:`~numpy.ndarray`
        The previous one in form of data vectors ready for the inversion.
    ``shifts`` : list of floats
        Shift values in seconds. List is ordered in the same way as the two previous list.
    ``mt_decomp`` : list
        Decomposition of the best centroid moment tensor solution calculated by :func:`decompose` or :func:`decompose_mopad`
    ``max_VR`` : tuple (VR, n)
        Variance reduction `VR` from `n` components of a subset of the closest stations
    ``logtext`` : dictionary
        Status messages for :func:`html_log`
    ``models`` : dictionary
        Crust models used for calculating synthetic seismograms
    ``rupture_length`` : float
        Estimated length of the rupture in meters
	"""

	from _input_crust import read_crust
	from _input_event import read_event_info, set_event_info
	from _input_network import read_network_info_DB, read_network_coordinates, create_station_index, write_stations
	from _input_seismo_files import add_NEZ, add_SAC, add_NIED, load_files, load_NIED_files, check_a_station_present
	from _input_seismo_remote import load_streams_ArcLink
	from _covariance_matrix import covariance_matrix, covariance_matrix_SACF, covariance_matrix_ACF
	from _plot import plot_MT, plot_uncertainty, plot_MT_uncertainty_centroid, plot_maps, plot_slices, plot_maps_sum, plot_map_backend, plot_3D, plot_seismo, plot_covariance_function, plot_noise, plot_spectra, plot_seismo_backend_1, plot_seismo_backend_2, plot_stations, plot_covariance_matrix
	from _html import html_log

	def __init__(self, location_unc=0, depth_unc=0, time_unc=0, deviatoric=False, step_x=500, step_z=500, max_points=100, grid_radius=0, grid_min_depth=0, grid_max_depth=0, grid_min_time=0, grid_max_time=0, threads=2, invert_displacement=False, circle_shape=True, use_precalculated_Green=False, rupture_velocity=1000, s_velocity=3000, decompose=True, outdir='output', logfile='$outdir/log.txt'):
		self.location_unc = location_unc # m
		self.depth_unc = depth_unc # m
		self.time_unc = time_unc # s
		self.deviatoric = deviatoric
		self.step_x = step_x # m
		self.step_z = step_z # m
		self.max_points = max_points
		self.grid_radius =    grid_radius # m
		self.grid_min_depth = grid_min_depth # m
		self.grid_max_depth = grid_max_depth # m
		self.grid_min_time =  grid_min_time # s
		self.grid_max_time =  grid_max_time # s
		self.threads = threads
		self.invert_displacement = invert_displacement
		self.circle_shape = circle_shape
		self.use_precalculated_Green = use_precalculated_Green
		self.rupture_velocity = rupture_velocity
		self.s_velocity = s_velocity
		self.decompose = decompose
		self.outdir = outdir
		self.logfile = open(logfile.replace('$outdir', self.outdir), 'w', 1)
		self.data = []
		self.data_raw = []
		#self.data_unfiltered = []
		self.noise = []
		self.Cd_inv = []
		self.Cd = []
		self.LT = []
		self.LT3 = []
		self.Cf = []
		self.Cd_inv_shifts = []
		self.Cd_shifts = []
		self.LT_shifts = []
		self.fmax = 0.
		self.data_deltas = [] # list of ``stats.delta`` values of traces in ``self.data`` or ``self.data_raw``
		self.mt_decomp = []
		self.max_VR = ()
		self.logtext = {}
		self.idx_use = {0:'useZ', 1:'useN', 2:'useE'}
		self.idx_weight = {0:'weightZ', 1:'weightN', 2:'weightE'}
		self.movie_writer = 'mencoder' # None for default
		self.models = {}
		
		self.log('Inversion of ' + {1:'deviatoric part of', 0:'full'}[self.deviatoric] + ' moment tensor (' + {1:'5', 0:'6'}[self.deviatoric] + ' components)')
		
	def __exit__(self, exc_type, exc_value, traceback):
		self.__del__()
		
	def __del__(self):
		self.logfile.close()
		del self.data
		del self.data_raw
		#del self.data_unfiltered
		del self.noise
		del self.Cd_inv
		del self.Cd
		del self.LT
		del self.LT3
		del self.Cf
		del self.Cd_inv_shifts
		del self.Cd_shifts
		del self.LT_shifts
		
	def log(self, s, newline=True, printcopy=False):
		"""
		Write text into log file
		
		:param s: Text to write into log
		:type s: string
		:param newline: if is ``True``, add LF symbol (\\\\n) at the end
		:type newline: bool, optional
		:param printcopy: if is ``True`` prints copy of ``s`` also to stdout
		:type printcopy: bool, optional
		"""
		self.logfile.write(s)
		if newline:
			self.logfile.write('\n')
		if printcopy:
			print(s)
	
	

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
		for stn in self.stations:
			dist = np.sqrt(stn['dist']**2 + self.event['depth']**2)
			stn['fmax'] = min(wavelengths * self.s_velocity / dist, fmax)
			stn['fmin'] = fmin
			self.fmax = max(self.fmax, stn['fmax'])
		#self.count_components()

	def count_components(self, log=True):
		"""
		Counts number of components, which should be used in inversion (e.g. ``self.stations[n]['useZ'] = True`` for `Z` component). This is needed for allocation proper size of matrices used in inversion.
		
		:param log: if true, write into log table of stations and components with information about component usage and weight
		:type log: bool, optional
		"""
		c = 0
		stn = self.stations
		for r in range(self.nr):
			if stn[r]['useZ']: c += 1
			if stn[r]['useN']: c += 1
			if stn[r]['useE']: c += 1
		self.components = c
		if log:
			out = '\nComponents used in inversion and their weights\nstation     \t   \t Z \t N \t E \tdist\tazimuth\tfmin\tfmax\n            \t   \t   \t   \t   \t(km)    \t(deg)\t(Hz)\t(Hz)\n'
			for r in range(self.nr):
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
				## vykreslime - DEBUG
				#if tr.stats.station == 'LAKA':
					#tr.plot()
					#tr2.plot()
					#t = np.arange(0, tr2.stats.npts / tr2.stats.sampling_rate, 1 / tr2.stats.sampling_rate)
					#fig, ax1 = plt.subplots()
					#ax2 = plt.twinx()
					#ax1.plot(t, tr2.data, 'k', label='orig')
					#ax2.plot(t, tr.data, 'r', label='corr')
					#plt.legend(loc='upper left')
					#ax1.set_ylabel('orig')
					#ax2.set_ylabel('corr')
					#plt.title(tr.stats.station + ' ' + tr.stats.channel)
					#plt.show()
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
	
	def evalute_noise(self):
		# compare spectrum of the signal and the noise
		pass
	
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


	def set_working_sampling(self, multiple8=False):
		"""
		Determine maximal working sampling as at least 8-multiple of maximal inverted frequency (``self.fmax``). If needed, increases the value to eneables integer decimation factor.
		
		:param multiple8: if ``True``, force the decimation factor to be such multiple, that decimation can be done with factor 8 (more times, if needed) and finaly with factor <= 8. The reason for this is decimation pre-filter unstability for higher decimation factor (now not needed).
		:type multiple8: bool, optional
		"""
		#min_sampling = 4 * self.fmax
		min_sampling = 8 * self.fmax # teoreticky 4*fmax aby fungovala L2 norma????
		SAMPRATE = 1. / lcmm(*self.data_deltas) # kazda stanice muze mit jine vzorkovani, bereme nejvetsiho spolecneho delitele (= 1. / nejmensi spolecny nasobek)
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
		# print(self.data_deltas) # DEBUG
		self.samprate = SAMPRATE / decimate
		self.logtext['samplings'] = samplings_str = ", ".join(["{0:5.1f} Hz".format(1./delta) for delta in  self.data_deltas])
		self.log('\nSampling frequencies:\n  Data sampling: {0:s}\n  Common sampling: {3:5.1f}\n  Decimation factor: {1:3d} x\n  Sampling used: {2:5.1f} Hz'.format(samplings_str, decimate, self.samprate, SAMPRATE))

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
		self.min_time(np.sqrt(self.stations[0]['dist']**2+self.depth_min**2))
		self.max_time(np.sqrt(self.stations[self.nr-1]['dist']**2+self.depth_max**2))
		self.t_min -= 20 # FIXED OPTION
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

	def set_Greens_parameters(self):
		"""
		Sets parameters for Green's function calculation:
		 - time window length ``self.tl``
		 - number of frequencies ``self.freq``
		 - spatial periodicity ``self.xl``
		 
		Writes used parameters to the log file.
		"""
		self.tl = self.npts_elemse/self.samprate
		#freq = int(math.ceil(fmax*tl))
		#self.freq = min(int(math.ceil(self.fmax*self.tl))*2, self.npts_elemse/2) # pocitame 2x vic frekvenci, nez pak proleze filtrem, je to pak lepe srovnatelne se signalem, ktery je kauzalne filtrovany
		self.freq = int(self.npts_elemse/2)+1
		self.xl = max(np.ceil(self.stations[self.nr-1]['dist']/1000), 100)*1e3*20 # `xl` 20x vetsi nez nejvetsi epicentralni vzdalenost, zaokrouhlena nahoru na kilometry, minimalne 2000 km
		self.log("\nGreen's function calculation:\n  npts: {0:4d}\n  tl: {1:4.2f}\n  freq: {2:4d}\n  npts for inversion: {3:4d}".format(self.npts_elemse, self.tl, self.freq, self.npts_slice))
	
	def write_Greens_parameters(self):
		"""
		Writes file grdat.hed - parameters for gr_xyz (Axitra)
		"""
		for model in self.models:
			if model:
				f = 'green/grdat' + '-' + model + '.hed'
			else:
				f = 'green/grdat.hed'
			grdat = open(f, 'w')
			grdat.write("&input\nnc=99\nnfreq={freq:d}\ntl={tl:1.2f}\naw=0.5\nnr={nr:d}\nns=1\nxl={xl:1.1f}\nikmax=100000\nuconv=0.1E-06\nfref=1.\n/end\n".format(freq=self.freq,tl=self.tl,nr=self.models[model], xl=self.xl)) # 'nc' is probably ignored in the current version of gr_xyz???
			grdat.close()
	
	def verify_Greens_parameters(self):
		"""
		Check whetrer parameters in file grdat.hed (probably used in Green's function calculation) are the same as used now.
		If it agrees, return True, otherwise returns False, print error description, and writes it into log.
		"""
		grdat = open('green/grdat.hed', 'r')
		if grdat.read() != "&input\nnc=99\nnfreq={freq:d}\ntl={tl:1.2f}\naw=0.5\nnr={nr:d}\nns=1\nxl={xl:1.1f}\nikmax=100000\nuconv=0.1E-06\nfref=1.\n/end\n".format(freq=self.freq,tl=self.tl,nr=self.nr, xl=self.xl):
			desc = 'Pre-calculated Green\'s functions calculated with different parameters (e.g. sampling) than used now, calculate Green\'s functions again. Exiting...'
			self.log(desc)
			print(desc)
			print ("&input\nnc=99\nnfreq={freq:d}\ntl={tl:1.2f}\naw=0.5\nnr={nr:d}\nns=1\nxl={xl:1.1f}\nikmax=100000\nuconv=0.1E-06\nfref=1.\n/end\n".format(freq=self.freq,tl=self.tl,nr=self.nr, xl=self.xl))
			return False
		grdat.close()
		return True
	
	def verify_Greens_headers(self):
		"""
		Checked whether elementary-seismogram-metadata files (created when the Green's functions were calculated) agree with curent grid points positions.
		Used to verify whether pre-calculated Green's functions were calculated on the same grid as used now.
		"""
		for g in range(len(self.grid)):
			gp = self.grid[g]
			point_id = str(g).zfill(4)
			meta  = open('green/elemse'+point_id+'.txt', 'r')
			lines = meta.readlines()
			if len(lines)==0:
				self.grid[g]['err'] = 1
				self.grid[g]['VR'] = -10
			elif lines[0] != '{0:1.3f} {1:1.3f} {2:1.3f}'.format(gp['x']/1e3, gp['y']/1e3, gp['z']/1e3):
				desc = 'Pre-calculated grid point {0:d} has different coordinates, probably the shape of the grid has changed, calculate Green\'s functions again. Exiting...'.format(g)
				self.log(desc)
				print(desc)
				return False
			meta.close()
		return True
	
	def calculate_or_verify_Green(self):
		"""
		If ``self.use_precalculated_Green`` is True, verifies whether the pre-calculated Green's functions were calculated on the same grid and with the same parameters (:func:`verify_Greens_headers` and :func:`verify_Greens_parameters`)
		Otherwise calculates Green's function (:func:`write_Greens_parameters` and :func:`calculate_Green`).
		
		:return: ``True`` if everything is OK, otherwise ``False``
		"""
		
		if not self.use_precalculated_Green: # calculate Green's functions in all grid points
			self.write_Greens_parameters()
			self.calculate_Green()
			return True
		else: # verify whether the pre-calculated Green's functions are calculated on the same grid and with the same parameters
			if not self.verify_Greens_parameters():
				return False
			if not self.verify_Greens_headers():
				return False
		return True

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
		rupture_length = self.rupture_length
		if self.grid_radius:
			self.radius = radius = self.grid_radius
		else:
			self.radius = radius = self.location_unc + rupture_length
		if self.grid_min_depth:
			self.depth_min = depth_min = max(min_depth, self.grid_min_depth)
		else:
			self.depth_min = depth_min = max(min_depth, self.event['depth'] - self.depth_unc - rupture_length)
		if self.grid_max_depth:
			self.depth_max = depth_max = self.grid_max_depth
		else:
			self.depth_max = depth_max = self.event['depth'] + self.depth_unc + rupture_length
		n_points = np.pi*(radius/step_x)**2 * (depth_max-depth_min)/step_z
		if n_points > max_points:
			step_x *= (n_points/max_points)**0.333
			step_z *= (n_points/max_points)**0.333
		n_steps = int(radius/step_x)
		n_steps_z = int((self.depth_unc + rupture_length)/step_z)
		depths = []
		for k in range(-n_steps_z, n_steps_z+1):
			z = self.event['depth']+k*step_z
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
		self.log('\nGrid parameters:\n  number of points: {0:4d}\n  horizontal step: {1:5.0f} m\n  vertical step: {2:5.0f} m\n  grid radius: {3:6.3f} km\n  minimal depth: {4:6.3f} km\n  maximal depth: {5:6.3f} km\nEstimated rupture length: {6:6.3f} km'.format(len(self.grid), step_x, step_z, radius/1e3, depth_min/1e3, depth_max/1e3, rupture_length/1e3))

	def set_time_grid(self):
		"""
		Sets equidistant time grid defined by ``self.shift_min``, ``self.shift_max``, and ``self.shift_step`` (in seconds). The corresponding values ``self.SHIFT_min``, ``self.SHIFT_max``, and ``self.SHIFT_step`` are (rounded) in samples related to the the highest sampling rate common to all stations.
		"""
		if self.grid_max_time:
			self.shift_max  = shift_max  = self.grid_max_time
		else:
			self.shift_max  = shift_max  = self.time_unc + self.rupture_length / self.rupture_velocity
		if self.grid_min_time:
			self.shift_min  = shift_min  = self.grid_min_time
		else:
			self.shift_min  = shift_min  = -self.time_unc
		self.shift_step = shift_step = 1./self.fmax * 0.01
		self.SHIFT_min = int(round(shift_min*self.max_samprate))
		self.SHIFT_max = int(round(shift_max*self.max_samprate))
		self.SHIFT_step = max(int(round(shift_step*self.max_samprate)), 1) # max() to avoid step beeing zero
		self.SHIFT_min = int(round(self.SHIFT_min / self.SHIFT_step)) * self.SHIFT_step # shift the grid to contain zero time shift
		self.log('\nGrid-search over time:\n  min = {sn:5.2f} s ({Sn:3d} samples)\n  max = {sx:5.2f} s ({Sx:3d} samples)\n  step = {step:4.2f} s ({STEP:3d} samples)'.format(sn=shift_min, Sn=self.SHIFT_min, sx=shift_max, Sx=self.SHIFT_max, step=shift_step, STEP=self.SHIFT_step))
		
	def skip_short_records(self, noise=False):
		"""
		Checks whether all records are long enough for the inversion and skips unsuitable ones.
		
		:parameter noise: checks also whether the record is long enough for generating the noise slice for the covariance matrix (if the value is ``True``, choose minimal noise length automatically; if it's numerical, take the value as minimal noise length)
		:type noise: bool or float, optional
		"""
		self.log('\nChecking record length:')
		for st in self.data_raw:
			for comp in range(3):
				stats = st[comp].stats
				if stats.starttime > self.event['t'] + self.t_min + self.shift_min or stats.endtime < self.event['t'] + self.t_max + self.shift_max:
					self.log('  ' + stats.station + ' ' + stats.channel + ': record too short, ignoring component in inversion')
					self.stations_index['_'.join([stats.network, stats.station, stats.location, stats.channel[0:2]])]['use'+stats.channel[2]] = False
				if noise:
					if type(noise) in (float,int):
						noise_len = noise
					else:
						noise_len = (self.t_max - self.t_min + self.shift_max + 10)*1.1 - self.shift_min - self.t_min
						#print stats.station, stats.channel, noise_len, '>', self.event['t']-stats.starttime # DEBUG
					if stats.starttime > self.event['t'] - noise_len:
						self.log('  ' + stats.station + ' ' + stats.channel + ': record too short for noise covariance, ignoring component in inversion')
						self.stations_index['_'.join([stats.network, stats.station, stats.location, stats.channel[0:2]])]['use'+stats.channel[2]] = False

	def set_parameters(self, fmax, fmin=0., wavelengths=5, min_depth=1000, log=True):
		"""
		Sets some technical parameters of the inversion.
		
		Technically, just runs following functions:
		 - :func:`set_frequencies`
		 - :func:`set_working_sampling`
		 - :func:`set_time_window`
		 - :func:`set_Greens_parameters`
		 - :func:`set_grid`
		 - :func:`set_time_grid`
		 - :func:`count_components`
		
		The parameters are parameters of the same name of these functions.
		"""
		self.set_frequencies(fmax, fmin, wavelengths)
		self.set_working_sampling()
		self.set_grid()
		self.set_time_window()
		self.set_Greens_parameters()
		self.set_time_grid()
		self.count_components(log)

	def calculate_Green(self):
		"""
		Runs :func:`Axitra_wrapper` (Green's function calculation) in parallel.
		"""
		logfile = self.outdir+'/log_green.txt'
		open(logfile, "w").close() # erase file contents
		# run `gr_xyz` aand `elemse`
		for model in self.models:
			if self.threads > 1: # parallel
				pool = mp.Pool(processes=self.threads)
				results = [pool.apply_async(Axitra_wrapper, args=(i, model, self.grid[i]['x'], self.grid[i]['y'], self.grid[i]['z'], self.npts_exp, self.elemse_start_origin, logfile)) for i in range(len(self.grid))]
				output = [p.get() for p in results]
				for i in range (len(self.grid)):
					if output[i] == False:
						self.grid[i]['err'] = 1
						self.grid[i]['VR'] = -10
			else: # serial
				for i in range (len(self.grid)):
					gp = self.grid[i]
					Axitra_wrapper(i, model, gp['x'], gp['y'], gp['z'], self.npts_exp, self.elemse_start_origin, logfile)

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
	
	def VR_of_components(self, n=1):
		"""
		Calculates the variance reduction from each component and the variance reduction from a subset of stations.
		
		:param n: minimal number of components used
		:type n: integer, optional
		:return: maximal variance reduction from a subset of stations
		
		Add the variance reduction of each component to ``self.stations`` with keys ``VR_Z``, ``VR_N``, and ``VR_Z``.
		Calculate the variance reduction from a subset of the closest stations (with minimal ``n`` components used) leading to the highest variance reduction and save it to ``self.max_VR``.
		"""
		npts = self.npts_slice
		data = self.data_shifts[self.centroid['shift_idx']]
		elemse = read_elemse(self.nr, self.npts_elemse, 'green/elemse'+self.centroid['id']+'.dat', self.stations, self.invert_displacement) # read elemse
		for r in range(self.nr):
			for e in range(6):
				my_filter(elemse[r][e], self.stations[r]['fmin'], self.stations[r]['fmax'])
				elemse[r][e].trim(UTCDateTime(0)+self.elemse_start_origin)
		MISFIT = 0
		NORM_D = 0
		COMPS_USED = 0
		max_VR = -99
		self.VRcomp = {}
		for sta in range(self.nr):
			SYNT = {}
			for comp in range(3):
				SYNT[comp] = np.zeros(npts)
				for e in range(6):
					SYNT[comp] += elemse[sta][e][comp].data[0:npts] * self.centroid['a'][e,0]
			comps_used = 0
			for comp in range(3):
				if self.Cd_inv and not self.stations[sta][{0:'useZ', 1:'useN', 2:'useE'}[comp]]:
					self.stations[sta][{0:'VR_Z', 1:'VR_N', 2:'VR_E'}[comp]] = None
					continue
				synt = SYNT[comp]
				d = data[sta][comp][0:npts]
				if self.LT3:
					d    = np.zeros(npts)
					synt = np.zeros(npts)
					x1 = -npts
					for COMP in range(3):
						if not self.stations[sta][{0:'useZ', 1:'useN', 2:'useE'}[COMP]]:
							continue
						x1 += npts; x2 = x1+npts
						y1 = comps_used*npts; y2 = y1+npts
						d    += np.dot(self.LT3[sta][y1:y2, x1:x2], data[sta][COMP].data[0:npts])
						synt += np.dot(self.LT3[sta][y1:y2, x1:x2], SYNT[COMP])
					
				elif self.Cd_inv:
					d    = np.dot(self.LT[sta][comp], d)
					synt = np.dot(self.LT[sta][comp], synt)
					
				else:
					pass
				comps_used += 1
				misfit = np.sum(np.square(d - synt))
				norm_d = np.sum(np.square(d))
				#t = np.arange(0, (npts-0.5) / self.samprate, 1. / self.samprate) # DEBUG
				#fig = plt.figure() # DEBUG
				#plt.plot(t, synt, label='synt') # DEBUG
				#plt.plot(t, d, label='data') # DEBUG
				#plt.plot(t, d-synt, label='difference') # DEBUG
				#plt.legend() # DEBUG
				#plt.show() # DEBUG
				VR = 1 - misfit / norm_d
				self.stations[sta][{0:'VR_Z', 1:'VR_N', 2:'VR_E'}[comp]] = VR
				if self.stations[sta][{0:'useZ', 1:'useN', 2:'useE'}[comp]]:
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

	def print_solution(self):
		"""
		Write into log the best solution ``self.centroid``.
		"""
		C = self.centroid
		t = self.event['t'] + C['shift']
		self.log('\nCentroid location:\n  Centroid time: {t:s}\n  Lat {lat:8.3f}   Lon {lon:8.3f}   Depth {d:5.1f} km'.format(t=t.strftime('%Y-%m-%d %H:%M:%S'), lat=C['lat'], lon=C['lon'], d=C['z']/1e3))
		self.log('  ({0:5.0f} m to the north and {1:5.0f} m to the east with respect to epicenter)'.format(C['x'], C['y']))
		if C['edge']:
			self.log('  Warning: the solution lies on the edge of the grid!')
		mt2 = a2mt(C['a'], system='USE')
		c = max(abs(min(mt2)), max(mt2))
		c = 10**np.floor(np.log10(c))
		MT2 = mt2 / c
		if C['shift'] >= 0:
			self.log('  time: {0:5.2f} s after origin time\n'.format(C['shift']))
		else:
			self.log('  time: {0:5.2f} s before origin time\n'.format(-C['shift']))
		if C['shift'] in (self.shifts[0], self.shifts[-1]):
			self.log('  Warning: the solution lies on the edge of the time-grid!')
		self.log('  VR: {0:4.0f} %\n  CN: {1:4.0f}'.format(C['VR']*100, C['CN']))
		#self.log('  VR: {0:8.4f} %\n  CN: {1:4.0f}'.format(C['VR']*100, C['CN'])) # DEBUG
		self.log('  MT [ Mrr    Mtt    Mpp    Mrt    Mrp    Mtp ]:\n     [{1:5.2f}  {2:5.2f}  {3:5.2f}  {4:5.2f}  {5:5.2f}  {6:5.2f}] * {0:5.0e}'.format(c, *MT2))

	def print_fault_planes(self, precision='3.0', tool=''):
		"""
		Decompose the moment tensor of the best grid point by :func:`decompose` and writes the result to the log.
		
		:param precision: precision of writing floats, like ``5.1`` for 5 letters width and 1 decimal place (default ``3.0``)
		:type precision: string, optional
		:param tool: tool for the decomposition, `mopad` for :func:`decompose_mopad`, otherwise :func:`decompose` is used
		"""
		mt = a2mt(self.centroid['a'])
		if tool == 'mopad':
			self.mt_decomp = decompose_mopad(mt)
		else:
			self.mt_decomp = decompose(mt)
		self.log('''\nScalar Moment: M0 = {{mom:5.2e}} Nm (Mw = {{Mw:3.1f}})
  DC component: {{dc_perc:{0:s}f}} %,   CLVD component: {{clvd_perc:{0:s}f}} %,   ISOtropic component: {{iso_perc:{0:s}f}} %
  Fault plane 1: strike = {{s1:{0:s}f}}, dip = {{d1:{0:s}f}}, slip-rake = {{r1:{0:s}f}}
  Fault plane 2: strike = {{s2:{0:s}f}}, dip = {{d2:{0:s}f}}, slip-rake = {{r2:{0:s}f}}'''.format(precision).format(**self.mt_decomp))

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

