#! /usr/bin/env python3
# -*- coding: utf-8 -*-

#import numpy as np
#import shutil
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
warnings.filterwarnings('ignore', '.*Conversion of the second argument of issubdtype from.*') # Avoids following warning:
		# /usr/lib/python3.7/site-packages/obspy/signal/detrend.py:31: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.

#import obspy
#from obspy import UTCDateTime

#from BayesISOLA.helpers import *
#from BayesISOLA.fileformats import *
#from BayesISOLA.MT_comps import *


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

	from BayesISOLA._input_crust import read_crust
	from BayesISOLA._input_event import read_event_info, set_event_info
	from BayesISOLA._input_network import read_network_info_DB, read_network_coordinates, create_station_index, write_stations
	from BayesISOLA._input_seismo_files import add_NEZ, add_SAC, add_NIED, load_files, load_NIED_files, check_a_station_present
	from BayesISOLA._input_seismo_remote import load_streams_ArcLink, load_streams_fdsnws
	from BayesISOLA._mouse import detect_mouse
	from BayesISOLA._covariance_matrix import covariance_matrix, covariance_matrix_SACF, covariance_matrix_ACF
	from BayesISOLA._grid import set_grid, set_time_grid
	from BayesISOLA._green import set_Greens_parameters, write_Greens_parameters, verify_Greens_parameters, verify_Greens_headers, calculate_or_verify_Green, calculate_Green
	from BayesISOLA._parameters import set_frequencies, count_components, set_working_sampling, min_time, max_time, set_time_window, set_parameters, skip_short_records
	from BayesISOLA._process_data import correct_data, trim_filter_data, prefilter_data, decimate_shift
	from BayesISOLA._inverse import run_inversion, find_best_grid_point
	from BayesISOLA._VR import VR_of_components
	from BayesISOLA._print import print_solution, print_fault_planes
	from BayesISOLA._save import save_seismo
	from BayesISOLA._plot import plot_stations, plot_covariance_matrix
	from BayesISOLA._plot_solution_summary import plot_MT, plot_uncertainty, plot_MT_uncertainty_centroid
	from BayesISOLA._plot_solution_maps import plot_maps, plot_slices, plot_maps_sum, plot_map_backend, plot_3D
	from BayesISOLA._plot_data import plot_seismo, plot_covariance_function, plot_noise, plot_spectra, plot_seismo_backend_1, plot_seismo_backend_2
	from BayesISOLA._html import html_log

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
	
	def evalute_noise(self):
		# compare spectrum of the signal and the noise
		pass

