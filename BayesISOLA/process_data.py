#! /usr/bin/env python3
# -*- coding: utf-8 -*-

class process_data:
	"""
    Process data for MT inversion.
    
    :param data: instance with raw data (event and seismograms)
    :type data: :class:`~BayesISOLA.load_data`
    :param grid: instance with space and time grid
    :type grid: :class:`~BayesISOLA.grid`
    :type s_velocity: float, optional
    :param s_velocity: characteristic S-wave velocity used for calculating number of wave lengths between the source and stations (default 3000 m/s)
    :type threads: integer, optional
    :param threads: number of threads for parallelization (default 2)
    :type invert_displacement: bool, optional
    :param invert_displacement: convert observed and modeled waveforms to displacement prior comparison (if ``True``), otherwise compare it in velocity (default ``False``)
    :type use_precalculated_Green: bool or ``'auto'``, optional
    :param use_precalculated_Green: use Green's functions calculated in the previous run (default ``False``), value ``'auto'`` for check whether precalculated Green's function exists and were calculated on the same grid
    :param correct_data: if ``True``, run :func:`correct_data`
    :type correct_data: bool, optional
    :param set_parameters: if ``True``, run :func:`set_parameters`; in this case setting parameters `fmax` and `fmin` is strongly recommended
	:param fmax: maximal inverted frequency for all stations
	:type fmax: float, optional
	:param fmin: minimal inverted frequency for all stations
	:type fmax: float, optional
	:parameter min_depth: minimal grid point depth in meters
	:type min_depth: float, optional
    :param skip_short_records: if not ``False``, run :func:`skip_short_records` with the value of the parameter
    :type skip_short_records: bool or int, optional
    :param calculate_or_verify_Green: if ``True``, run :func:`calculate_or_verify_Green`
    :type calculate_or_verify_Green: bool, optional
    :param trim_filter_data: if ``True``, run :func:`trim_filter_data`
    :type trim_filter_data: bool, optional
    :param decimate_shift: if ``True``, run :func:`decimate_shift`
    :type decimate_shift: bool, optional

    .. rubric:: _`Variables`

    ``data`` : list of :class:`~obspy.core.stream`
        Prepared data for the inversion. It's filled by function :func:`trim_filter_data`. The list is ordered ascending by epicentral distance of the station.
    ``noise`` : list of :class:`~obspy.core.stream`
        Before-event slice of ``data_raw`` for later noise analysis. Created by :func:`trim_filter_data`.
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
    ``components`` : integer
        Number of components of all stations used in the inversion. Created by :func:`count_components`.
    ``data_shifts`` : list of lists of :class:`~obspy.core.stream`
        Shifted and trimmed ``data`` ready.
    ``d_shifts`` : list of :class:`~numpy.ndarray`
        The previous one in form of data vectors ready for the inversion.
    ``shifts`` : list of floats
        Shift values in seconds. List is ordered in the same way as the two previous list.
    ``fmin`` : float
        Lower range of bandpass filter for data.
    ``fmax`` : float
        Higher range of bandpass filter for data.
	"""

	from BayesISOLA._green import set_Greens_parameters, write_Greens_parameters, verify_Greens_parameters, verify_Greens_headers, calculate_or_verify_Green, calculate_Green
	from BayesISOLA._parameters import set_frequencies, set_working_sampling, count_components, min_time, max_time, set_time_window, set_parameters, skip_short_records
	from BayesISOLA._process_data import correct_data, trim_filter_data, prefilter_data, decimate_shift

	def __init__(self, data, grid, s_velocity=3000, threads=2, invert_displacement=False, use_precalculated_Green=False, correct_data=True, set_parameters=True, fmax=1., fmin=0., min_depth=1000., skip_short_records=False, calculate_or_verify_Green=True, trim_filter_data=True, decimate_shift=True):
		self.d = data
		self.grid = grid
		self.s_velocity = s_velocity
		self.threads = threads
		self.invert_displacement = invert_displacement
		self.use_precalculated_Green = use_precalculated_Green
		self.data = []
		self.noise = []
		self.fmax = 0.
		self.log = data.log
		self.logtext = data.logtext
		self.idx_use = {0:'useZ', 1:'useN', 2:'useE'}
		self.idx_weight = {0:'weightZ', 1:'weightN', 2:'weightE'}
		
		if correct_data:
			self.correct_data()
		if set_parameters:
			self.set_parameters(fmax, fmin, min_depth)
		if not skip_short_records is False:
			self.skip_short_records(noise=True)
		if calculate_or_verify_Green:
			self.calculate_or_verify_Green()
		if trim_filter_data:
			self.trim_filter_data()
		if decimate_shift:
			self.decimate_shift()

	def __exit__(self, exc_type, exc_value, traceback):
		self.__del__()
		
	def __del__(self):
		del self.data
		del self.noise

	def evalute_noise(self):
		# compare spectrum of the signal and the noise
		pass
