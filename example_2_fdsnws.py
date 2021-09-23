#! /usr/bin/env python

import BayesISOLA

inputs = BayesISOLA.load_data(
	outdir = 'output/example_2_fdsnws'
	)
inputs.read_event_info('input/example_2_fdsnws/event.isl')
inputs.set_source_time_function('step')
inputs.read_network_coordinates('input/example_2_fdsnws/network.stn')
inputs.read_crust('input/example_2_fdsnws/crustal.dat')
inputs.load_streams_fdsnws(
	[
		'http://eida.ethz.ch/fdsnws/', 
		#'http://www.orfeus-eu.org/fdsnws/',
		#'http://erde.geophysik.uni-muenchen.de/fdsnws/',
		#'http://geofon.gfz-potsdam.de/fdsnws/',
		#'http://eida.bgr.de/fdsnws/',
		#'http://webservices.ingv.it/fdsnws/station/'
	],
	t_before=360, t_after=100)
inputs.detect_mouse(figures=True)

grid = BayesISOLA.grid(
	inputs,
	location_unc = 1000, # m
	depth_unc = 3000, # m
	time_unc = 1, # s
	step_x = 200, # m
	step_z = 200, # m
	max_points = 500,
	circle_shape = True,
	rupture_velocity = 1000 # m/s
	)

data = BayesISOLA.process_data(
	inputs,
	grid,
	threads = 8,
	use_precalculated_Green = 'auto',
	fmax = 0.15,
	fmin = 0.02
	)

cova = BayesISOLA.covariance_matrix(data)
cova.covariance_matrix_noise(crosscovariance=True, save_non_inverted=True)

solution = BayesISOLA.resolve_MT(data, cova, deviatoric=False)
	# deviatoric=True: force isotropic component to be zero

plot = BayesISOLA.plot(solution)
plot.html_log(h1='Example 2f (2013-12-12 00:59:18 Sargans)', mouse_figures='mouse/')
