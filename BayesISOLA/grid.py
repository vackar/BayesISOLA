#! /usr/bin/env python3
# -*- coding: utf-8 -*-

class grid:
	"""
    Creation of space and time grid for grid-searching MT solution.

    :param data: instance with data (event and seismograms)
    :type data: :class:`~BayesISOLA.load_data`
    :param location_unc: horizontal uncertainty of the location in meters (default 0)
    :type location_unc: float, optional
    :type depth_unc: float, optional
    :param depth_unc: vertical (depth) uncertainty of the location in meters (default 0)
    :type time_unc: float, optional
    :param time_unc: uncertainty of the origin time in seconds (default 0)
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
    :type circle_shape: bool, optional
    :param circle_shape: if ``True``, the shape of the grid is cylinder, otherwise it is rectangular box (default ``True``)
    :type rupture_velocity: float, optional
    :param rupture_velocity: rupture propagation velocity in m/s, used for estimating the difference between the origin time and the centroid time (default 1000 m/s)

    .. rubric:: _`Variables`
    
    The following description is useful mostly for developers. The parameters from the `Parameters` section are not listed, but they become also variables of the class. Two of them (``step_x`` and ``step_z``) are changed at some point, the others stay intact.

    ``grid`` : list of dictionaries
        Spatial grid on which is the inverse where is solved.
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
	"""

	from BayesISOLA._grid import set_grid, set_time_grid

	def __init__(self, data, location_unc=0, depth_unc=0, time_unc=0, step_x=500, step_z=500, max_points=100, grid_radius=0, grid_min_depth=0, grid_max_depth=0, grid_min_time=0, grid_max_time=0, circle_shape=True, add_rupture_length=True, rupture_velocity=1000):
		self.location_unc = location_unc # m
		self.depth_unc = depth_unc # m
		self.time_unc = time_unc # s
		self.step_x = step_x # m
		self.step_z = step_z # m
		self.max_points = max_points
		self.grid_radius =    grid_radius # m
		self.grid_min_depth = grid_min_depth # m
		self.grid_max_depth = grid_max_depth # m
		self.grid_min_time =  grid_min_time # s
		self.grid_max_time =  grid_max_time # s
		self.circle_shape = circle_shape
		self.rupture_velocity = rupture_velocity
		self.add_rupture_length = add_rupture_length
		self.data = data


