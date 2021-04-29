#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy.interpolate

from obspy.imaging.beachball import beach

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
#import matplotlib.cm as cm
#import matplotlib.colors as colors
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D

from BayesISOLA.MT_comps import a2mt, decompose

def plot_maps(self, outfile='$outdir/map.png', beachball_size_c=False):
	"""
	Plot figures showing how the solution is changing across the grid.
	
	:param outfile: Path to the file where to plot. If ``None``, plot to the screen. Because one figure is plotted for each depth, inserts an identifier before the last dot (`map.png` -> `map_1000.png`, `map_2000.png`, etc.).
	:type outfile: string, optional
	:param beachball_size_c: If ``True``, the sizes of the beachballs correspond to the posterior probability density function (PPD) instead of the variance reduction VR
	:type beachball_size_c: bool, optional
	
	Plot top view to the grid at each depth. The solutions in each grid point (for the centroid time with the highest VR) are shown by beachballs. The color of the beachball corresponds to its DC-part. The inverted centroid time is shown by a contour in the background and the condition number by contour lines.
	"""
	if len(isola.grid) == len(isola.depths): # just one point in the map, it has no sense to plot it
		return False
	outfile = outfile.replace('$outdir', self.outdir)
	r = self.grid.radius * 1e-3 * 1.1 # to km, *1.1
	if beachball_size_c:
		max_width = np.sqrt(self.MT.max_sum_c)
	for z in self.grid.depths:
		# prepare data points
		x=[]; y=[]; s=[]; CN=[]; MT=[]; color=[]; width=[]; highlight=[]
		for gp in self.grid.grid:
			if gp['z'] != z or gp['err']:
				continue
			x.append(gp['y']/1e3); y.append(gp['x']/1e3); s.append(gp['shift']); CN.append(gp['CN']) # NS is x coordinate, so switch it with y to be vertical
			MT.append(a2mt(gp['a'], system='USE'))
			VR = max(gp['VR'],0)
			if beachball_size_c:
				width.append(self.grid.step_x/1e3 * np.sqrt(gp['sum_c']) / max_width)
			else:
				width.append(self.grid.step_x/1e3*VR)
			if self.MT.decompose:
				dc = float(gp['dc_perc'])/100
				color.append((dc, 0, 1-dc))
			else:
				color.append('black')
			highlight.append(self.MT.centroid['id'] == gp['id'])
		if outfile:
			k = outfile.rfind(".")
			filename = outfile[:k] + "_{0:0>5.0f}".format(z) + outfile[k:]
		else:
			filename = None
		self.plot_map_backend(x, y, s, CN, MT, color, width, highlight, -r, r, -r, r, xlabel='west - east [km]', ylabel='south - north [km]', title='depth {0:5.2f} km'.format(z/1000), beachball_size_c=beachball_size_c, outfile=filename)

def plot_slices(self, outfile='$outdir/slice.png', point=None, beachball_size_c=False):
	"""
	Plot vertical slices through the grid of solutions in point `point`.
	If `point` not specified, use the best solution as a point.

	:param outfile: Path to the file where to plot. If ``None``, plot to the screen.
	:type outfile: string, optional
	:param point: `x` and `y` coordinates (with respect to the epicenter) of a grid point where the slices are placed through. If ``None``, uses the coordinates of the inverted centroid.
	:type point: tuple, optional
	:param beachball_size_c: If ``True``, the sizes of the beachballs correspond to the posterior probability density function (PPD) instead of the variance reduction VR
	:type beachball_size_c: bool, optional
	
	The legend is the same as at :func:`plot_maps`.
	"""
	if len(isola.depths) == 1: # just grid point(s) in a single depth, it has no sense to plot it
		return False
	outfile = outfile.replace('$outdir', self.outdir)
	if point:
		x0, y0 = point
	else:
		x0 = self.MT.centroid['x']; y0 = self.MT.centroid['y']
	depth_min = self.grid.depth_min / 1000; depth_max = self.grid.depth_max / 1000
	depth = depth_max - depth_min
	r = self.grid.radius * 1e-3 * 1.1 # to km, *1.1
	if beachball_size_c:
		max_width = np.sqrt(self.MT.max_sum_c)
	for slice in ('N-S', 'W-E', 'NW-SE', 'SW-NE'):
		x=[]; y=[]; s=[]; CN=[]; MT=[]; color=[]; width=[]; highlight=[]
		for gp in self.grid.grid:
			if   slice == 'N-S': X = -gp['x'];	Z = gp['y']-y0
			elif slice == 'W-E': X = gp['y'];	Z = gp['x']-x0
			elif slice=='NW-SE': X = (gp['y']-gp['x'])*1/np.sqrt(2);	Z = gp['x']+gp['y']-y0-x0
			elif slice=='SW-NE': X = (gp['y']+gp['x'])*1/np.sqrt(2);	Z = gp['x']-gp['y']+y0-x0
			Y = gp['z']
			if abs(Z) > 0.001 or gp['err']:
				continue
			x.append(X/1e3); y.append(Y/1e3); s.append(gp['shift']); CN.append(gp['CN'])
			MT.append(a2mt(gp['a'], system='USE'))
			VR = max(gp['VR'],0)
			if beachball_size_c:
				width.append(self.grid.step_x/1e3 * np.sqrt(gp['sum_c']) / max_width)
			else:
				width.append(self.grid.step_x/1e3*VR)
			if self.MT.decompose:
				dc = float(gp['dc_perc'])/100
				color.append((dc, 0, 1-dc))
			else:
				color.append('black')
			highlight.append(self.MT.centroid['id'] == gp['id'])
		if outfile:
			k = outfile.rfind(".")
			filename = outfile[:k] + '_' + slice + outfile[k:]
		else:
			filename = None
		xlabel = {'N-S':'north - south', 'W-E':'west - east', 'NW-SE':'north-west - south-east', 'SW-NE':'south-west - north-east'}[slice] + ' [km]'
		self.plot_map_backend(x, y, s, CN, MT, color, width, highlight, -r, r, depth_max + depth*0.05, depth_min - depth*0.05, xlabel, 'depth [km]', title='vertical slice', beachball_size_c=beachball_size_c, outfile=filename)

def plot_maps_sum(self, outfile='$outdir/map_sum.png'):
	"""
	Plot map and vertical slices through the grid of solutions showing the posterior probability density function (PPD).
	Contrary to :func:`plot_maps` and :func:`plot_slices`, the size of the beachball correspond not only to the PPD of grid-point through which is a slice placed, but to a sum of all grid-points which are before and behind.

	:param outfile: Path to the file where to plot. If ``None``, plot to the screen.
	:type outfile: string, optional
	
	The legend and properties of the function are similar as at function :func:`plot_maps`.
	"""
	outfile = outfile.replace('$outdir', self.outdir)
	if not self.cova.Cd_inv:
		return False # if the data covariance matrix is unitary, we have no estimation of data errors, so the PDF has good sense
	r = self.grid.radius * 1e-3 * 1.1 # to km, *1.1
	depth_min = self.grid.depth_min * 1e-3; depth_max = self.grid.depth_max * 1e-3
	depth = depth_max - depth_min
	Ymin = depth_max + depth*0.05
	Ymax = depth_min - depth*0.05
	#for slice in ('N-S', 'W-E', 'NW-SE', 'SW-NE', 'top'):
	for slice in ('N-S', 'W-E', 'top'):
		if slice == 'top' and len(isola.grid) == len(isola.depths): # just one point in the map, skip
			continue
		elif len(isola.depths) == 1: # just grid point(s) in a single depth, skip plotting
			continue
		X=[]; Y=[]; s=[]; CN=[]; MT=[]; color=[]; width=[]; highlight=[]
		g = {}
		max_c = 0
		for gp in self.grid.grid:
			if gp['err'] or gp['sum_c']<=0:   continue
			if   slice == 'N-S': x = -gp['x']
			elif slice == 'W-E': x = gp['y']
			elif slice=='NW-SE': x = (gp['y']-gp['x'])*1/np.sqrt(2)
			elif slice=='SW-NE': x = (gp['y']+gp['x'])*1/np.sqrt(2)
			x *= 1e-3
			y = gp['z']*1e-3
			if slice=='top':
				x = gp['y']*1e-3; y = gp['x']*1e-3 # NS is x coordinate, so switch it with y to be vertical
			if not x in g:    g[x] = {}
			if not y in g[x]: g[x][y] = {'c':0, 'max_c':0, 'highlight':False}
			g[x][y]['c'] += gp['sum_c']
			if g[x][y]['c'] > max_c:
				max_c = g[x][y]['c']
			if gp['sum_c'] > g[x][y]['max_c']:
				g[x][y]['max_c'] = gp['sum_c']
				g[x][y]['a'] = gp['a']
				#g[x][y]['CN'] = gp['CN']
				#g[x][y]['s'] = gp['shift']
				if self.MT.decompose:
					g[x][y]['dc'] = gp['dc_perc']
			if self.MT.centroid['id'] == gp['id']:
				g[x][y]['highlight'] = True
		for x in g:
			for y in g[x]:
				X.append(x)
				Y.append(y)
				#s.append(g[x][y]['s'])
				#CN.append(g[x][y]['CN'])
				MT.append(a2mt(g[x][y]['a'], system='USE'))
				if self.MT.decompose:
					dc = float(g[x][y]['dc'])*0.01
					color.append((dc, 0, 1-dc))
				else:
					color.append('black')
				highlight.append(g[x][y]['highlight'])
				width.append(self.grid.step_x*1e-3 * np.sqrt(g[x][y]['c'] / max_c))
		if outfile:
			k = outfile.rfind(".")
			filename = outfile[:k] + '_' + slice + outfile[k:]
		else:
			filename = None
		xlabel = {'N-S':'north - south', 'W-E':'west - east', 'NW-SE':'north-west - south-east', 'SW-NE':'south-west - north-east', 'top':'west - east'}[slice] + ' [km]'
		if slice == 'top':
			ymin = -r; ymax=r
			ylabel = 'south - north [km]'
			title = 'PDF sum: top view'
		else:
			ymin = Ymin; ymax = Ymax
			ylabel = 'depth [km]'
			title = 'PDF sum: side view'
		#self.plot_map_backend(X, Y, s, CN, MT, color, width, highlight, -r, r, ymin, ymax, xlabel, ylabel, title, True, filename)
		self.plot_map_backend(X, Y, None, None, MT, color, width, highlight, -r, r, ymin, ymax, xlabel, ylabel, title, True, filename)

def plot_map_backend(self, x, y, s, CN, MT, color, width, highlight, xmin, xmax, ymin, ymax, xlabel='', ylabel='', title='', beachball_size_c=False, outfile=None):
	"""
	The plotting back-end for functions :func:`plot_maps`, :func:`plot_slices` and :func:`plot_maps_sum`. There is no need for calling it directly.
	"""
	plt.rcParams.update({'font.size': 16})
	xdiff = xmax-xmin
	ydiff = ymax-ymin
	if xdiff > abs(1.3*ydiff):
		plt.figure(figsize=(16+bool(s)*2, abs(ydiff/xdiff)*14+3)) # FIXME
	else:
		plt.figure(figsize=(abs(xdiff/ydiff)*11+2+bool(s)*2, 14)) # FIXME
	ax = plt.gca()
	#if xmin != ymin or xmax != ymax:
	plt.axis('equal')
	plt.xlim(xmin, xmax)
	plt.ylim(ymin, ymax, int(np.sign(ydiff)))
	if xlabel: plt.xlabel(xlabel)
	if ylabel: plt.ylabel(ylabel)
	if title: plt.title(title)
	Xmin = min(x); Xmax = max(x); Ymin = min(y); Ymax = max(y)
	width_max = max(width)

	for i in range(len(x)):
		if highlight[i]:
			c = plt.Circle((x[i], y[i]), self.grid.step_x/1e3*0.5, color='r')
			c.set_edgecolor('r')
			c.set_linewidth(10)
			c.set_facecolor('none')  # "none" not None
			c.set_alpha(0.7)
			ax.add_artist(c)
		if width[i] > self.grid.step_x*1e-3 * 0.04:
			try:
				b = beach(MT[i], xy=(x[i], y[i]), width=(width[i], width[i]*np.sign(ydiff)), linewidth=0.5, facecolor=color[i], zorder=10)
			except:
				#print('Plotting this moment tensor in a grid point crashed: ', mt2, 'using mopad')
				try:
					b = beach(MT[i], xy=(x[i], y[i]), width=(width[i], width[i]*np.sign(ydiff)), linewidth=0.5, facecolor=color[i], zorder=10) # width: at side views, mirror along horizontal axis to avoid effect of reversed y-axis
				except:
					print('Plotting this moment tensor in a grid point crashed: ', MT[i])
				else:
					ax.add_collection(b)
			else:
				ax.add_collection(b)
		elif width[i] > self.grid.step_x*1e-3 * 0.001:
			b = plt.Circle((x[i], y[i]), width[i]/2, facecolor=color[i], edgecolor='k', zorder=10, linewidth=0.5)
			ax.add_artist(b)

	if CN and s:
		# Set up a regular grid of interpolation points
		xi = np.linspace(Xmin, Xmax, 400)
		yi = np.linspace(Ymin, Ymax, 400)
		xi, yi = np.meshgrid(xi, yi)

		# Interpolate
		rbf = scipy.interpolate.Rbf(x, y, s, function='linear')
		z1 = rbf(xi, yi)
		rbf = scipy.interpolate.Rbf(x, y, CN, function='linear')
		z2 = rbf(xi, yi)

		shift = plt.imshow(z1, cmap = plt.get_cmap('PRGn'), 
			vmin=self.grid.shift_min, vmax=self.grid.shift_max, origin='lower',
			extent=[Xmin, Xmax, Ymin, Ymax])
		levels = np.arange(1., 21., 1.)
		CN = plt.contour(z2, levels, cmap = plt.get_cmap('gray'), origin='lower', linewidths=1,
			extent=[Xmin, Xmax, Ymin, Ymax], zorder=4)
		plt.clabel(CN, inline=1, fmt='%1.0f', fontsize=10) # levels[1::2]  oznacit kazdou druhou caru
		CB1 = plt.colorbar(shift, shrink=0.5, extend='both', label='shift [s]')
		#CB2 = plt.colorbar(CN, orientation='horizontal', shrink=0.4, label='condition number', ticks=[levels[0], levels[-1]])
		l,b,w,h = plt.gca().get_position().bounds
		ll,bb,ww,hh = CB1.ax.get_position().bounds
		#CB1.ax.set_position([ll-0.2*w, bb+0.2*h, ww, hh])
		CB1.ax.set_position([ll, bb+0.2*h, ww, hh])
		#ll,bb,ww,hh = CB2.ax.get_position().bounds
		#CB2.ax.set_position([l+0.58*w, bb+0.07*h, ww, hh])
	
	# legend beachball's color = DC%
	if self.MT.decompose:
		x = y = xmin*2
		plt.plot([x],[y], marker='o', markersize=15, color=(1, 0, 0), label='DC 100 %')
		plt.plot([x],[y], marker='o', markersize=15, color=(.5, 0, .5), label='DC 50 %')
		plt.plot([x],[y], marker='o', markersize=15, color=(0, 0, 1), label='DC 0 %')
		mpl.rcParams['legend.handlelength'] = 0
		if CN and s:
			plt.legend(loc='upper left', numpoints=1, bbox_to_anchor=(1, -0.05), fancybox=True)
		else:
			plt.legend(loc='upper right', numpoints=1, bbox_to_anchor=(0.95, -0.05), fancybox=True)
	
	# legend beachball's area
	if beachball_size_c: # beachball's area = PDF
		r_max = self.grid.step_x/1e3/2
		r_half = r_max/1.4142
		text_max = 'maximal PDF'
		text_half = 'half-of-maximum PDF'
		text_area = 'Beachball area ~ PDF'
	else: # beachball's radius = VR
		VRmax = self.MT.centroid['VR']
		r_max = self.grid.step_x/1e3/2 * VRmax
		r_half = r_max/2
		text_max = 'VR {0:2.0f} % (maximal)'.format(VRmax*100)
		text_half = 'VR {0:2.0f} %'.format(VRmax*100/2)
		text_area = 'Beachball radius ~ VR'
	x_symb = [xmin+r_max, xmin][bool(CN and s)] # min(xmin, -0.8*ydiff)
	x_text = xmin+r_max*1.8
	y_line = ymin-ydiff*0.11
	VRlegend = plt.Circle((x_symb, y_line), r_max, facecolor=(1, 0, 0), edgecolor='k', clip_on=False)
	ax.add_artist(VRlegend)
	VRlegendtext = plt.text(x_text, y_line, text_max, verticalalignment='center')
	ax.add_artist(VRlegendtext)
	y_line = ymin-ydiff*0.20
	VRlegend2 = plt.Circle((x_symb, y_line), r_half, facecolor=(1, 0, 0), edgecolor='k', clip_on=False)
	ax.add_artist(VRlegend2)
	VRlegendtext2 = plt.text(x_text, y_line, text_half, verticalalignment='center')
	ax.add_artist(VRlegendtext2)
	y_line = ymin-ydiff*0.26
	VRlegendtext3 = plt.text(x_text, y_line, text_area, verticalalignment='center')
	ax.add_artist(VRlegendtext3)

	if outfile:
		plt.savefig(outfile, bbox_inches='tight')
	else:
		plt.show()
	plt.clf()
	plt.close()

def plot_3D(self, outfile='$outdir/animation.mp4'):
	"""
	Creates an animation with the grid of solutios. The grid points are labeled according to their variance reduction.
	
	:param outfile: path to file for saving animation
	:type outfile: string
	"""
	n = len(self.grid.grid)
	x = np.zeros(n); y = np.zeros(n); z = np.zeros(n); VR = np.zeros(n)
	c = np.zeros((n, 3))
	for i in range(len(self.grid.grid)):
		gp = self.grid.grid[i]
		if gp['err']:
			continue
		x[i] = gp['y']/1e3
		y[i] = gp['x']/1e3 # NS is x coordinate, so switch it with y to be vertical
		z[i] = gp['z']/1e3
		vr = max(gp['VR'],0)
		VR[i] = np.pi * (15 * vr)**2
		c[i] = np.array([vr, 0, 1-vr])
		#if self.MT.decompose:
			#dc = float(gp['dc_perc'])/100
			#c[i,:] = np.array([dc, 0, 1-dc])
		#else:
			#c[i,:] = np.array([0, 0, 0])
	# Create a figure and a 3D Axes
	fig = plt.figure()
	ax = Axes3D(fig)
	ax.set_xlabel('west - east [km]')
	ax.set_ylabel('south - north [km]')
	ax.set_zlabel('depth [km]')

	# Create an init function and the animate functions.
	# Both are explained in the tutorial. Since we are changing
	# the the elevation and azimuth and no objects are really
	# changed on the plot we don't have to return anything from
	# the init and animate function. (return value is explained
	# in the tutorial).
	def init():
		ax.scatter(x, y, z, marker='o', s=VR, c=c, alpha=1.)
	def animate(i):
		ax.view_init(elev=10., azim=i)
	anim = animation.FuncAnimation(fig, animate, init_func=init, frames=360, interval=20, blit=True) # Animate
	anim.save(outfile.replace('$outdir', self.outdir), writer=self.movie_writer, fps=30) # Save
	#anim.save(outfile.replace('$outdir', self.outdir), writer=self.movie_writer, fps=30, extra_args=['-vcodec', 'libx264']) 
 
