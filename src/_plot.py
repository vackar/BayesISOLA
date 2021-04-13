#! /usr/bin/env python3
# -*- coding: utf-8 -*-

from math import sin,cos,radians
import numpy as np
import scipy.interpolate

from obspy import UTCDateTime
from obspy.imaging.beachball import beach, beachball
#from obspy.imaging.mopad_wrapper import beach as beach2   # nofill parameter results in a black circle

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D

from MT_comps import a2mt, decompose
from histogram import histogram
from fileformats import read_elemse
from functions import my_filter

def align_yaxis(ax1, ax2, v1=0, v2=0):
	"""
	Adjust ax2 ylimit so that v2 in ax2 is aligned to v1 in ax1
	"""
	_, y1 = ax1.transData.transform((0, v1))
	_, y2 = ax2.transData.transform((0, v2))
	inv = ax2.transData.inverted()
	_, dy = inv.transform((0, 0)) - inv.transform((0, y1-y2))
	miny, maxy = ax2.get_ylim()
	ax2.set_ylim(miny+dy, maxy+dy)


def plot_MT(self, outfile='$outdir/centroid.png', facecolor='red'):
	"""
	Plot the beachball of the best solution ``self.centroid``.
	
	:param outfile: path to the file where to plot; if ``None``, plot to the screen
	:type outfile: string, optional
	:param facecolor: color of the colored quadrants/parts of the beachball
	"""
	outfile = outfile.replace('$outdir', self.outdir)
	fig = plt.figure(figsize=(5,5))
	ax = plt.axes()
	plt.axis('off')
	ax.axes.get_xaxis().set_visible(False)
	ax.axes.get_yaxis().set_visible(False)
	lw=2
	plt.xlim(-100-lw/2, 100+lw/2)
	plt.ylim(-100-lw/2, 100+lw/2)

	a = self.centroid['a']
	mt2 = a2mt(a, system='USE')
	#beachball(mt2, outfile=outfile)
	full = beach(mt2, linewidth=lw, facecolor=facecolor, edgecolor='black', zorder=1)
	ax.add_collection(full)
	if self.decompose:
		dc = beach((self.centroid['s1'], self.centroid['d1'], self.centroid['r1']), nofill=True, linewidth=lw/2, zorder=2)
		ax.add_collection(dc)
	if outfile:
		plt.savefig(outfile, bbox_inches='tight', pad_inches=0)
	else:
		plt.show()
	plt.clf()
	plt.close()
	#mt = [-a[3,0]+a[5,0], -a[4,0]+a[5,0], a[3,0]+a[4,0]+a[5,0], a[0,0], a[1,0], -a[2,0]] # [M11, M22, M33, M12, M13, M23] in NEZ system
	#beachball(mt, mopad_basis='NED')
	#mt = [mt[2], mt[0], mt[1], mt[4], -mt[5], -mt[3]]
	#beachball(mt)
	#beachball2(mt)

def plot_uncertainty(self, outfile='$outdir/uncertainty.png', n=200, reference=None, best=True, fontsize=None):
	"""
	Generates random realizations based on the calculated solution and its uncertainty and plots these mechanisms and histograms of its parameters.
	
	:param outfile: Path to the file where to plot. If ``None``, plot to the screen. Because more figures are plotted, inserts an identifier before the last dot (`uncertainty.png` -> `uncertainty_MT.png`, `intertanty_time-shift.png`, etc.).
	:type outfile: string, optional
	:param n: number of realizations
	:type n: integer, optional
	:param reference: plot a given reference solution too; the value should be length 6 array of moment tensor in 'NEZ' coordinates or a moment tensor decomposition produced by :func:`decompose`
	:type reference: array or dictionary
	:param best: show the best solutions together too
	:type best: boolean, optional
	:param fontsize: fontsize for histogram annotations
	:type fontsize: scalar, optional
	"""
	
	# Generate mechanisms
	shift = []; depth = []; NS = []; EW = []
	n_sum = 0
	A = []
	c = self.centroid
	for gp in self.grid:
		if gp['err']:
			continue
		for i in gp['shifts']:
			GP = gp['shifts'][i]
			n_GP = int(round(GP['c'] / self.sum_c * n))
			if n_GP == 0:
				continue
			n_sum += n_GP
			a = GP['a']
			if self.deviatoric:
				a = a[:5]
			cov = gp['GtGinv']
			A2 = np.random.multivariate_normal(a.T[0], cov, n_GP)
			for a in A2:
				a = a[np.newaxis].T
				if self.deviatoric:
					a = np.append(a, [[0.]], axis=0)
				A.append(a)
			shift += [self.shifts[i]] * n_GP
			depth += [gp['z']/1e3] * n_GP
			NS    += [gp['x']/1e3] * n_GP
			EW    += [gp['y']/1e3] * n_GP
	if n_sum <= 1:
		self.log('\nUncertainty evaluation: nothing plotted. Posterior probability density function too wide or prefered number of mechanism ({0:d}) too low.'.format(n))
		return False
	# Process mechanisms
	dc_perc = []; clvd_perc = []; iso_perc = []; moment = []; Mw = []; strike = []; dip = []; rake = []
	for a in A:
		mt = a2mt(a)
		MT = decompose(mt)
		dc_perc.append(MT['dc_perc'])
		clvd_perc.append(MT['clvd_perc'])
		iso_perc.append(MT['iso_perc'])
		moment.append(MT['mom'])
		Mw.append(MT['Mw'])
		strike += [MT['s1'], MT['s2']]
		dip    += [MT['d1'], MT['d2']]
		rake   += [MT['r1'], MT['r2']]
	
	# Compute standard deviation
	stdev = {'dc':np.std(dc_perc)/100, 'clvd':np.std(clvd_perc)/100, 'iso':np.std(iso_perc)/100, 'Mw':np.std(Mw)/0.2, 't':np.std(shift), 'x':np.std(NS), 'y':np.std(EW), 'z':np.std(depth)}
	
	# Compute standard deviation of strike/dip/rake # TODO
	strike1 = []; dip1 = []; rake1 = []
	count2 = 0
	for str in strike:
		if (str > 0) and (str < 50):
			strike1.extend([str])
			dip1.extend([dip[count2]])
		count2 = count2 + 1
	for rk in rake:
		if (rk > -50) and (rk < 50):
			rake1.extend([rk])
	stdev2 = {'strike':np.std(strike1), 'dip':np.std(dip1), 'rake':np.std(rake1)}
	
	# Plot centroid uncertainty
	fig = plt.figure(figsize=(5,5))
	ax = plt.axes()
	plt.axis('off')
	ax.axes.get_xaxis().set_visible('off')
	ax.axes.get_yaxis().set_visible('off')
	lw=0.5
	plt.xlim(-100-lw/2, 100+lw/2)
	plt.ylim(-100-lw/2, 100+lw/2)
	for a in A:
		mt2 = a2mt(a, system='USE')
		try:
			full = beach(mt2, linewidth=lw, nofill=True, edgecolor='black', alpha=0.1)
			ax.add_collection(full)
		except:
			print('plotting this moment tensor failed: ', mt2)
	if best:
		mt2 = a2mt(c['a'], system='USE')
		full = beach(mt2, linewidth=lw*3, nofill=True, edgecolor=(0.,1.,0.2))
		ax.add_collection(full)
	if reference and len(reference)==6:
		ref = decompose(reference)
		mt2 = (reference[2], reference[0], reference[1], reference[4], -reference[5], -reference[3])
		full = beach(mt2, linewidth=lw*3, nofill=True, edgecolor='red')
		ax.add_collection(full)
	elif reference:
		ref = reference
		if 'mom' in ref and not 'Mw' in ref:
			ref['Mw'] = 2./3. * np.log10(ref['mom']) - 18.1/3.
		elif 'Mw' in ref and not 'mom' in ref:
			ref['mom'] = 10**((ref['Mw']+18.1/3.)*1.5)
	else:
		ref = {'dc_perc':None, 'clvd_perc':None, 'iso_perc':None, 'mom':None, 'Mw':None, 's1':0, 's2':0, 'd1':0, 'd2':0, 'r1':0, 'r2':0}
	outfile = outfile.replace('$outdir', self.outdir)
	k = outfile.rfind(".")
	s1 = outfile[:k]+'_'; s2 = outfile[k:]
	if outfile:
		plt.savefig(s1+'MT'+s2, bbox_inches='tight', pad_inches=0)
	else:
		plt.show()
	plt.clf()
	plt.close()

	fig = plt.figure(figsize=(5,5))
	ax = plt.axes()
	plt.axis('off')
	ax.axes.get_xaxis().set_visible('off')
	ax.axes.get_yaxis().set_visible('off')
	lw=0.5
	plt.xlim(-100-lw/2, 100+lw/2)
	plt.ylim(-100-lw/2, 100+lw/2)
	for i in range(0, len(strike), 2):
		try:
			dc = beach((strike[i], dip[i], rake[i]), linewidth=lw, nofill=True, edgecolor='black', alpha=0.1)
			ax.add_collection(dc)
		except:
			print('plotting this moment strike / dip / rake failed: ', (strike[i], dip[i], rake[i]))
	if best and self.decompose:
		dc = beach((c['s1'], c['d1'], c['r1']), nofill=True, linewidth=lw*3, edgecolor=(0.,1.,0.2))
		ax.add_collection(dc)
	if reference:
			dc = beach((ref['s1'], ref['d1'], ref['r1']), linewidth=lw*3, nofill=True, edgecolor='red')
			ax.add_collection(dc)
	if outfile:
		plt.savefig(s1+'MT_DC'+s2, bbox_inches='tight', pad_inches=0)
	else:
		plt.show()
	plt.clf()
	plt.close()
	
	# Plot histograms
	histogram(dc_perc,   s1+'comp-1-DC'+s2,     bins=(10,100), range=(0,100), xlabel='DC %', reference=ref['dc_perc'], reference2=(None, c['dc_perc'])[best], fontsize=fontsize)
	histogram(clvd_perc, s1+'comp-2-CLVD'+s2,   bins=(20,200), range=(-100,100), xlabel='CLVD %', reference=ref['clvd_perc'], reference2=(None, c['clvd_perc'])[best], fontsize=fontsize)
	if not self.deviatoric:
		histogram(iso_perc,  s1+'comp-3-ISO'+s2,    bins=(20,200), range=(-100,100), xlabel='ISO %', reference=ref['iso_perc'], reference2=(None, c['iso_perc'])[best], fontsize=fontsize)
	#histogram(moment,    s1+'mech-0-moment'+s2, bins=20, range=(self.mt_decomp['mom']*0.7,self.mt_decomp['mom']*1.4), xlabel='scalar seismic moment [Nm]', reference=ref['mom'], fontsize=fontsize)
	histogram(moment,    s1+'mech-0-moment'+s2, bins=20, range=(self.mt_decomp['mom']*0.7/2,self.mt_decomp['mom']*1.4*2), xlabel='scalar seismic moment [Nm]', reference=ref['mom'], fontsize=fontsize)
	#histogram(Mw,        s1+'mech-0-Mw'+s2,     bins=20, range=(self.mt_decomp['Mw']-0.1,self.mt_decomp['Mw']+0.1), xlabel='moment magnitude $M_W$', reference=ref['Mw'], fontsize=fontsize)
	histogram(Mw,        s1+'mech-0-Mw'+s2,     bins=20, range=(self.mt_decomp['Mw']-0.1*3,self.mt_decomp['Mw']+0.1*3), xlabel='moment magnitude $M_W$', reference=ref['Mw'], reference2=(None, c['Mw'])[best], fontsize=fontsize)
	histogram(strike,    s1+'mech-1-strike'+s2, bins=72, range=(0,360), xlabel=u'strike [°]', multiply=2, reference=((ref['s1'], ref['s2']), None)[reference==None], reference2=(None, (c['s1'], c['s2']))[best], fontsize=fontsize)
	histogram(dip,       s1+'mech-2-dip'+s2,    bins=18, range=(0,90), xlabel=u'dip [°]', multiply=2, reference=((ref['d1'], ref['d2']), None)[reference==None], reference2=(None, (c['d1'], c['d2']))[best], fontsize=fontsize)
	histogram(rake,      s1+'mech-3-rake'+s2,   bins=72, range=(-180,180), xlabel=u'rake [°]', multiply=2, reference=((ref['r1'], ref['r2']), None)[reference==None], reference2=(None, (c['r1'], c['r2']))[best], fontsize=fontsize)
	if len(self.shifts) > 1:
		shift_step = self.SHIFT_step / self.max_samprate
		histogram(shift,     s1+'time-shift'+s2,    bins=len(self.shifts), range=(self.shifts[0]-shift_step/2., self.shifts[-1]+shift_step/2.), xlabel='time shift [s]', reference=[0., None][reference==None], reference2=(None, c['shift'])[best], fontsize=fontsize)
	if len (self.depths) > 1:
		min_depth = (self.depths[0]-self.step_z/2.)/1e3
		max_depth = (self.depths[-1]+self.step_z/2.)/1e3
		histogram(depth,     s1+'place-depth'+s2,   bins=len(self.depths), range=(min_depth, max_depth), xlabel='centroid depth [km]', reference=[self.event['depth']/1e3, None][reference==None], reference2=(None, c['z']/1e3)[best], fontsize=fontsize)
	if len(self.grid) > len(self.depths):
		x_lim = (self.steps_x[-1]+self.step_x/2.)/1e3
		histogram(NS,        s1+'place-NS'+s2,      bins=len(self.steps_x), range=(-x_lim, x_lim), xlabel=u'← to south : centroid place [km] : to north →', reference=[0., None][reference==None], reference2=(None, c['x']/1e3)[best], fontsize=fontsize)
		histogram(EW,        s1+'place-EW'+s2,      bins=len(self.steps_x), range=(-x_lim, x_lim), xlabel=u'← to west : centroid place [km] : to east →', reference=[0., None][reference==None], reference2=(None, c['y']/1e3)[best], fontsize=fontsize)

	self.log('\nUncertainty evaluation: plotted {0:d} mechanism of {1:d} requested.'.format(n_sum, n))
	self.log('Standard deviation :: dc: {dc:4.2f}, clvd: {clvd:4.2f}, iso: {iso:4.2f}, Mw: {Mw:4.2f}, t: {t:4.2f}, x: {x:4.2f}, y: {y:4.2f}, z: {z:4.2f}'.format(**stdev))
	return stdev

def plot_MT_uncertainty_centroid(self, outfile='$outdir/MT_uncertainty_centroid.png', n=100):
	"""
	Similar as :func:`plot_uncertainty`, but only the best point of the space-time grid is taken into account, so the uncertainties should be Gaussian.
	"""
	a = self.centroid['a']
	if self.deviatoric:
		a = a[:5]
	cov = self.centroid['GtGinv']
	A = np.random.multivariate_normal(a.T[0], cov, n)

	fig = plt.figure(figsize=(5,5))
	ax = plt.axes()
	plt.axis(False)
	ax.axes.get_xaxis().set_visible(False)
	ax.axes.get_yaxis().set_visible(False)
	lw=0.5
	plt.xlim(-100-lw/2, 100+lw/2)
	plt.ylim(-100-lw/2, 100+lw/2)

	for a in A:
		a = a[np.newaxis].T
		if self.deviatoric:
			a = np.append(a, [[0.]], axis=0)
		mt2 = a2mt(a, system='USE')
		#full = beach(mt2, linewidth=lw, nofill=True, edgecolor='black')
		try:
			full = beach(mt2, linewidth=lw, nofill=True, edgecolor='black')
		except:
			print(a)
			print(mt2)
		ax.add_collection(full)
	if outfile:
		plt.savefig(outfile.replace('$outdir', self.outdir), bbox_inches='tight', pad_inches=0)
	else:
		plt.show()
	plt.clf()
	plt.close()
	
def plot_maps(self, outfile='$outdir/map.png', beachball_size_c=False):
	"""
	Plot figures showing how the solution is changing across the grid.
	
	:param outfile: Path to the file where to plot. If ``None``, plot to the screen. Because one figure is plotted for each depth, inserts an identifier before the last dot (`map.png` -> `map_1000.png`, `map_2000.png`, etc.).
	:type outfile: string, optional
	:param beachball_size_c: If ``True``, the sizes of the beachballs correspond to the posterior probability density function (PPD) instead of the variance reduction VR
	:type beachball_size_c: bool, optional
	
	Plot top view to the grid at each depth. The solutions in each grid point (for the centroid time with the highest VR) are shown by beachballs. The color of the beachball corresponds to its DC-part. The inverted centroid time is shown by a contour in the background and the condition number by contour lines.
	"""
	outfile = outfile.replace('$outdir', self.outdir)
	r = self.radius * 1e-3 * 1.1 # to km, *1.1
	if beachball_size_c:
		max_width = np.sqrt(self.max_sum_c)
	for z in self.depths:
		# prepare data points
		x=[]; y=[]; s=[]; CN=[]; MT=[]; color=[]; width=[]; highlight=[]
		for gp in self.grid:
			if gp['z'] != z or gp['err']:
				continue
			x.append(gp['y']/1e3); y.append(gp['x']/1e3); s.append(gp['shift']); CN.append(gp['CN']) # NS is x coordinate, so switch it with y to be vertical
			MT.append(a2mt(gp['a'], system='USE'))
			VR = max(gp['VR'],0)
			if beachball_size_c:
				width.append(self.step_x/1e3 * np.sqrt(gp['sum_c']) / max_width)
			else:
				width.append(self.step_x/1e3*VR)
			if self.decompose:
				dc = float(gp['dc_perc'])/100
				color.append((dc, 0, 1-dc))
			else:
				color.append('black')
			highlight.append(self.centroid['id'] == gp['id'])
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
	outfile = outfile.replace('$outdir', self.outdir)
	if point:
		x0, y0 = point
	else:
		x0 = self.centroid['x']; y0 = self.centroid['y']
	depth_min = self.depth_min / 1000; depth_max = self.depth_max / 1000
	depth = depth_max - depth_min
	r = self.radius * 1e-3 * 1.1 # to km, *1.1
	if beachball_size_c:
		max_width = np.sqrt(self.max_sum_c)
	for slice in ('N-S', 'W-E', 'NW-SE', 'SW-NE'):
		x=[]; y=[]; s=[]; CN=[]; MT=[]; color=[]; width=[]; highlight=[]
		for gp in self.grid:
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
				width.append(self.step_x/1e3 * np.sqrt(gp['sum_c']) / max_width)
			else:
				width.append(self.step_x/1e3*VR)
			if self.decompose:
				dc = float(gp['dc_perc'])/100
				color.append((dc, 0, 1-dc))
			else:
				color.append('black')
			highlight.append(self.centroid['id'] == gp['id'])
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
	if not self.Cd_inv:
		return False # if the data covariance matrix is unitary, we have no estimation of data errors, so the PDF has good sense
	r = self.radius * 1e-3 * 1.1 # to km, *1.1
	depth_min = self.depth_min * 1e-3; depth_max = self.depth_max * 1e-3
	depth = depth_max - depth_min
	Ymin = depth_max + depth*0.05
	Ymax = depth_min - depth*0.05
	#for slice in ('N-S', 'W-E', 'NW-SE', 'SW-NE', 'top'):
	for slice in ('N-S', 'W-E', 'top'):
		X=[]; Y=[]; s=[]; CN=[]; MT=[]; color=[]; width=[]; highlight=[]
		g = {}
		max_c = 0
		for gp in self.grid:
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
				if self.decompose:
					g[x][y]['dc'] = gp['dc_perc']
			if self.centroid['id'] == gp['id']:
				g[x][y]['highlight'] = True
		for x in g:
			for y in g[x]:
				X.append(x)
				Y.append(y)
				#s.append(g[x][y]['s'])
				#CN.append(g[x][y]['CN'])
				MT.append(a2mt(g[x][y]['a'], system='USE'))
				if self.decompose:
					dc = float(g[x][y]['dc'])*0.01
					color.append((dc, 0, 1-dc))
				else:
					color.append('black')
				highlight.append(g[x][y]['highlight'])
				width.append(self.step_x*1e-3 * np.sqrt(g[x][y]['c'] / max_c))
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
			c = plt.Circle((x[i], y[i]), self.step_x/1e3*0.5, color='r')
			c.set_edgecolor('r')
			c.set_linewidth(10)
			c.set_facecolor('none')  # "none" not None
			c.set_alpha(0.7)
			ax.add_artist(c)
		if width[i] > self.step_x*1e-3 * 0.04:
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
		elif width[i] > self.step_x*1e-3 * 0.001:
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
			vmin=self.shift_min, vmax=self.shift_max, origin='lower',
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
	if self.decompose:
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
		r_max = self.step_x/1e3/2
		r_half = r_max/1.4142
		text_max = 'maximal PDF'
		text_half = 'half-of-maximum PDF'
		text_area = 'Beachball area ~ PDF'
	else: # beachball's radius = VR
		VRmax = self.centroid['VR']
		r_max = self.step_x/1e3/2 * VRmax
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
	n = len(self.grid)
	x = np.zeros(n); y = np.zeros(n); z = np.zeros(n); VR = np.zeros(n)
	c = np.zeros((n, 3))
	for i in range(len(self.grid)):
		gp = self.grid[i]
		if gp['err']:
			continue
		x[i] = gp['y']/1e3
		y[i] = gp['x']/1e3 # NS is x coordinate, so switch it with y to be vertical
		z[i] = gp['z']/1e3
		vr = max(gp['VR'],0)
		VR[i] = np.pi * (15 * vr)**2
		c[i] = np.array([vr, 0, 1-vr])
		#if self.decompose:
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

def plot_seismo(self, outfile='$outdir/seismo.png', comp_order='ZNE', cholesky=False, obs_style='k', obs_width=3, synt_style='r', synt_width=2, add_file=None, add_file_style='k:', add_file_width=2, add_file2=None, add_file2_style='b-', add_file2_width=2, plot_stations=None, plot_components=None, sharey=False):
	"""
	Plots the fit between observed and simulated seismogram.
	
	:param outfile: path to file for plot output; if ``None`` plots to the screen
	:type outfile: string, optional
	:param comp_order: order of component in the plot, supported are 'ZNE' (default) and 'NEZ'
	:type comp_order: string, optional
	:param cholesky: plots standardized seismogram instead of original ones
	:type cholesky: bool, optional
	:param obs_style: line style for observed data
	:param obs_width: line width for observed data
	:param synt_style: line style for simulated data
	:param synt_width: line width for simulated data
	:param add_file: path to a reference file generated by function :func:`save_seismo`
	:type add_file: string or None, optional
	:param add_file_style: line style for reference data
	:param add_file_width: line width for reference data
	:param add_file2: path to second reference file
	:type add_file2: string or None, optional
	:param add_file2_style: line style for reference data
	:param add_file2_width: line width for reference data
	:param plot_stations: list of stations to plot; if ``None`` plots all stations
	:type plot_stations: list or None, optional
	:param plot_components: list of components to plot; if ``None`` plots all components
	:type plot_components: list or None, optional
	:param sharey: if ``True`` the y-axes for all stations have the same limits, otherwise the limits are chosen automatically for every station
	:type sharey: bool, optional
	"""
	if cholesky and not len(self.LT) and not len(self.LT3):
		raise ValueError('Covariance matrix not set. Run "covariance_matrix()" first.')
	data = self.data_shifts[self.centroid['shift_idx']]
	npts = self.npts_slice
	samprate = self.samprate
	elemse = read_elemse(self.nr, self.npts_elemse, 'green/elemse'+self.centroid['id']+'.dat', self.stations, self.invert_displacement) # nacist elemse
	#if not no_filter:
	for r in range(self.nr):
		for e in range(6):
			my_filter(elemse[r][e], self.stations[r]['fmin'], self.stations[r]['fmax'])
			elemse[r][e].trim(UTCDateTime(0)+self.elemse_start_origin)

	plot_stations, comps, f, ax, ea = self.plot_seismo_backend_1(plot_stations, plot_components, comp_order, sharey=(cholesky or sharey), title_prefix=('','pseudo ')[cholesky and self.LT3!=[]], ylabel=('velocity [m/s]', None)[cholesky])
	
	t = np.arange(0, (npts-0.5) / samprate, 1. / samprate)
	if add_file:
		add = np.load(add_file)
	if add_file2:
		add2 = np.load(add_file2)
	d_max = 0
	for sta in plot_stations:
		r = plot_stations.index(sta)
		#if no_filter:
			#SAMPRATE = self.data_unfiltered[sta][0].stats.sampling_rate
			#NPTS = int(npts/samprate * SAMPRATE), 
			#SHIFT = int(round(self.centroid['shift']*SAMPRATE))
			#T = np.arange(0, (NPTS-0.5) / SAMPRATE, 1. / SAMPRATE)
		SYNT = {}
		for comp in range(3):
			SYNT[comp] = np.zeros(npts)
			for e in range(6):
				SYNT[comp] += elemse[sta][e][comp].data[0:npts] * self.centroid['a'][e,0]
		comps_used = 0
		for comp in comps:
			synt = SYNT[comp]
			#if no_filter:
				#D = np.empty(NPTS)
				#for i in range(NPTS):
					#if i+SHIFT >= 0:	
						#D[i] = self.data_unfiltered[sta][comp].data[i+SHIFT]
			#else:
			d = data[sta][comp][0:len(t)]
			if cholesky and self.stations[sta][{0:'useZ', 1:'useN', 2:'useE'}[comp]]:
				if self.LT3:
					#print(r, comp) # DEBUG
					d    = np.zeros(npts)
					synt = np.zeros(npts)
					x1 = -npts
					for COMP in range(3):
						if not self.stations[sta][{0:'useZ', 1:'useN', 2:'useE'}[COMP]]:
							continue
						x1 += npts; x2 = x1+npts
						y1 = comps_used*npts; y2 = y1+npts
						#print(self.LT3[sta][y1:y2, x1:x2].shape, data[sta][COMP].data[0:npts].shape) # DEBUG
						d    += np.dot(self.LT3[sta][y1:y2, x1:x2], data[sta][COMP].data[0:npts])
						synt += np.dot(self.LT3[sta][y1:y2, x1:x2], SYNT[COMP])
				else:
					d    = np.dot(self.LT[sta][comp], d)
					synt = np.dot(self.LT[sta][comp], synt)
				comps_used += 1
			c = comps.index(comp)
			#if no_filter:
				#ax[r,c].plot(T,D, color='k', linewidth=obs_width)
			if self.stations[sta][{0:'useZ', 1:'useN', 2:'useE'}[comp]] or not cholesky: # do not plot seismogram if the component is not used and Cholesky decomposition is plotted
				l_d, = ax[r,c].plot(t,d, obs_style, linewidth=obs_width)
				if self.stations[sta][{0:'useZ', 1:'useN', 2:'useE'}[comp]]:
					d_max = max(max(d), -min(d), d_max)
			else:
				ax[r,c].plot([0],[0], 'w', linewidth=0)
			if self.stations[sta][{0:'useZ', 1:'useN', 2:'useE'}[comp]]:
				l_s, = ax[r,c].plot(t,synt, synt_style, linewidth=synt_width)
				d_max = max(max(synt), -min(synt), d_max)
			else:
				if not cholesky:
					ax[r,c].plot(t,synt, color='gray', linewidth=2)
			if add_file:
				ax[r,c].plot(t, add[:, 3*sta+comp], add_file_style, linewidth=add_file_width)
			if add_file2:
				ax[r,c].plot(t, add2[:, 3*sta+comp], add_file2_style, linewidth=add_file2_width)
	ax[-1,0].set_ylim([-d_max, d_max])
	ea.append(f.legend((l_d, l_s), ('inverted data', 'modeled (synt)'), loc='lower center', bbox_to_anchor=(0.5, 1.-0.0066*len(plot_stations)), ncol=2, numpoints=1, fontsize='small', fancybox=True, handlelength=3)) # , borderaxespad=0.1
	ea.append(f.text(0.1, 1.06-0.004*len(plot_stations), 'x', color='white', ha='center', va='center'))
	self.plot_seismo_backend_2(outfile.replace('$outdir', self.outdir), plot_stations, comps, ax, extra_artists=ea)

def plot_covariance_function(self, outfile='$outdir/covariance.png', comp_order='ZNE', crosscovariance=False, style='k', width=2, plot_stations=None, plot_components=None):
	"""
	Plots the covariance functions on whose basis is the data covariance matrix generated
	
	:param outfile: path to file for plot output; if ``None`` plots to the screen
	:type outfile: string, optional
	:param comp_order: order of component in the plot, supported are 'ZNE' (default) and 'NEZ'
	:type comp_order: string, optional
	:param crosscovariance: if ``True`` plots also the crosscovariance between components
	:param crosscovariance: bool, optional
	:param style: line style
	:param width: line width
	:param plot_stations: list of stations to plot; if ``None`` plots all stations
	:type plot_stations: list or None, optional
	:param plot_components: list of components to plot; if ``None`` plots all components
	:type plot_components: list or None, optional
	"""
	if not len(self.Cf):
		raise ValueError('Covariance functions not calculated or not saved. Run "covariance_matrix(save_covariance_function=True)" first.')
	data = self.data_shifts[self.centroid['shift_idx']]
	
	plot_stations, comps, f, ax, ea = self.plot_seismo_backend_1(plot_stations, plot_components, comp_order, crosscomp=crosscovariance, yticks=False, ylabel=None)
	
	dt = 1. / self.samprate
	t = np.arange(-np.floor(self.Cf_len/2) * dt, (np.floor(self.Cf_len/2)+0.5) * dt, dt)
	COMPS = (1,3)[crosscovariance]
	for sta in plot_stations:
		r = plot_stations.index(sta)
		for comp in comps:
			c = comps.index(comp)
			for C in range(COMPS): # if crosscomp==False: C = 0
				d = self.Cf[sta][(comp,C)[crosscovariance],comp]
				#if len(t) != len(d): # DEBUG
					#t = np.arange(-np.floor(len(d)/2) * dt, (np.floor(len(d)/2)+0.5) * dt, dt) # DEBUG
					#print(len(d), len(t)) # DEBUG
				if type(d)==np.ndarray and self.stations[sta][{0:'useZ', 1:'useN', 2:'useE'}[comp]]:
					color = style
					if len(t) != len(d):
						t = np.arange(-np.floor(len(d)/2) * dt, (np.floor(len(d)/2)+0.5) * dt, dt)
					ax[COMPS*r+C,c].plot(t, d, color=style, linewidth=width)
				else:
					ax[COMPS*r+C,c].plot([0],[0], 'w', linewidth=0)
		if crosscovariance:
			ax[3*r,  0].set_ylabel(' \n Z ')
			ax[3*r+1,0].set_ylabel(data[sta][0].stats.station + '\n N ')
			ax[3*r+2,0].set_ylabel(' \n E ')
	self.plot_seismo_backend_2(outfile.replace('$outdir', self.outdir), plot_stations, comps, ax, yticks=False, extra_artists=ea)

def plot_noise(self, outfile='$outdir/noise.png', comp_order='ZNE', obs_style='k', obs_width=2, plot_stations=None, plot_components=None):
	"""
	Plots the noise records from which the covariance matrix is calculated together with the inverted data
	
	:param outfile: path to file for plot output; if ``None`` plots to the screen
	:type outfile: string, optional
	:param comp_order: order of component in the plot, supported are 'ZNE' (default) and 'NEZ'
	:type comp_order: string, optional
	:param obs_style: line style
	:param obs_width: line width
	:param plot_stations: list of stations to plot; if ``None`` plots all stations
	:type plot_stations: list or None, optional
	:param plot_components: list of components to plot; if ``None`` plots all components
	:type plot_components: list or None, optional
	"""
	samprate = self.samprate
	
	plot_stations, comps, f, ax, ea = self.plot_seismo_backend_1(plot_stations, plot_components, comp_order)
	
	t = np.arange(0, (self.npts_slice-0.5) / samprate, 1. / samprate)
	d_max = 0
	for sta in plot_stations:
		r = plot_stations.index(sta)
		for comp in comps:
			d = self.data_shifts[self.centroid['shift_idx']][sta][comp][0:len(t)]
			c = comps.index(comp)
			if self.stations[sta][{0:'useZ', 1:'useN', 2:'useE'}[comp]]:
				color = obs_style
				d_max = max(max(d), -min(d), d_max)
			else:
				color = 'gray'
			ax[r,c].plot(t, d, color, linewidth=obs_width)
			if len(self.noise[sta]) > comp:
				NPTS = len(self.noise[sta][comp].data)
				T = np.arange(-NPTS * 1. / samprate, -0.5 / samprate, 1. / samprate)
				ax[r,c].plot(T, self.noise[sta][comp], color, linewidth=obs_width)
				if self.stations[sta][{0:'useZ', 1:'useN', 2:'useE'}[comp]]:
					d_max = max(max(self.noise[sta][comp]), -min(self.noise[sta][comp]), d_max)
	ax[-1,0].set_ylim([-d_max, d_max])
	ymin, ymax = ax[r,c].get_yaxis().get_view_interval()
	for r in range(len(plot_stations)):
		for i in range(len(comps)):
			l4 = ax[r,i].add_patch(mpatches.Rectangle((-NPTS/samprate, -ymax), NPTS/samprate, 2*ymax, color=(1.0, 0.6, 0.4))) # (x,y), width, height
			l5 = ax[r,i].add_patch(mpatches.Rectangle((0, -ymax), self.npts_slice/samprate, 2*ymax, color=(0.7, 0.7, 0.7)))
	ea.append(f.legend((l4, l5), ('$C_D$', 'inverted'), 'lower center', bbox_to_anchor=(0.5, 1.-0.0066*len(plot_stations)), ncol=2, fontsize='small', fancybox=True, handlelength=3, handleheight=1.2)) # , borderaxespad=0.1
	ea.append(f.text(0.1, 1.06-0.004*len(plot_stations), 'x', color='white', ha='center', va='center'))
	self.plot_seismo_backend_2(outfile.replace('$outdir', self.outdir), plot_stations, comps, ax, extra_artists=ea)
	
def plot_spectra(self, outfile='$outdir/spectra.png', comp_order='ZNE', plot_stations=None, plot_components=None):
	"""
	Plots spectra of inverted data, standardized data, and before-event noise together
	
	:param outfile: path to file for plot output; if ``None`` plots to the screen
	:type outfile: string, optional
	:param comp_order: order of component in the plot, supported are 'ZNE' (default) and 'NEZ'
	:type comp_order: string, optional
	:param plot_stations: list of stations to plot; if ``None`` plots all stations
	:type plot_stations: list or None, optional
	:param plot_components: list of components to plot; if ``None`` plots all components
	:type plot_components: list or None, optional
	"""
	if not len(self.LT) and not len(self.LT3):
		raise ValueError('Covariance matrix not set. Run "covariance_matrix()" first.')
	data = self.data_shifts[self.centroid['shift_idx']]
	npts = self.npts_slice
	samprate = self.samprate

	plot_stations, comps, fig, ax, ea = self.plot_seismo_backend_1(plot_stations, plot_components, comp_order, yticks=False, xlabel='frequency [Hz]', ylabel='amplitude spectrum [m/s]')
	
	#plt.yscale('log')
	ax3 = np.empty_like(ax)
	fmin = np.zeros_like(ax, dtype=float)
	fmax = np.zeros_like(fmin)
	for i in range(len(plot_stations)):
		for j in range(len(comps)):
			#ax[i,j].set_yscale('log')
			ax3[i,j] = ax[i,j].twinx()
			#ax3[i,j].set_yscale('log')
	ax3[0,0].get_shared_y_axes().join(*ax3.flatten().tolist())

	dt = 1./samprate
	DT = 0.5*dt
	f = np.arange(0, samprate*0.5 * (1-0.5/npts), samprate / npts)
	D_filt_max = 0
	for sta in plot_stations:
		r = plot_stations.index(sta)
		SYNT = {}
		comps_used = 0
		for comp in comps:
			d = data[sta][comp][0:npts]
			d_filt = d.copy()
			c = comps.index(comp)
			if self.stations[sta][{0:'useZ', 1:'useN', 2:'useE'}[comp]]:
				if self.LT3:
					d_filt = np.zeros(npts)
					x1 = -npts
					for COMP in comps:
						if not self.stations[sta][{0:'useZ', 1:'useN', 2:'useE'}[COMP]]:
							continue
						x1 += npts; x2 = x1+npts
						y1 = comps_used*npts; y2 = y1+npts
						d_filt += np.dot(self.LT3[sta][y1:y2, x1:x2], data[sta][COMP].data[0:npts])
				else:
					d_filt = np.dot(self.LT[sta][comp], d)
				comps_used += 1
				fmin[r,c] = self.stations[sta]['fmin']
				fmax[r,c] = self.stations[sta]['fmax']
			ax[r,c].tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
			ax[r,c].yaxis.offsetText.set_visible(False) 
			ax3[r,c].get_yaxis().set_visible(False)
			if self.stations[sta][{0:'useZ', 1:'useN', 2:'useE'}[comp]]:
				noise = self.noise[sta][comp]
				NPTS = len(noise)
				NOISE  = np.sqrt(np.square(np.real(np.fft.fft(noise))*DT)*npts*dt / (NPTS*DT))
				f2 = np.arange(0, samprate*1. * (1-0.5/NPTS), samprate*2 / NPTS)
				D      = np.absolute(np.real(np.fft.fft(d))*dt)
				D_filt = np.absolute(np.real(np.fft.fft(d_filt))*dt)
				D_filt_max = max(D_filt_max, max(D_filt))
				l_d,     = ax[r,c].plot(f, D[0:len(f)],          'k', linewidth=2, zorder=2)
				l_filt, = ax3[r,c].plot(f, D_filt[0:len(f)],     'r', linewidth=1, zorder=3)
				l_noise, = ax[r,c].plot(f2, NOISE[0:len(f2)], 'gray', linewidth=4, zorder=1)
			else:
				ax[r,c].plot([0],[0], 'w', linewidth=0)
	#y3min, y3max = ax3[-1,0].get_yaxis().get_view_interval()
	ax3[-1,0].set_ylim([0, D_filt_max])
	#print (D_filt_max, y3max, y3min)
	align_yaxis(ax[0,0], ax3[0,0])
	ax[0,0].set_xlim(0, self.fmax*1.5)
	#ax[0,0].set_xscale('log')
	#f.legend((l4, l5), ('$C_D$', 'inverted'), 'upper center', ncol=2, fontsize='small', fancybox=True)
	ea.append(fig.legend((l_d, l_filt, l_noise), ('data', 'standardized data (by $C_D$)', 'noise'), loc='lower center', bbox_to_anchor=(0.5, 1.-0.0066*len(plot_stations)), ncol=3, numpoints=1, fontsize='small', fancybox=True, handlelength=3)) # , borderaxespad=0.1
	ea.append(fig.text(0.1, 1.06-0.004*len(plot_stations), 'x', color='white', ha='center', va='center'))
	ymin, ymax = ax[r,c].get_yaxis().get_view_interval()
	for r in range(len(plot_stations)):
		for c in range(len(comps)):
			if fmax[r,c]:
				ax[r,c].add_artist(Line2D((fmin[r,c], fmin[r,c]), (0, ymax), color='g', linewidth=1))
				ax[r,c].add_artist(Line2D((fmax[r,c], fmax[r,c]), (0, ymax), color='g', linewidth=1))
	self.plot_seismo_backend_2(outfile.replace('$outdir', self.outdir), plot_stations, comps, ax, yticks=False, extra_artists=ea)

def plot_seismo_backend_1(self, plot_stations, plot_components, comp_order, crosscomp=False, sharey=True, yticks=True, title_prefix='', xlabel='time [s]', ylabel='velocity [m/s]'):
	"""
	The first part of back-end for functions :func:`plot_seismo`, :func:`plot_covariance_function`, :func:`plot_noise`, :func:`plot_spectra`. There is no need for calling it directly.
	"""
	data = self.data_shifts[self.centroid['shift_idx']]
	
	plt.rcParams.update({'font.size': 22})
	
	if not plot_stations:
		plot_stations = range(self.nr)
	if plot_components:
		comps = plot_components
	elif comp_order == 'NEZ':
		comps = [1, 2, 0]
	else:
		comps = [0, 1, 2]
	
	COMPS = (1,3)[crosscomp]
	f, ax = plt.subplots(len(plot_stations)*COMPS, len(comps), sharex=True, sharey=('row', True)[sharey], figsize=(len(comps)*6, len(plot_stations)*2*COMPS))
	if len(plot_stations)==1 and len(comps)>1: # one row only
		ax = np.reshape(ax, (1,len(comps)))
	elif len(plot_stations)>1 and len(comps)==1: # one column only
		ax = np.reshape(ax, (len(plot_stations),1))
	elif len(plot_stations)==1 and len(comps)==1: # one cell only
		ax = np.array([[ax]])

	for c in range(len(comps)):
		ax[0,c].set_title(title_prefix+data[0][comps[c]].stats.channel[2])

	for sta in plot_stations:
		r = plot_stations.index(sta)
		ax[r,0].set_ylabel(data[sta][0].stats.station + u"\n{0:1.0f} km, {1:1.0f}°".format(self.stations[sta]['dist']/1000, self.stations[sta]['az']), fontsize=16)
		#SYNT = {}
		#comps_used = 0
		for comp in comps:
			c = comps.index(comp)
			for C in range(COMPS): # if crosscomp==False: C = 0
				ax[COMPS*r+C,c].set_frame_on(False)
				ax[COMPS*r+C,c].locator_params(axis='x',nbins=7)
				ax[COMPS*r+C,c].tick_params(labelsize=16)
				if c==0:
					if yticks:
						ax[r,c].ticklabel_format(axis='y', style='sci', scilimits=(-2,2))
						ax[r,c].get_yaxis().tick_left()
					else:
						ax[COMPS*r+C,c].tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
						ax[COMPS*r+C,c].yaxis.offsetText.set_visible(False) 
				else:
					ax[COMPS*r+C,c].get_yaxis().set_visible(False)
				if r == len(plot_stations)-1 and C==COMPS-1:
					ax[COMPS*r+C,c].get_xaxis().tick_bottom()
				else:
					ax[COMPS*r+C,c].get_xaxis().set_visible(False)
	extra_artists = []
	if xlabel:
		extra_artists.append(f.text(0.5, 0.04+0.002*len(plot_stations), xlabel, ha='center', va='center'))
	if ylabel:
		extra_artists.append(f.text(0.04*(len(comps)-1)-0.02, 0.5, ylabel, ha='center', va='center', rotation='vertical'))
	return plot_stations, comps, f, ax, extra_artists

def plot_seismo_backend_2(self, outfile, plot_stations, comps, ax, yticks=True, extra_artists=None):
	"""
	The second part of back-end for functions :func:`plot_seismo`, :func:`plot_covariance_function`, :func:`plot_noise`, :func:`plot_spectra`. There is no need for calling it directly.
	"""
	xmin, xmax = ax[0,0].get_xaxis().get_view_interval()
	ymin, ymax = ax[-1,0].get_yaxis().get_view_interval()
	if yticks:
		for r in range(len(plot_stations)):
			ymin, ymax = ax[r,0].get_yaxis().get_view_interval()
			ymax = np.round(ymax,int(-np.floor(np.log10(ymax)))) # round high axis limit to first valid digit
			ax[r,0].add_artist(Line2D((xmin, xmin), (0, ymax), color='black', linewidth=1))
			ax[r,0].yaxis.set_ticks((0., ymax))  
	for c in range(len(comps)):
		ax[-1,c].add_artist(Line2D((xmin, xmax), (ymin, ymin), color='black', linewidth=2))
	if outfile:
		if extra_artists:
			plt.savefig(outfile, bbox_extra_artists=extra_artists, bbox_inches='tight')
			#plt.savefig(outfile, bbox_extra_artists=(legend,))
		else:
			plt.savefig(outfile, bbox_inches='tight')
	else:
		plt.show()
	plt.clf()
	plt.close('all')



def plot_stations(self, outfile='$outdir/stations.png', network=True, location=False, channelcode=False, fontsize=0):
	"""
	Plot a map of stations used in the inversion.
	
	:param outfile: path to file for plot output; if ``None`` plots to the screen
	:type outfile: string, optional
	:param network: include network code into station label
	:type network: bool, optional
	:param location: include location code into station label
	:type location: bool, optional
	:param channelcode: include channel code into station label
	:type channelcode: bool, optional
	:param fontsize: font size for all texts in the plot; if zero, the size is chosen automatically
	:type fontsize: scalar, optional
	
	
	The stations are marked according to components used in the inversion.
	"""
	if fontsize:
		plt.rcParams.update({'font.size': fontsize})
	plt.figure(figsize=(16, 12))
	plt.axis('equal')
	plt.xlabel('west - east [km]')
	plt.ylabel('south - north [km]')
	plt.title('Stations used in the inversion')
	plt.plot(self.centroid['y']/1e3, self.centroid['x']/1e3, marker='*', markersize=75, color='yellow', label='epicenter', linestyle='None')
	
	L1 = L2 = L3 = True
	for sta in self.stations:
		az = radians(sta['az'])
		dist = sta['dist']/1000 # from meter to kilometer
		y = cos(az)*dist # N
		x = sin(az)*dist # E
		label = None
		if sta['useN'] and sta['useE'] and sta['useZ']:
			color = 'red'
			if L1: label = 'all components used'; L1 = False
		elif not sta['useN'] and not sta['useE'] and not sta['useZ']:
			color = 'white'
			if L3: label = 'not used'; L3 = False
		else:
			color = 'gray'
			if L2: label = 'some components used'; L2 = False
		if network and sta['network']: l = sta['network']+':'
		else: l = ''
		l += sta['code']
		if location and sta['location']: l += ':'+sta['location']
		if channelcode: l += ' '+sta['channelcode']
		#sta['weightN'] = sta['weightE'] = sta['weightZ']
		plt.plot([x],[y], marker='^', markersize=18, color=color, label=label, linestyle='None')
		plt.annotate(l, xy=(x,y), xycoords='data', xytext=(0, -14), textcoords='offset points', horizontalalignment='center', verticalalignment='top', fontsize=14)
		#plt.legend(numpoints=1)
	plt.legend(bbox_to_anchor=(0., -0.15-fontsize*0.002, 1., .07), loc='lower left', ncol=4, numpoints=1, mode='expand', fontsize='small', fancybox=True)
	if outfile:
		plt.savefig(outfile.replace('$outdir', self.outdir), bbox_inches='tight')
	else:
		plt.show()
	plt.clf()
	plt.close()

def plot_covariance_matrix(self, outfile=None, normalize=False, cholesky=False, fontsize=60, colorbar=False):
	"""
	Plots figure of the data covariance matrix :math:`C_D`.
	
	:param outfile: path to file for plot output; if ``None`` plots to the screen
	:type outfile: string, optional
	:param normalize: normalize each blok (corresponding to one station) of the :math:`C_D` to the same value
	:type normalize: bool, optional
	:param cholesky: plots Cholesky decomposition of the covariance matrix :math:`L^T` instead of the :math:`C_D`
	:type cholesky: bool, optional
	:param fontsize: font size for all texts in the plot
	:type fontsize: scalar, optional
	:param colorbar: show a legend for the color map
	:type colorbar: bool, optional
	"""
	plt.figure(figsize=(55, 50))
	fig, ax = plt.subplots(1, 1)
	if fontsize:
		plt.rcParams.update({'font.size': fontsize})
	Cd = np.zeros((self.components*self.npts_slice, self.components*self.npts_slice))
	if not len(self.Cd):
		raise ValueError('Covariance matrix not set or not saved. Run "covariance_matrix(save_non_inverted=True)" first.')
	i = 0
	if cholesky and self.LT3:
		matrix = self.LT3
	elif cholesky:
		matrix = [item for sublist in self.LT for item in sublist]
	else:
		matrix = self.Cd
	for C in matrix:
		if type(C)==int:
			continue
		if normalize and len(C):
			mx = max(C.max(), abs(C.min()))
			C *= 1./mx
		l = len(C)
		Cd[i:i+l,i:i+l] = C
		i += l
		
	Cd = Cd * 1000000 # from m2 to mm2
	
	values = []
	labels = []
	i = 0
	n = self.npts_slice
	for stn in self.stations:
		if cholesky and self.LT3:
			j = stn['useZ'] + stn['useN'] + stn['useE']
			if j:
				values.append(i*n + j*n/2)
				labels.append(stn['code'])
				i += j
		else:
			if stn['useZ']:
				values.append(i*n + n/2)
				labels.append(stn['code']+' '+'Z')
				i += 1
			if stn['useN']:
				values.append(i*n + n/2)
				labels.append(stn['code']+' '+'N')
				i += 1
			if stn['useE']:
				values.append(i*n + n/2)
				labels.append(stn['code']+' '+'E')
				i += 1
	ax = plt.gca()
	ax.invert_yaxis()
	ax.xaxis.tick_top()
	mx = max(Cd.max(), abs(Cd.min()))
	cax = plt.matshow(Cd, fignum=1, cmap=plt.get_cmap('seismic'), vmin=-mx, vmax=mx)
	#cb = plt.colorbar(shift, shrink=0.6, extend='both', label='shift [s]')

	if colorbar:
		#cbar = plt.colorbar(cax, shrink=0.6, label='correlation [$\mathrm{m}^2\,\mathrm{s}^{-2}$]')
		cbar = plt.colorbar(cax, shrink=0.6, label='correlation [$\mathrm{mm}^2$]') # TODO

		#cbar = plt.colorbar(cax, ticks=[-mx, 0, mx])

	plt.xticks(values, labels, rotation='vertical')
	plt.yticks(values, labels)
	# Turn off all the ticks
	for t in ax.xaxis.get_major_ticks():
		t.tick1On = False
		t.tick2On = False
	for t in ax.yaxis.get_major_ticks():
		t.tick1On = False
		t.tick2On = False
	
	if outfile:
		plt.savefig(outfile.replace('$outdir', self.outdir), bbox_inches='tight')
	else:
		plt.show()
	plt.clf()
	plt.close('all')
