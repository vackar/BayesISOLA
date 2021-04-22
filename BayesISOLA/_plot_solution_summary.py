#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from obspy.imaging.beachball import beach#, beachball
#from obspy.imaging.mopad_wrapper import beach as beach2   # nofill parameter results in a black circle
import matplotlib.pyplot as plt

from BayesISOLA.MT_comps import a2mt, decompose
from BayesISOLA.histogram import histogram

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
