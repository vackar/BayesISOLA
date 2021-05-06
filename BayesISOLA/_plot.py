#! /usr/bin/env python3
# -*- coding: utf-8 -*-

from math import sin,cos,radians
import numpy as np
import matplotlib.pyplot as plt

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
	plt.plot(self.MT.centroid['y']/1e3, self.MT.centroid['x']/1e3, marker='*', markersize=75, color='yellow', label='epicenter', linestyle='None')
	
	L1 = L2 = L3 = True
	for sta in self.inp.stations:
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
		outfile = outfile.replace('$outdir', self.outdir)
		plt.savefig(outfile, bbox_inches='tight')
		self.plots['stations'] = outfile
	else:
		plt.show()
	plt.clf()
	plt.close()

def plot_covariance_matrix(self, outfile='$outdir/covariance_matrix.png', normalize=False, cholesky=False, fontsize=60, colorbar=False):
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
	Cd = np.zeros((self.data.components*self.data.npts_slice, self.data.components*self.data.npts_slice))
	if not len(self.cova.Cd):
		raise ValueError('Covariance matrix not set or not saved. Run "covariance_matrix(save_non_inverted=True)" first.')
	i = 0
	if cholesky and self.cova.LT3:
		matrix = self.cova.LT3
	elif cholesky:
		matrix = [item for sublist in self.cova.LT for item in sublist]
	else:
		matrix = self.cova.Cd
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
	n = self.data.npts_slice
	for stn in self.inp.stations:
		if cholesky and self.cova.LT3:
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
		outfile = outfile.replace('$outdir', self.outdir)
		plt.savefig(outfile, bbox_inches='tight')
		self.plots['covariance_matrix'] = outfile
	else:
		plt.show()
	plt.clf()
	plt.close('all')
